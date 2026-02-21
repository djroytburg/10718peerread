## Stable Minima of ReLU Neural Networks Suffer from the Curse of Dimensionality: The Neural Shattering Phenomenon

Tongtong Liang UC San Diego ttliang@ucsd.edu

Dan Qiao

UC San Diego d2qiao@ucsd.edu

Yu-Xiang Wang UC San Diego yuxiangw@ucsd.edu

## Abstract

Westudy the implicit bias of flatness / low (loss) curvature and its effects on generalization in two-layer overparameterized ReLU networks with multivariate inputs-a problem well motivated by the minima stability and edge-of-stability phenomena in gradient-descent training. Existing work either requires interpolation or focuses only on univariate inputs. This paper presents new and somewhat surprising theoretical results for multivariate inputs. On two natural settings (1) generalization gap for flat solutions, and (2) mean-squared error (MSE) in nonparametric function estimation by stable minima, we prove upper and lower bounds, which establish that while flatness does imply generalization, the resulting rates of convergence necessarily deteriorate exponentially as the input dimension grows. This gives an exponential separation between the flat solutions compared to low-norm solutions (i.e., weight decay), which are known not to suffer from the curse of dimensionality. In particular, our minimax lower bound construction, based on a novel packing argument with boundary-localized ReLU neurons, reveals how flat solutions can exploit a kind of 'neural shattering' where neurons rarely activate, but with high weight magnitudes. This leads to poor performance in high dimensions. We corroborate these theoretical findings with extensive numerical simulations. To the best of our knowledge, our analysis provides the first systematic explanation for why flat minima may fail to generalize in high dimensions.

## 1 Introduction

Modern deep learning is inherently overparameterized. In this regime, there are typically infinitely many global (i.e., zero-loss or interpolating) minima to the training objective, yet gradient-descent (GD) training seems to successfully avoid 'bad' minima, finding those that generalize well. Understanding this phenomenon boils down to understanding the implicit biases of training algorithms [Zhang et al., 2021]. A large body of work has focused on understanding this phenomenon in the interpolation regime [Du et al., 2018, Liu et al., 2022], and the related concept of 'benign overfitting' [Belkin et al., 2019, Bartlett et al., 2020, Frei et al., 2022].

While these directions have been fruitful, there is increasing evidence that rectified linear unit (ReLU) neural networks do not benignly overfit [Mallinar et al., 2022, Haas et al., 2023], particularly in the case of learning problems with noisy data [Joshi et al., 2023, Qiao et al., 2024]. Furthermore, for noisy labels, it takes many iterations of GD to actually interpolate the labels [Zhang et al., 2021]. This discounts theories based on interpolation to explain the generalization performance of practical neural networks, which would have entered the so-called edge-of-stability regime [Cohen et al., 2020], or stopped long before interpolating the training data.

Rahul Parhi UC San Diego rahul@ucsd.edu

Figure 1: The 'neural shattering' phenomenon: From empirical observations to its geometric origin and theoretical consequences. Left panel: Training with a large learning rate and gradient descent empirically results in 'neural shattering': Neurons develop large weights despite activating on very few inputs, leading to a high MSE of ≈ 1 . 105 (red points). In contrast, explicit ℓ 2 -regularization prevents this, achieving a much lower MSE of ≈ 0 . 055 (orange points). Middle panel: The number of distinct directions, or 'caps', on a high-dimensional sphere grows exponentially. Consequently, the data sites are spread thinly across these caps. This makes it trivial for a ReLU neuron to find a direction that isolates only a few data points. This sparse activation pattern allows neurons to use large weight magnitudes for this local fitting without impacting the global loss curvature, thus 'tricking' the flatness criterion. Right panel: Visualization of 'hard-to-learn' function from our minimax lower bound construction, built from localized ReLU neurons described in the middle panel.

<!-- image -->

To that end, it has been observed that an important factor that affects/characterizes the implicit bias of GD training is the notion of dynamical stability [Wu et al., 2018]. Intuitively, the (dynamical) stability of a particular minimum refers to the ability of the training algorithm to 'stably converge' to that minimum. The stability of a minimum is intimately related to the flatness of the loss landscape about the minima [Mulayoff et al., 2021]. A number of recent works have focused on understanding linear stability , i.e., the stability of an algorithm's linearized dynamics about a minimum, in order to characterize the implicit biases of training algorithms [Wu et al., 2018, Nar and Sastry, 2018, Mulayoff et al., 2021, Ma and Ying, 2021, Nacson et al., 2023]. Minima that exhibit linear stability are often referred to as stable minima. In particular, Mulayoff et al. [2021] and Nacson et al. [2023] focus on the interpolation regime of two-layer overparameterized ReLU neural networks in the univariate input and multivariate input settings, respectively. Roughly speaking, the main takeaway from their work is that stability / flatness in parameter space implies a bounded-variation-type of smoothness in function space.

Moving beyond the interpolation regime, Qiao et al. [2024] extend the framework of Mulayoff et al. [2021] and provide generalization and risk bounds for stable minima in the non-interpolation regime for univariate inputs. They show that for univariate nonparametric regression, the functions realized by stable minima cannot overfit in the sense that the generalization gap vanishes as the number of training examples grows. Furthermore, they show that the learned functions achieve near-optimal estimation error rates for functions of second-order bounded variation on an interval strictly inside the data support. While this work is a good start, it begs the questions of (i) what happens in the multivariate / high-dimensional case and (ii) what happens off of this interval (i.e., how does the network extrapolate ). Indeed, these are key to understanding the implicit bias of GD trained neural networks, especially since learning high dimensions seems to always amount to extrapolation [Balestriero et al., 2021]. These two questions motivate the present paper in which we provide a precise answer to the following fundamental question.

How well do stable minima of two-layer overparameterized ReLU neural networks perform in the high-dimensional and non-interpolation regime?

We provide several new theoretical results for stable minima in this scenario, which are corroborated by numerical simulations. Some of our findings are surprising given the current state of understanding of stable minima. Notably, we show that, while flatness does imply generalization, the resulting sample complexity grows exponentially with the input dimension. This gives an exponential separation between flat solutions and low-norm solutions (weight decay) which are known not to suffer from the curse of dimensionality [Bach, 2017, Parhi and Nowak, 2023b].

## 1.1 Contributions

In this paper, we provide new theoretical results for stable minima of two-layer ReLU neural networks, particularly in the high-dimensional and non-interpolation regime. Our primary contributions lie in the rigorous analysis of the generalization and statistical properties of stable minima and the resulting insights into their high-dimensional behavior. In particular, our contributions include the following.

1. We establish that the functions realized by stable minima are regular in the sense of a weighted variation norm (Theorem 3.2 and Corollary 3.3). This norm defines a data-dependent function class that captures the inductive bias of stable minima. 1 Furthermore, this regularity admits an analytic description as a form of weighted total variation in the domain of the Radon transform. These results synthesize and extend previous work [cf., Nacson et al., 2023, Qiao et al., 2024] by removing interpolation assumptions and generalizing them to multivariate inputs.
2. We analyze the generalization properties of stable minima in both a statistical learning setting and a nonparametric regression setting defined using the smoothness class above.
- We establish that stable minima provably cannot overfit in the sense that their generalization gap (i.e., a uniform convergence bound) tends to 0 as the number of training examples n →∞ at a rate n -1 2 d +4 up to logarithmic factors (Theorem 3.5).
- For high-dimensional ( d &gt; 1 ) nonparametric regression, we show that stable minima (up to logarithmic factors) achieve an estimation error rate, in mean-squared error (MSE), upper bounded by n -1 2 d +4 (Theorem 3.6).
- We prove a minimax lower bound of rate n -2 d +1 up to a constant (Theorem 3.7) on both the MSE and the generalization gap, which certifies that stable minima are not immune to the curse of dimensionality. This gives an exponential separation between flat solutions and low-norm solutions (weight decay) [Bach, 2017, Parhi and Nowak, 2023b].
- By specializing the MSE upper bound to the univariate case ( d = 1 ), we show that stable minima (up to logarithmic factors) achieve an upper bound of n -1 6 . Furthermore, by a construction specific to the univariate case, we have a sharper lower bound of n -1 2 when d = 1 . These results should be contrasted to those of Qiao et al. [2024], who derive matching upper and lower bounds of n -4 5 on an interval strictly inside the data support. Note that our results hold over the full domain, therefore capturing how the networks extrapolate. Thus, our results provide a more realistic characterization of the statistical properties of stable minima in the univariate case than in prior work.
3. In Section 4, we corroborate our theoretical results with extensive numerical simulations. As a by-product, we uncover and characterize a phenomenon we refer to as 'neural shattering' that is inherent to stable minima in high dimensions. This refers to the observation that each neuron in a flat solution has very few activated data points, which means that the activation boundaries of the ReLU neurons in the solutions shatter the data set into small pieces. This leads to poor performance in high dimensions. We also highlight that this observation exactly matches the construction of 'hard-to-learn' functions for our minimax lower bound. Thus, our empirical validation combined with our theoretical analysis offers fresh insights into how high-dimensionality impacts neural network optimization and generalization. Indeed, our results reveal a subtle mechanism that leads to poor performance specifically in high dimensions.

These results are based on two novel technical innovations in the analysis of minima stability in comparison to prior works, which we summarize below.

Statistical bounds on the full input domain. The data-dependent nature of the stable minima function class implies that there are regions of the input domain where neuron activations are sparse for stable minima. This is because the functions in this class have local smoothness that can become arbitrarily irregular near the boundary of the data support. This makes it challenging to study the statistical performance of stable minima in the irregular regions. This was bypassed in the univariate case by Qiao et al. [2024] by restricting their attention to an interval strictly inside the data support, completely ignoring these hard-to-handle regions. Our analysis overcomes this via a novel technique

1 More specifically, this quantity defines a seminorm which correspondingly defines a kind of Banach space of functions called a weighted variation space [DeVore et al., 2025].

that balances the error strictly inside the data support with the error close to the boundary. This allows us to establish meaningful statistical bounds on the full input domain.

ReLU-specific minimax lower bound construction. We develop a novel minimax lower bound construction (see proof of Theorem 3.7) using functions built from sums of ReLU neurons. These neurons are strategically chosen to have activation regions near the boundary of the input domain. This exploits the 'on/off' nature of ReLUs and high-dimensional geometry to create 'hard-to-learn' functions. The data-dependent weighting allows these sparsely active, high-magnitude neurons to exist within the stable minima function class. This construction is fundamentally different from classical nonparametric techniques and is tightly linked to our experimental findings on neural shattering (see Figure 1).

## 1.2 Related Work

Stable minima and function spaces. Many works have investigated characterizations of the implicit bias of GD training from the perspective of dynamical stability [Wu et al., 2018, Nar and Sastry, 2018, Mulayoff et al., 2021, Ma and Ying, 2021, Nacson et al., 2023, Wang et al., 2022, Qiao et al., 2024]. In particular, Mulayoff et al. [2021] characterized the function-space implicit bias of minima stability for two-layer overparameterized univariate ReLU networks in the interpolation regime. This was extended to the multivariate case by Nacson et al. [2023] and, in the univariate case, this was extended to the non-interpolation regime by Qiao et al. [2024] with the addition of generalization guarantees. In this paper, we extend these works to the high-dimensional and non-interpolation regime and characterize the generalization and statistical properties of stable minima.

Nonparametric function estimation with neural networks. It is well known that neural networks are minimax optimal estimators for a wide variety of functions [Suzuki, 2018, Schmidt-Hieber, 2020, Kohler and Langer, 2021, Parhi and Nowak, 2023b, Zhang and Wang, 2023, Yang and Zhou, 2024, Qiao et al., 2024]. Outside of the univariate work of Qiao et al. [2024], all prior works construct their estimators via empirical risk minimization problems. Thus, they do not incorporate the training dynamics that arise when training neural networks in practice. Thus, the results of this paper provide more practically relevant results on nonparametric function estimation, providing estimation error rates achieved by local minima that GD training can stably converge to.

Loss curvature and generalization. A long-standing theory to explain why overparameterized neural networks generalize well is that the flat minima found by GD training generalize well [Hochreiter and Schmidhuber, 1997, Keskar et al., 2017]. Although there is increasing theoretical evidence for this phenomenon in various settings [Ding et al., 2024, Qiao et al., 2024], there is also evidence that sharp minima can also generalize [Dinh et al., 2017]. Thus, this paper adds complementary results to this list where we establish that, while flatness does imply generalization for two-layer ReLU networks, the resulting sample complexity grows exponentially with the input dimension.

## 2 Preliminaries, Notation, and Problem Setup

We investigate learning with two-layer ReLU neural networks. Our focus is on understanding the generalization and statistical performance of solutions obtained through GD training, particularly those that are stable.

Neural networks. We consider two-layer ReLU neural networks with K neurons. Such a network implements a function f θ : R d → R of the form

<!-- formula-not-decoded -->

where θ = { K } ∪ { v k , w k , b k } K k =1 ∪ { β } denotes the collection of all neural network parameters, including the width K ∈ N . Here, v k ∈ R denotes the output-layer weights, w k ∈ R d denotes the input-layer weights, b k ∈ R denotes the input-layer biases, and β ∈ R denotes the output-layer bias.

Data fitting and loss function. We consider the problem of fitting the data D = { ( x i , y i ) } n i =1 , where x i ∈ R d and y i ∈ R . We consider the empirical risk minimization problem with squared-error loss L ( θ ) = 1 2 n ∑ n i =1 ( y i -f θ ( x i )) 2 .

Gradient descent and minima stability. We aim to minimize L ( · ) via GD training, i.e., we consider the iteration θ t +1 = θ t -η ∇ θ L ( θ t ) , for t = 0 , 1 , 2 , . . . , where η &gt; 0 is the step size / learning rate, ∇ θ denotes the gradient operator with respect to θ , ∇ 2 θ denotes the Hessian operator with respect to θ , and the iteration is initialized with some initial condition θ 0 . The analysis of these dynamics in generality is intractable in most cases. Thus, following the work of Wu et al. [2018], many works [e.g., Nar and Sastry, 2018, Mulayoff et al., 2021, Ma and Ying, 2021, Wang et al., 2022, Nacson et al., 2023, Qiao et al., 2024] have considered the behavior of this iteration using linearized dynamics about a minimum. Following Mulayoff et al. [2021], we consider the Taylor series expansion of the loss function about a minimum θ ⋆ . 2 That is,

<!-- formula-not-decoded -->

As the GD iteration approaches a minimum θ ⋆ , it is well approximated by the linearized dynamics

<!-- formula-not-decoded -->

Aminimum is said to be linearly stable if the GD iterates are 'trapped' once they enter a neighborhood of the minimum. See Wu et al. [2018], Ma and Ying [2021], or Chemnitz and Engel [2025] for various rigorous definitions of linear stability that have appeared in the literature. It turns out that stability is tightly connected to the flatness of the minimum. Indeed, many equivalences have been proven, e.g., Mulayoff et al. [2021, Lemma 1], Qiao et al. [2024, Lemma 2.2], or Chemnitz and Engel [2025, Section 2.3]. We have the following proposition from Chemnitz and Engel [2025, p. 7].

Proposition 2.1. Suppose that η &lt; 2 . A minimum θ ⋆ is linearly stable 3 if and only if

<!-- formula-not-decoded -->

Thus, we see that the stability of a minimum is equivalent to the flatness of the minimum under the assumption that the step size η satisfies η &lt; 2 . Thus, we make this assumption in the remainder of this paper. Given a data set D , we refer to the class of neural network parameters

<!-- formula-not-decoded -->

as the collection of flat / stable minima or flat / stable solutions. This parameter class is further motivated by empirical observations that GD often operates in the edge-of-stability regime , where λ max ( ∇ 2 θ L ( θ t )) hovers around 2 /η [Cohen et al., 2020, Damian et al., 2024].

## 3 Main Results

In this section, we characterize the implicit bias of stable solutions. It turns out that every function f θ , with θ ∈ Θ flat ( η ; D ) , is regular in the sense of a weighted variation norm. In particular, the weight function is a data-dependent quantity. This weight function reveals that neural networks can learn features that are intrinsic within the structure of the training data. To that end, given a data set D = { ( x i , y i ) } n i =1 ⊂ R d × R , we consider a weight function g : S d -1 × R → R , where S d -1 := { u ∈ R d : ∥ u ∥ = 1 } denotes the unit sphere. This weight is defined by g ( u , t ) := min { ˜ g ( u , t ) , ˜ g ( -u , -t ) } , where

<!-- formula-not-decoded -->

2 Technically, we require that the loss is twice differentiable at θ ⋆ . Due to the ReLU activation, there is a measure 0 set in the parameter space where this is not true. However, if we randomly initialize the weights with a density and use gradient descent with non-vanishing learning rate, then with probability 1 the GD iterations do not visit such non-differentiable points. For the interest of generalization bounds, the behaviors of non-differentiable points are identical to their infinitesimally perturbed neighbor, which is differentiable. For these reasons, this assumption will be implicitly assumed at each candidate θ in the remainder of the paper.

3 In particular, this holds for the definition of linear stability where µ ( θ ⋆ ) ≤ 0 in the notation of Chemnitz and Engel [2025, p. 7], which is a strictly weaker notion of linear stability than that of Wu et al. [2018] and Ma and Ying [2021] [see the discussion in Chemnitz and Engel, 2025, Appendix A].

Here, X is a random vector drawn uniformly at random from the training examples { x i } n i =1 . Note that the distribution P X from which { x i } n i =1 are drawn i.i.d. controls the regularity of g .

With this weight function in hand, we define a (semi)norm on functions of the form

<!-- formula-not-decoded -->

where R &gt; 0 , c ∈ R d , and c 0 ∈ R . Functions of this form are 'infinite-width' neural networks. We define the weighted variation (semi)norm as

<!-- formula-not-decoded -->

where, if there does not exist a representation of f in the form of (7), then the seminorm 4 is understood to take the value + ∞ . Here, M ( S d -1 × [ -R,R ]) denotes the Banach space of (Radon) measures and, for µ ∈ M ( S d -1 × [ -R,R ]) , ∥ µ ∥ M := ∫ S d -1 × [ -R,R ] d | µ | ( u , t ) is the measure-theoretic total-variation norm.

With this seminorm, we define the Banach space of functions V g ( B d R ) on the ball B d R := { x ∈ R d : ∥ x ∥ 2 ≤ R } as the set of all functions f such that | f | V g is finite. When g ≡ 1 , | · | V g and V g ( B d R ) coincide with the variation (semi)norm and variation norm space of Bach [2017].

Example 3.1. Since we are interested in functions defined on B d R , for a finite-width neural network f θ ( x ) = ∑ K k =1 v k ϕ ( w T k x -b k ) + β , we observe that it has the equivalent implementation as f θ ( x ) = ∑ J j =1 a j ϕ ( u T j x -t j ) + c T x + c 0 , where a j ∈ R , u j ∈ S d -1 , t j ∈ R , c ∈ R d , and c 0 ∈ R . Indeed, this is due to the fact that the ReLU is homogeneous, which allows us to absorb the magnitude of the input weights into the output weights (i.e., each a j = | v k j ∥ w k j ∥ 2 for some k j ∈ { 1 , . . . , K } ). Furthermore, any ReLUs in the original parameterization whose activation threshold 5 is outside B d R can be implemented by an affine function on B d R , which gives rise to the c T x + c 0 term in the implementation. If this new implementation is in 'reduced form', i.e., the collection { ( u j , t j ) } J j =1 are distinct, then we have that | f θ | V g = ∑ J j =1 | a j | g ( u j , t j ) .

This example reveals that this seminorm is a weighted path norm of a neural network and, in fact, coincides with the path norm when g ≡ 1 [Neyshabur et al., 2015]. It also turns out that the datadependent regularity induced by this seminorm is tightly linked to the flatness of a neural network minimum. We summarize this fact in the next theorem.

Theorem 3.2. Suppose that f θ is a two-layer neural network such that the loss L ( · ) is twice differentiable at θ . Then, | f θ | V g ≤ λ max ( ∇ 2 θ L ( θ )) 2 -1 2 +( R +1) √ 2 L ( θ ) .

The proof of this theorem appears in Appendix C. This theorem reveals that flatness implies regularity in the sense of the variation space V g ( B d R ) . In particular, we also have an immediate corollary for stable minima thanks to Proposition 2.1.

<!-- formula-not-decoded -->

The main takeaway messages from Theorem 3.2 and Corollary 3.3 are that flat / stable solutions are smooth in the sense of V g ( B d R ) . In particular, we see that the Banach space V g ( B d R ) is the natural function space to study stable minima. This framework provides the mathematical foundation and sets the stage to investigate the generalization and statistical performance of stable minima.

We also note that, from Corollary 3.3 and Example 3.1, for stable solutions f θ , as the step size η grows, the function f θ becomes smoother, eventually approaching an affine function as η →∞ . This can be viewed as an example of the simplicity bias phenomenon of GD training [Arpit et al., 2017, Kalimeris et al., 2019, Valle-Perez et al., 2019].

4 We use the notation | · | instead of ∥ · ∥ to highlight that this quantity is a seminorm. This quantity is a seminorm since affine functions are in its null space. See K˚ urková and Sanguineti [2001, 2002], Mhaskar [2004], Bach [2017], Siegel and Xu [2023], Shenouda et al. [2024] for more details about variation spaces.

5 The activation threshold of a neuron ϕ ( w T x -b ) is the hyperplane { x ∈ R d : w T x = b } .

Finally, we note that the function-space regularity induced by V g ( B d R ) has an equivalent analytic description via a weighted norm in the domain of the Radon transform of the function. This analytic description is based on the R -(semi)norm/second-order Radon-domain total variation inductive bias of infinite-width two-layer neural networks [Ongie et al., 2020, Parhi and Nowak, 2021, Bartolucci et al., 2023]. Before stating our theorem, we first recall the definition of the Radon transform. The Radon transform of a function f ∈ L 1 ( R d ) is given by

<!-- formula-not-decoded -->

where the integration is against the ( d -1) -dimensional Lebesgue measure on the hyperplane u T x = t . Thus, we see that the Radon transform integrates functions along hyperplanes.

Theorem 3.4. For every f ∈ V g ( B d R ) , consider the canonical extension 6 f ext : R d → R via its integral representation (7) . It holds that | f | V g = ∥ g · R ( -∆) d +1 2 f ext ∥ M , where fractional powers of the Laplacian are understood via the Fourier transform.

The proof of this theorem appears in Appendix D. We remark that the operators that appear in the theorem must be understood in the distributional sense (i.e., by duality). We refer the reader to Parhi and Unser [2024] for rigorous details about the distributional extension of the Radon transform. We also remark that a version of this theorem also appeared in Nacson et al. [2023, Theorem 1], but we note that their problem setting was the implicit bias of minima stability in the interpolation regime.

## 3.1 Stable Solutions Generalize But Suffer the Curse of Dimensionality

In the remainder of this paper, we focus on the scenario where inputs { x i } n i =1 are drawn i.i.d. uniformly from the unit ball B d 1 (i.e., R = 1 ). Under this assumption, the population version of the weight function, which we denote as g P , has a well-defined asymptotic behavior. As detailed in Appendix E, for | t | ≥ 1 , g P ( u , t ) = 0 , and as | t |→ 1 -, g P ( u , t ) ≍ (1 -| t | ) d +2 . While the actual weight function g in our analysis remains the empirical one derived from the data, this population behavior serves as a crucial analytical guide. Our proofs will show that the empirical g concentrates around this population version (Appendix E.2). For clarity in expressing our main results and their dependence on dimensionality, we will characterize the function space of stable minima with respect to a canonical weight function g ( u , t ) := (1 - | t | ) d +2 , which captures this essential asymptotic property.

With this in hand, we can now characterize the generalization gap of stable minima, which is defined to be the absolute difference between the training loss and the population risk. We are able to characterize the generalization gap under mild conditions on the joint distribution of the training examples and the labels.

Theorem 3.5. Let P denote the joint distribution of ( x , y ) . Assume that P is supported on B d 1 × [ -D,D ] for some D &gt; 0 and that the marginal distribution of x is Uniform( B d 1 ) . Fix a data set D = { ( x i , y i ) } n i =1 , where each ( x i , y i ) is drawn i.i.d. from P . Then, with probability ≥ 1 -δ we have that for the plug-in risk estimator ˆ R ( f ) := 1 n ∑ n i =1 ( f ( x i ) -y i ) 2

<!-- formula-not-decoded -->

where M := max { D, ∥ f θ ∥ L ∞ ( B d 1 ) , 1 } and ⪅ d hides constants (which could depend on d ) and logarithmic factors in n and (1 /δ ) . Furthermore, for any L ≥ D , it holds that

<!-- formula-not-decoded -->

6 Since functions in V g ( B d R ) are only defined on B d R , we must consider their extension to R d when working with the Radon transform. See Parhi and Nowak [2023b, Section IV] for more details.

where the inf is over all risk estimators, ≳ d hides constants (which could depend on d ), and the sup is over all distributions that satisfy the above hypotheses.

The proof of this theorem appears in Appendix F. While this theorem does show that as n →∞ , the generalization gap vanishes, it reveals that the sample complexity grows exponentially with the input dimension (as seen by the n -1 2 d +4 term in the upper bound and the n -2 d +1 term in the lower bound). This suggests that the curse-of-dimensionality is intrinsic to the stable minima set Θ flat ( η ; D ) -not an artifact of our mathematical analysis nor the naive plug-in empirical risk estimator being suboptimal. On the other hand, for low-norm solutions (in the sense that they minimize the weightdecay objective), it can be shown that the generalization gap decays at a rate of ˜ O ( n -1 4 ) , where ˜ O ( · ) hides logarithmic factors [cf., Bach, 2017, Parhi and Nowak, 2023b]. This uncovers an exponential gap between flat and low-norm solutions, and, in particular, that stable solutions suffer the curse of dimensionality. When d = 1 , this result also provides a strict generalization of Qiao et al. [2024, Theorem 4.3], as they measure the error strictly inside the input domain, rather than on the full input domain. Thus, our result also characterizes how stable solutions extrapolate .

## 3.2 Nonparametric Function Estimation With Stable Minima

We now turn to the problem of nonparametric function estimation. As we have seen that V g ( B d 1 ) is a natural model class for stable minima, this raises two fundamental questions: (i) How well do stable minima estimate functions in V g ( B d 1 ) from noisy data? and (ii) What is the best performance any estimation method could hope to achieve for functions in V g ( B d 1 ) . In this section we provide answers to both these questions by deriving a mean-squared error upper bound for stable minima and a minimax lower bound for this function class.

Theorem 3.6. Fix a step size η &gt; 0 and noise level σ &gt; 0 . Given a ground truth function f 0 ∈ V g ( B d 1 ) such that ∥ f 0 ∥ L ∞ ≤ B and | f 0 | V g ≤ ˜ O ( 1 η -1 2 +2 σ ) , suppose that we are given a data set y i = f 0 ( x i ) + ε i , where x i are i.i.d. Uniform( B d 1 ) and ε i are i.i.d. N (0 , σ 2 ) . Then, with probability ≥ 1 -δ , we have that

<!-- formula-not-decoded -->

for any θ ∈ Θ flat ( η ; D ) that is optimized, i.e., ( f θ ( x i ) -y i ) 2 ≤ ( f 0 ( x i ) -y i ) 2 , for i = 1 , . . . , n .

The proof of this theorem appears in Appendix G. This theorem shows that optimized stable minima incur an estimation error rate that decays as ˜ O ( n -1 2 d +4 ) , which suffers the curse of dimensionality. The optimized assumption is mild as it only asks that the error for each data point is smaller than the label noise σ 2 , which is easy to achieve in practice with GD training, especially in the overparameterized regime. The next theorem shows that the curse of dimensionality is actually necessary for this function class.

Theorem 3.7. Consider the same data-generating process as in Theorem 3.6. We have the following minimax lower bounds.

<!-- formula-not-decoded -->

where ≳ d hides constants (that could depend on d ).

The proof of this theorem appears in Appendix H. Our proof relies on two high-dimensional constructions. The first construction is to pack the unit sphere S d -1 with M = exp(Ω( d )) pairwise-disjoint spherical caps, each specified by a unit vector u i as its center. Then, for every center u i the ReLU neuron φ i ( x ) = cϕ ( u T i x -t ) is active only on its outward-facing cap, and attains its peak value min { B,C } by choosing a suitable t . The second construction is to observe that since the weight function g ( u , t ) decreases quickly as | t |→ 1 (see Appendix E), the regularity constraint | · | V g ≤ C allows us to combine an exponential number of such atoms to construct a family of 'hard-to-learn' functions. Traditional lower-bound constructions satisfy regularity by shrinking bump amplitudes

(vertical changes), whereas our approach fundamentally differs by shifting and resizing bump supports (horizontal changes). Our experiments reveal that stable minima actually favor these kinds of hard-to-learn functions and we refer to this as the neural shattering phenomenon.

## 4 Experiments

In this section, we empirically validate our claims that (i) stable minima are not immune to the curse of dimensionality and (ii) the 'neural shattering' phenomenon occurs. All synthetic data points are generated by uniformly sampling x from B d 1 and y i = f 0 ( x i ) + N (0 , σ 2 ) , where the ground-truth function f 0 ( x ) = w T x for some fixed vector w with ∥ w ∥ = 1 . All the models are two-layer ReLU neural networks with width four times the training data size. The networks are randomly initialized by the standard Kaiming initialization [He et al., 2015]. We also use gradient clipping with threshold 50 to avoid divergence for large learning rates. 7

Curse of dimensionality. In this experiment, we train neural networks with GD and vary the data set sizes in { 32 , 64 , 128 , 256 , 512 } and dimensions in { 1 , 2 , 3 , 4 , 5 } , with noise level σ = 1 . For each data set size, dimension, and training parameters ( η = 0 . 2 without weight decay and η = 0 . 01 with weight-decay 0 . 1 ), we conduct 5 experiments and take the median. The log-log curves are displayed in Figure 2.

Figure 2: Empirical validation of the curse of dimensionality. Left panel: The slope of log MSE versus log n for training with vanilla gradient descent rapidly decreases with dimension, falling to about 0.1 at d = 5 . Right panel: Training with ℓ 2 (weight decay) results in slopes above 0 . 5 in the log-log scale.

<!-- image -->

Neural shattering. As briefly illustrated in the right panel of Figure 1, Figure 3 presents more detailed experiments. We train a two-layer ReLU network of width 2048 on 512 noisy samples ( σ = 1 ) of a 10-dimensional linear target. Under a large step size η = 0 . 9 (no weight decay), gradient descent enters a flat / stable minimum ( λ max ( ∇ 2 θ L ( θ )) oscillates around 2 /η ≈ 2 . 2 , signaling edgeof-stability dynamics). This drastically reduces each neuron's data-activation rate to ≤ 10%, rather than reducing their weight norms. The network overfits (train MSE ≈ 1 . 105 , matching the noise level). In contrast, with η = 0 . 01 plus ℓ 2 -weight-decay λ = 0 . 1 , all neurons remain active and weight norms stay tightly bounded, so the model avoids overfitting (train MSE ≈ 0 . 055 ).

## 5 Discussion and Conclusion

This paper presents a nuanced conclusion on the link between minima stability and generalization: Stable solutions do generalize, but when data is distributed uniformly on a ball, this generalization ability is severely weakened by the Curse of Dimensionality (CoD). Our analysis pinpoints the mechanism behind this failure. The implicit regularization from GD is not uniform across the input

7 We monitor clipping during the training, and the clipping only occurs in the first 10 epochs. Gradient clipping does not prevent the training dynamics from entering edge-of-stability regime.

Figure 3: The top-left plot illustrates the neural shattering phenomenon: after large-step training each ReLU neuron (orange) is active on only a tiny fraction of the data (small horizontal support) yet its weight norm remains large, exactly as in our sphere-packing lower-bound construction where each outward-facing ReLU atom fires on very few inputs but retains full peak amplitude.

<!-- image -->

domain. While it imposes strong regularity in the strict interior of the data support, this guarantee collapses at the boundary. This localized failure of regularization is precisely what enables 'neural shattering', a phenomenon where neurons satisfy the stability condition not by shrinking their weights, but by minimizing their activation frequency. This causes the CoD: The intrinsic geometry of a high-dimensional ball provides an exponential increase in available directions for shattering to occur, while the boundary regularization simultaneously weakens exponentially as the input dimension d grows. This mechanism, confirmed by both our lower bounds and experiments, explains why stable solutions exhibit poor generalization in high dimensions.

Several simplifications limit the scope of these results. The theory treats only two-layer ReLU networks and relies on the idealized assumption that samples are drawn uniformly from the unit ball. For more general distributions, the induced weight function g inherits the full geometry of the data and becomes harder to describe and interpret. Understanding this effect, together with extending the analysis to deeper architectures and adaptive algorithms, will take substantial effort, which we leave as future work.

## 6 Acknowledgments

The research was partially supported by NSF Award # 2134214. The authors acknowledge early discussion with Peter Bartlett at the Simons Foundation that motivated us to consider the problem. Tongtong Liang thanks Zihan Shao for providing helpful suggestions on the implementation of the experiments.

## References

Devansh Arpit, Stanisław Jastrz˛ ebski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder S Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, et al. A closer look at memorization in deep networks. In International Conference on Machine Learning , pages 233-242. PMLR, 2017.

Richard Arratia, Larry Goldstein, and Louis Gordon. Two moments suffice for poisson approximations: the chen-stein method. Annals of Probability , 17(1):9-25, 1989.

- Francis Bach. Breaking the curse of dimensionality with convex neural networks. Journal of Machine Learning Research , 18(1):629-681, 2017.
- Randall Balestriero, Jerome Pesenti, and Yann LeCun. Learning in high dimension always amounts to extrapolation. arXiv preprint arXiv:2110.09485 , 2021.
- A. D. Barbour, L. Holst, and S. Janson. Poisson Approximation , volume 2 of Oxford Studies in Probability . Oxford University Press, 1992.
- Peter L Bartlett, Philip M Long, Gábor Lugosi, and Alexander Tsigler. Benign overfitting in linear regression. Proceedings of the National Academy of Sciences , 117(48):30063-30070, 2020.
- Francesca Bartolucci, Ernesto De Vito, Lorenzo Rosasco, and Stefano Vigogna. Understanding neural networks with reproducing kernel Banach spaces. Applied and Computational Harmonic Analysis , 62:194-236, 2023.
- Mikhail Belkin, Alexander Rakhlin, and Alexandre B Tsybakov. Does data interpolation contradict statistical optimality? In The 22nd International Conference on Artificial Intelligence and Statistics , pages 1611-1619. PMLR, 2019.
- Lucien Le Cam. An approximation theorem for the poisson binomial distribution. Pacific Journal of Mathematics , 10(4):1181-1197, 1960.
- Dennis Chemnitz and Maximilian Engel. Characterizing dynamical stability of stochastic gradient descent in overparameterized learning. Journal of Machine Learning Research , 26(134):1-46, 2025.
- Jeremy Cohen, Simran Kaur, Yuanzhi Li, J Zico Kolter, and Ameet Talwalkar. Gradient descent on neural networks typically occurs at the edge of stability. In International Conference on Learning Representations , 2020.
- Alex Damian, Eshaan Nichani, and Jason D Lee. Self-stabilization: The implicit bias of gradient descent at the edge of stability. In International Conference on Learning Representations , 2024.
- Ronald DeVore, Robert D. Nowak, Rahul Parhi, and Jonathan W. Siegel. Weighted variation spaces and approximation by shallow ReLU networks. Applied and Computational Harmonic Analysis , 74:101713, 2025.
- Lijun Ding, Dmitriy Drusvyatskiy, Maryam Fazel, and Zaid Harchaoui. Flat minima generalize for low-rank matrix recovery. Information and Inference: A Journal of the IMA , 13(2):iaae009, 2024.
- Laurent Dinh, Razvan Pascanu, Samy Bengio, and Yoshua Bengio. Sharp minima can generalize for deep nets. In International Conference on Machine Learning , pages 1019-1028. PMLR, 2017.
- Simon S Du, Xiyu Zhai, Barnabas Poczos, and Aarti Singh. Gradient descent provably optimizes over-parameterized neural networks. In International Conference on Learning Representations , 2018.
- Spencer Frei, Niladri S Chatterji, and Peter Bartlett. Benign overfitting without linearity: Neural network classifiers trained by gradient descent for noisy linear data. In Conference on Learning Theory , pages 2668-2703. PMLR, 2022.
- Moritz Haas, David Holzmüller, Ulrike Luxburg, and Ingo Steinwart. Mind the spikes: Benign overfitting of kernels and neural networks in fixed dimension. Advances in Neural Information Processing Systems , 36, 2023.
- David Haussler. Decision-theoretic generalizations of the pac model for neural net and other learning applications. Information and Computation , 100(1):78-150, 1992.
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision (ICCV) , pages 1026-1034, 2015.
- Sepp Hochreiter and Jürgen Schmidhuber. Flat minima. Neural computation , 9(1):1-42, 1997.

- Nirmit Joshi, Gal Vardi, and Nathan Srebro. Noisy interpolation learning with shallow univariate ReLU networks. In International Conference on Learning Representations , 2023.
- Dimitris Kalimeris, Gal Kaplun, Preetum Nakkiran, Benjamin Edelman, Tristan Yang, Boaz Barak, and Haofeng Zhang. SGD on neural networks learns functions of increasing complexity. Advances in Neural Information Processing Systems , 32, 2019.
- Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. In International Conference on Learning Representations , 2017.
- Michael Kohler and Sophie Langer. On the rate of convergence of fully connected deep neural network regression estimates. Annals of Statistics , 49(4):2231-2249, 2021.
- Vˇ era K˚ urková and Marcello Sanguineti. Bounds on rates of variable-basis and neural-network approximation. IEEE Transactions on Information Theory , 47(6):2659-2665, 2001.
- Vˇ era K˚ urková and Marcello Sanguineti. Comparison of worst case errors in linear and neural network approximation. IEEE Transactions on Information Theory , 48(1):264-275, 2002.
- Chaoyue Liu, Libin Zhu, and Mikhail Belkin. Loss landscapes and optimization in over-parameterized non-linear systems and neural networks. Applied and Computational Harmonic Analysis , 59: 85-116, 2022.
- Chao Ma and Lexing Ying. On linear stability of sgd and input-smoothness of neural networks. Advances in Neural Information Processing Systems , 34:16805-16817, 2021.
- Neil Mallinar, James Simon, Amirhesam Abedsoltan, Parthe Pandit, Misha Belkin, and Preetum Nakkiran. Benign, tempered, or catastrophic: Toward a refined taxonomy of overfitting. Advances in Neural Information Processing Systems , 35:1182-1195, 2022.
- Hrushikesh N. Mhaskar. On the tractability of multivariate integration and approximation by neural networks. Journal of Complexity , 20(4):561-590, 2004.
- Rotem Mulayoff, Tomer Michaeli, and Daniel Soudry. The implicit bias of minima stability: A view from function space. Advances in Neural Information Processing Systems , 34:17749-17761, 2021.
- Mor Shpigel Nacson, Rotem Mulayoff, Greg Ongie, Tomer Michaeli, and Daniel Soudry. The implicit bias of minima stability in multivariate shallow ReLU networks. In International Conference on Learning Representations , 2023.
- Kamil Nar and Shankar Sastry. Step size matters in deep learning. Advances in Neural Information Processing Systems , 31, 2018.
- Behnam Neyshabur, Russ R. Salakhutdinov, and Nati Srebro. Path-SGD: Path-normalized optimization in deep neural networks. Advances in Neural Information Processing Systems , 28, 2015.
- Greg Ongie, Rebecca Willett, Daniel Soudry, and Nathan Srebro. A function space view of bounded norm infinite width relu nets: The multivariate case. In International Conference on Learning Representations , 2020.
- Rahul Parhi and Robert D. Nowak. Banach space representer theorems for neural networks and ridge splines. Journal of Machine Learning Research , 22(43):1-40, 2021.
- Rahul Parhi and Robert D. Nowak. What kinds of functions do deep neural networks learn? Insights from variational spline theory. SIAM Journal on Mathematics of Data Science , 4(2):464-489, 2022.
- Rahul Parhi and Robert D. Nowak. Deep learning meets sparse regularization: A signal processing perspective. IEEE Signal Processing Magazine , 40(6):63-74, 2023a.
- Rahul Parhi and Robert D. Nowak. Near-minimax optimal estimation with shallow ReLU neural networks. IEEE Transactions on Information Theory , 69(2):1125-1139, 2023b.

- Rahul Parhi and Michael Unser. Distributional extension and invertibility of the k -plane transform and its dual. SIAM Journal on Mathematical Analysis , 56(4):4662-4686, 2024.
- Dan Qiao, Kaiqi Zhang, Esha Singh, Daniel Soudry, and Yu-Xiang Wang. Stable minima cannot overfit in univariate ReLU networks: Generalization by large step sizes. In Advances in Neural Information Processing Systems , volume 37, pages 94163-94208, 2024.
- Johannes Schmidt-Hieber. Nonparametric regression using deep neural networks with ReLU activation function. Annals of Statistics , 48(4):1875-1897, 2020.
- Joseph Shenouda, Rahul Parhi, Kangwook Lee, and Robert D. Nowak. Variation spaces for multioutput neural networks: Insights on multi-task learning and network compression. Journal of Machine Learning Research , 25(231):1-40, 2024.
- Jonathan W. Siegel and Jinchao Xu. Characterization of the variation spaces corresponding to shallow neural networks. Constructive Approximation , pages 1-24, 2023.
- Taiji Suzuki. Adaptivity of deep relu network for learning in besov and mixed smooth besov spaces: optimal rate and curse of dimensionality. arXiv preprint arXiv:1810.08033 , 2018.
- Alexandre B. Tsybakov. Introduction to Nonparametric Estimation . Springer Series in Statistics. Springer, New York, 1st edition, 2009. ISBN 9780387790511.
- Guillermo Valle-Perez, Chico Q. Camargo, and Ard A. Louis. Deep learning generalizes because the parameter-function map is biased towards simple functions. In International Conference on Learning Representations , 2019.
- Vladimir N. Vapnik. Statistical Learning Theory . Wiley, 1998.
- Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science . Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2018. ISBN 978-1108415194. doi: 10.1017/9781108231596.
- Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint , volume 48. Cambridge university press, 2019.
- Yuqing Wang, Minshuo Chen, Tuo Zhao, and Molei Tao. Large learning rate tames homogeneity: Convergence and balancing effect. In International Conference on Learning Representations (ICLR) , 2022.
- Larry Wasserman. Minimax theory lecture notes. https://www.stat.cmu.edu/~larry/=sml/ minimax.pdf , 2020. Accessed: 2025-05-21.
- Lei Wu, Chao Ma, and Weinan E. How SGD selects the global minima in over-parameterized learning: A dynamical stability perspective. Advances in Neural Information Processing Systems , 31, 2018.
- Yunfei Yang and Ding-Xuan Zhou. Optimal rates of approximation by shallow ReLU k neural networks and applications to nonparametric regression. Constructive Approximation , pages 1-32, 2024.
- Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. Communications of the ACM , 64(3):107-115, 2021.
- Kaiqi Zhang and Yu-Xiang Wang. Deep learning meets nonparametric regression: Are weightdecayed DNNs locally adaptive? In International Conference on Learning Representations , 2023.

## Supplementary Materials

| A Additional Experiments   | A Additional Experiments                                             | A Additional Experiments                                                                                                              | 15    |
|----------------------------|----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|-------|
|                            | A.1                                                                  | Empirical Evidence: High Dimensionality Yields Neural Shattering                                                                      | 15    |
|                            | A.2                                                                  | Neural Shattering and Learning Rate (dim=5) . . . . . . . . . . .                                                                     | 16    |
|                            | A.3                                                                  | Neural Shattering in the Underparametrized Regime . . . . . . . .                                                                     | 17    |
|                            | A.4                                                                  | Neural Shattering for GELU Networks . . . . . . . . . . . . . . .                                                                     | 18    |
|                            | A.5                                                                  | Empirical Analysis of the Curse of Dimensionality (I) . . . . . . .                                                                   | 19    |
|                            | A.6                                                                  | Empirical Analysis of the Curse of Dimensionality (II) . . . . . .                                                                    | 20    |
|                            | A.7                                                                  | Empirical Analysis of the Curse of Dimensionality (III) . . . . . .                                                                   | 22    |
| B                          | Overview of the Proofs                                               | Overview of the Proofs                                                                                                                | 24    |
|                            | B.1                                                                  | Proof Overview of Theorem 3.2. . . . . . . . . . . . . . . . . . .                                                                    | 24    |
|                            | B.2                                                                  | Proof Overview of Theorem 3.5 . . . . . . . . . . . . . . . . . .                                                                     | 25    |
|                            | B.3                                                                  | Proof Overview of Theorem 3.6 . . . . . . . . . . . . . . . . . .                                                                     | 25    |
|                            | B.4                                                                  | Proof Overview of Theorem 3.7 . . . . . . . . . . . . . . . . . .                                                                     | 26    |
|                            | B.5                                                                  | Discussion of the Proofs . . . . . . . . . . . . . . . . . . . . . .                                                                  | 27    |
| C                          | Proof of Theorem 3.2: Stable Minima Regularity                       | Proof of Theorem 3.2: Stable Minima Regularity                                                                                        | 28    |
| D                          | Proof of Theorem 3.4: Radon-Domain Characterization of Stable Minima | Proof of Theorem 3.4: Radon-Domain Characterization of Stable Minima                                                                  | 30    |
| E                          | Characterization of the Weight Function for the Uniform Distribution | Characterization of the Weight Function for the Uniform Distribution                                                                  | 30    |
|                            | E.1                                                                  | The Computation of the Population Weight Function . . . . . . .                                                                       | 30    |
|                            | E.2                                                                  | Empirical Process for the Weight Function . . . . . . . . . . . . .                                                                   | 34    |
| F                          | Proof of Theorem 3.5: Generalization Gap of Stable Minima            | Proof of Theorem 3.5: Generalization Gap of Stable Minima                                                                             | 36    |
|                            | F.1                                                                  | Definition of the Variation Space of ReLU Neural Networks . . .                                                                       | 36    |
|                            | F.2                                                                  | Metric Entropy and Variation Spaces . . . . . . . . . . . . . . . .                                                                   | 36    |
|                            | F.3                                                                  | Generalization Gap of Unweighted Variation Function Class . . .                                                                       | 37    |
|                            | F.4                                                                  | Concentration Property on the Ball: Uniform Distribution . . . . .                                                                    | 38    |
|                            | F.5                                                                  | Upper Bound of Generalization Gap of Stable Minima . . . . . .                                                                        | 38    |
| G                          | Proof of Theorem 3.6: Estimation Error Rate for Stable Minima        | Proof of Theorem 3.6: Estimation Error Rate for Stable Minima                                                                         | 42    |
|                            | G.1                                                                  | Computation of Local Gaussian Complexity . . . . . . . . . . . .                                                                      | 42    |
|                            | G.2                                                                  | Proof of the Estimation Error Upper Bound . . . . . . . . . . . .                                                                     | 43    |
| H                          | Proof of Theorem 3.7: Minimax Lower Bound                            | Proof of Theorem 3.7: Minimax Lower Bound                                                                                             | 44    |
|                            | H.1                                                                  | The Multivariate Case . . . . . . . . . . . . . . . . . . . . . . . .                                                                 | 44    |
|                            | H.2 H.3                                                              | Why Classical Bump-Type Constructions Are Ineffective . . . . . The Univariate Case . . . . . . . . . . . . . . . . . . . . . . . . . | 48 49 |
| I                          | Lower Bound on Generalization Gap                                    | Lower Bound on Generalization Gap                                                                                                     |       |
|                            | I.1                                                                  | Construction Can be Realized by Stable Minima                                                                                         |       |
|                            |                                                                      |                                                                                                                                       | 51    |
|                            | The Lower Bound                                                      | The Lower Bound                                                                                                                       | 51    |
| J                          | Technical Lemmas                                                     | Technical Lemmas                                                                                                                      | 56    |
|                            | J.1 Information-Theoretic tools . . . .                              | . . . . . . . . . . . . . . . . .                                                                                                     | 56    |
|                            | J.2                                                                  | Poissonization and Le Cam's Inequality . . . . . . . . . . . . . .                                                                    | 57    |

## A Additional Experiments

## A.1 Empirical Evidence: High Dimensionality Yields Neural Shattering

Figure 4: Comparison across input dimension d for a two-layer ReLU network of width 1024 trained on 512 samples for 20000 epochs with learning rate η = 0 . 5 . At d = 1 , all neurons extrapolate (0% active), while as d increases the fraction of neurons surviving training rises dramatically (up to 65% at d = 6 ). Simultaneously, the training loss monotonically decreases whereas the training MSE increases with d , demonstrating that neural shattering under large learning rates may be the key driver of the curse of dimensionality in stable minima.

<!-- image -->

## A.2 Neural Shattering and Learning Rate (dim=5)

<!-- image -->

Training Curves (

=0.4,  =0)

Figure 5: Effect of increasing learning rate η on shattering ( η × epochs = 10000 ): as η grows, the stability/flatness constraint forces an ever larger fraction of neurons to activate only on a small subset of the data (neural shattering). To further decrease the training loss, gradient descent correspondingly increases the weight norms of the remaining active neurons.

## A.3 Neural Shattering in the Underparametrized Regime

<!-- image -->

(d)

W

= 32

, weight decay

= 0

.

05

(e)

W

= 48

, weight decay

= 0

.

05

(f)

W

= 64

, weight decay

= 0

.

05

Figure 6: Persistence of neural shattering across width in the underparametrized regime. Each panel plots per-neuron activation rate versus weight norm after training a two-layer ReLU network on n = 1024 samples with learning rate η = 0 . 5 . Columns correspond to widths W ∈ { 32 , 48 , 64 } ; the top row uses no weight decay, and the bottom row uses mild decay ( λ = 0 . 05 ). The parameter count is Wd + W + 1 , so W = 64 is mildly overparameterized while W = 32 , 48 are strictly underparameterized. Across all widths, we observe clear neural shattering: neurons with smaller activation fractions carry larger weight norms. This monotone trend is especially visible in the stable-minima panels ( λ = 0 ), exactly as predicted by our theory. The weight-decay panels serve only as a high-activation baseline to calibrate what 'few activations' means, underscoring how exceptionally low the activation rates are at stable minima.

<!-- image -->

Figure 7: Each panel shows the relation between neuron activation rate and weight norm after training a two-layer ReLU network of width W = 16 on n = 1024 samples drawn from a linear target with learning rate η = 0 . 5 for d ∈ { 64 , 48 , 32 } . This observation indicates that neural shattering is a generic feature of stable minima, robust even when in small network.

## A.4 Neural Shattering for GELU Networks

This set of stable minima serves as a prism through which we can understand the emergent behaviors of the learning process. The data-dependent weight function g ( u , t ) , which is central to our analysis, arises directly from the structure of the loss Hessian and provides a static characterization of the implicit bias of gradient dynamics. A smaller value of g ( u , t ) for a neuron ( u , t ) implies that the stability condition imposes a weaker regularization, allowing for larger weight magnitudes for that neuron.

This static view is intimately connected to the underlying learning dynamics. In high-dimensional spaces, a neuron's activation boundary can easily drift to a region where it activates on only a small fraction of the data points. For such a neuron, the gradients it receives are sparse and localized. If the few data points it activates on are already well-fitted, the local gradient signal can vanish, causing the neuron's parameters to become effectively 'stuck' or stable. The small value of g ( u , t ) in these boundary regions creates 'space' within the class of stable functions for these trapped, high-magnitude, yet sparsely-activating neurons to exist, a possibility our lower bound construction then formalizes and exploits.

The ReLU activation function is analytically convenient for this analysis because its hard-sparsity property: a strictly zero gradient for non-activating inputs. This leads to a sparse loss Hessian, allowing for a clean derivation of the weight function g ( u , t ) . However, the underlying 'stuck neuron' dynamic is not necessarily unique to ReLU. Activations like GELU provide a non-zero gradient for negative inputs, but this signal is weak and decays quickly away from the activation boundary. It is therefore plausible that this weak gradient is insufficient to pull a 'stuck' neuron back from the data boundary once it has drifted there and its activation rate has diminished. This suggests that the fundamental mechanism enabling 'neural shattering' may persist. This hypothesis motivates an empirical investigation into whether the same phenomena of neural shattering also manifests in networks trained with GELU activations.

Figure 8: Comparison across input dimension d for a two-layer GELU network of width 1024 trained on 512 samples for 20000 epochs with learning rate η = 0 . 5 . The neural shattering behavior observed for ReLU networks in Figure 4 also appears here with GeLU activations. In particular, we can see the trend more clearly: neurons with lower activation rates tend to develop larger weight norms, highlighting that the neural shattering mechanism extends beyond piecewise-linear activations.

<!-- image -->

## A.5 Empirical Analysis of the Curse of Dimensionality (I)

We conduct the following experiments in the setting where the ground-truth function is linear with Gaussian noise σ 2 = 1 . The width of neural network is 4 times of the number of samples.

Figure 9: Log-log plots of the mean squared error (MSE) versus sample size n (Part I). Each curve is regressed by the median result over five random initializations (lighter markers), while the shallow markers denote the other runs. As the input dimension increases, the slope of the fitted regression line becomes progressively shallower, indicating slower error decay.

<!-- image -->

Figure 10: Log-log plots of the mean squared error (MSE) versus sample size n , illustrating the curse of dimensionality in stable minima (Part II). We can see in dimension 5, the slope is almost flat and even the large-step size cannot save the results (even worse than small step-size).

<!-- image -->

## A.6 Empirical Analysis of the Curse of Dimensionality (II)

We conduct the following experiments in the setting where the ground-truth function is linear with Gaussian noise σ 2 = 0 . 25 . The width of neural network is 2 times of the number of samples.

Figure 11: Log-log plots of the mean squared error (MSE) versus sample size n (Part III). Compared to the previous experiments, this setup reduces the noise level to σ = 0 . 5 , applies weight decay λ = 0 . 01 , and constrains the model width to 2 n .

<!-- image -->

Figure 12: Log-log plots of the mean squared error (MSE) versus sample size n (Part IV). The log-log MSE vs. n curves still exhibit progressively flattening slopes as the input dimension grows, demonstrating the enduring curse of dimensionality in stable minima.

<!-- image -->

## A.7 Empirical Analysis of the Curse of Dimensionality (III)

We conduct the following experiments in the setting where the ground-truth function is Hölder(1/2) f ( x ) = 1 d ∑ d i =1 | u T j x | 1 / 2 +1 with Gaussian noise σ 2 = 0 . 25 , where u j is uniformly sampled from S d -1 . The width of neural network is 2 times of the number of samples.

Figure 13: Log-log plots of the mean squared error (MSE) versus sample size n (Part V). We can see the generalization slopes of stable minima degrades as dimension increase from 1 to 4.

<!-- image -->

Figure 14: Log-log plots of the mean squared error (MSE) versus sample size n (Part VI). The panels on the left are the log-log plots for stable minima trained in η ∈ { 0 . 05 , 0 . 1 , 0 . 2 , 0 . 4 } , while the panels on the left are the log-log plots for low-norm solutions trained in weight decay λ = 0 . 01 .

<!-- image -->

Figure 15: Log-log plots of the mean squared error (MSE) versus sample size n (Part VII). The panels on the left are the log-log plots for stable minima trained in different learning rate η ∈ { 0 . 05 , 0 . 1 , 0 . 2 , 0 . 4 } , while The panels on the left are the log-log plots for low-norm solutions trained in weight decay λ = 0 . 01 .

<!-- image -->

## B Overview of the Proofs

In this section, we provide an overview of the proofs of the claims in the paper. The full proofs are deferred to later appendices. We introduce the following notations we use in our proofs and their overviews.

- Let φ ( ε ) and ψ ( ε ) be two functions in variable of ε . For constants a, b ∈ R (independent of ε ), the notation

<!-- formula-not-decoded -->

means that φ ( ε ) ≤ aψ ( ε ) and b ψ ( ε ) ≤ φ ( ε ) . We may directly use the notation ≍ if the constants are hidden (we may use the simplified version when the constants are justified).

- f ( x ) = O ( g ( x )) means there exist constants c &gt; 0 and x 0 &gt; 0 such that

<!-- formula-not-decoded -->

Intuitively, for sufficiently large x , f ( x ) grows at most as fast as g ( x ) , up to a constant factor. We may also use f ( x ) ≲ g ( x ) .

- f ( x ) = Ω( g ( x )) means there exist constants c ′ &gt; 0 and x 1 &gt; 0 such that

<!-- formula-not-decoded -->

Intuitively, for sufficiently large x , f ( x ) grows at least as fast as g ( x ) , up to a constant factor.

- f ( x ) = Θ( g ( x )) means there exist constants c 1 , c 2 &gt; 0 and x 2 &gt; 0 such that

<!-- formula-not-decoded -->

Equivalently,

<!-- formula-not-decoded -->

Intuitively, for sufficiently large x , f ( x ) grows at the same rate as g ( x ) , up to constant factors.

## B.1 Proof Overview of Theorem 3.2.

We consider the neural network of the form:

<!-- formula-not-decoded -->

The Hessian matrix of the loss function, obtained through direct computation, is expressed as:

<!-- formula-not-decoded -->

Consider v to be the unit eigenvector (i.e., ∥ v ∥ 2 = 1 ) corresponding to the largest eigenvalue of the matrix 1 n ∑ n i =1 ( ∇ θ f θ ( x i ))( ∇ θ f θ ( x i )) T . Consequently, the maximum eigenvalue of the Hessian of the loss can be lower-bounded as follows:

<!-- formula-not-decoded -->

Regarding (Term A), its maximum eigenvalue at a given θ can be related to the V g seminorm of the associated function f = f θ . Letting Ω = B d ( 0 , R ) , Nacson et al. [2023, Appendix F.2] demonstrate that:

<!-- formula-not-decoded -->

where ¯ w k = w k / ∥ w k ∥ 2 ∈ S d -1 , ¯ b k = b k / ∥ w k ∥ 2 and

<!-- formula-not-decoded -->

For (Term B), an upper bound can be established using the training loss L ( θ ) via the Cauchy-Schwarz inequality. This also employs a notable uniform upper bound for v T ∇ 2 θ f θ ( x n ) v , as detailed in Lemma C.1:

<!-- formula-not-decoded -->

## B.2 Proof Overview of Theorem 3.5

This proof establishes an upper bound on the generalization gap for stable minima in Θ flat ( η ; D ) . The strategy leverages the structural properties of these solutions, which are captured by a data-dependent weighted variation norm.

First, we recall from Corollary 3.3 that any stable solution f θ has a bounded norm, | f θ | V g ≤ A . The weight function g is determined by the training data. This data-dependent nature is central to our analysis. To bound the generalization gap, we must translate the constraint on the weighted norm | f | V g into a bound on the standard, unweighted norm | f | V . This is possible only in regions where the weight function g is bounded away from zero. This naturally suggests a decomposition of the input domain into two parts: a 'well-behaved' region where g has a positive lower bound, and the remaining region where g may be arbitrarily close to zero.

To facilitate a tractable analysis, we introduce the deterministic population weight function g P as a reference. We then bridge the two using empirical process theory. As established in Appendix E.2 (Theorem E.5), the uniform deviation between g and its population counterpart is bounded by a statistical error term ϵ n = ˜ O ( √ d/n ) with high probability. This allows us to leverage the wellbehaved properties of g P to characterize the behavior of the empirical function g .

For inputs from Uniform( B d 1 ) , this population function behaves like (1 -| t | ) d +2 (see Appendix E), where | t | is the distance of a neuron's activation boundary from the origin. This behavior motivates a specific geometric decomposition: an inner core B d 1 -ε (where | t | &lt; 1 -ε ) and an outer annulus A d ε .

For the inner core B d 1 -ε , the key step is to translate the bound on | f | V g to a bound on the standard (unweighted) norm | f | V . To do this, we need a lower bound on the empirical weight g within the core. Using g P as our analytical proxy, we establish that g min ≥ g P, min -ε n ≈ ε d +2 -ε n . This step requires a validity condition: ε must be large enough such that the geometric term ε d +2 dominates the statistical error ε n . With the unweighted norm now bounded, we utilize metric entropy arguments (e.g., Proposition F.4 and results from Parhi and Nowak [2023b]) to bound the generalization error in the core that scales with O ( ε -d ( d +2) 2 d +3 n -d +3 4 d +6 ) . In the annulus A d ε , the contribution is small, scaling with O ( ε ) .

## B.3 Proof Overview of Theorem 3.6

The proof for Theorem 3.6 establishes an upper bound on the mean squared error (MSE) for estimating a true function f 0 using a stable minimum f θ . The overall strategy shares similarities with the proof of the generalization gap, particularly in its treatment of the data-dependent function class.

The argument begins by leveraging the property that a stable minimum θ ∈ Θ flat ( η ; D ) corresponds to a neural network f θ with a bounded weighted variation norm | f θ | V g , where g is the empirical

weight function. The theorem also assumes the ground truth f 0 lies in a similar space. A key condition is that f θ is 'optimized' such that its empirical loss is no worse than that of f 0 . This is crucial as it allows us to bound the empirical MSE primarily by an empirical process term involving the noise terms ε i .

To bound this empirical process, the proof again decomposes the input domain B d 1 into a strict interior ball B d 1 -ε and an annulus A d ε . In the outer shell, the contribution to the MSE is controlled by the function's L ∞ bound. For the strict interior B d 1 -ε , we analyze the difference function f ∆ = f θ -f 0 . Consistent with our generalization analysis, we use the results from Appendix E.2 to ensure the empirical weight function g can be reliably analyzed via its population counterpart. This allows us to bound the unweighted variation norm of f ∆ over the core, which is then used to bound the empirical process via local Gaussian complexities (as detailed in Appendix G).

The bounds from the annulus and the core are then summed. The resulting expression for the total MSE is minimized by choosing an optimal ε . This balancing yields the final estimation error rate presented in Theorem 3.6, connecting the stability-induced regularity and the 'optimized' nature of the solution to its statistical performance.

## B.4 Proof Overview of Theorem 3.7

The proof establishes the minimax lower bound by constructing a packing set of functions within the specified function class V g ( B d 1 ) and then applying Fano's Lemma. The construction differs for multivariate ( d &gt; 1 ) and univariate ( d = 1 ) cases.

Multivariate Case ( d &gt; 1 ) The core idea is to use highly localized ReLU atoms that have a small V g norm due to the weighting g ( u , t ) vanishing near the boundary ( | t |→ 1 ), yet can be combined to form a sufficiently rich and separated set of functions.

Figure 16: The ReLU atoms only activate on the localized spherical cap and with L ∞ ( B d 1 ) -norm equal to 1 . As dimension increases, more data points will concentrate on the boundary region and the choice of directions increase exponentially.

<!-- image -->

1. Atom Construction: We utilize ReLU atoms Φ u ,ε 2 ( x ) = ε -2 ϕ ( u T x -(1 -ε 2 )) as defined in Construction H.4 (see Eq. (105) for the unnormalized version). These atoms are L ∞ -normalized, have an L 2 ( B d 1 ) -norm ∥ Φ u ,ε 2 ∥ L 2 ≍ ε d +1 2 (Lemma H.1), and a weighted variation norm | Φ u ,ε 2 | V g = ε 2 d +2 (Lemma H.2, Eq. (116)). The small V g norm is crucial.
2. Packing Set: Using a packing of K ≍ ε -( d -1) disjoint spherical caps on S d -1 (Lemma H.3), we construct a family of functions f ξ ( x ) = ∑ K i =1 ξ i Φ u i ,ε 2 ( x ) for ξ ∈ {-1 , 1 } K . By Varshamov-Gilbert lemma (Lemma J.2), we can find a subset Ξ ⊂ {-1 , 1 } K such that log | Ξ | ≍ K ≍ ε -( d -1) and for any distinct f ξ , f ξ ′ ∈ { f ζ } ζ ∈ Ξ , their L 2 -distance is ∥ f ξ -f ξ ′ ∥ L 2 ≳ ε . The total variation norm | f ξ | V g ≤ Kε 2 d +2 ≍ ε d +3 , which is significantly smaller than 1 when ε &lt; 1 .

3. Leveraging Fano's Lemma: (Proposition H.5) The KL divergence between distributions induced by f ξ and f ξ ′ is KL( P ξ ∥ P ξ ′ ) ≍ nε 2 /σ 2 . To apply Fano's Lemma (see Lemma J.1), we need to satisfy the condition (141) that nε 2 /σ 2 ≲ log | Ξ | ≍ ε -( d -1) , which implies ε ≍ ( σ 2 /n ) 1 d +1 and the minimax risk is then given by E ∥ ˆ f -f ∥ 2 L 2 ≳ ε 2 ≍ ( σ 2 /n ) 2 d +1 .

Univariate Case ( d = 1 ) The high-dimensional spherical cap packing is not applicable. Instead, we use scaled bump functions and exploit the simplified 1D V g norm.

1. Function Class: For d = 1 , if we assume f is smooth, then | f | V g = ∥ f ′′ · g ∥ M = ∫ 1 -1 | f ′′ ( x ) | (1 -| x | ) 3 d x (from Theorem 3.4 and leading to the class in Eq. (121)).
2. Atom Construction: We construct functions Φ k ( x ) as smooth bump functions, each supported on a distinct interval of width ε 2 near the boundary (e.g., x ∈ [1 -ε +( k -1) ε 2 , 1 -ε + kε 2 ] ). These are scaled such that ∥ Φ k ∥ L 2 ≍ ε . Due to the (1 -| x | ) 3 ≲ ε 3 factor in the V g norm and ∫ 1 -1 | Φ ′′ k ( x ) | d x ≍ 1 /ε 2 , the weighted variation is | Φ k | V g ≍ ε 3 · (1 /ε 2 ) = ε .
3. Packing Set: A family f ξ ( x ) = ∑ K k =1 ξ k Φ k ( x ) is formed with K ≍ 1 /ε terms. Using Varshamov-Gilbert (Lemma J.2), we find a subset Ξ with log | Ξ | ≍ K ≍ 1 /ε such that for distinct f ξ , f ξ ′ , the L 2 -distance is ∥ f ξ -f ξ ′ ∥ L 2 ≳ √ Kε ≍ √ 1 /ε · ε = √ ε .
4. Leveraging Fano's Lemma: The KL divergence is KL( P ξ ∥ P ξ ′ ) ≍ n ( √ ε ) 2 /σ 2 = nε/σ 2 . Fano's condition (141) nε/σ 2 ≲ log | Ξ | ≍ 1 /ε implies ε ≍ ( σ 2 /n ) 1 / 2 . The minimax risk is then E ∥ ˆ f -f ∥ 2 L 2 ≳ ( √ ε ) 2 = ε ≍ ( σ 2 /n ) 1 / 2 .

## B.5 Discussion of the Proofs

A notable feature in the proofs for the generalization gap upper bound (Theorem 3.5) and the MSE upper bound (Theorem 3.6) is the strategy of decomposing the domain B d 1 into an inner core B d 1 -ε and an annulus A d ε . This decomposition, involving a trade-off by treating the boundary region differently, is not merely a technical convenience but is fundamentally motivated by the characteristics of the function class V g ( B d 1 ) and the nature of 'hard-to-learn' functions within it.

The necessity for this approach is starkly illustrated by our minimax lower bound construction in Theorem 3.7 (see Appendix H for construction details) and Proposition I.1. The hard-to-learn functions used to establish this lower bound are specifically constructed using ReLU neurons that activate only near the boundary of the unit ball (i.e., for x such that u T x ≈ 1 ). The crucial insight here is the behavior of the weight function g ( u , t ) ≍ (1 - | t | ) d +2 (see Appendix E). For these boundary-activating neurons, | t | is close to 1 , making g ( u , t ) exceptionally small. This allows for functions that are potentially complex or have large unweighted magnitudes near the boundary (the annulus) to still possess a small weighted variation norm | f | V g , thus qualifying them as members of the function class under consideration. Our lower bound construction focuses almost exclusively on these boundary phenomena, as they represent the primary source of difficulty for estimation within this specific weighted variation space.

The upper bound proofs implicitly acknowledge this. By isolating the annulus A d ε , the analysis effectively concedes that this region might harbor complex behavior. The error contribution from this annulus is typically bounded by simpler means, often proportional to its small volume (controlled by ε ) and the L ∞ norm of the functions. The more sophisticated analysis, involving metric entropy or Gaussian complexity arguments (which depend on an unweighted variation norm that becomes large as | f | V g /ε d +2 when restricted to the strict interior B d 1 -ε ), is applied to the 'better-behaved' interior region. The parameter ε is then chosen optimally to balance the error from the boundary (which increases with ε ) against the error from the interior (where the complexity term effectively increases as ε shrinks).

This methodological alignment between our upper and lower bounds underscores a self-consistency in our analysis. Both components of the argument effectively exploit the geometric properties stemming from the uniform data distribution on a sphere and the specific decay characteristics of the data-dependent weight function g near the boundary. The strategy of 'sacrificing the boundary' in the upper bounds is thus a direct and necessary consequence of where the challenging functions identified by the lower bound constructions.

## C Proof of Theorem 3.2: Stable Minima Regularity

In this section, we prove the regularity constraint of stable minima. We begin by upper bounding the operator norm of the Hessian matrix. In other words, we upper bound | v T ∇ 2 θ f θ ( x ) v | under the constraint that ∥ v ∥ 2 = 1 .

Lemma C.1. Assume f θ ( x ) = ∑ K k =1 v k ϕ ( w T k x + b k ) + β is a two-layer ReLU network with input x ∈ R d such that ∥ x ∥ 2 ≤ R . Let θ represent all parameters { w k , b k , v k , β } K k =1 . Assume f θ ( x ) is twice differentiable with respect to θ at x . Then for any vector v corresponding to a perturbation in θ such that ∥ v ∥ 2 = 1 , it holds that:

<!-- formula-not-decoded -->

Proof. Let the parameters be θ = ( w T 1 , ..., w T K , b 1 , ..., b K , v 1 , ..., v K , β ) T . The total number of parameters is N = K × d + K + K +1 = K ( d +2)+1 . Let the corresponding perturbation vector be v ∈ R N , structured as: v = ( α T 1 , ..., α T K , δ 1 , ..., δ K , γ 1 , ..., γ K , ι ) T , where α k ∈ R d corresponds to w k , δ k ∈ R corresponds to b k , γ k ∈ R corresponds to v k , and ι ∈ R corresponds to β . The normalization constraint is

<!-- formula-not-decoded -->

̸

We need to compute the Hessian matrix ∇ 2 θ f θ ( x ) . Let z k = w T k x + b k and 1 k = 1( z k &gt; 0) . Since we assume twice differentiability, z k = 0 for all k , the Hessian ∇ 2 θ f θ ( x ) is block diagonal, with K blocks corresponding to each neuron. The k -th block, ∇ 2 ( θ k ) f θ ( x ) , involves derivatives with respect to θ k = ( w T k , b k , v k ) T . The relevant non-zero second partial derivatives defining this block are:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

All other second derivatives within the block are zero, as are derivatives between different blocks or involving β . The k -th block of the Hessian is thus:

<!-- formula-not-decoded -->

where 0 d × d is the d × d zero matrix and 0 d is the d -dimensional zero vector. The quadratic form v T ∇ 2 θ f θ ( x ) v becomes:

<!-- formula-not-decoded -->

Now, we bound the absolute value:

<!-- formula-not-decoded -->

Now we are ready to prove Theorem 3.2.

Proof of Theorem 3.2. Without loss of generality, we consider neural networks of the following form:

<!-- formula-not-decoded -->

The Hessian matrix of the loss function, obtained through direct computation, is expressed as:

<!-- formula-not-decoded -->

Let v be the unit eigenvector (i.e., ∥ v ∥ 2 = 1 ) corresponding to the largest eigenvalue of the matrix 1 n ∑ n i =1 ( ∇ θ f θ ( x i ))( ∇ θ f θ ( x i )) T , the maximum eigenvalue of the Hessian matrix of the loss can be lower-bounded as follows:

<!-- formula-not-decoded -->

Regarding (Term A), its maximum eigenvalue at a given θ can be related to the V g norm of the associated function f = f θ . Considering the domain B d R , Nacson et al. [2023, Appendix F.2] demonstrate that:

<!-- formula-not-decoded -->

where ¯ w k = w k / ∥ w k ∥ 2 ∈ S d -1 , ¯ b k = b k / ∥ w k ∥ 2 and

<!-- formula-not-decoded -->

For (Term B), an upper bound can be established using the training loss L ( θ ) via the Cauchy-Schwarz inequality. This also employs a notable uniform upper bound for | v T ∇ 2 θ f θ ( x n ) v | , as detailed in Lemma C.1:

<!-- formula-not-decoded -->

Finally, the proof of Theorem 3.2 is complete by plugging (27) and (29) into (26).

## D Proof of Theorem 3.4: Radon-Domain Characterization of Stable Minima

In this part, we prove Theorem 3.4 by extending the unweighted case to the weighted one.

Proof of Theorem 3.4. In the unweighted scenario, i.e., g ≡ 1 , it was established by Parhi and Nowak [2023b, Lemma 2] that if f ∈ V( B d R ) := V 1 ( B d R ) with integral representation

<!-- formula-not-decoded -->

where ν , c , and c 0 solve (8) (with g ≡ 1 ) that

<!-- formula-not-decoded -->

where we recall that f ext is the canonical extension of f from B d R to R d via the formula (30) and ν ∈ M ( S d -1 × R ) with supp ν = S d -1 × [ -R,R ] (i.e., we can identify ν with a measure in M ( S d -1 × [ -R,R ]) . Since the weighted variation seminorm | · | V g is simply (cf., (8))

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

g d +1

Remark D.1. The unweighted variation seminorm exactly corresponds to the second-order Radondomain total variation of the function [Ongie et al., 2020, Parhi and Nowak, 2021, 2022, 2023b,a]. Thus, the weighted variation seminorm is a weighted variant of the second-order Radon-domain total variation.

## E Characterization of the Weight Function for the Uniform Distribution

Recall that, given a data set D = { ( x i , y i ) } n i =1 ⊂ R d × R , we consider a weight function g : S d -1 × R → R , where S d -1 := { u ∈ R d : ∥ u ∥ = 1 } denotes the unit sphere. This weight is defined by g ( u , t ) := min { ˜ g ( u , t ) , ˜ g ( -u , -t ) } , where

<!-- formula-not-decoded -->

Here, X is a random vector drawn uniformly at random from the training examples { x i } n i =1 . Note that the distribution P X for which the { x i } n i =1 are drawn i.i.d. from controls the regularity of g .

In this section, We first analyze the properties of the population version g P by assuming that the random vector X is uniformly sampled from the d -dimensional unit ball B d 1 = { x ∈ R d : ∥ x ∥ 2 ≤ 1 } . Then we analyze the gap between the empirical g and the population g P using the empricial process.

## E.1 The Computation of the Population Weight Function

We focus on the marginal distribution of a single coordinate and related conditional expectations. Let X 1 be the first coordinate of X . Due to symmetry, all coordinates have the same marginal distribution.

The following proposition calculates the marginal probability density function of the first coordinate (and also other coordinates) of the random vector X .

Proposition E.1 (Marginal PDF of a Coordinate) . Let X follow the uniform distribution in B d 1 . The probability density function (PDF) of its first coordinate X 1 is given by:

<!-- formula-not-decoded -->

where α = d -1 2 and the normalization constant is

<!-- formula-not-decoded -->

Proof. The volume of the unit ball is V d = π d/ 2 Γ( d/ 2+1) . The uniform density is f X ( x ) = 1 /V d for x ∈ B d 1 . The marginal PDF is found by integrating out the other coordinates:

<!-- formula-not-decoded -->

where Vol d -1 ( R ) is the volume of a ( d -1) -ball of radius R . Using V d -1 = π ( d -1) / 2 Γ(( d -1) / 2+1) = π ( d -1) / 2 Γ(( d +1) / 2) , we get

<!-- formula-not-decoded -->

which simplifies to the stated result. For d = 2 , α = 1 / 2 , c 1 (2) = Γ(2) √ π Γ(3 / 2) = 1 √ π ( √ π/ 2) = 2 π . For d = 3 , α = 1 , c 1 (3) = Γ(5 / 2) √ π Γ(2) = 3 √ π/ 4 √ π = 3 4 .

Given the marginal probability density function, the tail probability follows from direct calculation.

Proposition E.2 (Tail Probability) . Let X be a random vector uniformly distributed in the d -dimensional unit ball B d 1 = { x ∈ R d : ∥ x ∥ 2 ≤ 1 } . Let X 1 be its first coordinate whose tail probability is defined as Q ( x ) = P ( X 1 &gt; x ) for x ∈ [ -1 , 1] . Then there exists a fixed x 0 ∈ [0 , 1) (specifically, we choose x 0 = 3 / 4 , which implies (1 -x ) ∈ (0 , 1 / 4] ) such that for all x ∈ [ x 0 , 1) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the constants c 2 ( d ) and c 3 ( d ) are given by:

<!-- formula-not-decoded -->

and c 1 ( d ) = Γ ( d 2 +1 ) √ π Γ ( d +1 2 ) is the normalization constant from the marginal PDF of X 1 .

Proof. The tail probability Q ( x ) is given by the integral of the marginal PDF f X 1 ( t ) = c 1 ( d )(1 -t 2 ) α for t ∈ [ -1 , 1] , where α = d -1 2 .

<!-- formula-not-decoded -->

Let s = t 2 , so d t = d s/ (2 √ s ) . The limits of integration for s become x 2 to 1 .

<!-- formula-not-decoded -->

Or equivalently,

Now, let u = 1 -s . Then s = 1 -u and d s = -d u . Let δ s = 1 -x 2 . The limits for u become δ s to 0 .

<!-- formula-not-decoded -->

For u ∈ [0 , δ s ] , we have 1 ≤ (1 -u ) -1 / 2 ≤ (1 -δ s ) -1 / 2 , because (1 -u ) is decreasing and non-negative. The integral ∫ δ s 0 u α d u = δ α +1 s α +1 . Substituting these bounds for the term (1 -u ) -1 / 2 :

<!-- formula-not-decoded -->

We use α = d -1 2 , so 2( α +1) = 2( d -1 2 +1) = 2( d +1 2 ) = d +1 . The exponent α +1 = d +1 2 . Thus,

<!-- formula-not-decoded -->

We choose x 0 = 3 / 4 , so we consider x ∈ [3 / 4 , 1) , which means (1 -x ) ∈ (0 , 1 / 4] . For x ∈ [3 / 4 , 1) , we have 1 + x ∈ [7 / 4 , 2) . The term δ s = 1 -x 2 = (1 -x )(1 + x ) . Given the range for 1 + x , for (1 -x ) ∈ (0 , 1 / 4] :

<!-- formula-not-decoded -->

- Lower Bound for Q ( x ) : Using δ s ≥ 7 4 (1 -x ) from the above range:

<!-- formula-not-decoded -->

This establishes the lower bound with c 2 ( d ) = c 1 ( d ) d +1 ( 7 4 ) d +1 2 .

- Upper Bound for Q ( x ) : For the term δ d +1 2 s , we use δ s &lt; 2(1 -x ) , so δ d +1 2 s &lt; (2(1 -x )) d +1 2 . For the term (1 -δ s ) -1 / 2 : Since (1 -x ) ∈ (0 , 1 / 4] , δ s &lt; 2(1 -x ) ≤ 2(1 / 4) = 1 / 2 . So, 1 -δ s &gt; 1 -1 / 2 = 1 / 2 . This implies (1 -δ s ) -1 / 2 &lt; (1 / 2) -1 / 2 = √ 2 . Combining these for the upper bound of Q ( x ) :

<!-- formula-not-decoded -->

This establishes the upper bound with c 3 ( d ) = c 1 ( d ) d +1 2 d +2 2 .

Thus, for x ∈ [3 / 4 , 1) (i.e., 1 -x ∈ (0 , 1 / 4] ):

<!-- formula-not-decoded -->

This corresponds to Q ( x ) ≍ c 3 ( d ) c 2 ( d ) (1 -x ) d +1 2 . The constants c 2 ( d ) and c 3 ( d ) depend only on the dimension d (via c 1 ( d ) and the exponents derived from d ) and are valid for the specified range of x .

Based on the tail probability, we calculate the expectation conditional on the tail events.

Proposition E.3 (Conditional Expectation) . For x ∈ [3 / 4 , 1) , the conditional expectation E [ X 1 | X 1 &gt; x ] is bounded by

<!-- formula-not-decoded -->

where the constants c 4 ( d ) and c 5 ( d ) are given by:

<!-- formula-not-decoded -->

Proof. We consider 1 -E [ X 1 | X 1 &gt; x ] = E [1 -X 1 | X 1 &gt; x ]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let I 1 ( x ) = ∫ 1 x (1 -t ) α +1 (1 + t ) α dt . We consider x ∈ [3 / 4 , 1) . For t ∈ [ x, 1] , we have 1 + t ∈ [1 + x, 2] . Since x ≥ 3 / 4 , 1 + x ≥ 7 / 4 . Thus, (7 / 4) α ≤ (1 + t ) α ≤ 2 α for t ∈ [ x, 1] (assuming α ≥ 0 , which holds for d ≥ 1 ). The integral ∫ 1 x (1 -t ) α +1 d t = [ -(1 -t ) α +2 α +2 ] 1 x = (1 -x ) α +2 α +2 . So, I 1 ( x ) is bounded by:

<!-- formula-not-decoded -->

Let N 1 ( x ) = c 1 ( d ) I 1 ( x ) . Then, using α +2 = ( d +3) / 2 :

<!-- formula-not-decoded -->

From Proposition E.2 , for x ∈ [3 / 4 , 1) , Q ( x ) is bounded by:

<!-- formula-not-decoded -->

where c 2 ( d ) = c 1 ( d ) d +1 ( 7 4 ) d +1 2 and c 3 ( d ) = c 1 ( d ) d +1 2 d +2 2 . Therefore, E [1 -X 1 | X 1 &gt; x ] = N 1 ( x ) Q ( x ) is bounded by:

- Lower bound:

<!-- formula-not-decoded -->

- Upper bound:

<!-- formula-not-decoded -->

So, for x ∈ [3 / 4 , 1) :

This implies:

<!-- formula-not-decoded -->

This completes the proof.

Finally, we combine the results and characterize the asymptotic behavior of the weight function g . Proposition E.4 (Asymptotic Behavior of g + ( x ) ) . Let the function g + ( x ) be defined as: for x ∈ ( -1 , 1) ,

<!-- formula-not-decoded -->

Then for x ∈ [3 / 4 , 1) , we have:

<!-- formula-not-decoded -->

where c ( g ) L ( d ) and c ( g ) U ( d ) are positive constants depending on dimension d , defined in the proof (39) .

<!-- formula-not-decoded -->

Proof. Let Q ( x ) = P ( X 1 &gt; x ) and E ( x ) = E [ X 1 | X 1 &gt; x ] . The function is g + ( x ) = Q ( x ) 2 · ( E ( x ) -x ) · √ 1 + E ( x ) 2 . Now, we establish precise bounds for x ∈ [3 / 4 , 1) . Let (1 -x ) be the variable.

1. Bounds for Q ( x ) 2 : From Propostion E.2, c 2 ( d )(1 -x ) d +1 2 ≤ Q ( x ) ≤ c 3 ( d )(1 -x ) d +1 2 . So, A L ( d )(1 -x ) d +1 ≤ Q ( x ) 2 ≤ A U ( d )(1 -x ) d +1 , where

<!-- formula-not-decoded -->

2. Bounds for E ( x ) -x = E [ X 1 -x | X 1 &gt; x ] : From Propostion E.3, we have (1 -x ) -c 5 ( d )(1 -x ) ≤ E ( x ) -x ≤ (1 -x ) -c 4 ( d )(1 -x ) . So, B L ( d )(1 -x ) ≤ E ( x ) -x ≤ B U ( d )(1 -x ) , where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since E ( x ) -x = E [ X 1 -x | X 1 &gt; x ] must be positive (as X 1 &gt; x ), we take B L ( d ) = max(0 , 1 -c 5 ( d )) .

3. Bounds for √ 1 + E ( x ) 2 : We know 1 -c 5 ( d )(1 -x ) ≤ E ( x ) ≤ 1 -c 4 ( d )(1 -x ) . For x ∈ [3 / 4 , 1) , (1 -x ) ∈ (0 , 1 / 4] and the upper bound of E ( x ) is given by

<!-- formula-not-decoded -->

The lower bound is also given in this way

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Combining these bounds, for x ∈ [3 / 4 , 1) :

<!-- formula-not-decoded -->

The constants are:

<!-- formula-not-decoded -->

## E.2 Empirical Process for the Weight Function

In this section, we discuss the empirical process of g P . Now we may relax the assumption by just assuming that X is random variable with supp( X ) ⊆ B d 1 . Fix dimension d ∈ N , sample size n ∈ N , and let X 1 , . . . , X n be i.i.d. copies of X . We use the notation ˆ g n to denote empirical weight function g as we defined previously.

For u ∈ S d -1 and t ∈ [ -1 , 1] , define

<!-- formula-not-decoded -->

Therefore, we deduce that

<!-- formula-not-decoded -->

and recall the population weight g P ( u , t ) . By the bounds 0 ≤ ( X T u -t ) + ≤ 2 and ∥ E [ X | X T u &gt; t ] ∥ ≤ 1 (valid on B d 1 ), we have the pointwise comparison

<!-- formula-not-decoded -->

i.e., there exist universal c, C ∈ (0 , ∞ ) such that c p s ≤ g P ≤ C p s for all ( u , t ) ∈ S d -1 × [ -1 , 1] . Consider the empirical plug-ins

<!-- formula-not-decoded -->

Note that ˆ g n involves no division by ˆ p n , hence avoids any small-mass instability. We now give a self-contained proof of a sharp, distribution-free uniform deviation bound.

Theorem E.5 (Distribution-free uniform deviation for ˆ g n ) . There exists a universal constant C &gt; 0 such that, for every δ ∈ (0 , 1) ,

<!-- formula-not-decoded -->

Proof. Using (40), it is enough (up to absolute constants) to control ∣ ∣ ˆ p n ( u , t )ˆ s n ( u , t ) -p ( u , t ) s ( u , t ) ∣ ∣ uniformly over ( u , t ) ∈ S d -1 × [ -1 , 1] . Observe that 0 ≤ s, ˆ s n ≤ 2 and 0 ≤ p, ˆ p n ≤ 1 , so

<!-- formula-not-decoded -->

We thus seek uniform bounds for the two empirical processes appearing on the right-hand side. The argument proceeds in two steps:

- Halfspaces. The class { x ↦→ 1 { x T u &gt; t } : u ∈ S d -1 , t ∈ R } has VC-dimension d +1 . Hence, by the VC uniform convergence inequality for { 0 , 1 } -valued classes (e.g., [Vapnik, 1998]), there exists a universal constant C 1 &gt; 0 such that, for all δ ∈ (0 , 1) ,

<!-- formula-not-decoded -->

- ReLU class. Let F := { f u ,t ( x ) = ( u T x -t ) + : u ∈ S d -1 , t ∈ [ -1 , 1] } . Since ∥ x ∥ ≤ 1 and t ∈ [ -1 , 1] , we have f ∈ [0 , 2] . Consider the subgraph family

<!-- formula-not-decoded -->

For y ≤ 0 membership is automatic; for y &gt; 0 it is equivalent to the affine halfspace condition u T x -t -y ≥ 0 in R d +1 . Thus VCdim( subG ( F )) ≤ d +2 , whence Pdim( F ) ≤ d + 2 . Standard pseudo-dimension bounds (see [Haussler, 1992, Thm. 3, 6, 7]) give a universal C 2 &gt; 0 with

<!-- formula-not-decoded -->

Combining Step (I) and Step (II) with a union bound and the previous inequality yields

<!-- formula-not-decoded -->

for an absolute constant C ′ &gt; 0 . Finally, the equivalence (40) transfers this bound to sup u ,t ∣ ∣ ˆ g n -g P ∣ ∣ , up to a universal multiplicative factor and the same failure probability.

## F Proof of Theorem 3.5: Generalization Gap of Stable Minima

Let P denote the joint distribution of ( x , y ) . Assume that P is supported on B d 1 × [ -D,D ] for some D &gt; 0 . Let f be a function. The population risk or expected risk of f is defined to be

<!-- formula-not-decoded -->

Let D = { ( x i , y i ) } n i =1 be a data set where each ( x i , y i ) is drawn i.i.d. from P . Then the empirical risk is defined to be

<!-- formula-not-decoded -->

The generalization gap is defined to be

<!-- formula-not-decoded -->

The generalization gap measures the difference between the train loss and the expected testing error. The smaller the generalization gap, the less likely the model overfits.

## F.1 Definition of the Variation Space of ReLU Neural Networks

Recall the notion in Section 3, the weighted variation (semi)norm is defined to be

<!-- formula-not-decoded -->

and now we define the unweighted variation norm or simply variation norm to be

<!-- formula-not-decoded -->

This definition is identical to the one in [Parhi and Nowak, 2023b, Section V.B]. The following example for unweighted variation norm is similar to Example 3.1.

Example F.1. Since we are interested in functions defined on B d R , for a finite-width neural network f θ ( x ) = ∑ K k =1 v k ϕ ( w T k x -b k ) + β , we observe that it has the equivalent implementation as f θ ( x ) = ∑ J j =1 a j ϕ ( u T j x -t j ) + c T x + c 0 , where a j ∈ R , u j ∈ S d -1 , t j ∈ R , c ∈ R d , and c 0 ∈ R . Indeed, this is due to the fact that the ReLU is homogeneous, which allows us to absorb the magnitude of the input weights into the output weights (i.e., each a j = | v k j ∥ w k j ∥ 2 for some k j ∈ { 1 , . . . , K } ). Furthermore, any ReLUs in the original parameterization whose activation threshold 8 is outside B d R can be implemented by an affine function on B d R , which gives rise to the c T x + c 0 term in the implementation. If this new implementation is in 'reduced form', i.e., the collection { ( u j , t j ) } J j =1 are distinct, then we have that | f θ | V = ∑ J j =1 | a j | .

The bounded variation function class is defined w.r.t. the unweighted variation norm.

Definition F.2. For the compact region Ω = B d R , we define the bounded variation function class as

<!-- formula-not-decoded -->

## F.2 Metric Entropy and Variation Spaces

Metric entropy quantifies the compactness of a set A in a metric space ( X,ρ X ) . Below we introduce the definition of covering numbers and metric entropy.

8 The activation threshold of a neuron ϕ ( w T x -b ) is the hyperplane { x ∈ R d : w T x = b } .

Definition F.3 (Covering Number and Entropy) . Let A be a compact subset of a metric space ( X,ρ X ) . For t &gt; 0 , the covering number N ( A,t, ρ X ) is the minimum number of closed balls of radius t needed to cover A :

<!-- formula-not-decoded -->

where B ( x i , t ) = { y ∈ X : ρ X ( y, x i ) ≤ t } . The metric entropy of A at scale t is defined as:

<!-- formula-not-decoded -->

The metric entropy of the bounded variation function class has been studied in previous works. More specifically, we will directly use the one below in future analysis.

Proposition F.4 (Parhi and Nowak 2023b, Appendix D) . The metric entropy of V C ( B d R ) (see Definition F .2) with respect to the L ∞ ( B d R ) -distance ∥ · ∥ ∞ satisfies

<!-- formula-not-decoded -->

where ⪅ d hides constants (which could depend on d ) and logarithmic factors.

## F.3 Generalization Gap of Unweighted Variation Function Class

As a middle step towards bounding the generalization gap of the weighted variation function class, we first bound the generalization gap of the unweighted variation function class according to a metric entropy analysis.

Lemma F.5. Let F M,C = { f ∈ V C ( B d R ) | ∥ f ∥ ∞ ≤ M } with M ≥ D where D refers to Theorem 3.5. Then with probability at least 1 -δ :

<!-- formula-not-decoded -->

Proof. According to Proposition F.4, one just needs N ( t ) balls to cover F in ∥· ∥ ∞ with radius t &gt; 0 such that where

<!-- formula-not-decoded -->

Then for any f, g ∈ F M,C and any ( x , y ) ,

<!-- formula-not-decoded -->

Hence replacing f by a centre f i within t changes both the empirical and true risks by at most 4 Mt .

For any fixed centre ¯ f in the covering, Hoeffding's inequality implies that with probability at least ≥ 1 -δ , we have

<!-- formula-not-decoded -->

because each squared error lies in [0 , 4 M 2 ] . Then we take all the centers with union bound to deduce that with probability at least 1 -δ/ 2 , for any center ¯ f in the set of covering index, we have

<!-- formula-not-decoded -->

According to the definition of covering sets, for any f ∈ F M,C , we have that ∥ f -¯ f ∥ ∞ ≤ t for some center ¯ f . Then we have

<!-- formula-not-decoded -->

After tuning t to be the optimal choice, we deduce that (54).

## F.4 Concentration Property on the Ball: Uniform Distribution

In the following analysis, we will handle the interior and boundary of the unit ball separately. In this part, we define the annulus of a ball rigorously and provide a high-probability bound on the number of samples falling in the annulus.

Definition F.6. Let B d 1 be the unit ball. The ε -annulus is a subset of B d 1 defined as

<!-- formula-not-decoded -->

and the closure of its complement is called ε -strict interior and denoted by I d ε .

Lemma F.7 (High-Probability Upper Bound on Annulus) . Let d ∈ N and ε ∈ (0 , 1) . Let

<!-- formula-not-decoded -->

Define n A := |{ i | x i ∈ A d ε }| and p = P ( X ∈ A d ε ) = 1 -(1 -ε ) d = Θ( ε ) . Then for any δ ∈ (0 , 1) , with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Proof. For each i = 1 , . . . , n , consider a Bernoulli random variable

<!-- formula-not-decoded -->

so that E [ U ] = p and regards U i as a sample. Then we may take n A = ∑ n i =1 U i . By the multiplicative Chernoff bound for the upper tail of a sum of independent Bernoulli variables,

<!-- formula-not-decoded -->

Set the right-hand side equal to δ and solve for γ :

<!-- formula-not-decoded -->

If γ &gt; 1 , note that trivially n A /n ≤ 1 ≤ p + √ 3 p ln(1 /δ ) n , so the claimed bound holds in all cases. Otherwise, plugging this choice of γ into the Chernoff bound gives

<!-- formula-not-decoded -->

i.e. with probability at least 1 -δ ,

<!-- formula-not-decoded -->

and dividing by n yields the stated inequality.

## F.5 Upper Bound of Generalization Gap of Stable Minima

Let f = f θ be a stable solution of the loss function L ( θ ) , trained by gradient descent with learning rate η . Then we have

<!-- formula-not-decoded -->

For (Term A), we have

<!-- formula-not-decoded -->

For (Term B), we have

<!-- formula-not-decoded -->

Let M = max {∥ f ∥ ∞ , D, 1 } . Then we have

<!-- formula-not-decoded -->

Combining these inequalities together, we may deduce that

<!-- formula-not-decoded -->

With all the preparations, we are ready to prove the generalization gap upper bound for stable minima. Theorem F.8. (First part of Theorem 3.5) Let P denote the joint distribution of ( x , y ) . Assume that P is supported on B d 1 × [ -D,D ] for some D &gt; 0 and that the marginal distribution of x is Uniform( B d 1 ) . Fix a data set D = { ( x i , y i ) } n i =1 , where each ( x i , y i ) is drawn i.i.d. from P , and D yields the empirical weight function g defined in (6) . Then, with probability at least 1 -δ , we have that for the plug-in risk estimator ˆ R ( f ) := 1 n ∑ n i =1 ( f ( x i ) -y i ) 2 ,

<!-- formula-not-decoded -->

where B is assumed &gt; 1 and ⪅ d hides constants (which could depend on d ) and logarithmic factors in n and (1 /δ ) . In particular, Theorem 3.2 and (62) imply that that

<!-- formula-not-decoded -->

for every

<!-- formula-not-decoded -->

Therefore, we may conclude that

<!-- formula-not-decoded -->

where M := max { D, ∥ f θ ∥ L ∞ ( B d 1 ) , 1 } .

Proof. For any fixed ε &lt; 1 / 4 , we may decompose B d 1 into ε -annulus and ε -strict interior

<!-- formula-not-decoded -->

According to the law of total expectation, the population risk is decomposed into

<!-- formula-not-decoded -->

where E A means that { x , y } is a new sample from the data distribution conditioned on x ∈ A d ε and E I means that ( x , y ) is a new sample from the data distribution conditioned on x ∈ I d ε .

Similarly, we also have this decomposition for empirical risk

<!-- formula-not-decoded -->

where I is the set of data points with x i ∈ I d ε and A is the set of data points with x i ∈ A d ε . Then the generalization gap can be decomposed into

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using the property that the marginal distribution of x is Uniform( B d 1 ) and its concentration property (see Lemma F.7), with probability at least 1 -δ/ 2 :

<!-- formula-not-decoded -->

where ⪅ d hides the constants that could depend on d and logarithmic factors of 1 /δ .

For the term (69), with probability 1 -δ/ 3

<!-- formula-not-decoded -->

so we may also conclude that

<!-- formula-not-decoded -->

For the part of the interior (70), the scalar P ( x ∈ I d ε ) is less than 1 with high-probability. Therefore, we just need to deal with the term

<!-- formula-not-decoded -->

Since both the distribution and sample points only support in I d ε , we may consider f by its restrictions in I d ε , which are denoted by f ε . Furthermore, according to the definition, we have

<!-- formula-not-decoded -->

where the Annulus ReLU term is totally linear in the strictly interior i.e. there exists c ′ , b ′ such that

<!-- formula-not-decoded -->

Therefore, we may write

<!-- formula-not-decoded -->

The core of the argument is to rigorously bound the interior generalization gap. Recall that a stable minima θ ∈ Θ flat ( η ; D ) satisfies | f | V g ≤ A with respect to the empirical weight function g . To analyze the complexity of its restriction f ε on the core I d ε , we need a lower bound on g ε min := inf | t |≤ 1 -ε g ( u , t ) . This quantity is a random variable.

From empirical process we discussed in Section E.2, especially Theorem E.5, we know that with probability at least 1 -δ/ 3 ,

<!-- formula-not-decoded -->

This implies a lower bound on the empirical minimum weight in the core with probability at least 1 -δ/ 3 ,

<!-- formula-not-decoded -->

Here, g ε P, min ≍ ε d +2 is the minimum of the population weight function in the core.

For the bound | f ε | V ≤ A/g ε min ≤ A/ ( g P, min -ϵ n ) to be meaningful with high probability, we must operate in a regime where g min ≥ ϵ n . We enforce a stricter validity condition for our proof

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Under this condition, we have g ε min ≥ g ε P, min -ϵ n ≥ g ε P, min / 2 ≍ ε d +2 . Thus, for any stable solution f , its restriction f ε has a controlled unweighted variation norm with high probability:

<!-- formula-not-decoded -->

We can now apply the generalization bound from Lemma F.5 to the class V C ε ( B d 1 -ε ) by plugging in (81), with probability 1 -δ/ 3 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ⪅ d hides the constants that could depend on d and logarithmic factors of 1 /δ .

Now we combine the upper bounds (71), (73) and (84) to deduce an upper bound of the generalization gap. With probability 1 -δ , we have

<!-- formula-not-decoded -->

Since n -1 2 d +4 &gt; n -3 4 d +6 and B 2 &gt; B 3( d +2) 2 d +3 with the assumption M ≥ 1 , we conclude that

<!-- formula-not-decoded -->

which finishes the proof.

Remark F.9. For the generalization gap lower bound (second part of Theorem 3.5), we defer the proof to Appendix I as it relies on a construction that is used to prove Theorem 3.7 from Appendix H.

Therefore, we may choose

## G Proof of Theorem 3.6: Estimation Error Rate for Stable Minima

## G.1 Computation of Local Gaussian Complexity

It is known from Wainwright 2019 that a tight analysis of MSE results from local gaussian complexity . We begin with the following proposition that connects the local gaussian complexity to the critical radius.

Proposition G.1 (Wainwright 2019, Chapter 13) . Let F be a convex model class that contains the constant function 1 . Fix design points x 1 , . . . , x n in the region of interest and denote the empirical norm

For any radius r &gt; 0 write

<!-- formula-not-decoded -->

where ε 1 , . . . , ε n i.i.d. ∼ N (0 , σ 2 ) and G n ( r, F ) := E ̂ G n ( r, F ) .

If δ satisfies the integral inequality

<!-- formula-not-decoded -->

where ∂ F := { f 1 -f 2 : f 1 , f 2 ∈ F } , then the local empirical Gaussian complexity obeys

<!-- formula-not-decoded -->

Moreover, with probability at least 1 -δ one has

<!-- formula-not-decoded -->

As a result, we can derive an upper bound for the local empirical Gaussian complexity of the variation function class through a careful analysis of the critical radius.

Lemma G.2. Let F B,C ( B d R ) = { f ∈ V C ( B d R ) | ∥ f ∥ L ∞ ( B d R ) ≤ B } . Then with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

for any two f 1 , f 2 ∈ F B,C

Proof. As ∂ F B,C = 2 F B,C ⊂ F 2 B, 2 C , bounding the entropy of F 2 B, 2 C suffices. Using ∥ f ∥ n ≤ ∥ f ∥ L ∞ ( B d 1 ) and referring to Proposition F.4, we have, up to logarithmic factors,

<!-- formula-not-decoded -->

Plugging this entropy bound into the left side of (87) and integrating,

<!-- formula-not-decoded -->

Hence inequality (87) is met provided

<!-- formula-not-decoded -->

Solving for r 2 (and keeping only dominant terms) yields

<!-- formula-not-decoded -->

With this choice of r n , Proposition G.1 guarantees

<!-- formula-not-decoded -->

and the high-probability version (89) holds verbatim.

<!-- formula-not-decoded -->

## G.2 Proof of the Estimation Error Upper Bound

Given the local gaussian complexity upper bound, together with the assumption of solutions being 'optimized', we can prove the following MSE upper bound.

Theorem G.3 (Restate Theorem 3.6) . Fix a step size η &gt; 0 and noise level σ &gt; 0 . Given a ground truth function f 0 ∈ V g ( B d 1 ) such that ∥ f 0 ∥ L ∞ ≤ B and | f 0 | V g ≤ ˜ O ( 1 η -1 2 +2 σ ) , suppose that we are given a data set y i = f 0 ( x i ) + ε i , where x i are i.i.d. Uniform( B d 1 ) and ε i are i.i.d. N (0 , σ 2 ) . Then, with probability at least 1 -δ , we have that

<!-- formula-not-decoded -->

for any θ ∈ Θ flat ( η ; D ) that is optimized, i.e., ( f θ ( x i ) -y i ) 2 ≤ ( f 0 ( x i ) -y i ) 2 , for i = 1 , . . . , n . Here, ⪅ d hides constants (that could depend on d ) and logarithmic factors in n and (1 /δ ) .

Proof of Theorem 3.6. The empirical Mean Squared Error (MSE) we want to bound is MSE( f θ , f 0 ) = 1 n ∑ n i =1 ( f θ ( x i ) -f 0 ( x i )) 2 .

First, we establish bounds on the regularity of f θ ( x ) -f 0 ( x ) . The condition that f θ is "optimized" means ( f θ ( x i ) -y i ) 2 ≤ ( f 0 ( x i ) -y i ) 2 for all i . Summing over i and dividing by n , we have 1 n ∑ n i =1 ( f θ ( x i ) -y i ) 2 ≤ 1 n ∑ n i =1 ( f 0 ( x i ) -y i ) 2 = 1 n ∑ n i =1 ε 2 i . Since ε i ∼ N (0 , σ 2 ) , E [ 1 n ∑ n i =1 ε 2 i ] = σ 2 . Standard concentration inequalities (e.g., for sums of χ 2 (1) scaled variables) show that 1 n ∑ n i =1 ε 2 i ≲ σ 2 with high probability (hiding logarithmic factors in 1 /δ , which are absorbed into ⪅ d ). Thus, 2 L ( θ ) ≲ σ 2 . For θ ∈ Θ flat ( η ; D ) , by Corollary 3.3 (with R = 1 for B d 1 , so R +1 = 2 ), we have

<!-- formula-not-decoded -->

Let C := 1 η -1 2 +2 σ . The theorem assumes | f 0 | V g ≤ C . Thus, we have | f 0 | V g ≲ C . The difference f θ ( x ) -f 0 ( x ) then satisfies

<!-- formula-not-decoded -->

Also, ∥ f θ -f 0 ∥ L ∞ ( B d 1 ) ≤ ∥ f θ ∥ L ∞ ( B d 1 ) + ∥ f 0 ∥ L ∞ ( B d 1 ) ≤ B + B = 2 B .

The optimized condition ( f θ ( x i ) -y i ) 2 ≤ ( f 0 ( x i ) -y i ) 2 implies (( f θ ( x i ) -f 0 ( x i )) -ε i ) 2 ≤ ε 2 i . Expanding this gives ( f θ ( x i ) -f 0 ( x i )) 2 -2( f θ ( x i ) -f 0 ( x i )) ε i + ε 2 i ≤ ε 2 i , which simplifies to

<!-- formula-not-decoded -->

This inequality is crucial and holds for each data point.

We decompose the MSE based on the location of data points. Let A d ε := { x ∈ B d 1 : ∥ x ∥ 2 ≥ 1 -ε } be the annulus and B d 1 -ε be the inner core. Let S A := { i : x i ∈ A d ε } and S I := { i : x i ∈ B d 1 -ε } . The total empirical MSE is

<!-- formula-not-decoded -->

The contribution from the shell, MSE S , is bounded using the L ∞ norm of f θ -f 0 and the concentration of points in the shell. Let n A := | S A | . By Lemma F.7, n A /n ⪅ ε with high probability.

<!-- formula-not-decoded -->

For the inner core's contribution, MSE I , we use Equation (94):

<!-- formula-not-decoded -->

Let n I := | S I | . The empirical process term is 2 n I n ( 1 n I ∑ i ∈ S I ( f θ ( x i ) -f 0 ( x i )) ε i ) . The function f θ -f 0 restricted to B d 1 -ε has an unweighted variation norm. As shown in Appendix E, for x ∈ Uniform( B d 1 ) , the population weight function g P ( u , t ) ≍ (1 -| t | ) d +2 . For activation hyperplanes relevant to B d 1 -ε (i.e., | t | ≤ 1 -ε ), g P ( u , t ) ≳ ε d +2 . Thus, the unweighted variation of f θ -f 0 on B d 1 -ε is

<!-- formula-not-decoded -->

We apply Lemma G.2 to bound 1 n I ∑ i ∈ S I ( f θ ( x i ) -f 0 ( x i )) ε i . The function h ( x ) = f θ ( x ) -f 0 ( x ) has unweighted variation ≲ C/ε d +2 and L ∞ norm ≤ 2 B . Therefore, we have that

<!-- formula-not-decoded -->

Combining the bounds for MSE S and MSE I :

<!-- formula-not-decoded -->

Similarly to the proof of Theorem 3.5 in Appendix F.5, we require that

<!-- formula-not-decoded -->

to filling the gap between the empirical weighted function g and the population g P with high probability, becase with high probability,

<!-- formula-not-decoded -->

where ⪅ d hides constants (which could depend on d ) and logarithmic factors, as stated by by Theorem E.5 in Section E.2. Therefore, we may choose

<!-- formula-not-decoded -->

and plug it into (100) to have

<!-- formula-not-decoded -->

Since 1 2 d +4 &lt; 3 2 d +3 , we conclude that

<!-- formula-not-decoded -->

which completes the proof.

## H Proof of Theorem 3.7: Minimax Lower Bound

## H.1 The Multivariate Case

In this section, we assume that d &gt; 1 and all the norms and semi-norms are restricted to the unit ball B d 1 . Let u ∈ S d -1 be a unit vector. Let ε ∈ R + be a constant with ε ≤ 1 / 2 . Consider the ReLU atom:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma H.1. The L 2 -norm of φ u ,ε 2 over B d 1 is given by

<!-- formula-not-decoded -->

where c 7 ( d ) and c 8 ( d ) are constants depends on d (the conconcrete definition is (113) ). Recall that ≍ c 8 ( d ) c 7 ( d ) means

<!-- formula-not-decoded -->

Proof. The squared L 2 norm of φ u ,ε 2 over the unit ball B d 1 is defined as:

<!-- formula-not-decoded -->

Substituting the definition of φ ε 2 ( w , x ) and using the property of the ReLU function that ϕ ( z ) = z for z &gt; 0 and ϕ ( z ) = 0 for z ≤ 0 , we get:

<!-- formula-not-decoded -->

To simplify the integral, we perform a rotation of the coordinate system such that w aligns with the d -th standard basis vector e d = (0 , . . . , 0 , 1) . In these new coordinates, u T x = X d . The unit ball remains the unit ball under rotation. The integral becomes:

<!-- formula-not-decoded -->

We can write the volume element d X as d X ′ d X d , where X ′ ∈ R d -1 represents the first d -1 coordinates. The condition X ∈ B d 1 translates to ∥ X ′ ∥ 2 2 + X 2 d ≤ 1 . The integral can be written as an iterated integral:

<!-- formula-not-decoded -->

The inner integral is over a ( d -1) -dimensional ball in R d -1 with radius R = √ 1 -X 2 d . The integrand ( X d -(1 -ε 2 )) 2 is constant with respect to X ′ . Therefore, the inner integral evaluates to:

<!-- formula-not-decoded -->

where Vol d -1 ( R ) is the volume of the ( d -1) -dimensional ball of radius R . This volume is given by V d -1 R d -1 , with V d -1 = π ( d -1) / 2 Γ( d +1 2 ) . So, the inner integral is ( X d -(1 -ε 2 )) 2 V d -1 (1 -X 2 d ) ( d -1) / 2 , and the outer integral becomes:

<!-- formula-not-decoded -->

Let X d = 1 -δ performing a change of variable. Then d X d = -d δ . The integration limits change

<!-- formula-not-decoded -->

Since we assumed ε 2 &lt; 1 4 , for the integration range [0 , ε 2 ] , we may write 2 δ -δ 2 = (2 -δ ) δ ≍ 2 7 / 4 2 δ .

<!-- formula-not-decoded -->

The integral is approximated by:

<!-- formula-not-decoded -->

Consider another change of variable: δ = ε 2 s . Then d δ = ε 2 d s . The limits change

<!-- formula-not-decoded -->

The L 2 norm is the square root of I is given by

<!-- formula-not-decoded -->

where c 7 ( d ) and c 8 ( d ) are constants defined by

<!-- formula-not-decoded -->

This completes the proof.

Lemma H.2. Let φ u ,ε 2 be a ReLU atom defined in (105) . Then

<!-- formula-not-decoded -->

Proof. We decode the definiton (see Example 3.1) and compute directly the weighted function g ( u , 1 -ε 2 ) = ( ε 2 ) d +2 = ε 2 d +4 .

Let S d -1 be the unit sphere in R d . For 0 &lt; ε &lt; 1 and w ∈ S d -1 , define the spherical cap C ( u , ε 2 ) as

<!-- formula-not-decoded -->

Lemma H.3. Let N max ( ε, d ) denote the maximum number of points u 1 , . . . , u N ∈ S d -1 such that the caps C ( u i , ε 2 ) are mutually disjoint. Then, as ε → 0 ,

<!-- formula-not-decoded -->

where the implicit constants depend only on the dimension d .

Proof. The spherical cap C ( u , ε 2 ) has an angular radius ϑ = arccos(1 -ε 2 ) , satisfying ϑ = Θ( ε ) for small ε . The condition that caps C ( u i , ε 2 ) and C ( u j , ε 2 ) are disjoint requires the angular separation ϕ ij between their centers w i and w j to be at least 2 ϑ . Thus, N max ( ε, d ) is the maximum size M ( S d -1 , 2 ϑ ) of a 2 ϑ -separated set (packing number) on S d -1 .

The upper bound N max ( ε, d ) = O ( ε -( d -1) ) follows from a surface area argument: N disjoint caps C ( u i , ε 2 ) , each with surface area Θ( ϑ d -1 ) = Θ( ε d -1 ) , must fit within the total surface area of S d -1 .

For the lower bound, we relate the packing number M ( S d -1 , α ) to the covering number N ( S d -1 , α ) , the minimum number of caps of angular radius α needed to cover S d -1 . It is a standard result that these quantities are closely related, for instance, M ( S d -1 , α ) ≥ N ( S d -1 , α ) can be shown via a greedy packing argument [Vershynin, 2018, see discussions in Chapter 4]. Furthermore, the asymptotic behavior of the covering number is known to be N ( S d -1 , α ) ≍ α -( d -1) for small α [Vershynin, 2018, Corollary 4.2.14]. Setting the minimum separation α = 2 ϑ = Θ( ε ) , we obtain the lower bound:

<!-- formula-not-decoded -->

where the implicit constants depend only on the dimension d . Combining the upper and lower bounds, we conclude that N max ( ε, d ) ≍ ε -( d -1) .

Construction H.4. We construct a suitable packing set in F = { f ∈ V g ( B d 1 ) : ∥ f ∥ L ∞ ≤ 1 , | f | V g ≤ 1 } based on a weighted ReLU atoms. Let φ u ,ε 2 be the ReLU atom defined in (105), and according to Lemma H.1 and Lemma H.2:

<!-- formula-not-decoded -->

According to Lemma H.1, there exists N = c N ( d ) ε -d +1 spherical caps u 1 , · · · , u N such that the caps C ( u i , ε 2 ) are mutually disjoint, for some constant c N ( d ) ≤ 1 that may depend on the dimension d . For convenience, we simply denote Φ i = Φ u i ,ε 2 . Therefore, we have | N Φ i | V g = c N ( d ) ε d +3 &lt; 1 referring to (116). For each ξ = ( ξ 1 , . . . , ξ N ) ∈ {-1 , 1 } N , define

<!-- formula-not-decoded -->

According to the conventions, each f ξ belongs to F . Since the supports of the ridge functions are disjoint, for any ξ, ξ ′ ∈ {-1 , 1 } N we have

<!-- formula-not-decoded -->

̸

where d H ( ξ, ξ ′ ) denotes the Hamming distance between ξ and ξ ′ , and S is the set of indices where ξ i = ξ ′ i . By the Varshamov-Gilbert lemma, there exists a subset Ξ ⊂ {-1 , 1 } N with

<!-- formula-not-decoded -->

for some constant, and such that for any distinct ξ, ξ ′ ∈ Ξ , the Hamming distance d H

<!-- formula-not-decoded -->

Thus, for any distinct ξ, ξ ′ ∈ Ξ , we obtain

<!-- formula-not-decoded -->

Proposition H.5 (Minimax Lower Bound via Fano's Lemma) . Consider the problem of estimating a function f ∈ F = { f ∈ V g ( B d 1 ) : ∥ f ∥ L ∞ ≤ 1 , | f | V g ≤ 1 } with

<!-- formula-not-decoded -->

where { ε i } n i =1 are i.i.d. N (0 , σ 2 ) random variables and { x i } n i =1 ⊂ B d 1 are i.i.d. uniform random variables on B d 1 . The lower bound of the minimax non-parametric risk is given by

<!-- formula-not-decoded -->

Proof. We use the standard Fano's lemma argument. By our Construction (105), we have a packing set { f ξ : ξ ∈ Ξ } in F with the following properties:

1. The L 2 -distance between any two distinct functions is at least δ , where δ ≍ ε .
2. The size of the packing set satisfies log | Ξ | ≳ K ≍ ε -( d -1) .

For Gaussian noise with variance σ 2 , the Kullback-Leibler divergence between the distributions induced by two functions f ξ and f ξ ′ is

<!-- formula-not-decoded -->

In order to use Fano's lemma J.1 effectively, we need to satisfy the requirement (141), where in this context is

<!-- formula-not-decoded -->

for some small constant α &gt; 0 , then the minimax risk is bounded from below by a constant multiple of δ 2 .

Substituting δ ≍ ε and log | Ξ | ≳ ε -( d -1) , the condition becomes

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, Fano's lemma J.1 (particularly (140)) yields

<!-- formula-not-decoded -->

or equivalently,

Solving for ε , we have

Then, the separation becomes which is the desired result.

Corollary H.6. Let { f ∈ V g ( B d 1 ) : ∥ f ∥ L ∞ ≤ B, | f | V g ≤ C } . Then

<!-- formula-not-decoded -->

Proof. We just need to replace f ξ in Construction 105 by min( B,C ) f ξ and adapt it to the the proof of Proposition H.5.

## H.2 Why Classical Bump-Type Constructions Are Ineffective

The minimax lower bound construction in this paper crucially hinges on exploiting the properties of the data-dependent weighted variation norm, denoted as | · | V g , where the weight function is g ( u , t ) . A key characteristic of g ( u , t ) (when data is, for instance, uniform on the unit ball B d 1 ) is that g ( u , t ) ≍ (1 - | t | ) d +2 . This implies that g ( u , t ) becomes very small as | t | → 1 , i.e., for activations near the boundary of the domain. This property allows for the construction of functions with significant magnitudes near the boundary using a relatively small variation norm. Therefore, any effective construction for the lower bound must create functions that are highly localized near this boundary.

<!-- formula-not-decoded -->

Figure 17: Isotropic Locality is Costly: An isotropic bump function, by definition, must be localized (decay rapidly) in all directions around its center. Suppose we place such a bump centered at a point x 0 near the boundary in direction u 0 (i.e., u T 0 x 0 ≈ 1 -ε 2 , here u 0 = (1 , 0) ). To achieve localization in directions orthogonal to u 0 , one would need to combine ReLU activations whose ridges are oriented appropriately. More critically, to achieve localization in the direction parallel to u 0 (i.e., to ensure the bump decays as we move radially inward from x 0 ), we would need ReLU activations whose ridges { x : u T 0 x = t } have t &lt; 1 -ε 2 and are potentially much closer to the origin (i.e., t is significantly smaller than 1).

<!-- image -->

For these ReLU activations whose ridges are not very close to the boundary (i.e., t is not close to 1), the weight function g ( u 0 , t ) will not be small. Consequently, constructing a sharply localized bump isotropically would require a substantial sum of weighted coefficients in the V g norm to cancel out the function in regions away from its intended support while maintaining a significant peak. This large variation norm would make such functions "too regular" or "too expensive" to serve as effective elements in a packing set for Fano's Lemma, especially when aiming to show a rate degradation due to dimensionality.

In essence, isotropic bump functions do not efficiently leverage the anisotropic nature of the ReLU activation and the specific properties of the g ( u , t ) weighting. The construction used in this paper, which employs ReLU atoms active only on thin spherical caps near the boundary (an anisotropic construction), is far more effective. It allows for localization and significant function magnitude primarily by choosing the activation threshold t to be very close to 1 (making g ( u , t ) small), rather than by intricate cancellations of many neurons with large weighted coefficients. This is why such anisotropic, boundary-localized constructions are essential for revealing the curse of dimensionality in this setting.

## H.3 The Univariate Case

The minimax lower bound construction detailed above, which leverages a packing argument with boundary-localized ReLU neurons exploiting the multiplicity of available directions on S d -1 , is particularly effective in establishing the curse of dimensionality for d &gt; 1 . However, the geometric foundation of this approach, specifically the ability to pack an exponential number of disjoint spherical caps, does not directly translate to the univariate case ( d = 1 ) where the notion of distinct directional activation regions fundamentally changes. Consequently, the lower bound for d = 1 necessitates a separate construction or modification of the argument. Fortunately, in the one-dimensional setting, the distinction between isotropic and anisotropic function characteristics, which posed challenges for classical approaches in higher dimensions under the specific data-dependent weighted norm, becomes moot. This simplification allows us to directly employ classical bump function constructions, suitably adapted to the function class, to establish the minimax rates in one dimension.

According to Theorem 3.4, we have

<!-- formula-not-decoded -->

When d = 1 and f is smooth, (118) is simplified to be

<!-- formula-not-decoded -->

and so is the unweighted variation seminorm

<!-- formula-not-decoded -->

which is also known as the second-order total variation seminorm. Therefore, the function class of stable minima in univariate case is characterized into

<!-- formula-not-decoded -->

Using this characterization, it is more convenient to smooth bump functions to construct a minimax risk lower bound for stable minima class.

Construction H.7. Consider a smooth compact support function:

<!-- formula-not-decoded -->

By adjusting the constant c , we may assume

<!-- formula-not-decoded -->

and let D be a constant such that ∥ Φ( x ) ∥ L 2 = √ 2 D . We can construct a translated and scaled version:

<!-- formula-not-decoded -->

and in particular, Φ a,b has the following properties by directly computations:

<!-- formula-not-decoded -->

Proposition H.8. Consider the problem of estimating a function f ∈ F 1 , 1 with

<!-- formula-not-decoded -->

where { ε i } n i =1 are i.i.d. N (0 , σ 2 ) random variables and { x i } N i =1 ⊂ [ -1 , 1] are i.i.d. uniform random variables on [ -1 , 1] . The lower bound of the minimax non-parametric risk is given by

<!-- formula-not-decoded -->

Proof. For any ε &gt; 0 , we may construct a family. Let a k = 1 -ε + kε 2 , k = 0 , ..., ⌊ 1 ε ⌋ . We denote K = ⌊ 1 ε ⌋ . For each k = 1 , ..., K , we define Φ k := Φ a k -1 ,a k Since a k -a k -1 = ε 2 , we have the following properties

- ∥ Φ k ∥ L 2 = D · ε ;

<!-- formula-not-decoded -->

Let { Φ 1 , ..., Φ K } , K ≍ ⌊ 1 ε ⌋ be such a family of function classes, and any K -terms combination { Φ 1 , ..., Φ K } is in F 1 , 1 . Then we let

<!-- formula-not-decoded -->

For any two indexes ξ 1 , ξ 2 in {± 1 } K , we have that

<!-- formula-not-decoded -->

̸

where d H is the Hamming distance. Then, using Varshamov-Gilbert's lemma (Lemma J.2), the pruned cube of { f 1 , ..., f M } has a size M ≥ 2 K/ 8 , and each has the property that if i = j ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and thus for any i = j

̸

On the other hand, to satisfy the Fano inequality (141):

we let

<!-- formula-not-decoded -->

and thus Fano's lemma (Lemma J.1, particularly (140)) implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that by rescale the functions in the lower bound construction, we can deduce a more general result.

Corollary H.9. For general case F B,C , we can scale the construction functions by min( B,C ) to deduce the result:

<!-- formula-not-decoded -->

## I Lower Bound on Generalization Gap

In this section, we derive a lower bound for the generalization gap.

## I.1 The Lower Bound Construction Can be Realized by Stable Minima

Recall the notations in Construction H.4, for ε ∈ (0 , 1) and a unit vector u ∈ S d -1 , define the (ball) cap

<!-- formula-not-decoded -->

Fix a dimension d ≥ 2 and a fixed cap C = C ( u , ε 2 ) , the mass under Uniform( B d 1 ) satisfies the two-sided bound

<!-- formula-not-decoded -->

where writing v k := Vol ( Bb k 1 ) .

<!-- formula-not-decoded -->

Indeed, writing h = ε 2 and parameterizing x = t u + √ 1 -t 2 z with t ∈ [1 -h, 1] and z ∈ S d -2 , we have

where s = 1 -t . Using 1 ≤ 2 -s ≤ 2 on [0 , h ] yields

<!-- formula-not-decoded -->

Dividing by v d = Vol ( B d 1 ) and recalling h = ε 2 gives the stated probability bounds. Proposition I.1 (Many caps does not have a sample point w.h.p. via Poissonization) . Fix d ≥ 2 and ε = κn -1 / ( d +1) with a constant κ ∈ (0 , 1] . Let { C ( u j , ε 2 ) } j m =1 be any family of pairwisedisjoint caps as in (129) , and draw X 1 , . . . , X n i . i . d . ∼ Uniform( B d 1 ) . For each j , write Z j := # { i ≤ n : X i ∈ C ( u j , ε 2 ) } and p j := P ( X ∈ C ( u j , ε 2 )) . Then there exist absolute constants c ∗ , C ∗ &gt; 0 (depending only on d and κ ) such that, for any δ ∈ (0 , 1) ,

<!-- formula-not-decoded -->

where q ∗ = e -λ + and λ + = U cap d κ d +1 is a constant independent of n . In particular, with probability at least 1 -exp( -cm ) (for some c &gt; 0 ), there exists a subset Γ ⊂ { 1 , . . . , m } with | Γ | ≥ c 0 m such that Z j ≤ 1 for every j ∈ Γ , where c 0 ∈ (0 , q ∗ ) depends only on d, κ .

Proof. Introduce a Poisson variable N ∼ Poi( n ) independent of the data. Conditionally on N , draw X 1 , . . . , X N i . i . d . ∼ Uniform( B d 1 ) . Let ˜ Z j := # { i ≤ N : X i ∈ C ( u j , ε 2 ) } . By standard Poisson thinning, ˜ Z 1 , . . . , ˜ Z m are independent with ˜ Z j ∼ Poi( λ j ) and

<!-- formula-not-decoded -->

At the critical scaling ε = κn -1 / ( d +1) , we thus have constants

<!-- formula-not-decoded -->

For each j , set A j := 1 { ˜ Z j = 0 } . Then A 1 , . . . , A m are independent Bernoulli random variables with

<!-- formula-not-decoded -->

Therefore, by Hoeffding's inequality for independent bounded variables,

<!-- formula-not-decoded -->

Equivalently,

<!-- formula-not-decoded -->

To proceed de-Poissonization, we compare ( Z 1 , . . . , Z m ) under the fixedn model (a multinomial random variable) to the corresponding joint Poisson variable ( ˜ Z 1 , . . . , ˜ Z m ) . By Le Cam's inequality for Poisson approximation, the total variation distance between the joint law of the Bernoulli multivariables ( Z 1 , . . . , Z m ) and that of independent Poi( λ j ) variables is bounded by

<!-- formula-not-decoded -->

Applying this to the event E = { # { j : Z j = 0 } ≥ ( q ∗ -δ ) m } and combining with (130) yields

<!-- formula-not-decoded -->

Setting c ∗ := 2 and C ∗ := ( U cap d ) 2 gives the stated bound. In particular, choosing any fixed δ ∈ (0 , q ∗ ) and defining c 0 = q ∗ -δ ∈ (0 , q ∗ ) proves that with probability at least 1 -exp( -cm ) -C ∗ mε 2( d +1) there exists Γ ⊂ { 1 , . . . , m } , | Γ | ≥ c 0 m , such that Z j ≤ 1 for all j ∈ Γ .

Proposition I.1 ensures that, at the critical scaling ε ≍ n -1 / ( d +1) , there exists (with overwhelmingly high probability) a large subcollection of caps, each containing no sample.

We now show that if a neural network does not have any activated datapoint, the operator norm of its Hessian is constantly 1.

Proposition I.2. Let f θ ( x ) = ∑ K k =1 v k ϕ ( w T k x -b k ) + β be network defined in (1) . Let D = { ( x i , y i ) } n i =1 be a data set such that each neuron of f θ contains no activated datapoint, i.e for each k , ∑ n i =1 1 { w T k x i -b k } = 0 , and f θ interpolates D in the sense that f θ ( x i ) = y i = 0 for each i . Then λ max ( ∇ 2 θ L ) = 1 .

Proof. By direct computation, the Hessian ∇ 2 θ L is given by

<!-- formula-not-decoded -->

Since the model interpolates f θ ( x i ) = y i for all i , we have

<!-- formula-not-decoded -->

Consider the tangent features matrix that is defined by

<!-- formula-not-decoded -->

Then we have ∇ 2 θ L = ΦΦ T /n , and the operator norm is computed by

<!-- formula-not-decoded -->

Furthermore, we have

<!-- formula-not-decoded -->

For the parameters [ w k , b k , v k ] associated to the neuron of index k ,

<!-- formula-not-decoded -->

Since there is no data point activating, we have that

<!-- formula-not-decoded -->

After subsistion by (136), (134) is of the form

<!-- formula-not-decoded -->

Let u = ( u 1 , · · · , u n ) ∈ S n -1 and plug (137) in (134) to have

<!-- formula-not-decoded -->

We now establish that such a specially constructed interpolation solution is indeed stable. As shown in the proof, for an interpolation solution where none of the hidden neurons are active on the training data, the Hessian of the loss has an operator norm of exactly 1, i.e., λ max ( ∇ 2 L ( θ )) = 1 . The primary contribution to this norm comes from the gradient of the output layer bias. According to the stability condition defined in Proposition 2.1 ( λ max ≤ 2 /η ), this solution is guaranteed to be in Θ flat ( η ; D ) so long as the step size satisfies η ≤ 2 . Since we assume that η &lt; 2 in this paper (cf. Proposition 2.1 and the discussion below it), we have that this interpolating solution is indeed stable.

For brevity, we write F flat ( η ; D ) := { f θ | θ ∈ Θ flat ( η ; D ) } in the sequel.

Corollary I.3 (Stronger Version of Minimax Lower Bound) . Consider the problem of estimating a function f ∈ F = { f | f ∈ F flat ( η ; D ) , ∥ f ∥ L ∞ ( B d 1 ) ≤ L } with

<!-- formula-not-decoded -->

where { x i } n i =1 ⊂ B d 1 are i.i.d. uniform random variables on B d 1 . The lower bound of the minimax nonparametric risk is given by

<!-- formula-not-decoded -->

where ˆ f refers to a estimator and ˆ f ( D ) means the estimation based on the data set D .

Proof. The core of the proof is to construct two functions, f 1 and f 2 , which belong to the function class F but are far apart in L 2 norm. We will show that for a typical random data set { x i } n i =1 , any estimator ˆ f ( D ) cannot distinguish between them, as they produce identical observations on D . This implies a lower bound on the minimax risk.

We set the critical scaling for our construction to be:

<!-- formula-not-decoded -->

Following the geometric packing argument from Lemma H.3, we can find a set of N ≍ ε -( d -1) pairwise-disjoint spherical caps { C j } N j =1 , where each cap C j = C ( u j , ε 2 ) is defined by a unique direction u j ∈ S d -1 .

Let { x i } n i =1 be the randomly drawn data set's inputs. Let E be the event that there exists a subset of indices Γ ⊂ { 1 , . . . , N } such that:

- (i) | Γ | ≥ c 0 N for some constant c 0 &gt; 0 .
- (ii) For every j ∈ Γ , the cap C j is empty, i.e., C j ∩ { x i } n i =1 = ∅ .

According to Proposition I.1, this event E occurs with high probability, i.e., P ( E ) ≥ 1 -exp( -c 1 N ) for some constant c 1 &gt; 0 . From now on, we condition our entire analysis on this high-probability event E occurring.

Conditioned on the event E , we now define two functions. Let j ∈ Γ be an index corresponding to one of the empty caps, C j .

## 1. Let the first function be the zero function:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

2. Let the second function be a combination appropriately scaled ReLU atoms supported on the empty cap C j :

<!-- formula-not-decoded -->

On the event E , both functions produce the exact same observations. For any x i :

<!-- formula-not-decoded -->

Therefore, the data set generated by both functions is identical: D = { ( x 1 , 0) , . . . , ( x n , 0) } .

Since the data set D is fixed by the condition E and the cap C j is empty for j ∈ Γ , the neuron implementing f Γ is never active on any data point x i ∈ { x i } n i =1 . This implies f 2 ( x i ) = 0 for all x i . The corresponding labels are y i = 0 . Therefore, f 2 perfectly interpolates the data ( x i , 0) . According to Proposition I.2, any such network that interpolates the data and has no active neurons on the data set has λ max ( ∇ 2 L ( θ 2 )) = 1 , where θ 2 is the parameter vector that implements f 2 . Since η &lt; 2 , this solution is stable. Thus, f 2 ∈ F flat ( η ; D ) . Moreover, ∥ f 2 ∥ L ∞ = L , since the spherical caps { C j } are disjoint, at any point x at most one of the scaled ReLU atoms is non-zero. In summary, both f 1 and f 2 are valid functions in the class F .

An estimator ˆ f takes as input the data set D , which is identical for both potential ground-truth functions f 1 and f 2 , and produces an estimate function, which we denote by ˆ f ( D ) . The performance of this estimator is measured by its population risk. The estimator's objective is to minimize this risk under a worst-case choice of the ground truth from the set { f 1 , f 2 } .

For a given estimate ˆ f ( D ) , the worst-case risk over this set is

<!-- formula-not-decoded -->

The minimax risk for this problem is the minimal possible worst-case risk achievable by any estimator. It is lower-bounded by considering the optimal decision rule conditioned on the event E :

<!-- formula-not-decoded -->

The function ˆ f ( D ) ∗ that minimizes max { ∥ ∥ ∥ ˆ f ( D ) -f 1 ∥ ∥ ∥ 2 L 2 ( B d 1 ) , ∥ ∥ ∥ ˆ f ( D ) -f 2 ∥ ∥ ∥ 2 L 2 ( B d 1 ) } is the average of f 1 and f 2 in the Hilbert space L 2 ( B d 1 ) . This optimal estimate is ˆ f ( D ) ∗ = ( f 1 + f 2 ) / 2 . The minimal possible worst-case risk is thus achieved at this midpoint:

<!-- formula-not-decoded -->

According to the computation in Construction H.4, we may conclude that

<!-- formula-not-decoded -->

This completes the proof.

Theorem I.4. Let P denote any joint distribution of ( x , y ) where the marginal distribution of x is Uniform( B d 1 ) and y satisfies the P P [ -D ≤ y ≤ D ] = 1 .

Let D = { ( x j , y j ) } n j =1 be a data set of n i.i.d. samples from P , and that ˜ R is any risk estimator that takes any f and D as input, then outputs a scalar that aims at estimating the risk R P ( f ) := E ( x ,y ) ∼P [ ( f ( x ) -y ) 2 ) ] . Moreover, let F be the function class we defined in Corollary I.3.

Then

<!-- formula-not-decoded -->

where we assume that L ≥ D .

Proof. Let the E [ · ] be the short-hand for the expectation over the random training data set D .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The first inequality restricts P further to deterministic labels with labeling functions in F . Check that any function in F is bounded between [ -M,M ] . The second inequality uses the fact that f 0 ∈ F flat ( η ; D ) , and the following decomposition

<!-- formula-not-decoded -->

where we used ˜ R ( ˆ f ERM ; D ) -˜ R ( f 0 ; D ) ≤ 0 from the definition of ˆ f ERM

<!-- formula-not-decoded -->

The third inequality enlarges the set of ERM estimators to any function of the data ˆ f that output. The subsequent identity uses the fact that R P ( f 0 ) = 0 .

This completes the proof for the lower bound on generalization gap stated in Theorem 3.5.

## J Technical Lemmas

## J.1 Information-Theoretic tools

Fano's Lemma provides a powerful method for establishing such minimax lower bounds by relating the estimation problem to a hypothesis testing problem. It leverages information-theoretic concepts, particularly the Kullback-Leibler (KL) divergence.

̸

Lemma J.1 (Fano's Lemma (Statistical Estimation Context)) . Consider a finite set of functions (or parameters) { f 1 , f 2 , . . . , f M } ⊂ F , with N ≥ 2 . Let P f j denote the probability distribution of the observed data D when the true underlying function is f j . Suppose that for any estimator ˆ f , the loss function L ( f j , ˆ f ) satisfies L ( f j , ˆ f ) ≥ s 2 / 2 &gt; 0 if ˆ f is not close to f j (e.g., if we make a wrong decision in a multi-hypothesis test where closeness is defined by a metric d ( f j , f k ) ≥ s ). More specifically, for function estimation with squared L 2 -norm loss, if we have a packing set { f 1 , . . . , f M } ⊂ F such that ∥ f j -f k ∥ 2 L 2 ≥ s 2 for all j = k , then the minimax risk is bounded as:

̸

<!-- formula-not-decoded -->

provided the term in the parenthesis is positive. KL( P f j ∥ P f k ) denotes the Kullback-Leibler divergence between the distributions P f j and P f k . For this bound to be non-trivial (e.g., ≳ s 2 ), we typically require that the number of well-separated functions M is large enough such that

̸

<!-- formula-not-decoded -->

One can refer to Wasserman [2020, Theorem 12, Corollary 13] or Tsybakov [2009, Chapter 2] for more details.

Our application of Fano's Lemma (for proving Proposition H.5) involves:

1. Constructing a suitable finite subset of functions { f 1 , . . . , f M } within the class F such that they are well-separated in the metric defined by the loss function (e.g., pairwise L 2 -distance s ). This is often achieved using techniques like the Varshamov-Gilbert lemma (Lemma J.2) for constructing packings.
2. Bounding the KL divergence (or another information measure like χ 2 -divergence) between the probability distributions generated by pairs of these functions. For n i.i.d. observations with additive Gaussian noise N (0 , σ 2 ) , and if using the empirical L 2 norm ∥ · ∥ L 2 ( P n ) based on fixed data points x i , this divergence is often related to 1 2 σ 2 ∑ n i =1 ( f j ( x i ) -f k ( x i )) 2 . More generally, for population norms, it's often n ∥ f j -f k ∥ 2 L 2 2 σ 2 .
3. Choosing M and s (or the parameters defining the packing) to maximize the lower bound, typically by ensuring that the KL divergence term does not dominate log M .

Lemma J.2 (Varshamov-Gilbert Lemma) . Let

<!-- formula-not-decoded -->

Suppose N ≥ 8 . Then there exist such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

3. for all 0 ≤ j &lt; k ≤ M , the Hamming distance satisfies

<!-- formula-not-decoded -->

We call { ξ 0 , ξ 1 , . . . , ξ M } a pruned hypercube .

One can refer to Tsybakov [2009, Lemma 2.9] and Wasserman [2020, Lemma 15] for more details.

## J.2 Poissonization and Le Cam's Inequality

For a random variable S on a probability space (Ω , F , P ) with values in a measurable space ( E, E ) , the law is L ( S ) := P ◦ S -1 . For two laws µ, ν on the same space, define the total variation distance

<!-- formula-not-decoded -->

When E is countable, d TV ( µ, ν ) = 1 2 ∑ x ∈ E | µ ( { x } ) -ν ( { x } ) | .

Lemma J.3 (Poissonization [Barbour et al., 1992, Ch. 1]) . Let N ∼ Poi( λ ) and, conditional on N , let W 1 , . . . , W N be i.i.d. taking values in { 0 , 1 , . . . , m } with P { W = j } = p j for j = 0 , 1 , . . . , m , where p 0 := 1 -∑ m j =1 p j ≥ 0 . Define ˜ Z j := # { 1 ≤ i ≤ N : W i = j } for j = 1 , . . . , m . Then ˜ Z 1 , . . . , ˜ Z m are independent and ˜ Z j ∼ Poi( λp j ) .

Remark J.4. This standard Poissonization trick replaces the fixed sample size n by a Poisson random size N ∼ Poi( n ) , making the cell counts independent. See Barbour, Holst and Janson [Barbour et al., 1992, Ch. 1] for a general treatment and applications in occupancy problems.

<!-- formula-not-decoded -->

LemmaJ.5 (Le Cam's inequality for Poisson approximation [Cam, 1960, Arratia et al., 1989, Barbour et al., 1992]) . Let ( Z 1 , . . . , Z m ) ∼ Mult( n ; p 1 , . . . , p m , p 0 ) with p 0 = 1 -∑ m j =1 p j . Let Y 1 , . . . , Y m be independent with Y j ∼ Poi( np j ) . Then there exists a universal constant C &gt; 0 such that

<!-- formula-not-decoded -->

Remark J.6. Lemma J.5 is a classical result of Le Cam [Cam, 1960], with modern proofs and refinements given by [Arratia et al., 1989] and by [Barbour et al., 1992, Sec. 1.3]. It provides a quantitative control of the total variation distance between the multinomial occupancy vector and the independent Poisson approximation. We use this bound to justify the de-Poissonization step in the proof of Proposition I.1.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claims in the abstract and introduction accurately reflect the paper's contributions and scope.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Please refer to the last paragraph in Section 5.

## Guidelines:

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

Justification: The assumptions are clearly stated in the theorems while the proof is stated in the appendix.

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

Justification: The details can be found in Section 5 and Appendix.

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

Justification: We will release the code.

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

Justification: The details can be found in both Section 4 and the Appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We sweep the random seeds for the initializations of neural networks and take the median for performance metrics such as MSE. Details are discussed in the Appendix.

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

Justification: All the experiments can run on Mac Air M1.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in the paper conform, in every respect, with the Neuips Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This is a paper about theory of neural network, where we believe there is no possible negative societal impact.

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

Justification: The paper does have have such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper does not use existing assets.

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

Answer: [No]

Justification: This paper does not involve this.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing experiments.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification:

Guidelines: The paper does not involve crowdsourcing nor research with human subjects.

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development of this paper comes from human brains.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.