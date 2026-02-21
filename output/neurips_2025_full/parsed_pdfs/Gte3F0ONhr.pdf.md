## A Principled Path to Fitted Distributional Evaluation

Sungee Hong Texas A&amp;M University

Jiayi Wang University of Texas at Dallas

Raymond K. W. Wong Texas A&amp;M University

## Abstract

In reinforcement learning, distributional off-policy evaluation (OPE) focuses on estimating the return distribution of a target policy using offline data collected under a different policy. This work focuses on extending the widely used fitted Q-evaluation-developed for expectation-based reinforcement learning-to the distributional OPE setting. We refer to this extension as fitted distributional evaluation (FDE). While only a few related approaches exist, there remains no unified framework for designing FDE methods. To fill this gap, we present a set of guiding principles for constructing theoretically grounded FDE methods. Building on these principles, we develop several new FDE methods with convergence analysis and provide theoretical justification for existing methods, even in non-tabular environments. Extensive experiments, including simulations on linear quadratic regulators and Atari games, demonstrate the superior performance of the FDE methods.

## 1 Introduction

In reinforcement learning (RL), the return, defined as the cumulative sum of (discounted) rewards, is a fundamental measure for how well the underlying policy performs. Traditional RL assesses policies by calculating the expected return, whereas distributional RL [5] focuses on the full distribution of return, providing a richer and more complete understanding of the policy behavior. Leveraging distributional properties beyond the expectation (e.g., risk, multimodality) enables the handling of more general tasks, such as risk assessment [27] and risk-sensitive policy learning [e.g., 15, 32]. Therefore, distributional RL has found applications across various domains [e.g., 8, 4, 64, 17, 12].

In this paper, we focus on addressing off-policy evaluation (OPE) problems within distributional RL, referred to as distributional OPE. The goal is to estimate the return distribution of a target policy using data collected under a different policy known as the behavior policy. Many distributional OPE methods extend existing OPE techniques from traditional RL, including temporal difference (TD) methods [e.g., 7, 14, 65, 52] and model-based approaches [e.g., 31]. Bellman residual minimization, a well-studied method in traditional RL [e.g., 33, 21, 48], was recently adapted for distributional OPE by [26]. Another key methodology, fitted Q-evaluation (FQE)-an iterative algorithm extensively analyzed in traditional RL [e.g., 10, 69, 46, 59]-has also been recently extended to the distributional setting [32, 63]. In this work, we focus on the distributional extension of FQE.

Unlike FQE, where the discrepancy function (squared ℓ 2 -loss) is defined over real numbers, distributional extensions operate directly on distributions, making the choice of discrepancy non-trivial. Although statistical distances are natural candidates for measuring discrepancy between distributions, some well-known distances (e.g., total variation distance) perform poorly. Existing distributional extensions of FQE [32, 63] are developed based on specific discrepancy functions. [32] utilizes the p -powered Wassersteinp metric, whereas [63] is based on log-likelihood (closely related to Kullback-Leibler (KL) divergence). A general guideline for selecting appropriate discrepancy functions

Zhengling Qi

George Washington University

in FDE remains unclear, leaving practitioners without systematic guidance for developing valid FQE extensions tailored to their specific needs. In this paper, we address this gap by proposing a unified framework for FDE that clarifies the role of discrepancy function selection and enables analysis under a broad class of discrepancy functions and general conditions (including non-tabular settings, i.e., inifinite state-action space). Along the way, we establish several guiding principles and provide theoretical support for our framework. Beyond these general principles, we present concrete examples of discrepancy functions derived from our framework, accompanied by their statistical analyses, offering readily applicable discrepancy functions for practitioners. Additionally, our general framework gives rise to FDE methods that overcome some challenges faced by existing distributional OPE approaches (beyond those FQE extensions). In particular, many of such FDE methods support multi-dimensional returns, which are incompatible with widely used quantile-based methods [e.g., 15, 32, 47]. Furthermore, we provide examples of FDE methods with theoretical guarantees for unbounded distributions, in contrast to conventionally assumed bounded distributions [e.g., 63, 45].

Our key contributions are summarized as follows. (1) Principled Framework for Fitted Distributional Evaluation: We introduce guiding principles for constructing theoretically grounded extensions of FQE to the distributional OPE problems, which we term fitted distributional evaluation (FDE) methods. In particular, we show that functional Bregman divergences are natural candidates for the discrepancy measures within this framework. See Section 2; (2) Survey of Valid Discrepancy Measures: Leveraging the proposed framework, we derive many examples and discover novel discrepancy measures that have not previously been considered as objective functions for distributional OPE. See Tables 1 and 2; (3) Unified Statistical Convergence Analysis: We provide a comprehensive convergence analysis covering a broad class of FDE methods in tabular (i.e., finite state-action space) and non-tabular settings. This significantly expands the set of distributional OPE methods with theoretical guarantees in non-tabular scenarios. See Section 3.

## 2 Fitted distributional evaluation in off-policy settings

Consider a homogeneous Markov decision process ( S , A , ˜ p, γ ) where S and A refer to the state and action spaces, ˜ p is the transition probability, and γ ∈ [0 , 1) is the discount factor. More specifically, the transition probability ˜ p represents the conditional distribution of reward R ∈ R d and next state S ′ ∈ S conditional on the current state S ∈ S and action A ∈ A . We shall focus on the evaluation of a stationary target policy π : S → ∆( A ) under infinite-horizon setting. Consider a trajectory { ( S t , A t , R t ) } t ≥ 0 generated by iteratively sampling from the target policy and the transition probability: A t ∼ π ( ·| S t ) and R t , S t +1 ∼ ˜ p ( ·| S t , A t ) (and some initial distribution of S 0 ). The return of the trajectory is defined by ∑ t ≥ 0 γ t R t . Unlike traditional RL which focuses on the policy value, i.e., the expectation of return, distributional RL considers the whole distribution of return. Analogously to Q-function estimation in traditional RL, the policy evaluation problem in distributional RL, known as distributional policy evaluation, aims to estimate the conditional distribution of return Υ π ∈ P S×A (with P ⊆ ∆( R d ) as a convex set of probability measures 1 ), where Υ π ( s, a ) ∈ P refers to the probability measure of ∑ t ≥ 0 γ t R t under π starting from the initial state-action pair s, a . We are interested in distributional policy evaluation in the off-policy setup, where data are collected under a different policy, resulting in a distributional mismatch. More specifically, we shall assume that the data consisting of tuples in the form ( S, A, R, S ′ ) that are generated by S ∼ µ , A ∼ b ( ·| S ) with b being the behavior policy, and followed by ( R,S ′ ) ∼ ˜ p ( ·| S, A ) . We further let ( S, A ) ∼ ρ = µ × b .

## 2.1 Fitted distributional evaluation algorithm

In traditional RL, fitted Q-evaluation (FQE) [e.g., 10, 30, 58, 29, 46, 59] is a popular approach to estimate Q -function, i.e., Q π ( s, a ) := E { ∑ t ≥ 0 γ t R t | s, a } . It is motivated by the Bellman equation

<!-- formula-not-decoded -->

where B π : R S×A → R S×A is known as the Bellman operator. Based on the contractive property of B π with respect to L ∞ -norm, iterative application of the Bellman operator yields convergence towards

1 For our objective functions to be well-defined, P should be closed under push-forward mapping with respect to affine maps: g ( x ) = r + γx for any r in the support of the reward distribution. See Appendix C.1.2.

the unique target, i.e., lim T →∞ ( B π ) T Q 0 = Q π in L ∞ -norm for an initial value Q 0 . However, the Bellman operator is often unknown and so is the evaluation of the RHS in (1), rendering this approach impractical. Instead, given a data set D = { ( s i , a i , r i , s ′ i ) } N i =1 , FQE aims to iteratively approximate Q t ≈ B π Q t -1 by minimizing the mean squared error

<!-- formula-not-decoded -->

where Q ⊆ R S×A is a chosen function class. Succinctly stated, FQE consists of iteratively solving regression problems with predictions Q ( s, a ) and the Bellman backup r + γ · E A ′ ∼ π ( ·| s ′ ) ( Q t -1 ( s ′ , A ′ )) , where the squared ℓ 2 -loss is used to measure the discrepancy.

We aim to extend FQE for distributional policy evaluation, where the quantity of interest is the whole conditional distribution Υ π ∈ P S×A instead of the conditional mean Q π ∈ R S×A . To this end, we define the distributional Bellman operator T π : P S×A →P S×A [7] by

<!-- formula-not-decoded -->

where ( g r,γ ) # : P → P is the push-forward mapping with respect to the function g r,γ ( x ) := r + γx , that maps the distribution of any random vector X to the distribution of r + γX . Analogously to (1), Υ π is the solution to the distributional Bellman equation: Υ = T π Υ . As in FQE, we would like to form a sequence Υ t ≈ T π Υ t -1 based on the data. In a similar spirit of FQE, given a single observation ( s, a, r, s ′ ) , we compute the distributional Bellman backup

<!-- formula-not-decoded -->

as the target. Then we aim to optimize the prediction Υ( s, a ) such that some discrepancy measure is minimized. Specifically, consider an appropriate mapping d : P × P → [0 , ∞ ) , we can then formulate a distributional extension of FQE by performing iterative minimization:

<!-- formula-not-decoded -->

where M⊆P S×A is a chosen set of conditional distributions. We call the resulting method fitted distributional evaluation (FDE). See Algorithm 1 in Appendix A for its full algorithm. In passing, [32] also discussed a distributional FQE extension with the same name, but there are some key differences as will be explained in Remark 2.1.

In FQE, it is straightforward to choose squared ℓ 2 -loss for minimization as the Bellman backup is real number. However, in FDE, the distributional Bellman backup is a distribution. Thus, to construct a FDE algorithm, a key question is:

## How to choose d correctly to build a theoretically grounded FDE?

This question is indeed non-trivial. While many known statistical distances are natural candidates for measuring discrepancy between two probabilities, a number of them fail. For instance, total variation distance can perform poorly as shown in Section 4 (also see Figure 1). A major reason is that (the expected-extended or supremum-extended) total variation distance does not lead to contraction of the distributional Bellman operator [5]. See Section 2.3 for more details. This may not be surprising, as the core motivation behind FQE is the contraction argument, which likely extends to its distributional counterpart, FDE. However, a metric that guarantees contraction of the distributional Bellman operator may also fail. One example is the (expectation-extended or supremun-extended) Wasserstein distance due to biased gradient [7]. The issue originates from the sample approximation based on the Bellman backup in (5), which will be explained in Section 2.4. In Section 2.2, we will outline several principles to choose a valid discrepancy d , based on a motivating error-bound analysis. Remark 2.1 . Two existing methods share a similar construction with FDE. Crucially, both focus exclusively on specific objective functions, without any general guideline on how to select an appropriate divergence d . First, [32] formulated their objective using powered Wassersteinp metric (see their Equation (4)). Instead of applying the distributional Bellman backup Ψ π (4) based on a

single sample, [32] estimates T π by leveraging the entire data to approximate the transition dynamics with conditional empirical distributions. However, this approach generally fails to yield consistent estimates in non-tabular settings (i.e., |S × A| = ∞ ), introducing potential bias. The second method is fitted likelihood evaluation (FLE) [63], which is based on log-likelihood. Unlike FDE, FLE uses only a single draw from the distribution Bellman backup, leading to information loss. Moreover, FLE is restricted to the cases where the return distributions have densities. Theoretically, their statistical convergence rate (see their Corollary 4.14) is also slower than ours; see details in Section 3.2.

## 2.2 A motivating error-bound analysis

To motivate a theoretically grounded FDE method, we begin with an error-bound analysis. Firstly, the metric under which the distributional Bellman operator is contractive plays a crucial role.

Definition 2.2. (Contraction-inducing metric) Let ˜ η be a metric over P S×A . We call it a contractioninducing metric if T π is ζ -contraction with respect to ˜ η , i.e., ˜ η ( T π Υ 1 , T π Υ 2 ) ≤ ζ · ˜ η (Υ 1 , Υ 2 ) for a constant ζ ∈ (0 , 1) .

We will focus exclusively on contraction-inducing metrics under which T π is contractive. Given a contraction-inducing metric ˜ η , we can derive the inequality for any sequence Υ 0 , . . . , Υ T ∈ P S×A (see Appendix C.2):

<!-- formula-not-decoded -->

which provides an upper bound for ˜ η (Υ T , Υ π ) , referred to as the ˜ η -error of Υ T . Roughly speaking, the second term ζ T · ˜ η (Υ 0 , Υ π ) in the error upper bound (6) becomes negligible as we increase T , due to ζ ∈ (0 , 1) . Then, to ensure small ˜ η -error of Υ T , it suffices to additionally require that ˜ η (Υ t , T π Υ t -1 ) converges. This is the essential goal of the minimization (5). To build a successful FDE method, we lay out the following principles to choose a valid pair (˜ η, d ) :

- (P1) ˜ η is a contraction-inducing metric;
- (P2) For any given ˜ Υ ∈ P S×A , the unique 2 minimizer of the population objective F (Υ; ˜ Υ) := E S,A ∼ ρ,R,S ′ ∼ ˜ p ( ·| S,A ) { d (Υ( S, A ) , Ψ π ( R,S ′ , ˜ Υ)) } is Υ = T π ˜ Υ .
- (P3) For any ˜ Υ ∈ P S×A , there exists a function g : R → [0 , ∞ ) such that: (i) g ( ξ ) = 0 and g is continuous at ξ = min Υ F (Υ; ˜ Υ) ; (ii) ˜ η (Υ , T π ˜ Υ) ≤ g ( F (Υ; ˜ Υ)) for any Υ ∈ P S×A .

(P2) ensures that the minimizer of the population objective F (Υ; Υ t -1 ) of (5) is the target T π Υ t -1 . In addition, (P3) indicates that sufficiently small value of F (Υ; Υ t -1 ) implies closeness between Υ and T π Υ t -1 with respect to the metric ˜ η . In the following subsections, we will center our discussion around these three principles. With the systematic construction based on these principles, we are able to construct a wide range of FDEs, even with choices of d that have never been used in the current literature of distributional RL (see Table 2). The above principles do not address the finite-sample error, but we will provide a unified statistical analysis for a broad class of FDEs in Section 3.

## 2.3 Contraction-inducing metrics

In this subsection, we will focus on the choices of contraction-inducing metrics (P1). To broaden the choices of valid discrepancy d (see (P3)), we describe two classes of contraction-inducing metrics over P S×A . The first class comprises supremum-extended metrics (7), which are well studied in the literature [e.g., 42, 40, 6], whereas the second class consists of expectation-extended metrics (8), which have received significantly less attention but are particularly useful for large or continuous state-action spaces.

Supremum extension Given a probability metric η over P , its supremum-extended metric is defined as

<!-- formula-not-decoded -->

2 Uniqueness is only required to hold almost surely over the data generating distribution ρ .

If the individual probability metric η satisfies (i) scale-sensitivity ( c -sensitivity), (ii) locationinsensitivity (regularity) and (iii) q -convexity , collectively denoted by (S-L-C) with definitions deferred to Appendix C.4, then η ∞ is a contraction-inducing metric with ζ = γ c (proof in Appendix C.3.2). Our result is a slight extension from Theorem 4.25 of [6], not requiring that R and S ′ to be independent conditioned on S, A . We have listed three examples that satisfy (S-L-C) in Table 3 of Appendix C.4, along with a survey of other well-known probability metrics that fail to satisfy (S-L-C). Examples include total variation distance (TVD) whose supremum extension is not guaranteed to be a contraction-inducing metric [5].

However, controlling supremum-extended metric (7) requires uniform control of probability metrics over all state-action pairs. For large or infinite state-action space, supremum-extended metrics can be challenging to control, which limits the choice of discrepancy d in view of (P3). This is related to the fundamental difficulty for using an expectation-based criterion (or the sample version in (5)) to control a supermum-based quantity (7). Indeed, existing analyses of statistical error bounds in η ∞ for distributional OPE methods mainly focus on tabular setting ( |S × A| &lt; ∞ ) [e.g., 47, 22, 45].

Expectation extension In view of the above explanation, we also introduce expectation-extended metrics which are more compatible with the expectation-based criterion (P3). Given a probability metric η over P and a parameter q ≥ 1 , an expectation-extended metric is defined as

<!-- formula-not-decoded -->

where d π ∈ ∆( S × A ) is defined as d π = (1 -γ ) -1 ∑ ∞ h =1 γ h -1 d h π with d h π ( E ) := P (( S h , A h ) ∈ E | S 0 , A 0 ∼ ρ, A t ∼ π ( ·| A t ) for t ≥ 1) . The distribution d π has appeared commonly in the RL literature [e.g., 39, 68, 66], and is important for ensuring contraction-inducing property as in Theorem 2.3 below. Note that the supremum-extended metric η ∞ (7) can be regarded as a special case of ¯ η d π ,q when q →∞ (under appropriate conditions of d π ). Expectation-extended metrics (based on possibly different distributions in the expectation) have been recently used for distributional OPE [63, 26]. Here, we provide a new result that facilitates the construction of a contraction-inducing expectation-extended metric as follows, with the corresponding proof given in Appendix C.3.1.

Theorem 2.3. Suppose that a probability metric η over P satisfies (S-L-C) with convexity parameter q ≥ 1 and scale-sensitivity parameter c &gt; 1 / (2 q ) (see (15) of Appendix C.4). Then the expectationextended metric ¯ η d π ,q is a contraction-inducing metric with ζ = γ c -1 2 q defined in Definition 2.2.

## 2.4 Discrepancy measures

We will now provide some guideline on the choice of discrepancy d . Based on (P2), we would ask how to choose d such that, for any Υ ∈ P S×A ,

<!-- formula-not-decoded -->

where the minimizer is unique up to almost surely equivalence. Note that E { Ψ π ( R,S ′ , Υ) | S = s, A = a } = T π Υ( s, a ) for any ( s, a ) ∈ S × A . As such, we hope that this conditional expectation of random measure is the minimizer of the expected discrepancy. In the case of FQE where the discrepancy is defined between two scalar values (see (2)), it is well known that minimizing the expected squared loss yields the conditional expectation. However, extending this property to settings where the discrepancy is defined between two measures is less obvious. Nevertheless, we show that a broad family of discrepancies-the functional extension of Bregman divergences-does satisfy this desirable property. This result significantly expands the possible construction of FDE. The formal definition of functional Bregman divergence [43] is technically involved and thus deferred to Definition C.1 of Appendix C.1.1. Before further discussion of functional Bregman divergences, we present the following key result, which we prove in Appendix C.1.2:

Theorem 2.4. A functional Bregman divergence d satisfies (9) for any ρ ∈ ∆( S × A ) , transition p , and target policy π .

Despite the technically involved definition of functional Bregman divergence, it has a close relationship with strictly proper scoring rule, which has been broadly studied in statistics literature. A scoring rule S ( · , ∗ ) : P × Ω X → R (with Ω X being the corresponding support space of P , say R d ) is strictly

proper if ¯ S ( Q,Q ) ≥ ¯ S ( P, Q ) where ¯ S ( P, Q ) := ∫ Ω X S ( P, x )d Q ( x ) [23] with equality holding only when P = Q . Given a strictly proper scoring rule S , we can always build a functional Bregman divergence by letting d ( P, Q ) = ¯ S ( Q,Q ) -¯ S ( P, Q ) ≥ 0 , and vice versa (see Definition 3.8 and Theorem 4.1 of [43]). This linkage, together with Theorem 2.4, provides justifications to many examples that have been used in distributional RL, including logarithmic scoring rule [e.g., 63, 61] that corresponds to Kullback-Leibler (KL) divergence, and squared maximum mean discrepancy (MMD) with specific kernels [40]. More interestingly, we also find (and analyze) various examples that have never been used in distributional RL, including squared MMD with additional kernels and L 2 distance based on density functions (see Table 2). Finally, our result also provides justifications for adopting other strictly proper scoring rules (e.g., survival, spheric, Hyvärinen, Tsallis, Brier scoring rules) [e.g., 23, 43, 16], which are not analyzed in this work.

Remark 2.5 . With appropriate differentiability condition, the property (9) also implies that expected gradient of d (Υ( S, A ) , Ψ π ( R,S ′ , Υ t -1 )) (with respect to Υ ) becomes zero at Υ = T π Υ t -1 . This is a crucial unbiased gradient property for building TD-based or more general gradient-based algorithms. Despite our focus on FDE, we note that Theorem 2.4 also provides justifications for building such algorithms (e.g., TD update based on squared-MMD in [40, 67]) via functional Bregman divergence.

Next, we discuss the last principle (P3). Unlike (P1) and (P2), which involve a single quantity (either ˜ η or d ), (P3) requires establishing an appropriate relationship between the contraction-inducing metric ˜ η and the discrepancy d . Since each discrepancy d has its own relationship with different probability metrics η (prior to their extension to ˜ η ), the discussion of (P3) becomes specific to each pair (˜ η, d ) . While a certain level of generalization is possible-for the squared form of some probability metric (i.e., d = m 2 ), ˜ η can be controlled under conditions such as Assumption 3.2-this framework does not accommodate other divergences (e.g., KL divergence). Additionally, depending on the cardinality of S × A , different forms of ˜ η may be employed (e.g., ˜ η = η ∞ or ˜ η = ¯ η d π ,q ). Consequently, unlike (P1) and (P2), it is challenging to establish a concise guideline to choose (˜ η, d ) based on (P3). Instead, we study a number of examples (Table 1 for ˜ η = η ∞ and Table 2 for ˜ η = ¯ η d π ,q ). Section 3 provides statistical convergence analyses of FDE methods based on different choices of d .

## 3 Theoretical results

First, we make the modeling in (5) explicit and write M = M Θ := { Υ θ ∈ P S×A : θ ∈ Θ } . We assume that the offline dataset D = { ( s i , a i , r i , s ′ i ) } N i =1 consists of N independently and identically distributed draws according to: ( s i , a i ) ∼ ρ and r i , s ′ i ∼ ˜ p ( ·| s i , a i ) . To simplify the theoretical analysis, we will slightly modify the objective function (5) so that we use non-overlapping subsets of data in each iteration. That is, the data is first split into T equally sized partitions, i.e., D = ∪ t ∈ [ T ] D t with |D t | = n = N/T (assuming that N is divisible by T , without loss of generality), and, at the t -th iteration, we obtain the estimator ˆ θ n,t := arg min θ ∈ Θ ˆ F n,t ( θ | ˆ θ n,t -1 ) where

<!-- formula-not-decoded -->

For convergence of iteration-level error in (6), i.e., ˜ η (Υ ˆ θ n,t , T π Υ ˆ θ n,t -1 ) P → 0 , T π Υ ˆ θ n,t -1 should be accommodated in the chosen model, which is implied by the completeness assumption:

Assumption 3.1. (Completeness) For ∀ θ ∈ Θ , T π Υ θ = Υ θ ′ for some θ ′ ∈ Θ .

This assumption is common in the analysis of many iteration-based RL algorithms [e.g., 13, 61, 63, 60, 24]. By Assumption 3.1, there exists a value θ ∗ ,t ∈ Θ such that Υ θ ∗ ,t = T π Υ ˆ θ n,t -1 .

Our results are divided into the following two subsections. In Section 3.1, we focus exclusively on divergences that can be expressed as squared metrics and consider the tabular setting. In this case, we can obtain near-minimax optimal convergence rates for various FDEs (see Table 1). In Section 3.2, we broaden the analysis to cover a wider range of divergences in both tabular and non-tabular settings (see Table 2), establishing theoretical guarantees for many FDEs under more general conditions.

## 3.1 Squared metric divergence under tabular setting

Under tabular setting (i.e., |S × A| &lt; ∞ ), we shall assume that every state-action pair can be observed with non-zero probability, which is at least p min := min s,a ρ ( s, a ) &gt; 0 , where ρ represents

the probability mass function of ( S, A ) in the data generating distribution. In this subsection, we focus on a class of functional Bregman divergences that are squared metrics, i.e., d = m 2 for some induced metric m associated with an appropriate inner product space ( G , ⟨· , ·⟩ m ) . More specifically, we require that each probability distribution µ ∈ P admits a unique representation G ( ·| µ ) ∈ G , and that m ( µ 1 , µ 2 ) = ∥ G ( ·| µ 1 ) -G ( ·| µ 2 ) ∥ m holds for any µ 1 , µ 2 ∈ P . (It is possible that some element in G does not correspond to any element in P .) The inner product structure supports a stronger theoretical analysis, leading to near-minimax optimal convergence rate established in Theorem 3.3. The precise requirement (including an additional technical condition) for the metric m are formally stated in Definition C.2 of Appendix C.6. We will refer to such metrics as inner-product-space (IPS) metrics. Under mild conditions, squared IPS metrics d = m 2 are functional Bregman divergences (see Appendix C.9.1). Examples of functional Bregman divergence d that can be represented as squared IPS metrics are listed in Table 1. See Appendix C.5 for the their technical constructions.

Regarding (P3), we assume that η can be well-bounded by m in the following assumption.

Assumption 3.2. The probabaility metric η satisfies (S-L-C). For any µ 1 , µ 2 ∈ P , there exist some constants C surr &gt; 0 , δ &gt; 0 , ϵ 0 ≥ 0 such that

<!-- formula-not-decoded -->

Theorem 3.3. Suppose Assumptions 3.1 and 3.2 hold. Moreover, given a probability metric η that satisfies (S-L-C) with convexity parameter q ≥ 1 and scale-sensitivity parameter c &gt; 0 (i.e., (15) of Appendix C.4), consider the FDE with a functional Bregman divergence d = m 2 where m is an IPS metric (Definition C.2 of Appendix C.6). Assume that there exists C max,m ∈ (0 , ∞ ) such that

<!-- formula-not-decoded -->

Then for any δ 0 ∈ (0 , 1) , by letting T = ⌊ δ 2 · 1 c · log 1 /γ ( N log(4 |S×A| /δ 0 ) ) ⌋ , we have, with probability larger than 1 -δ 0 ,

<!-- formula-not-decoded -->

for a constant C &gt; 0 that does not depend on any of γ, N, |S × A| , δ 0 .

See Appendix C.6 for its proof and more detailed bound (21). Table 1 shows valid examples of ( d = m 2 , η ) , along with the parameters presented in Theorem 3.3. The second last column determines the convergence rate ( N δ/ 2 , up to a logarithmic order), which depends on the moment degree r ≥ 1 (defined in Table 1) for those that can cover unbounded distributions. We can see that m = l 2 , d L 2 , MMD cou achieves O P ( N -1 / (2 p ) ) (up to a logarithmic order) in W p, ∞ -error for bounded distributions (i.e., r = ∞ ), where W p, ∞ is the supremum extension (7) of W p metric. For p = 1 , the convergence rate is (near-)optimal, aligning with the minimax optimal convergence rate for W 1 , ∞ -error shown in Theorem B.1 of [45] (up to a logarithmic order). To our knowledge, there is no corresponding minimax result for distributional OPE problems when p &gt; 1 . However, N -1 / (2 p ) is comparable to the optimal convergence rate of empirical probability measure (Theorem 1 of [18]).

Table 1: Comparison of different FDE methods under tabular setting. See Appendix A for definitions of the suggested examples of d = m 2 and η ∞ , and the corresponding probability space P , such that M Θ ⊆ P S×A . p ≥ 1 is an integer, and r &gt; p is such that sup θ ∈ Θ sup s,a E Z ∼ Υ θ ( s,a ) {∥ Z ∥ r } &lt; ∞ .

| d (= m 2 )                                                                                                                                                | η ∞                                                | c                | δ                                                                                   | ϵ 0                                                                           |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------|------------------|-------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| Cramér : l 2 2 MMD-Energy : MMD 2 β PDF- L 2 : d 2 L 2 MMD-Matern : MMD 2 ν MMD-RBF : MMD 2 σ RBF MMD-RBF-transformed : MMD 2 σ,f MMD-Coulomb : MMD 2 cou | W p, ∞ MMD β, ∞ W p, ∞ W p, ∞ W p, ∞ W p, ∞ W p, ∞ | 1 β/ 2 1 1 1 1 1 | 1 /p 1 2( r - p ) ( d +2 r ) p r - p ( d +2 r ) p 2( r - p ) ( d +2 r ) p 1 /p 1 /p | 0 0 0 0 3 / 2 · Γ(( p + d ) / 2) Γ( p/ 2) 3 / 2 · Γ(( p + d ) / 2) Γ( p/ 2) 0 |

## 3.2 General divergence under possibly non-tabular setting

In this subsection, we will discuss general state-action spaces that can be either tabular or non-tabular. As explained in Section 2.3, controlling η ∞ -error (like Theorem Theorem 3.3) by an expectationbased objective function is challenging and requires strong assumptions for non-tabular settings. We now shift our focus to the more compatible (expectation-based) ¯ η d π ,q -bound. In this subsection, we move beyond squared IPS metrics discussed in Section 3.1 and allow d to be more general functional Bregman divergence. Some standard conditions are needed to establish the convergence. Specifically, we assume that the data distribution ( ρ ) sufficiently covers the target distribution ( d π ) (see Assumption C.4), and that the model class satisfies an entropy condition characterized by a complexity coefficient α ∈ (0 , 2) (see (23) of Assumption C.3). The larger the α , the more complex the model class.

Table 2 summarizes the theoretical guarantee for a broad class of d based on our theorems. All error bounds are developed using empirical process theory for M-estimation (see Theorem D.1). However, different classes of d require separate ways to verify the required modulus condition (Condition 2 of Theorem D.1), resulting in different convergence rates. To apply the empirical process theory, we introduce a surrogate metric for the construction of entropy conditions (e.g., see m in Appendices C.8 and C.15). Informally, this surrogate metric is required to be able to control the differences in the objective function. However, unlike Theorem 3.3, we do not require the divergence to be explicitly constructed from the metric, hence accommodating a broader class of divergences. Based on this, we present two general theorems. Theorem C.5 assumes that the surrogate metric is induced by a normed space, while Theorem C.11 applies to general surrogate metrics. Although Theorem C.5 supports a more limited set of divergences, it provides stronger convergence rate guarantees compared to Theorem C.11. In terms of the scope of applicable divergence choices, we have the following inclusion relationships:

Theorem 3.3 (tabular) ⊆ Theorem C.5 (general) ⊆ Theorem C.11 (general) .

However, the strength of the convergence rate guarantees follows the reverse order. Note that Theorem 3.3 only applies to the tabular case. We remark that some important divergences (e.g., KL divergence) are not covered in the aforementioned theorems, requiring separate analysis (see Corollary B.9).

Due to space limitation, we only present a simplified version of Theorem C.5 for reference. The error bound in Theorem C.11 shares a similar form.

Theorem 3.4 (Simplified version of Theorem C.5) . Let η be a probability metric that satisfies (S-L-C) with q ≥ 1 and c &gt; 1 / (2 q ) . Suppose Assumptions 3.1, 3.2, C.3 and C.4 hold. By letting T = ⌊ 1 c -1 2 q · δ ′ 2( l -1)+ α · log 1 /γ N ⌋ with δ ′ = min { δ, 1 /q } , the corresponding FDE achieves

<!-- formula-not-decoded -->

We now turn to Table 2, which provides several examples of divergences d . (See Appendix A for definitions and Corollaries B.1-B.9 in Appendix B for detailed bounds.) The table summarizes the pairs ( d , ¯ η d π ,q ) in Columns 1-2, along with their ability to handle unbounded return distributions, distributions without densities, and their associated return dimensionality considerations (Columns 3-5). If convergence cannot be guaranteed due to a nonzero ϵ 0 , the corresponding convergence rate is omitted. To the best of our knowledge, [63] is the only existing work that provides a statistical convergence analysis for distributional OPE in non-tabular settings. In comparison, our FDE method using d = l 2 2 (the first row of Table 2) consistently achieves faster convergence rates, as compared to the rate of their FLE method obtained in [63], across all degrees of Wasserstein metric p ≥ 1 and model complexities α ∈ (0 , 2) . See (12) of Appendix B.1 for details.

## 4 Experiments

We evaluate our FDE methods with baselines through two experiments 3 : linear quadratic regulator (LQR) and Atari games. LQR is a parametric setting widely studied in RL literature [e.g., 9, 63, 37].

3 Codes are available at https://github.com/hse1223/Fitted-Distributional-Evaluation.git

Table 2: Comparison of different FDE methods under general state-action space. See Appendix A for definitions of examples of d and ¯ η d π ,q . p ≥ 1 is an integer, and r &gt; p satisfies sup θ ∈ Θ sup s,a E Z ∼ Υ θ ( s,a ) {∥ Z ∥ r } &lt; ∞ . Convergence rates are displayed up to a logarithmic order.

̸

| d           | ¯ η d π ,q    | unbounded   | density-free   | d   | rate                                 | reference     |
|-------------|---------------|-------------|----------------|-----|--------------------------------------|---------------|
| l 2 2       | W p,d π ,p    | ✗           | ✓              | 1   | N - 1 p · 1 2+ α - 1                 | Theorem 3.4   |
| MMD 2 β =1  | W 1 ,d π , 1  | ✓           | ✓              | ≥ 1 | N 4( d +1) r r - 1 + α               | Theorem 3.4   |
| MMD 2 β> 1  | MMD β,d π , 1 | ✗           | ✓              | ≥ 1 | N - 1 2+ α                           | Theorem C.11  |
| d 2 L 2     | W p,d π ,p    | ✓           | ✗              | ≥ 1 | N - 2( r - p ) ( d +2 r ) p · 1 2+ α | Theorem 3.4   |
| MMD 2 ν     | W p,d π ,p    | ✓           | ✗              | ≥ 1 | N - ( r - p ) ( d +2 r ) p · 1 2+ α  | Theorem 3.4   |
| MMD 2 σ RBF | W p,d π ,p    | ✓           | ✓              | ≥ 1 | -                                    | Theorem 3.4   |
| MMD 2 σ,f   | W p,d π ,p    | ✗           | ✓              | ≥ 1 | -                                    | Theorem 3.4   |
| MMD 2 cou   | W p,d π ,p    | ✗           | ✗              | = 1 | N - 1 p · (6 - α ) 16                | Theorem C.11  |
| KL          | W p,d π ,p    | ✗           | ✗              | ≥ 1 | N - 1 p · 1 2+ α                     | Corollary B.9 |

For the LQR experiment, we have compared FDE methods (PDFL2, RBF, Matern, Energy, KL). All of these FDE methods outperformed the baseline FLE [63]. Due to space limitations, we defer the details to Appendix E.1.

Atari games are common testbeds for distributional RL methods [e.g., 5, 15, 40, 52, 65]. We compared the proposed FDE methods with FLE [63], QRDQN [15] and IQN [14]. Among these methods, QRDQN and IQN rely on quantile-based modeling, and we adopted Gaussian mixture modeling (GMM, see details in Appendix E.2.1) to model distributions for all other methods. In our experiments, we tested different numbers of mixtures / quantiles ( M = 10 , 100 , 200 ) and considered multiple Atari games: Atlantis, Breakout, Enduro, KunfuMaster, Pong, Qbert, SpaceInvader. We estimated Υ π ∈ P S×A in two different environments (deterministic / random reward), each having different target policies. For each environment, we collected offline data from two different behavior policies (representing strong and weak coverage), with three different sample sizes ( N = 2 K, 5 K, 10 K ). See Appendices E.2.2 and E.2.4 for algorithmic details.

In our simulations, we have included the following methods: (i) baseline methods (FLE, QRDQN, IQN), (ii) our FDE methods (KL, Energy, PDF-L2, RBF), (iii) non-functional Bregman divergence (TVD) that we did not study. Figure 1, which contains these methods in order from left to right for four different games, shows the inaccuracy comparison with N = 10 K and M = 200 under a specific environment and behavior policy. TVD shows no improvement of accuracy through iterations, corroborating our proposal to build the objective functions based on functional Bregman divergence in Section 2.4. On the contrary, our FDE methods (particularly, KL and Energy) not only decrease the inaccuracy, but also outperform the baseline methods in most games with respect to mean inaccuracy and variance.

Figure 1: Mean (dots) and confidence region (mean ± STD) for 5 seeds based on N = 10 K samples: W 1 -inaccuracy (Y-axis) for each method (X-axis) for the games with M = 200 . See Figures 3-5 for simulation results in all seven games.

<!-- image -->

Besides the four settings of Figure 1, we have more results for a wide range of different settings (i.e., seven games, two environments, two behavior policies, three mixture sizes and three sample sizes). Overall, FDE methods outperformed the baseline methods. More details can be found in Appendix E.2. Moveover, we also demonstrated that another functional Bregman divergence-the Hyvärinen divergence, which was not part of our main study-performs well in practice, supporting our discussion in Section 2.4. See Figures 3-5 and Tables 6-33 of Appendix E.2.5.

## 5 Discussion

A central goal of this work is to explore how to choose d for constructing a theoretically sound FDE method. At a high level, different d respond differently to deviation between Υ and Ψ π in (5), much like how various loss functions behave differently in empirical risk minimization. Therefore, the choice of divergence should be tailored to the specific problem at hand. To move toward a principled framework for choosing d in specific applications, we believe the first step is to identify what constitutes a valid candidate. To date, this question remains largely unaddressed in the literature, and only a few valid choices have been identified - let alone systematically compared. Our work makes substantial progress on this front. While comparative analyses of certain functional Bregman divergences do exist and may provide practical guidance, we acknowledge that this work does not yet offer a comprehensive guide for selecting among them in practice.

Besides, we acknowledge several other technical limitations of our study. First, while our framework establishes the sufficiency of using functional Bregman divergence to ensure the property (9) (Theorem 2.4), the necessity remains an open question. Second, for simplicity, our theoretical analysis assumes data splitting, leading to independent samples at each iteration. It would be valuable to extend the analysis to scenarios where data are reused across iterations. Finally, our theorems rely on the completeness assumption (Assumption 3.1), which may be overly restrictive in practical settings. We could introduce a term that quantifies violations of this assumption via the inherent Bellman error [38] and incorporate it into our final bound. We leave such an extension for future work.

## Acknowledgments and Disclosure of Funding

Portions of this research were conducted with the advanced computing resources provided by Texas A&amp;MHigh Performance Research Computing. The work of Jiayi Wang is partly supported by the National Science Foundation (DMS-2401272) and Texas Artificial Intelligence Research Institute (TAIRI).

## References

- [1] A. Agarwal, S. M. Kakade, J. D. Lee, and G. Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. Journal of Machine Learning Research , 22(98):1-76, 2021.
- [2] P. Alquier and M. Gerber. Universal robust regression via maximum mean discrepancy. Biometrika , 111(1):71-92, 2024.
- [3] A. Banerjee, X. Guo, and H. Wang. On the optimality of conditional expectation as a bregman predictor. IEEE Transactions on Information Theory , 51(7):2664-2669, 2005.
- [4] M. G. Bellemare, S. Candido, P. S. Castro, J. Gong, M. C. Machado, S. Moitra, S. S. Ponda, and Z. Wang. Autonomous navigation of stratospheric balloons using reinforcement learning. Nature , 588(7836):77-82, 2020.
- [5] M. G. Bellemare, W. Dabney, and R. Munos. A distributional perspective on reinforcement learning. In International conference on machine learning , pages 449-458. PMLR, 2017.
- [6] M. G. Bellemare, W. Dabney, and M. Rowland. Distributional reinforcement learning . MIT Press, 2023.
- [7] M. G. Bellemare, I. Danihelka, W. Dabney, S. Mohamed, B. Lakshminarayanan, S. Hoyer, and R. Munos. The cramer distance as a solution to biased wasserstein gradients. arXiv preprint arXiv:1705.10743 , 2017.

- [8] C. Bodnar, A. Li, K. Hausman, P. Pastor, and M. Kalakrishnan. Quantile qt-opt for risk-aware vision-based robotic grasping. arXiv preprint arXiv:1910.02787 , 2019.
- [9] S. Bradtke. Reinforcement learning applied to linear quadratic regulation. Advances in neural information processing systems , 5, 1992.
- [10] S. J. Bradtke and A. G. Barto. Linear least-squares algorithms for temporal difference learning. Machine learning , 22:33-57, 1996.
- [11] D. Chafaï, A. Hardy, and M. Maïda. Concentration for coulomb gases and coulomb transport inequalities. Journal of Functional Analysis , 275(6):1447-1483, 2018.
- [12] S. Chaudhari, D. Arbour, G. Theocharous, and N. Vlassis. Distributional off-policy evaluation for slate recommendations. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 8265-8273, 2024.
- [13] J. Chen and N. Jiang. Information-theoretic considerations in batch reinforcement learning. In International Conference on Machine Learning , pages 1042-1051. PMLR, 2019.
- [14] W. Dabney, G. Ostrovski, D. Silver, and R. Munos. Implicit quantile networks for distributional reinforcement learning. In International conference on machine learning , pages 1096-1105. PMLR, 2018.
- [15] W. Dabney, M. Rowland, M. Bellemare, and R. Munos. Distributional reinforcement learning with quantile regression. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 32, 2018.
- [16] A. P. Dawid and M. Musio. Theory and applications of proper scoring rules. Metron , 72(2):169183, 2014.
- [17] A. Fawzi, M. Balog, A. Huang, T. Hubert, B. Romera-Paredes, M. Barekatain, A. Novikov, F. J. R. Ruiz, J. Schrittwieser, G. Swirszcz, et al. Discovering faster matrix multiplication algorithms with reinforcement learning. Nature , 610(7930):47-53, 2022.
- [18] N. Fournier and A. Guillin. On the rate of convergence in wasserstein distance of the empirical measure. Probability theory and related fields , 162(3):707-738, 2015.
- [19] B. A. Frigyik, S. Srivastava, and M. R. Gupta. Functional bregman divergence and bayesian estimation of distributions. IEEE Transactions on Information Theory , 54(11):5130-5139, 2008.
- [20] S. A. Geer. Empirical Processes in M-estimation , volume 6. Cambridge university press, 2000.
- [21] M. Geist, B. Piot, and O. Pietquin. Is the bellman residual a bad proxy? Advances in Neural Information Processing Systems , 30, 2017.
- [22] J. Gerstenberg, R. Neininger, and D. Spiegel. On policy evaluation algorithms in distributional reinforcement learning. arXiv preprint arXiv:2407.14175 , 2024.
- [23] T. Gneiting and A. E. Raftery. Strictly proper scoring rules, prediction, and estimation. Journal of the American statistical Association , 102(477):359-378, 2007.
- [24] N. Golowich and A. Moitra. Linear bellman completeness suffices for efficient online reinforcement learning with few actions. In The Thirty Seventh Annual Conference on Learning Theory , pages 1939-1981. PMLR, 2024.
- [25] A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and A. Smola. A kernel two-sample test. The Journal of Machine Learning Research , 13(1):723-773, 2012.
- [26] S. Hong, Z. Qi, and R. K. Wong. Distributional off-policy evaluation with bellman residual minimization. arXiv preprint arXiv:2402.01900 , 2024.
- [27] A. Huang, L. Leqi, Z. Lipton, and K. Azizzadenesheli. Off-policy risk assessment for markov decision processes. In International Conference on Artificial Intelligence and Statistics , pages 5022-5050. PMLR, 2022.

- [28] A. Korba, F. Bach, and C. Chazal. Statistical and geometrical properties of the kernel kullbackleibler divergence. Advances in Neural Information Processing Systems , 37:32536-32569, 2024.
- [29] I. Kostrikov and O. Nachum. Statistical bootstrapping for uncertainty estimation in off-policy evaluation. arXiv preprint arXiv:2007.13609 , 2020.
- [30] H. Le, C. Voloshin, and Y. Yue. Batch policy learning under constraints. In International Conference on Machine Learning , pages 3703-3712. PMLR, 2019.
- [31] C. E. Luis, A. G. Bottero, J. Vinogradska, F. Berkenkamp, and J. Peters. Value-distributional model-based reinforcement learning. Journal of Machine Learning Research , 25(298):1-42, 2024.
- [32] Y. Ma, D. Jayaraman, and O. Bastani. Conservative offline distributional reinforcement learning. Advances in neural information processing systems , 34:19235-19247, 2021.
- [33] O.-A. Maillard, R. Munos, A. Lazaric, and M. Ghavamzadeh. Finite-sample analysis of bellman residual minimization. In Proceedings of 2nd Asian Conference on Machine Learning , pages 299-314. JMLR Workshop and Conference Proceedings, 2010.
- [34] E. Mammen and S. van de Geer. Penalized quasi-likelihood estimation in partial linear models. The Annals of Statistics , 25(3):1014-1035, 1997.
- [35] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al. Human-level control through deep reinforcement learning. nature , 518(7540):529-533, 2015.
- [36] T. Modeste and C. Dombry. Characterization of translation invariant mmd on rd and connections with wasserstein distances. Journal of Machine Learning Research , 25(237):1-39, 2024.
- [37] A. N. Moghaddam, A. Olshevsky, and B. Gharesifard. Sample complexity of the linear quadratic regulator: A reinforcement learning lens. arXiv preprint arXiv:2404.10851 , 2024.
- [38] R. Munos and C. Szepesvári. Finite-time bounds for fitted value iteration. Journal of Machine Learning Research , 9(5), 2008.
- [39] O. Nachum, Y. Chow, B. Dai, and L. Li. Dualdice: Behavior-agnostic estimation of discounted stationary distribution corrections. Advances in neural information processing systems , 32, 2019.
- [40] T. Nguyen-Tang, S. Gupta, and S. Venkatesh. Distributional reinforcement learning via moment matching. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 35, pages 9144-9152, 2021.
- [41] S. Nietert, Z. Goldfeld, and K. Kato. Smooth p -wasserstein distance: structure, empirical approximation, and statistical applications. In International Conference on Machine Learning , pages 8172-8183. PMLR, 2021.
- [42] E. Odin and A. Charpentier. Dynamic Programming in Distributional Reinforcement Learning . PhD thesis, Université du Québec à Montréal, 2020.
- [43] E. Y. Ovcharov. Proper scoring rules and bregman divergence. Bernoulli , 24(1):53-79, 2018.
- [44] V. M. Panaretos and Y. Zemel. Statistical aspects of wasserstein distances. Annual review of statistics and its application , 6:405-431, 2019.
- [45] Y. Peng, L. Zhang, and Z. Zhang. Near minimax-optimal distributional temporal difference algorithms and the freedman inequality in hilbert spaces. CoRR , 2024.
- [46] J. C. Perdomo, A. Krishnamurthy, P. Bartlett, and S. Kakade. A complete characterization of linear estimators for offline policy evaluation. Journal of machine learning research , 24(284):150, 2023.

- [47] M. Rowland, R. Munos, M. G. Azar, Y. Tang, G. Ostrovski, A. Harutyunyan, K. Tuyls, M. G. Bellemare, and W. Dabney. An analysis of quantile temporal-difference learning. Journal of Machine Learning Research , 25(163):1-47, 2024.
- [48] E. Saleh and N. Jiang. Deterministic bellman residual minimization. In Proceedings of Optimization Foundations for Reinforcement Learning Workshop at NeurIPS , 2019.
- [49] D. Sejdinovic, B. Sriperumbudur, A. Gretton, and K. Fukumizu. Equivalence of distance-based and rkhs-based statistics in hypothesis testing. The annals of statistics , pages 2263-2291, 2013.
- [50] R. Selten. Axiomatic characterization of the quadratic scoring rule. Experimental Economics , 1(1):43-61, 1998.
- [51] B. Sen. A gentle introduction to empirical process theory and applications. Lecture Notes, Columbia University , 11:28-29, 2018.
- [52] K. Sun, Y. Zhao, Y. Liu, W. Liu, B. Jiang, and L. Kong. Distributional reinforcement learning via sinkhorn iterations. arXiv preprint arXiv:2202.00769 , 2022.
- [53] R. S. Sutton and A. G. Barto. Reinforcement learning: An introduction . MIT press, 2018.
- [54] S. van de Geer. Regression analysis and empirical processes . CWI, 1988.
- [55] A. W. Van der Vaart and J. A. Wellner. Weak convergence . Springer, 1996.
- [56] T. Vayer and R. Gribonval. Controlling wasserstein distances by kernel norms with application to compressive statistical learning. Journal of Machine Learning Research , 24(149):1-51, 2023.
- [57] R. Vershynin. High-dimensional probability: An introduction with applications in data science , volume 47. Cambridge university press, 2018.
- [58] C. Voloshin, H. M. Le, N. Jiang, and Y. Yue. Empirical study of off-policy policy evaluation for reinforcement learning. arXiv preprint arXiv:1911.06854 , 2019.
- [59] J. Wang, Z. Qi, and R. K. Wong. A fine-grained analysis of fitted q-evaluation: beyond parametric models. arXiv preprint arXiv:2406.10438 , 2024.
- [60] K. Wang, O. Oertell, A. Agarwal, N. Kallus, and W. Sun. More benefits of being distributional: Second-order bounds for reinforcement learning. arXiv preprint arXiv:2402.07198 , 2024.
- [61] K. Wang, K. Zhou, R. Wu, N. Kallus, and W. Sun. The benefits of being distributional: Smallloss bounds for reinforcement learning. Advances in Neural Information Processing Systems , 36, 2023.
- [62] R. Wang, S. S. Du, L. Yang, and R. R. Salakhutdinov. On reward-free reinforcement learning with linear function approximation. Advances in neural information processing systems , 33:17816-17826, 2020.
- [63] R. Wu, M. Uehara, and W. Sun. Distributional offline policy evaluation with predictive error guarantees. arXiv preprint arXiv:2302.09456 , 2023.
- [64] P. R. Wurman, S. Barrett, K. Kawamoto, J. MacGlashan, K. Subramanian, T. J. Walsh, R. Capobianco, A. Devlic, F. Eckert, F. Fuchs, et al. Outracing champion gran turismo drivers with deep reinforcement learning. Nature , 602(7896):223-228, 2022.
- [65] D. Yang, L. Zhao, Z. Lin, T. Qin, J. Bian, and T.-Y. Liu. Fully parameterized quantile function for distributional reinforcement learning. Advances in neural information processing systems , 32, 2019.
- [66] A. Zanette. When is realizability sufficient for off-policy reinforcement learning? In International Conference on Machine Learning , pages 40637-40668. PMLR, 2023.
- [67] P. Zhang, X. Chen, L. Zhao, W. Xiong, T. Qin, and T.-Y. Liu. Distributional reinforcement learning for multi-dimensional reward functions. Advances in Neural Information Processing Systems , 34:1519-1529, 2021.

- [68] R. Zhang, B. Dai, L. Li, and D. Schuurmans. Gendice: Generalized offline estimation of stationary values. arXiv preprint arXiv:2002.09072 , 2020.
- [69] R. Zhang, X. Zhang, C. Ni, and M. Wang. Off-policy fitted q-evaluation with differentiable function approximators: Z-estimation and inference theory. In International Conference on Machine Learning , pages 26713-26749. PMLR, 2022.
- [70] Y. Zhang, X. Cheng, and G. Reeves. Convergence of gaussian-smoothed optimal transport distance with sub-gamma distributions and dependent samples. In International Conference on Artificial Intelligence and Statistics , pages 2422-2430. PMLR, 2021.

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1-2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

IMPORTANT, please:

- Delete this instruction block, but keep the section heading 'NeurIPS Paper Checklist" ,
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers .

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Abstract contains the summary of the paper's contributions.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We mentioned limitations of our research in the last section of our main text.

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

Justification: Every theorems and lemmas explicitly mention the necessary assumptions.

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

Justification: We have elaborated the algorithmic details used in constructing experiment settings.

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

Justification: The data that we used are either simulated on our own, or already made public by OpenAI. All the simulation details are in Appendices E.1 and E.2. After acceptance of the paper, we disclosed the URL to the github repository that contains our Python codes (that we have created) which are used for simulations.

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

Justification: We have specified all the algorithmic details including hyperparameters in Appendices E.1 and E.2.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The simulation results contain the mean inaccuracy and standard deviation, thereby providing an error bar.

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

Justification: We specified all the computation details in Appendix E.3.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our research complies with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our research is purely about methodology of machine learning. Our goal lies in presenting methods and proving their theoretical convergences.

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

Justification: Our research does not contain any data or models that could be risky. We have used only public open-source data that is available from OpenAI.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The data that we use are open-source data, available from OpenAI.

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

Justification: The data that we use are open-source data, available from OpenAI. There are new assets whose documentations are needed. After acceptance, we only added URL to the github repository that contains our Python codes (that we have created) which are used in simulations.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our research does not contain any crowdsourcing or study over human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: In our study, there was no participation of human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

## Answer: [NA]

Justification: LLM is not used in developing the core methodology.

## Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Table of contents in Appendix

| A Definitions of metrics and divergences   | A Definitions of metrics and divergences   | A Definitions of metrics and divergences                         | 24   |
|--------------------------------------------|--------------------------------------------|------------------------------------------------------------------|------|
| A.1                                        |                                            | Contraction-inducing metrics . . . . . . . . . . . . . . . . . . | 24   |
|                                            | A.2 . . . . .                              | Divergences . . . . . . . . . . . . . . . . . . . . . .          | 24   |
| B                                          | FDE methods for general state-action space |                                                                  | 26   |
| B.1                                        | Cramér . . . . . .                         | FDE . . . . . . . . . . . . . . . . . . . . .                    | 26   |
| B.2                                        |                                            | Energy FDE . . . . . . . . . . . . . . . . . . . . . . . . . . . | 26   |
| B.3                                        | . .                                        | PDF-L2 FDE . . . . . . . . . . . . . . . . . . . . . . . .       | 27   |
|                                            | B.4 .                                      | MMD-Matern FDE . . . . . . . . . . . . . . . . . . . . . .       | 27   |
|                                            | B.5 . .                                    | MMD-RBF FDE . . . . . . . . . . . . . . . . . . . . . .          | 28   |
|                                            | B.6                                        | MMD-RBF-transformed FDE . . . . . . . . . . . . . . . . .        | 28   |
|                                            | B.7                                        | MMD-Coulomb FDE . . . . . . . . . . . . . . . . . . . . . .      | 29   |
|                                            | B.8 .                                      | KL FDE . . . . . . . . . . . . . . . . . . . . . . . . . . . .   | 29   |
| C                                          | Technical                                  | proofs                                                           | 30   |
|                                            | C.1                                        | Functional Bregman divergence . . . . . . . . . . . . . . . .    | 30   |
|                                            |                                            | C.1.1 .                                                          |      |
|                                            |                                            | Technical definition . . . . . . . . . . . . . . . . . .         | 30   |
|                                            |                                            | C.1.2 Proof of Theorem 2.4 . . . . . . . . . . . . . . . . . .   | 30   |
|                                            | C.2                                        | How bounding iteration-level error leads to final inaccuracy . . | 31   |
| C.3                                        |                                            | How extended metrics become contraction-inducing metrics .       | 31   |
|                                            | C.3.1                                      | Expectation-extension . . . . . . . . . . . . . . . . .          | 31   |
|                                            | C.3.2                                      | Supremum-extension . . . . . . . . . . . . . . . . . .           | 32   |
| C.4                                        | Examples                                   | of probability metric . . . . . . . . . . . . . . . . .          | 32   |
| C.5                                        | Functional spaces and corresponding norms  | . . . . . . . . . .                                              | 33   |
| C.6                                        | Proof for Theorem 3.3                      | . . . . . . . . . . . . . . . . . . . . .                        | 34   |
|                                            | C.6.1                                      | Outline of proof . . . . . . . . . . . . . . . . . . . .         | 34   |
|                                            | C.6.2 . . . . .                            | Proof . . . . . . . . . . . . . . . . . . . . .                  | 35   |
| C.7                                        | Parameters of Examples in Table 1          | . . . . . . . . . . . . . . .                                    | 37   |
| C.8                                        | Theorem C.5 and its proof                  | . . . . . . . . . . . . . . . . . . .                            | 38   |
|                                            | C.8.1                                      | Outline of proof . . . . . . . . . . . . . . . . . . . . .       | 39   |
|                                            | C.8.2                                      | Convergence in a single step . . . . . . . . . . . . . .         | 39   |
|                                            | C.8.3                                      | Convergence through multiple steps . . . . . . . . . .           | 40   |

|         | C.8.4 Satisfaction of the condition of Lemma C.7 in Theorem C.5 . .                                                                            | . . . . . . 41                |
|---------|------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| C.9     | About IPS metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                    | . . . . . . 43                |
|         | C.9.1 How squared IPS metric becomes a functional Bregman divergence                                                                           | . . . . 43                    |
|         | C.9.2 How IPS Properties imply NS Properties . . . . . . . . . . . .                                                                           | . . . . . . 43                |
| C.10    | Proof for Corollary B.1 . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                    | . . . . . . 45                |
| C.11    | Proof of Corollary B.2 . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                     | . . . . . . 46                |
| C.12    | Proof for Corollary B.4 . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                    | . . . . . . 48                |
| C.13    | Proof of Corollary B.5 . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                     | . . . . . . 48                |
| C.14    | Proofs based on Gaussian regularizer . . . . . . . . . . . . . . . . . . .                                                                     | . . . . . . 48                |
|         | C.14.1 Proof for Lemma C.9 . . . . . . . . . . . . . . . . . . . . . . .                                                                       | . . . . . . 49                |
|         | C.14.2 Proof for Corollary B.6 . . . . . . . . . . . . . . . . . . . . . .                                                                     | . . . . . . 49                |
|         | C.14.3 Proof for Corollary B.7 . . . . . . . . . . . . . . . . . . . . . .                                                                     | . . . . . . 49                |
| C.15    | Alternative for violation of (NS4) . . . . . . . . . . . . . . . . . . . . .                                                                   | . . . . . . 50                |
|         | C.15.1 Proof for Theorem C.11 . . . . . . . . . . . . . . . . . . . . .                                                                        | . . . . . . 51                |
|         | C.15.2 Proof for Corollary B.8 . . . . . . . . . . . . . . . . . . . . . .                                                                     | . . . . . . 53                |
|         | C.15.3 Proof for Corollary B.3 . . . . . . . . . . . . . . . . . . . . . .                                                                     | . . . . . . 54                |
| C.16    | Proof for Corollary B.9 . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                    | . . . . . . 54                |
|         | C.16.1 Quadratic lower bound . . . . . . . . . . . . . . . . . . . . . .                                                                       | . . . . . . 55                |
|         | C.16.2 Convergence of modulus . . . . . . . . . . . . . . . . . . . . .                                                                        | . . . . . . 55                |
|         | C.16.3 Consistence . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                     | . . . . . . 57                |
|         | C.16.4 Finalization . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                    | . . . . . . 58                |
| C.17    | Comparison and examples on Cramér FDE . . . . . . . . . . . . . . .                                                                            | . . . . . . 59                |
|         | C.17.1 Comparison between Cramér FDE and FLE . . . . . . . . . . .                                                                             | . . . . . . 59                |
|         | C.17.2 Example 1: Linear MDP . . . . . . . . . . . . . . . . . . . . .                                                                         | . . . . . . 60                |
|         | C.17.3 Example 2: Linear Quadratic Regulator . . . . . . . . . . . . .                                                                         | . . . . . . 61                |
| D       | Auxiliary proofs for Appendix C                                                                                                                | 63                            |
| D.1     | Supporting theorem . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                     | . . . . . . 63                |
| D.2 D.3 | Proof of Lemma C.7 . . . . . . . . . . . . . . . . . . . . . . . . . . . Proof of Lemma C.8                                                    | . . . . . . 63                |
|         | . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                          | . . . . . . 63                |
| D.4     | Proof of Lemma C.12 . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                      | . . . . . . 64                |
|         | D.4.1 Conditioned event . . . . . . . . . . . . . . . . . . . . . . . . D.4.2 Concentration of modulus . . . . . . . . . . . . . . . . . . . . | . . . . . . 64 . . . . . . 65 |
|         | Satisfaction of the condition of Lemma C.7 in Theorem C.11                                                                                     |                               |
| D.5     | . . . . . .                                                                                                                                    | . . . . . . 66                |
| E       | Experiment Details                                                                                                                             | 67                            |
| E.1     | E.1.1 . . . . . . . . . . . . . . . . . . . .                                                                                                  | . . . . . . 67                |
|         | Data collection and model E.1.2 Data splitting . . . . . . . . . . . . . . . . . . . . . . . . . . .                                           | . . . . . . 67                |
|         | E.1.3 gaussian distributions . . . . . . . . . . . . . .                                                                                       | . . . . . . 68                |
|         | Closed form for                                                                                                                                |                               |

|     | E.1.4                                                                                     | Simulation results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 68         |
|-----|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| E.2 | Details in Atari games . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 70 | Details in Atari games . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 70 |
|     | E.2.1                                                                                     | Deep Neural Network structure . . . . . . . . . . . . . . . . . . . . . . . 70            |
|     | E.2.2                                                                                     | Algorithmic details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 70        |
|     | E.2.3                                                                                     | Closed / approximated form . . . . . . . . . . . . . . . . . . . . . . . . . 71           |
|     | E.2.4                                                                                     | Simulation settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 72        |
|     | E.2.5                                                                                     | Simulation results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 72         |
| E.3 | Computation resources . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 84  | Computation resources . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 84  |

## A Definitions of metrics and divergences

Following is the generalized algorithm of FDE methods.

Input: Functional Bregman divergence d (Tables 1-2), Model M , Initial Υ 0 ∈ M , Data D Output: Υ

Algorithm 1 Fitted distributional evaluation T for t = 1 to T do Perform the minimization (5) to obtain Υ t . end for

For Tables 1 and 2 of Section 3, we shall define the necessary concepts needed to understand the objective functions and contraction-inducing metrics.

## A.1 Contraction-inducing metrics

Following are the individual metrics that satisfy (S-L-C) and their supremum (7) and expectationextensions (8) (which are contraction-inducing metrics (Definition 2.2)) that we used in Tables 1 and 2.

Wasserstein-p metric is defined as follows, with J ( µ 1 , µ 2 ) being the possible joint distributions of X ∼ µ 1 and Y ∼ µ 2 :

<!-- formula-not-decoded -->

Its supremum and expectation extensions are defined as follows, with their contractive factors ( ζ of Definition 2.2)

<!-- formula-not-decoded -->

MMD-Energy is defined as the square-rooted form of following MMD-squared (with X,X ′ ∼ µ 1 and Y, Y ′ ∼ µ 2 all being indepenent)

<!-- formula-not-decoded -->

Its supremum and expectation extensions are defined as follows, with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A.2 Divergences

Functional Bregman divergences (choices suggested in Tables 1 and 2) are what determine the objective functions and their corresponding choices of P , whose elements can be used as inputs of d . Our model should satisfy M Θ ⊆ P S×A .

For Cramér FDE, we use d = l 2 2 and corresponding P :

<!-- formula-not-decoded -->

P = ∆([ b 1 , b 2 ]) := { probability measures of distributions bounded in [ b 1 , b 2 ] ⊊ R } .

For Energy FDE, we use d = MMD 2 β and corresponding P ( X,X ′ ∼ µ 1 and Y, Y ′ ∼ µ 2 being mutually independent):

<!-- formula-not-decoded -->

P = { µ ∈ ∆( R d ) ( d ≥ 1) : ∥ E X ∼ µ { k β ( X, · ) }∥ H &lt; ∞} with H β being the RKHS of kernel k β .

For PDF-L2 FDE, we use d = d 2 L 2 and corresponding P :

<!-- formula-not-decoded -->

For MMD-Matern FDE, we use d = MMD 2 ν and corresponding P ( X,X ′ ∼ µ 1 and Y, Y ′ ∼ µ 2 being mutually independent):

with

<!-- formula-not-decoded -->

For MMD-RBF FDE, we use d = MMD 2 σ RBF and corresponding P ( X,X ′ ∼ µ 1 and Y, Y ′ ∼ µ 2 being mutually independent):

<!-- formula-not-decoded -->

For MMD-RBF-transformed FDE, we use d = MMD 2 ( σ,f ) and corresponding P (with conditions of function f and definition of I f are mentioned in Equations 8 and 10 of [70]) ( X,X ′ ∼ µ 1 and Y, Y ′ ∼ µ 2 being mutually independent):

<!-- formula-not-decoded -->

For MMD-Coulomb FDE, we use d = MMD 2 cou and corresponding P ( X,X ′ ∼ µ 1 and Y, Y ′ ∼ µ 2 being mutually independent):

<!-- formula-not-decoded -->

P = { µ ∈ ∆( R d ) ( d ≥ 2) : ∥ E X ∼ µ { k cou ( X, · ) }∥ H &lt; ∞} with H cou being the RKHS of k cou .

For KL FDE, we use d ( µ 1 , µ 2 ) = KL ( µ 2 , µ 1 ) and corresponding P :

<!-- formula-not-decoded -->

## B FDE methods for general state-action space

## B.1 Cramér FDE

We construct our objective function (10) based on Cramér distance d = l 2 2 , and refer to this as Cramér FDE. Assuming bounded variables in R , i.e., Z ( s, a ; θ ) ∈ [ a, b ] for all s, a ∈ S × A and θ ∈ Θ , we obtain the following based on Theorem C.5, with proof and detailed bound in Appendix C.10.

Corollary B.1. Under Assumptions 3.1, C.4, our estimator (10) based on d = l 2 2 achieves the following in bounded support in R , say [ a, b ] .

<!-- formula-not-decoded -->

Here, ≲ means upper-bounded by RHS multiplied by some constant that does not depend on γ and N .

Although Cramér FDE is limited in bounded uni-dimensional distributions, we can provide direct comparison with FLE under exactly the same conditions, at least for d = 1 . Assuming bounded return distribution with the same model complexity log N [ ] ( ˜ F Θ , ∥ · ∥ L 1 , ∞ , ϵ ) ≲ ϵ -α (where ˜ F Θ represents the conditional pdf family modeled by Θ ), Cramér FDE achieves faster convergence rate than FLE uniformly for all p ≥ 1 and α ∈ (0 , 2) . See Appendix C.17.1 for its proof.

<!-- formula-not-decoded -->

Here, ˜ O P indicating the convergence rate, allowing up to logarithmic difference from the conventional O P . We have suggested two examples for Cramér FDE, in which FLE cannot provide a valid bound due to no conditional densities. The first example is linear MDP (Appendix C.17.2) that is frequently assumed in both traditional and distributional RL [e.g., 62, 46, 61]. Cramér FDE achieves W p,d π ,p -convergence rate of ˜ O P ( N -1 / (3 p ) ) . The second example is Linear Quadratic Regulator (Appendix C.17.3), which has been also been frequently mentioned in traditional RL [e.g., 9, 37], and was recently applied in distributional RL under finite-horizontal setting [63]. We extended this towards the infinite-horizontal setting. For bounded distributional families, Cramér FDE achieves W p,d π ,p -convergence rate of ˜ O P ( N -1 / (2 p ) ) . For unbounded distributional families, its generalized extension Energy FDE (introduced in the following subsection) achieves W 1 ,d π , 1 -convergence rate of ˜ O P ( N -1 / 8 ) .

## B.2 Energy FDE

Now, we will assume multi-dimensional return ( d ≥ 1 ), but still bounded. Here, we use d = MMD 2 β with β ∈ (1 , 2) , which we name as Energy FDE. MMD β is the MMD (17) with the following

<!-- formula-not-decoded -->

Unlike the tabular setting (supported by Theorem 3.3) where we could apply β ∈ (0 , 1) , we have to limit to β ∈ (0 , 1) . But we can extend the result into β = 1 (second statement of Corollary B.2). Considering that l 2 2 = 1 2 MMD 2 β =1 for d = 1 [7], Energy FDE can be viewed as an extension of Cramér FDE. Proofs and detailed bounds are in Appendix C.11.

Corollary B.2. Under Assumptions 3.1, C.4, our estimator (10) based on d = MMD 2 β with β ≥ 1 can achieve the following bound in bounded support of R d ,

<!-- formula-not-decoded -->

This result can further be used to bound W 1 ,d π , 1 -inaccuracy based on the relationship between MMD β and W 1 based on (36), which is introduced by [36]. Based on W p ( µ, ν ) ≤ D 1 -1 p · W 1 /p 1 ( µ, ν ) (suggested by [44]) where D ∈ (0 , ∞ ) represents the diameter of support of bounded distributions, we can further bound W p,d π ,p (Υ θ T , Υ π ) ≤ D 2(1 -1 p ) · W 1 /p 1 ,d π , 1 (Υ θ T , Υ π ) . However, these are still restricted in bounded distributions.

Therefore, let us introduce the case where Energy FDE can bound the inaccuracy for unbounded distributions even without s, a -conditional densities, based on W 1 ,d π , 1 -inaccuracy. Towards that end, we assume that the r -th moment ( r &gt; 1) of the distributions are uniformly bounded, i.e., M r := sup θ ∈ Θ ( E ∥ Z ( s, a ; θ ) ∥ r ) 1 /r &lt; ∞ . See Appendix C.15.3 for proof, which is based on the alternative Theorem C.11.

Corollary B.3. Under Assumptions 3.1, C.4, our estimator (10) based on d = MMD 2 β ( β = 1 ), we have following. Assuming that sup θ ∈ Θ sup s,a E ∥ Z ( s, a ; θ ) ∥ &lt; ∞ and sup s,a ∥ R ( s, a ) ∥ ψ 2 &lt; ∞ where R ( s, a ) indicates the reward vector conditioned on s, a , we have

<!-- formula-not-decoded -->

Here, having finite values of higher moments (i.e., larger r &gt; 1 ) leads to tighter convergence rate. Bounded variables ( r = ∞ ) will give us l = 2 d +3 , leading to W 1 ,d π , 1 (Υ θ T , Υ π ) = ˜ O P ( N -1 4( d +1)+ α ) . In case of d = 1 and r = ∞ , it does not degenerate into that of Corollary B.1, since they are based upon different proof structures.

## B.3 PDF-L2 FDE

Now let us construct the objective function (10) with d = d 2 L 2 , namely PDF-L2 method. By assuming existence of conditional densities, PDF-L2 method can achieve faster convergence rate for unbounded distributions than Energy FDE (Corollary B.3). Proof and detailed bound are in Appendix C.12.

Corollary B.4. Under Assumptions 3.1, C.4, our estimator (10) based on d = d 2 L 2 can achieve the following bound in R d . Assuming M r := sup θ ∈ Θ sup s,a ( E ∥ Z ( s, a ; θ ) ∥ r ) 1 /r &lt; ∞ with r &gt; p and sup θ ∈ Θ sup z,s,a | f θ ( z | s, a ) | &lt; ∞ .

<!-- formula-not-decoded -->

We can see that PDF-L2 method not only provides W p,d π ,p -bound for all p ≥ 1 , but also provides faster convergence rate than Energy FDE (Corollary B.3) in the case of p = 1 , for all r ≥ 1 , d ≥ 1 , and α ∈ (0 , 2) .

## B.4 MMD-Matern FDE

[56] has shown that MMD based on translation invariant kernels, i.e., k ( x , y ) = κ 0 ( x -y ) , can bound Wasserstein metrics when the distributions have smooth enough densities (see their Theorem 15). One such example is MMD-Matern, which is the MMD (17) with following kernel with parameters ν &gt; 0 and σ Mat &gt; 0 (assumed to be fixed) and K ν being modified Bessel function,

<!-- formula-not-decoded -->

We will treat σ Mat &gt; 0 as a fixed constant since we are more interested in ν . It is well-known that ν = 1 corresponds to Laplace kernel and ν →∞ corresponds to RBF kernel. It leads to following result (proof and detailed bound in Appendix C.13).

Corollary B.5. Under Assumptions 3.1, C.4, our estimator (10) based on d = MMD 2 ν can achieve the following bound in R d . Assuming M r := sup θ ∈ Θ sup s,a ( E ∥ Z ( s, a ; θ ) ∥ r ) 1 /r &lt; ∞ with r &gt; p and its conditional densities having Sobolev norms ∥ f θ ( ·| s, a ) ∥ H ν + d/ 2 ( R d ) ≤ B ( &lt; ∞ ) , we have

<!-- formula-not-decoded -->

Although this can be applied for unbounded distributions as PDF-L2 method (Corollary B.4), its bound is always looser than that of PDF-L2 method, as shown in its convergence rate which is half times that of PDF-L2 method. This is since the proof of bounding Wasserstein metric by MMD-Matern is based on the following logic.

<!-- formula-not-decoded -->

See Section 2.4 of [56] for details. This applies for all possible MMD's associated with translationinvariant kernels, other than MMD-Matern.

## B.5 MMD-RBF FDE

Since MMD-Matern that is shown above covers the case of 0 &lt; ν &lt; ∞ and requires smooth enough densities. MMD-RBF (corresponding to MMD-Matern with ν = ∞ ) is used a lot in many machine learning areas (e.g., kernel support vector machine).

<!-- formula-not-decoded -->

It is also used in distributional reinforcement learning (MMDRL by [40]). They only used it in simulations, without any theoretical justification due to their failure to provide a contraction-inducing metric based on MMD-RBF. Following lemma can provide justification of its practice (proof and detailed bound in Appendix C.14.2), not requiring the existence of conditional densities.

Corollary B.6. Under Assumptions 3.1, C.4, assuming M r := sup θ ∈ Θ sup s,a ( E ∥ Z ( s, a ; θ ) ∥ r ) 1 /r &lt; ∞ with r &gt; p , our estimator (10) based on d = MMD 2 σ RBF with σ RBF let = σ N &gt; 0 can achieve the following bound in R d ,

<!-- formula-not-decoded -->

Fixing σ N = σ RBF &gt; 0 leads to an irreducible term at the end. Of course, we can shrink it by letting σ N → 0 . However, other terms (e.g., C ( p, d, σ N , p ) , C brack ( σ N ) /σ d N ) can increase with σ N → 0 , leaving the optimal rate of σ N to be intractable.

Although we cannot show that MMD-RBF can shrink Wasserstein inaccuracy to zero, it can bound gaussian-smoothed Wasserstein metric W σ p ( µ, ν ) := W p ( µ ∗ α σ , ν ∗ α σ ) where ∗ indicates convolution and α σ is the probability measure of N ( 0 , σ 2 I d ) . This is widely used as an alternative for Wasserstein metric that is difficult to be computed in high dimensions d &gt; 1 [e.g., 70, 41]. Thus, we have made a separate lemma (Lemma C.9) that can accommodate many other probability distances that can bound smoothed gaussain Wasserstein metric, which includes MMD-Two-Moment.

## B.6 MMD-RBF-transformed FDE

[70] suggested an MMD (17) with the following kernel which is RBF kernel multiplied with an extra term. We will refer to its corresponding MMD as MMD-RBF-transformed, denoting it as MMD ( σ,f ) .

<!-- formula-not-decoded -->

The conditions of density function f and definition of I f are mentioned in Equations 8 and 10 of [70]. Assuming bounded kernel k ( σ,f ) ( · , · ) &lt; ∞ for fixed value of σ &gt; 0 , we can achieve the following. Proof and detailed bound are in Appendix C.14.3.

Corollary B.7. Under Assumptions 3.1, C.4, our estimator (10) based on d = d 2 L 2 can achieve the following bound in R d , if we have k ( σ,f ) &lt; ∞ for fixed value of σ &gt; 0 .

<!-- formula-not-decoded -->

We can let σ N → 0 as N →∞ . However, the optimal rate is intractable.

One example of such f is generalized beta-prime distributions: f ( x ) = ϵ 2 πx · (( x λ ) -ϵ +( x λ ) ϵ ) -1 with ϵ ∈ (0 , d +2 p ] and λ ∈ (0 , ∞ ) , which leads to Two-moment kernel (see Definition 1 of [70]). They have shown that such k ( σ,f ) is bounded in their Section 3.2 for bounded distributions.

## B.7 MMD-Coulomb FDE

MMD-Coulumb is an MMD which corresponds to the following kernel.

<!-- formula-not-decoded -->

We can see that k ( · , · ) &lt; ∞ does not hold, leading to violation of the first statement of (23). Since we cannot apply Theorem C.5, we should resort to Theorem C.11 instead, which gives us the following result for MMD-Coulomb. Proof and detailed bound are in Appendix C.15.2.

Corollary B.8. Under Assumptions 3.1, C.4, our estimator (10) based on d = MMD 2 cou can achieve the following bound. Assuming bounded support in R d ( d ≥ 2 ) and sup θ ∈ Θ sup s,a E cou (Υ θ ( s, a )) &lt; ∞ with E cou ( µ ) := k cou ( x , y ) µ (d x ) µ (d y ) , we have

<!-- formula-not-decoded -->

We can further bound W p,d π ,p (Υ θ T , Υ π ) ≤ D 2(1 -1 p ) · W 1 /p 1 ,d π , 1 (Υ θ T , Υ π ) if we assume bounded conditional distributions. Note that the condition E (Υ θ ( s, a )) &lt; ∞ does not allow nonzero probability to any points.

## B.8 KL FDE

Interestingly enough, convergence of FLE [63] can also be shown by Theorem D.1. However, since it is based on maximium likelihood, which is not a squared metric, we could fit this into the structure of neither Theorem C.5 nor C.11. Instead, we developed a method that uses log-likelihood (or equivalently, KL divergence), based on Theorem D.1. See Appendix C.16 for its proof. Note that this is different from FLE in the sense that it fully utilizes the closed form density of Υ ˆ θ n,t -1 ( s ′ , a ′ ) and π ( a ′ | s ′ ) needed for Ψ π ( r, s ′ , Υ ˆ θ n,t -1 ) in computing the objective function (10). This is free from MCerror caused by sampling from z ∼ Ψ π ( r, s ′ , Υ ˆ θ n,t -1 ) as [63] did.

Corollary B.9. Under Assumptions 3.1, C.4, our estimator (10) with d = KL (Kullback-Leibler divergence) achieves the following bound. Assuming bounded distributions and existence of conditional densities, we have following with T = ⌊ 1 1 -1 2 p · 1 /p 2+ α · log 1 /γ N ⌋ ,

<!-- formula-not-decoded -->

Here, we have log N [ ] ( ˜ F 1 / 2 Θ , ∥ · ∥ L 2 ,ρ , ϵ ) ≤ C brack · ϵ -α for some α ∈ (0 , 2) with ˜ F 1 / 2 Θ being the squared conditional densities, i.e., ˜ F 1 / 2 Θ := { f 1 / 2 θ ( ·| · · · ) : θ ∈ Θ } .

## C Technical proofs

## C.1 Functional Bregman divergence

## C.1.1 Technical definition

Bregman divergence was originally developed to measure discrepancy between two real vectors. It measures the difference between the value of the convex function ϕ at x ∈ R d and its first-order approximation at y ∈ R d :

<!-- formula-not-decoded -->

After defining a scalar-valued convex function Φ over a functional inner product space, they can also be extended to functional objects based on Frechet derivative [19].

<!-- formula-not-decoded -->

Based on the fact that each probability measure can have its own functional representation (e.g., density functions), we can also extend functional Bregman divergence into discrepancy between two probability measures. Following is the technical definition of a functional Bregman divergence for two probability measures P, Q ∈ P (Section 3.2 of [43]).

Definition C.1. Linear span of P , namely span P , is a collection of signed measures, and let U ⊆ span P be a convex subset that contains P , i.e., P ⊆ U . Let L ( P ) be the functional space where its arbitrary element f satisfies ∫ Ω X ∣ ∣ f ∣ ∣ d µ &lt; ∞ for ∀ µ ∈ P with Ω X being the support space. Now, assume a strictly convex function Φ : U → R that has a subgradient Φ ∗ : U → L ( P ) which satisfies Φ( Q ) ≥ Φ ∗ ( P ) · ( Q -P ) + Φ( P ) for all P, Q ∈ U , where f · µ := ∫ Ω X f d µ for any f ∈ L ( P ) and µ ∈ span P . Then we can build a functional Bregman divergence d : U × U → R + on U as follows (Definition 3.8 of [43]),

<!-- formula-not-decoded -->

Since functional Bregman divergence in Definition C.1 is based on a strictly convex function Φ , equality d ( P, Q ) = 0 holds only when P = Q . This is needed to ensure that T π Υ becomes the unique minimizer of (9). Note that this is stronger than general definition of functional Bregman divergence (e.g., Definition 3.8 of [43]) that is based on standard convex function (which may not be strictly convex).

## C.1.2 Proof of Theorem 2.4

We will treat both S, A ∼ ρ and R,S ′ ∼ ˜ p ( · · · | S, A ) to be random. Let Υ 1 , Υ 2 ∈ P S×A be arbitrary. We obtain the following, by using the same logic that [3] used in their Theorem 1,

<!-- formula-not-decoded -->

Letting Υ 2 = Υ and Υ 1 = T π Υ , we have

<!-- formula-not-decoded -->

Then, the following holds based on T π Υ 2 ( s, a ) = E { Ψ π ( R,S ′ , Υ 2 ) | s, a } ,

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

where ¯ d ρ (Υ 1 , Υ 2 ) := E { d (Υ 1 ( S, A ) , Υ 2 ( S, A )) } . Note that we definition of our empirical and population objective functions (5) and (9) already require Ψ π ( r, s ′ ; Υ) ∈ P . Otherwise, they will not be defined. Then, by convexity of P , we have T π Υ( s, a ) ∈ P , since it is a convex combination over r, s ′ ∼ p ( ·| s, a ) (see definition (3)). This leads to satisfaction of (9), since P is a convex set.

To ensure Ψ π ( r, s ′ ; Υ) ∈ P , we can assume that P is closed under push-forward mapping with respect to single-sample Bellman backup functions { g r,γ } for all r ∈ R d that can be observed. For a single-sampe Bellman backup function defined as g r,γ ( x ) := r + γx and X ∼ µ , its result of push-forward mapping ( g r,γ ) # µ is defined as the distribution of r + γX . Being closed under push-forward mapping means that ( g r,γ ) # µ ∈ P holds for any probability measure µ ∈ P and any r ∈ R d that can be observed. Then, for an arbitrary Υ ∈ P S×A , we can ensure that Ψ π ( r, s ′ , Υ) ∈ P holds for any target policy π : S → ∆( A ) and observable ( r, s ′ ) ∈ R d ×S , by its definition (4) and convexity of P . Then, our objective functions (5) and (9) can always be defined.

## C.2 How bounding iteration-level error leads to final inaccuracy

By Definition 2.2, we can prove (6) based on Υ π = T π Υ π :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This leads to (6).

## C.3 How extended metrics become contraction-inducing metrics

## C.3.1 Expectation-extension

Let us show that the expectation-extended (8) metric satisfies Definition 2.2.

<!-- formula-not-decoded -->

Here, the (A) part can be bounded as follows based on Properties (15). Letting ˜ p π ( r, s ′ , a ′ | s, a ) = ˜ p ( r, s ′ | s, a ) · π ( a ′ | s ′ ) and temporarily using it as a probability measure, we have

<!-- formula-not-decoded -->

where we have used the (C) property of (15) in the second line, and the (S) and (L) properties of (15) in the last line. Then, we have

<!-- formula-not-decoded -->

We acknowledge that the last line can be shown by the trick used by [63], i.e., E ˜ s, ˜ a ∼ d π ˜ p π ( s, a | ˜ s, ˜ a ) ≤ γ -1 · d π ( s, a ) . This finally leads to

<!-- formula-not-decoded -->

## C.3.2 Supremum-extension

We can apply the same logic to supremum-extension (7). Taking up from (14), we have

<!-- formula-not-decoded -->

Taking supremum over S × A on the LHS gives us η q ∞ ( T π Υ 1 , T π Υ 2 ) ≤ γ cq · η ∞ (Υ 1 , Υ 2 ) , leading to η ∞ ( T π Υ 1 , T π Υ 2 ) ≤ γ c · η ∞ (Υ 1 , Υ 2 ) . Although Theorem 4.25 of [6] presented the same statement, their proof requires an additional assumption, i.e., R and S ′ (generated from ˜ p ( · · · | s, a ) ) being independent. We did not require this assumption.

## C.4 Examples of probability metric

To construct a metric ˜ η over P S×A that satisfies Definition 2.2, we will assume to have a probability metric η over P that satisfies the following three properties of (15) with some c &gt; 0 and q ≥ 1 . We copied the terminologies from [6] (see their Definitions 4.22, 4.23, 4.24). In the first two properties, L ( X ) denotes the probability measure of random vector X ∈ R d , and µ 1 , µ 2 : X → P , ν ∈ ∆( X ) for some space X can be arbitrary in the third property.

<!-- formula-not-decoded -->

- (L) location-insensitive : η ( L ( X + z ) , L ( Y + z )) ≤ η ( L ( X ) , L ( Y )) for an arbitrary z ∈ R d

<!-- formula-not-decoded -->

Note that if some q ≥ 1 satisfies the Property-3, then any q ′ ≥ q also satisfies it by Jensen's inequality. Here are the examples that satisfy the properties in (15). We specify the values of c, q &gt; 0 for each η . Then, we specify the contractive factor ζ = γ c -1 2 q for their corresponding expectation-extended metrics (8). (Note that the supremum-extension (7) has ζ = γ c .)

<!-- formula-not-decoded -->

Here are the definition of above metrics. W p is the Wasserstein metric, which is defined as follows with J ( µ 1 , µ 2 ) being the possible joint distributions of X ∼ µ 1 and Y ∼ µ 2 ,

<!-- formula-not-decoded -->

It is straightforward to see Properties 1 and 2 hold with c = 1 . Property 3 is shown to hold by [26] (see their Appendix A.3.1).

MMD(maximum-mean discrepancy) [25] associated with a kernel k ( · , · ) defined as

<!-- formula-not-decoded -->

with X , X ′ ∼ µ 1 and Y , Y ′ ∼ µ 2 are all independent. MMD β is associated with Energy kernel k β ( x , y ) := ∥ x ∥ β + ∥ y ∥ β -∥ x -y ∥ β ( 0 &lt; β &lt; 2 ), Properties 1 and 2 are straightforward t show with c = β/ 2 . Property 3 can be shown with q = 1 , since MMD can be expressed as the norm of difference between two mean embeddings, i.e., MMD k ( µ 1 , µ 2 ) = ∥ κ µ 1 -κ µ 2 ∥ H where κ µ is the mean embedding of µ and ∥ · ∥ H is the corresponding RKHS norm of RKHS H .

<!-- formula-not-decoded -->

l p is the metric between cdf's of 1-dimensional random variable. Letting F i be the cdf of µ i , we define

<!-- formula-not-decoded -->

Properties 1 and 2 are shown with c = 1 /p by Proposition 3.2 of [42]. Property 3 can be shown by using the fact that l p is a functional L p norm between cdf's. The same logic holds for d L p .

Table 3 also contains the survey of other well-known probability metrics. We will skip proof for their satisfaction or non-satisfaction of each property. With f i and F i being the pdf and cdf of probability measure µ i ,

<!-- formula-not-decoded -->

Table 3: Survey of metrics: If it satisfies c -scale-sensitive or q -convexity, the corresponding value of c &gt; 0 or q ≥ 1 are specified.

|                                              | Location-insensitive   | Scale-sensitive   | Convex                          |
|----------------------------------------------|------------------------|-------------------|---------------------------------|
| Wasserstein-p ( W p )                        | ✓                      | c = 1             | q ≥ p q ≥ 1 q ≥ 1 q ≥ 1 q ≥ 1 ✗ |
| MMD-Energy ( MMD β with 0 < β < 2 )          | ✓                      | c = β/ 2          |                                 |
| MMD-Matern ( MMD ν with 0 < ν ≤∞ )           | ✓                      | ✗                 |                                 |
| CDF- L p ( l p )                             | ✓                      | c = 1 /p          |                                 |
| PDF- L p ( d L p ) : p = 1 refers to 2 × TVD | ✓                      | ✗                 |                                 |
| Hellinger metric                             | ✓                      | c = d/ 4          |                                 |
| Discrepancy metric                           | ✓                      | ✗                 | q ≥ 1                           |
| Kolmogorov metric                            | ✓                      | ✗                 | q ≥ 1                           |

In Table 3, MMD's associated with any kernel satisfy convexity with q = 1 . However, MMD-Energy was so far the only example that we have found, which satisfies all three properties. In the definitions of Wasserstein metric and MMD-Energy (Appendix C.4), the Euclidean norm ∥ · ∥ used can be replaced with another norm on R d . See Example 11 of [23] and Proposition 3 of [49] for extensions of MMD-Energy, and [44] for extensions of Wasserstein metrics.

## C.5 Functional spaces and corresponding norms

Following are examples of the functional inner product space ( G , ⟨· , ·⟩ m ) and its corresponding probability space P that we have used for our methods in Section 3.1.

For m = l 2 (only applying for d = 1 ), the functional representer is the cdf, denoted by F ( ·| µ ) . The functional space is the functional L 2 space, that is G := { F : X → R : ∫ | F ( z ) | 2 d z &lt; ∞} with X = [ b 1 , b 2 ] ⊊ R being a bounded support. The equipped inner product ⟨· , ·⟩ m = ⟨· , ·⟩ L 2 is defined as ⟨ F 1 , F 2 ⟩ L 2 := ∫ X F 1 ( z ) F 2 ( z )d z . The functional representer of Υ θ ( s, a ) ∈ P , namely F θ ( ·| s, a ) will be an element of the following set F :

<!-- formula-not-decoded -->

For m = d L 2 , the functional representer is the pdf, denoted by f ( ·| µ ) . The functional space is the functional L 2 space, that is G := { f : X → R : ∫ | f ( z ) | 2 d z &lt; ∞} with X ⊆ R d being the support. The equipped inner product ⟨· , ·⟩ m = ⟨· , ·⟩ L 2 is defined as ⟨ f 1 , f 2 ⟩ L 2 := ∫ X f 1 ( z ) f 2 ( z )d z . The functional representer of Υ θ ( s, a ) ∈ P , namely f θ ( ·| s, a ) will be an element of the following set ˜ F :

<!-- formula-not-decoded -->

For m = MMD k (maximum mean discrepancy associated with some kernel), the functional representer is the kernel mean embedding, κ ( ·| µ ) = E Z ∼ µ { k ( Z, · ) } . The functional space is the RKHS corresponding to the given kernel, i.e., G = { ∑ n i =1 α i · k ( x i , · ) | α i ∈ R , x i ∈ X , n ∈ N } with X ⊆ R d being the support. The equipped inner product is the corresponding RKHS inner product, ⟨· , ·⟩ m = ⟨· , ·⟩ H that is defined as ⟨ ∑ n i =1 α i k ( x i , · ) , ∑ m j =1 β j k ( x ′ j , · ) ⟩ H := ∑ n i =1 ∑ m j =1 α i β j k ( x i , x ′ j ) . The functional representer of Υ θ ( s, a ) ∈ P , namely κ θ ( ·| s, a ) will be an element of the following set H :

<!-- formula-not-decoded -->

## C.6 Proof for Theorem 3.3

Definition C.2 (Inner-product-space metric) . The probability metric m over P is called an innerproduct-space metric (IPS metric, in short) if there exists an inner product space ( G , ⟨· , ·⟩ m ) where any element µ ∈ P can be uniquely (up to equivalence under zero induced norm ∥ · ∥ m ) represented by the corresponding element G ( ·| µ ) ∈ G such that

- (IPS1) The probability m is expressed as the norm of difference between functional representers, i.e., m ( µ 1 , µ 2 ) := ∥ G ( ·| µ 1 ) -G ( ·| µ 2 ) ∥ m , where ∥ · ∥ m is the induced norm by the inner product;
- (IPS2) Functional representers of probability measures are unbiased with respect to probability mixture. That is, for µ ( · ) : Z → P and arbitrary probability P over Z , we have G ( ·| ∫ µ ( z )d P ( z )) = ∫ G ( ·| µ ( z ))d P ( z ) .

It is possible that there exists an element in G that does not represent any µ ∈ P .

## C.6.1 Outline of proof

The first step is to obtain the convergence of single iteration at t -th step, i.e., η ∞ (Υ ˆ θ n,t , T π Υ ˆ θ n,t -1 ) = O P (1 / √ n ) . This is similar to Lemma 10 of [2], which demonstrates that empirical probability measure converges towards the underlying probability measure in O P (1 / √ n ) rate with respect to MMD. We can generalize the argument to IPS metrics (listed in Appendix C.5) that have corresponding inner product spaces of functional elements. We can form the analogy to the empirical probability measure under tabular setting to simplify the arguments.

The second step is to obtain convergence of T different steps' estimations. Here, there are two things that play a role in determining t -th estimation, i.e., the previous iterate and the data used in the t -th iteration (denoted as D t , or the t -th subdataset). Although the previous iterate depends on all previous datasets, i.e., D 1 , · · · , D t -1 , we can utilize the independence among datasets.

## C.6.2 Proof

Assuming |S × A| &lt; ∞ , each state-action pair can be observed with probability at least p min := inf s,a ρ ( s, a ) where ρ ( s, a ) = P { S, A = s, a } . We can always assume p min &gt; 0 without loss of generality, since we can exclude s, a with ρ ( s, a ) &gt; 0 from S × A . Denoting the empirical estimate of ρ ( s, a ) and ˜ p ( r, s ′ | s, a ) as ˆ ρ ( s, a ) and ˆ p ( r, s ′ | s, a ) , we can copy the logic of (13) to obtain the following equivalence for ˆ θ n,t defined in (10),

<!-- formula-not-decoded -->

Here, ˆ T π is defined accordingly to (3), only replacing ˜ p ( r, s ′ | s, a ) with ˆ p ( r, s ′ | s, a ) .

Now, we restrict d = m 2 into squared form of probability metric so that it satisfies relaxed triangular inequality, that is, m 2 ( P, Q ) ≤ 2 · ( m 2 ( P, R ) + m 2 ( R,Q )) . With fixed ˆ θ n,t -1 and its corresponding value fo θ ∗ ,t such that Υ θ ∗ ,t = T π Υ ˆ θ n,t -1 , we have following,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Given that Υ θ ∗ ,t = T π Υ ˆ θ n,t -1 , this comes down to showing ˆ T π Υ ˆ θ n,t -1 ( s, a ) converges in m towards T π Υ ˆ θ n,t -1 ( s, a ) for each s, a . Towards that end, we will assume that each s, a is observed sufficiently many times. We define the following event along with two probability vectors in R S×A ,

<!-- formula-not-decoded -->

By Lemma A.3 of [26], we have

<!-- formula-not-decoded -->

Note that conditioning on Ω 0 has no effect on the randomness of r, s ′ ∼ ˜ p ( · · · | s, a ) for each s, a . Denote its conditional expectation and probability as ˜ E ( · · · ) := E ( · · · | Ω 0 ) and ˜ P ( · · · ) := P ( · · · | Ω 0 ) . Based on the IPS Properties of Definition C.2 and (11), we obtain the following for a fixed s, a , assuming that we have collected { ( s, a, r i , s ′ i ) } n ( s,a ) i =1 for the s, a . We acknowledge that this proof is highly based on Lemma 10 of [2].

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

Then, by McDiarmid's inequality (e.g., Theorem 2.9.1 of [57]), we obtain

<!-- formula-not-decoded -->

Now considering the conditional probabilty for Ω 0 , we have sup s,a m { Υ θ ∗ ,t ( s, a ) , ˆ T π Υ ˆ θ n,t -1 ( s, a ) } ≤ 1 √ n · √ 2 p min · C 1 / 2 max,m + t with probability larger than

<!-- formula-not-decoded -->

Letting t 2 = 4 C 2 max,m n · p min · log( 4 |S×A| δ 0 ) for arbitrary δ 0 ∈ (0 , 1) , given that we have sufficiently large n such that

<!-- formula-not-decoded -->

we have following with probability larger than 1 -δ 0 ,

<!-- formula-not-decoded -->

Note that we have following under Ω 0 that leads to ρ ( s, a ) ≤ 2ˆ ρ ( s, a ) ,

<!-- formula-not-decoded -->

Then, we have following with probability larger than 1 -δ 0 ,

<!-- formula-not-decoded -->

Under Assumption 3.2, we have η ∞ ( ˆ θ n,t , θ ∗ ,t ) ≤ C surr · m δ ∞ ( ˆ θ n,t , θ ∗ ,t ) + ϵ 0 , which allows us to bound η ∞ (Υ θ t , T π Υ ˆ θ n,t -1 ) for all t ∈ [ T ] . Now letting n = N T , we have following with probability larger than 1 -T · δ 0 , the inaccuracy η ∞ (Υ θ T , Υ π ) can be bounded by following based on (6),

<!-- formula-not-decoded -->

under the premise that (20) holds with n = N/T . This is possible since different subdatasets ( D 1 , · · · , D T ) are non-overlapping and independently collected.

Now we choose T = ⌊ δ 2 · 1 c · log 1 /γ ( N log(4 |S×A| /δ 0 ) ) ⌋ , which also leads to ζ T ≈ ( 1 N · log( 4 |S×A| δ 0 )) δ/ 2 . For sufficiently large N , (20) holds with n = N/T and δ 0 replaced with δ 0 /T for sufficiently large N . Note that we have

<!-- formula-not-decoded -->

Thus, with probability larger than 1 -δ 0 , we have

<!-- formula-not-decoded -->

## C.7 Parameters of Examples in Table 1

For all other examples except for the first two examples of Table 1, the values of C surr and C max,m are the same with what we have derived in the proofs of corollaries of Appendix B. For their values, refer to the proof of each corollary of its subsections (which are shown in Appendices C.10-C.16).

Here, we will only present the first two examples of Table 1, whose values of C surr and C max,m are different from those of Appendix B.

For m = l 2 , we need to assume that return distributions are 1-dimension and bounded, i.e., Z ( s, a ; θ ) ∈ [ a, b ] for all s, a ∈ S × A and θ ∈ Θ . By Lemma C.8 and E = 2 l 2 2 , we have C surr = ( b -a ) 1 -1 2 p and δ = 1 /p ,

<!-- formula-not-decoded -->

We also have C max,m = C · ( b -a ) 1 / 2 .

For m = MMD β , we need to assume D &lt; ∞ (where D &gt; 0 is the diameter of support). Since the corresponding kernel can be considered as k ( x , y ) = -∥ x -y ∥ β , we have C max,m = C · D β/ 2 . We have C surr = 1 and δ = 1 since η = m .

## C.8 Theorem C.5 and its proof

Now we shall present and prove the non-simplified version of Theorem 3.4. For our analysis, we define the population objective function and the target parameter at each iteration:

<!-- formula-not-decoded -->

̸

where Z t := { ( s, a ) | ( s, a, r, s ′ ) ∈ D t } . Note that we have θ ∗ ,t = arg min θ ∈ Θ F n,t ( θ | ˆ θ n,t -1 ) . Like Section 3.1, our objective functions should have connection with the surrogate metric m , although d may not necessarily be the squared form of m (i.e., d = m 2 ). With m being a probability metric for P and extended metrics ¯ m ρ, 1 , ¯ m n, 1 , m ∞ for M⊆P S×A (defined in (24)), they should satisfy following Assumption C.3 with ∥ · ∥ ψ 2 ( n ) being Z t -conditioned subgaussian norm (defined in (30) of Appendix C.10).

Assumption C.3 (Norm-space association) . The functional Bregman divergence d and probability metric m over P form a norm-space association (NS association, in short) if there exists an norm space ( G , ∥ · ∥ m ) where any element µ ∈ P can be uniquely (up to equivalence under zero norm ∥ · ∥ m ) represented by the corresponding element G ( ·| µ ) ∈ G such that

- (NS1) The Z t -conditioned modulus ∆ n ( θ | ˆ θ n,t -1 ) := { ˆ F n,t ( θ | ˆ θ n,t -1 ) -F n,t ( θ | ˆ θ n,t -1 ) } -{ ˆ F n,t ( θ ∗ ,t | ˆ θ n,t -1 ) -F n,t ( θ ∗ ,t | ˆ θ n,t -1 ) } is a mean-zero sub-gaussian process with respect to ¯ m n, 1 . That is, for any Z t ⊂ S × A and ˆ θ n,t -1 ∈ Θ , we have

<!-- formula-not-decoded -->

- (NS2) We have F n,t ( θ | ˆ θ n,t -1 ) -F n,t ( θ ∗ ,t | ˆ θ n,t -1 ) ≥ λ · ¯ m l n, 1 ( θ, θ ∗ ,t ) for some l ≥ 2 . Here, the value of λ &gt; 0 does not depend on Z t , θ , ˆ θ n,t -1 (or θ ∗ ,t such that Υ θ ∗ ,t = T π Υ ˆ θ n,t -1 ).
- (NS3) ¯ m n, 1 ( ˆ θ n,t , θ ∗ ,t ) → P 0 converges in a rate that does not depend on ˆ θ n,t -1 (and thereby θ ∗ ,t ) and the deterministic sequence Z t .
- (NS4) Elements of M Θ ⊆ P S×A has a corresponding element in G Θ ⊆ G S×A where G is a functional norm space with ∥ · ∥ m , such that ∥ G µ 1 -G µ 2 ∥ m := m ( µ 1 , µ 2 ) . With the extended norm ∥ · ∥ m, ∞ corresponding to m ∞ (see (24)) and some α ∈ (0 , 2) , it satisfies

<!-- formula-not-decoded -->

It is possible that there exists an element in G that does not represent any µ ∈ P .

Under coverage assumption (Assumption C.4), we can obtain Theorem C.5, with its proof in the subsections of Appendix C.8.

Assumption C.4. (Coverage) C cover := sup θ, ˜ θ ∈ Θ ¯ m 2 d π , 1 (Υ θ , T π Υ ˜ θ ) / ¯ m 2 ρ, 1 (Υ θ , T π Υ ˜ θ ) &lt; ∞ .

Theorem C.5. Suppose Assumptions 3.1, 3.2, C.4 hold. Moreover, given a probability metric η that satisfies (S-L-C) with convexity parameter q ≥ 1 and scale-sensitivity parameter c &gt; 1 / (2 q ) (i.e., (15) of Appendix C.4), consider the FDE with a functional Bregman divergence d satisfying Assumption C.3 of Appendix C.8. Then, by letting T = ⌊ 1 c -1 2 q · δ ′ 2( l -1)+ α · log 1 /γ N ⌋ with δ ′ = min { δ, 1 /q } , we have

<!-- formula-not-decoded -->

where C 0 &gt; 0 is a constant that contains the effect of the parameters C surr , C cover , C brack , C subg , etc (defined in (27) of Appendix C.8). Note that we need q &gt; 1 / (2 c ) to ensure γ c -1 2 q ∈ (0 , 1) .

## C.8.1 Outline of proof

The first step (Appendix C.8.2) is to obtain the convergence of single iteration at t -th step. Temporarily assuming that the state-action pairs Z t are given (i.e., fixed or conditioned), we obtain the convergence with respect to the semi-metric ¯ m n, 1 (24) that is based on Z t . Towards that end, we resort to standard M-estimation theory (Theorem D.1) Unlike the tabular case (Theorem 3.3), the model complexity plays a role in the convergence rate. Then, we can obtain the same convergence rate in (sample-free metric) ¯ m ρ, 1 , based on the fact that the probability metric m has a corresponding functional norm space (see Theorem 2.3 of [34]).

The second step (Appendix C.8.3) is to combine the convergences of multiple iterations into the final bound for the T -th iteration, just like we did in Theorem 3.3 (Appendix C.6). So, it shares many similar ideas. However, unlike the previous Theorem 3.3, this requires us to handle the coverage. The inaccuracy metric ¯ η d π ,q is constructed based on d π , whereas the collected state-action pairs are generated by ρ . We need to take into account the misalignment between the two underlying distributions, quantified by C cover (Assumption C.4).

## C.8.2 Convergence in a single step

Lemma C.6. Assume that our objective functions (10) and (22) and η satisfy Assumption C.3. Conditioning on any given value of ˆ θ n,t -1 (previous iteration value), under Assumptions 3.1, C.4, we have

<!-- formula-not-decoded -->

The proof of Lemma C.6 is as follows. Let D t : { ( s i , a i , r i , s ′ i ) } n i =1 .

With the norm representation of the surrogate metric m , i.e., m ( µ 1 , µ 2 ) := ∥ G µ 1 ( · ) -G µ 2 ( · ) ∥ m , then we can build its extension as follows:

<!-- formula-not-decoded -->

Define Θ δ n := { θ ∈ Θ : ¯ m n, 1 ( θ, θ ∗ ,t ) ≤ δ } . By (NS1), we can apply Theorem 8.1.3 of [57] to obtain

<!-- formula-not-decoded -->

Let C bound := sup θ ∈ Θ sup z,s,a | G θ ( z | s, a ) | ∈ (0 , ∞ ) . Here, we let G ′ θ ( ·| · · · ) := 1 C bound · G θ ( ·| · · · ) and G ′ Θ := { G ′ θ | G θ ∈ G Θ } . Note that we have G ′ θ ( z | s, a ) ≤ 1 and

<!-- formula-not-decoded -->

This leads to

<!-- formula-not-decoded -->

We have consistency ¯ m n, 1 ( ˆ θ n,t , θ ∗ ,t ) → P 0 by (NS3). Then, we can apply Theorem D.1 to obtain

<!-- formula-not-decoded -->

Using 1 2+ α ≥ 1 2( l -1)+ α due to l ≥ 2 , we can apply logic of Theorem 6.3.2 of [54] (or equivalently, Theorem 2.3 of [34]), based on G ′ θ ( z | s, a ) ≤ 1 and (25). This gives us the same asymptotic bound for ¯ m ρ, 1 ( ˆ θ n,t , θ ∗ ,t ) , and further applying Assumption C.4 gives us Lemma C.6.

## C.8.3 Convergence through multiple steps

Now let us bound ¯ η d π ,q ( ˆ θ n,t , θ ∗ ,t ) with q ≥ 1 , with Assumption 3.2. We use ( a + b ) 2 q ≤ 2 2 q -1 · ( a 2 q + b 2 q ) to obtain the following with q ′ let = qδ :

<!-- formula-not-decoded -->

Assume a bounded random variable X such that | X | ≤ C maxim . Since | X/C maxim | ≤ 1 , we have the following if 1 ≤ p 1 ≤ p 2 .

<!-- formula-not-decoded -->

If we have 1 &lt; q ′ , letting p 1 = 2 , p 2 = 2 q ′ gives us

<!-- formula-not-decoded -->

If we have q ′ ≤ 1 , then we have ¯ η d π ,q ( ˆ θ n,t , θ ∗ ,t ) ≤ 2 C surr · ¯ m δ d π , 1 ( ˆ θ n,t , θ ∗ ,t )+2 1 -1 2 q · ϵ 0 by Jensen's inequality (even for q ′ &lt; 1 ). Thus, we have following with δ ′ = min { δ, 1 /q } ,

<!-- formula-not-decoded -->

Restoring the iteration index t ∈ [ T ] (i.e., θ t = ˆ θ n,t ), we have following by (6),

<!-- formula-not-decoded -->

Recall that the data (which are independently collected) are split into non-overlapping subdatasets D = ∪ T t =1 D t (see Section 3). This ensures that ˆ θ n,t -1 can be considered as a given constant at each t -th iteration.

Lemma C.7. With X n,t = ˜ C · O P ( n -1 /b ) for some b &gt; 0 which can be possibly dependent for different t ∈ N , assuming lim sup n →∞ sup t ∈ N E | n 1 /b · X n,t | &lt; ∞ , then we have ∑ T t =1 ζ T -t · X n,t = ˜ C · O P ( 1 1 -ζ · n -1 /b ) . See Appendix D.2 for proof.

In our case, letting X n,t = ¯ m δ ′ d π , 1 (Υ ˆ θ n,t , T π Υ ˆ θ n,t -1 ) (where n affects ˆ θ n,t = arg min θ ∈ Θ ˆ F n,t ( θ | ˆ θ n,t -1 ) ), we can show that the condition of Lemma C.7 can be satisfied. We will defer the proof to Appendix C.8.4 for the sake of simplicity. Then, combining with Lemma C.6 and letting n = N/T with T = ⌊ 1 c -1 2 q · δ ′ 2( l -1)+ α · log 1 /γ N ⌋ , we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

with C maxim := sup θ, ˜ θ ∈ Θ m ∞ (Υ θ , T π Υ ˜ θ ) . For C maxim = ∞ , we let C 0 maxim = 1 . This validates Theorem C.5.

## C.8.4 Satisfaction of the condition of Lemma C.7 in Theorem C.5

Stage-1: Apply peeling argument Arbitrarily choose an integer M ∈ N and a sequence ( δ n ) ≥ 1 . we can use the logic of standard proof of convergence in M-estimation (e.g., Section 5.1 of [51]). Define Z t := { ( s i , a i ) | ( s i , a i , r i , s ′ i ) ∈ D t } and S n,j := { θ ∈ Θ : 2 j -1 · δ n &lt; ¯ m n, 1 ( θ, θ ∗ ,t ) ≤ 2 j · δ n } , and we can obtain the following,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For arbitrary ϵ &gt; 0 , we can choose M ∈ N and δ n &gt; 0 be such that 2 M · δ n = ϵ . Then, we have

<!-- formula-not-decoded -->

Now letting ϵ 1 -α/ 2 -l = 1 √ n · ˜ ϵ for arbitrary ˜ ϵ &gt; 0 , we have following with b = 2( l -1) + α &gt; 2 and C 2 &gt; 0 not depending on ˆ θ n,t -1 (and thereby θ ∗ ,t ),

<!-- formula-not-decoded -->

Based on the above tail probability, we can bound the expectation as follows,

<!-- formula-not-decoded -->

Note that the bound does not depend on Z t .

Stage-2: Convert the ¯ m n, 1 -bound into ¯ m ρ, 1 -bound First fix c ∈ (0 , 1) . Based on Lemma 6.3.4 of [54], we have L c , α c &gt; 0 (that do not depend on ˆ θ n,t -1 ) such that following holds

<!-- formula-not-decoded -->

Denoting the event as Ω E and note that this only regards to the observation of Z t and not the r i , s ′ i ∼ ˜ p ( · · · | s i , a i ) . Then, we have

<!-- formula-not-decoded -->

The last line holds due to (28), since Ω E only regards to Z t . Also, C 3 &gt; 0 is some large value such that C maxim · exp( -α c · L 2 c · n b -2 b ) ≤ C 3 · n -1 /b holds for sufficiently large n .

Stage-3: Showing the satisfaction We can further derive the following.

<!-- formula-not-decoded -->

for some constant C 4 &gt; 0 that does not depend on n or t . Then, the condition of Lemma C.7 is satisfied with degree ˜ b = 2( l -1)+ α δ ′ .

## C.9 About IPS metrics

## C.9.1 How squared IPS metric becomes a functional Bregman divergence

Assume an IPS metric (Definition C.2) where G ( ·| P ) can be defined for dirac delta P = δ x . This accommodates (i) G ( ·| P ) being cdf with ⟨· , ·⟩ L 2 and (ii) G ( ·| P ) being kernel mean embedding with ⟨· , ·⟩ H . See Appendix C.5 for their detailed explanation. Define the score S ( P, x ) := -∥ G ( ·| P ) -G ( ·| δ x ) ∥ 2 m where δ x denotes dirac delta at x . Then we have

<!-- formula-not-decoded -->

Since we have -∫ S ( Q, x )d Q ( x ) = -∥ G ( ·| Q ) ∥ 2 m + E x ∼ Q ∥ G ( ·| δ x ) ∥ 2 m , this leads to a valid functional Bregman divergence as follows (based on equivalence stated before (5)),

<!-- formula-not-decoded -->

Of course, there are cases where G ( ·| P ) does not exist for dirac delta's, most representatively G ( ·| P ) being a pdf with ⟨· , ·⟩ m . However, this case is shown to make a functional Bregman divergence (Section 4.1 of [23]).

## C.9.2 How IPS Properties imply NS Properties

Under (IPS1), (IPS2) of Definition C.2, and (11), we shall show ( d , m ) with d = m 2 satisfies NS Properties of Section 3.2 with l = 2 , λ = 1 , and C subg = C max,m .

(NS1) can be shown as follows. With A θ i = m 2 (Υ θ ( s i , a i ) , Ψ π ( R i , S ′ i , Υ ˆ θ n,t -1 )) where Ψ π ( r, s ′ , Υ) := ∫ A ( g γ,r ) # Υ( s ′ , a ′ ) π (d a ′ | s ′ ) ∈ P (due to convexity of P , provided that P satisfies the assumption of Theorem 2.4),

<!-- formula-not-decoded -->

Since we have ∥ ∆ n ( θ 1 | ˆ θ n,t -1 ) -∆ n ( θ 2 | ˆ θ n,t -1 ) ∥ 2 ψ 2 ( n ) ≤ C · 1 n 2 ∑ n i =1 ∥ A θ 1 i -A θ 2 i ∥ 2 ψ 2 ( n ) by Proposition 2.6.1 and Lemma 2.6.8 of [57], we shall bound | A θ 1 i -A θ 2 i | .

<!-- formula-not-decoded -->

Here, C max,m &gt; 0 is a constant (that depends on the corresponding kernel k ) such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus we have

<!-- formula-not-decoded -->

(NS2) can be shown as follows,

<!-- formula-not-decoded -->

The terms within the curly bracket can be rewritten as follows with temporarily letting Ψ i := Ψ π ( R i , S ′ i , Υ ˆ θ n,t -1 ) ,

<!-- formula-not-decoded -->

Taking its expectation so that E ( G ( ·| Ψ)) = G ( ·| Υ θ ∗ ,t ( s i , a i )) , we have λ = 1 as follows,

<!-- formula-not-decoded -->

(NS3) can be shown as follows. Let us define the following,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Based on our derivations for F n,t ( θ | ˆ θ n,t -1 ) -F n,t ( θ ∗ ,t | ˆ θ n,t -1 ) , we can see F n,t ( θ | ˆ θ n,t -1 ) -F n,t ( θ ∗ ,t | ˆ θ n,t -1 ) = F ′ n,t ( θ | ˆ θ n,t -1 ) -F ′ n,t ( θ ∗ ,t | ˆ θ n,t -1 ) . Likewise, we have ˆ F n,t ( θ | ˆ θ n,t -1 ) -ˆ F n,t ( θ ∗ ,t | ˆ θ n,t -1 ) = ˆ F ′ n,t ( θ | ˆ θ n,t -1 ) -ˆ F ′ n,t ( θ ∗ ,t | ˆ θ n,t -1 ) . Then, we have the following based on our derivations for (NS2),

<!-- formula-not-decoded -->

Using subgaussianity that we have shown for (NS1), since subgaussian constant does not depend on ˆ θ n,t -1 , we can apply Theorem 8.1.6 of [57] (probabilistic tail for Dudley's inequality) to obtain probabilistic convergence of the term sup θ ∈ Θ | ∆ n ( θ | ˆ θ n,t -1 ) | . Regarding the second term, we have

<!-- formula-not-decoded -->

Letting X i = ⟨ G ( ·| Υ θ ∗ ,t ( s i , a i )) , G ( ·| Υ θ ∗ ,t ( s i , a i )) ⟩ m , we have | X i | ≤ C 2 max,m by CauchySchwartz inequality. Then, we can apply Hoeffiding's inequality for bounded random variables (e.g., Theorem 2.2.6 of [57]) to obtain probabilistic bound that does not depend on ˆ θ n,t -1 .

(NS4) is straightforward. However, the two statements of (23) need to be assumed.

## C.10 Proof for Corollary B.1

For the case of ( d , m, η ) = ( l 2 2 , W 1 , W p ) , we have q = p , δ = 1 p , l = 2 , c = 1 , C subg = C , C surr = D 1 -1 p , ϵ 0 = 0 .

Based on the fact that l 2 2 = 1 2 E [7] for d = 1 , where energy distance E = MMD 2 β (17) with β = 1 , it is equivalent to let d = MMD 2 β =1 .

Let us first verify (NS1). We obtain the following for F n,t ( θ | ˆ θ n,t -1 ) . Temporarily ignoring ˆ θ n,t -1 , since we have ∆ n ( θ 1 ) -∆ n ( θ 2 ) = { ˆ F n,t ( θ 1 | ˆ θ n,t -1 ) -F n,t ( θ 1 | ˆ θ n,t -1 ) } - { ˆ F n,t ( θ 2 | ˆ θ n,t -1 ) -F n,t ( θ 2 | ˆ θ n,t -1 ) } , we can let ∆ n ( θ ) = ˆ F n,t ( θ | ˆ θ n,t -1 ) -F n,t ( θ | ˆ θ n,t -1 ) . We will ignore the notation ˆ θ n,t -1 for the sake of convenience.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where f θ i = E ∥ Z α ( s i , a i ; θ ) -r i -γ · Z β ( s ′ i , A ′ i ; ˆ θ n,t -1 ) ∥ and g θ i = E ∥ Z α ( s i , a i ; θ ) -Z β ( s i , a i ; θ ) ∥ , with A ′ i ∼ π ( ·| s ′ i ) . Let us only deal with the first term, since the second term can be handled via analogous approach. Letting x θ i := f θ i -E ( f θ i ) , we have ∆ (1) n ( θ ) = 1 n ∑ n i =1 x i . In order to derive its probabilistic bound, we shall bound the following term,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the inequality holds by Proposition 2.6.1 of [57]. In defining ∥ · ∥ ψ 2 ( n ) , X and x mean random variable ( d = 1) and vector ( d ≥ 1 ). Further using the fact that ∥ · ∥ ψ 2 ( n ) is a valid norm, we can derive the following where E n := E ( · · · |Z t ) ,

<!-- formula-not-decoded -->

where the last inequality is based on Proposition 2.6.1 of [57]. Based on technique suggested in Appendix A.6 of [26], we can derive | f θ 1 1 -f θ 2 1 | ≤ W 1 { Υ θ 1 ( s 1 , a 1 ) , Υ θ 2 ( s 1 , a 1 ) } , which leads to following based on Example 2.5.8 of [57],

<!-- formula-not-decoded -->

Applying the same logic to ∆ (2) n ( θ ) , we obtain C subg = C . It is trivial to show ∆ n ( θ ) is mean-zero. (NS2) can be shown as follows

<!-- formula-not-decoded -->

Here, E is energy disance defined as E ( P, Q ) := 2 E ∥ X -Y ∥ -E ∥ X -X ′ ∥ -E ∥ Y -Y ′ ∥ for X,X ′ iid ∼ Q and Y, Y ′ iid ∼ P . Using the following lemma (proof in Appendix D.3), we obtain the desired result with λ = 2 / ( b -a ) .

Lemma C.8. Assume 1 ≤ p ≤ r , and let P, Q be probability measures for arbitrary random vectors of d ≥ 1 . We have

<!-- formula-not-decoded -->

If P, Q ∈ P [ a, b ] , we have

<!-- formula-not-decoded -->

(NS3) can be shown by following. ˆ θ n,t is equivalent to ˆ θ n,t = arg min θ ∈ Θ ˆ F ′ n,t ( θ | ˆ θ n,t -1 ) where ˆ F ′ n,t ( θ | ˆ θ n,t -1 ) is defined as follows (instead of what we defined in Appendix C.9.2),

<!-- formula-not-decoded -->

with R i , S ′ i ∼ ˜ p ( · · · | s i , a i ) . Then we have following with the same logic shown in Appendix C.9.2,

<!-- formula-not-decoded -->

Just as Appendix C.9.2, we can show | ∆ n ( θ | ˆ θ n,t -1 ) |→ P 0 by applying Dudley's inequality (Theorem 8.1.6 of [57]) and show | ˆ F ′ n,t ( θ ∗ ,t | ˆ θ n,t -1 ) -F ′ n,t ( θ ∗ ,t | ˆ θ n,t -1 ) |→ P 0 with Hoeffding's inequality for bounded random variables (Theorem 2.2.6 of [57]).

Now let us show (NS4). Every µ ∈ ∆([ a, b ]) has a corresponding function function F ∈ F , where F represents the cdf family. We let F Θ := { F θ ( ·| · · · ) ∈ F S×A | F θ ( s, a ) is the cdf of Υ θ ( s, a ) } , and we have | F θ ( ·| · · · ) | ≤ 1 . Based on W 1 ( µ 1 , µ 2 ) = ∥ F 1 -F 2 ∥ L 1 with functional norm ∥ F ∥ L p := ( ∫ | F ( z ) | p d z ) 1 /p , we can define extended norms (24) with ∥ · ∥ m = ∥ · ∥ L 1 .

Lastly, by Section 2.3 of [44], we have W p ≤ D 1 -1 p · W 1 /p 1 , where D indicates the diameter of the support of return distribution ( D = b -a in 1-dimensional case). This leads to C surr = ( b -a ) 1 -1 /p and δ = 1 /p . Then, by applying Theorem C.5. we obtain the following bound for W p,d π ,p (Υ θ T , Υ π ) ,

<!-- formula-not-decoded -->

## C.11 Proof of Corollary B.2

For the case of ( d , m, η ) = (MMD 2 β , MMD β , MMD β ) , we have q = 1 , δ = 1 , l = 2 , c = β 2 , C subg = C · ( r max + z max ) β/ 2 let = C ( β ) max with sup θ ∈ Θ sup s,a ∥ Z ( s, a ; θ ) ∥ ≤ z max and sup s,a ∥ R ( s, a ) ∥ ≤ r max , C surr = 1 , ϵ 0 = 0 . We need β &gt; 1 for q = 1 as we mentioned in (16).

Any pair of ( d , m ) = (MMD 2 , MMD) satisfies NS Properties ((NS1)-(NS4)) for bounded kernels (see Appendix C.9.2). Then, we can apply Theorem C.5 to obtain the following bound for MMD β,d π , 1 (Υ θ T , Υ π ) .

<!-- formula-not-decoded -->

If we want to obtain the result for β = 1 , we let c = 1 / 2 , q = 1 + ϵ N for some monotonically decreasing sequence ϵ N → 0 as N →∞ , and C ( β =1) max = C · ( r max + z max ) 1 / 2 . Applying this to Theorem C.5, we obtain

<!-- formula-not-decoded -->

Letting y N := ϵ N 2(1+ ϵ N ) → 0 , we have

<!-- formula-not-decoded -->

Therefore, the convergent part can be bounded by constant multiplied by following for sufficiently large N ,

<!-- formula-not-decoded -->

where the second last line holds by following.

<!-- formula-not-decoded -->

Then, we finally have

<!-- formula-not-decoded -->

## C.12 Proof for Corollary B.4

For the case of ( d , m, η ) = ( d 2 L 2 , d L 2 , W p ) , we have q = p , δ = 2( r -p ) ( d +2 r ) p , l = 2 , c = 1 C subg = C · (1 + γ -d/ 2 ) · C L 2 , C surr = 2 · (max { V d , 1 } ) 1 2 p · M ( d +2 p ) r ( d +2 r ) p r , ϵ 0 = 0 . Here, we have C L 2 := sup s,a sup θ ∈ Θ ∥ f θ ( ·| s, a ) ∥ L 2 , M r := sup θ ∈ Θ sup s,a E {∥ Z ( s, a ; θ ) ∥ r } 1 /r &lt; ∞ and V d is the volume of a unit ball in R d .

,

Due to Appendix C.9.2, it suffices to show (IPS1) and (IPS2) of Definition C.2 and (11). IPS properties are trivial to show.

(11) can be shown as follows. Letting f ( ·| s, π ) := ∫ a ∈A f ( ·| s, a )d π ( a | s ) , we have ∥ 1 /γ d · f ˆ θ n,t -1 (( · -r ) /γ | s ′ , π ) ∥ L 2 = 1 /γ d/ 2 · sup θ ∈ Θ sup s,a ∥ f θ ( ·| s, a ) ∥ L 2 . Thus we have C max,m = C · (1+ γ -d/ 2 ) · C L 2 . Wecan see C L 2 &lt; ∞ by following, by letting L := sup θ ∈ Θ sup z,s,a | f θ ( z | s, a ) | &lt; ∞ ,

<!-- formula-not-decoded -->

Lastly, by Proposition 13 of [56], we have following under 1 ≤ p &lt; r

<!-- formula-not-decoded -->

with E X ∼ µ 1 {∥ X ∥ r } 1 /r , E X ∼ µ 2 {∥ X ∥ r } 1 /r ≤ M and V d being the volume of d -dimensional unit ball. Thus, we have δ = 2( r -p ) ( d +2 r ) p with

<!-- formula-not-decoded -->

Applying Theorem C.5 gives us the following bound for W p,d π ,p (Υ θ T , Υ π ) ,

<!-- formula-not-decoded -->

## C.13 Proof of Corollary B.5

NS Properties ((NS1)-(NS4) of Assumption C.3) are satisfied by Appendix C.9.2, since k ν is bounded. For the case of ( d , m, η ) = (MMD 2 ν , MMD ν , W p ) , we have q = p , l = 2 , c = 1 , C subg = C ( k ) max let = C ( ν ) max . Here, we have C ( ν ) max &lt; ∞ since k ν is a bounded function with a fixed value of σ Mat &gt; 0 . Assuming that ∥ f θ ( ·| s, a ) ∥ H ν + d/ 2 ( R d ) &lt; B and M r &lt; ∞ , Theorem 15 and Example 4 of [56] gives us C surr = C ( B,r, M r , s, d, p, σ Mat ) , ϵ 0 = 0 , δ = r -p ( d +2 r ) p in Assumption 3.2. See the detailed form of C ( B,r, M r , s, d, p, σ Mat ) &gt; 0 in their Theorem 15. Then, applying Theorem C.5 gives us following bound for W p,d π ,p (Υ θ T , Υ π ) ,

<!-- formula-not-decoded -->

## C.14 Proofs based on Gaussian regularizer

Here, we introduce a method by which we can bound W p,d π ,p -inaccuracy based on regularizer α σ ( · ) (see Definition 20 of [56]). In the following lemma, we use gaussian regularizer α σ due to its convenience, however it can be extended to other regularizers (see their Lemma 21). Since our surrogate metric m may depend on the value of σ &gt; 0 , we will denote it as m σ .

Lemma C.9. Let α σ ( · ) be the density (or probability measure) of N ( 0 , σ 2 I d ) and ∗ indicate convolution. Assuming W p ( µ ∗ α σ , ν ∗ α σ ) ≤ C ( σ, p ) · m δ σ ( µ, ν ) with p ∈ (0 , 1 p ] and m σ satisfies (NS4), letting d = m 2 σ achieves the following bound (with detailed bound in (33) ),

<!-- formula-not-decoded -->

## C.14.1 Proof for Lemma C.9

By Lemma 21 of [56], we have the following,

<!-- formula-not-decoded -->

Then, we can apply Assumption 3.2 with C surr = C ( σ, p ) , and ϵ 0 = 2 3 / 2 · Γ(( p + d ) / 2) Γ( p/ 2) · σ . Further letting m = m σ , l = 2 , q = p , c = 1 , we can apply Theorem C.5 to obtain the following,

<!-- formula-not-decoded -->

## C.14.2 Proof for Corollary B.6

The case ( d , m, η ) = (MMD 2 σ RBF , MMD σ RBF , W p ) satisfies Property (NS4), having δ = 2( r -p ) ( d +2 r ) p ∈ (0 , 1 p ) , C ( σ RBF , p ) = C ( p, d, r, M r ) (see Theorem 24 of [56] for its detailed form). We also have C subg = C ( k ) max = C 2 d/ 2 · π d/ 4 · σ RBF -d/ 2 by (29), and C brack = C brack ( σ RBF ) whose dependence on σ RBF &gt; 0 is not clear. Further letting σ RBF = σ N → 0 to a monotonically decreasing sequence, applying Lemma C.9 gives us

<!-- formula-not-decoded -->

## C.14.3 Proof for Corollary B.7

The case ( d , m, η ) = (MMD 2 ( σ,f ) , MMD ( σ,f ) , W p ) satisfies (NS4), having δ = 1 p , C ( σ, p ) = 2 1 -1 p · σ by Theorem 2 of [70], which can be applied since the kernel k ( σ,f ) is bounded. We also have C subg = C ( k ) max let = C ( σ,f ) max , which is bounded due to k ( σ,f ) ( · , · ) &lt; ∞ by (29), along with C brack = C brack ( σ ) whose dependence on σ &gt; 0 is not clear. Further letting σ = σ N → 0 to a monotonically decreasing sequence, applying Lemma C.9 gives us

<!-- formula-not-decoded -->

Note that for fixed σ &gt; 0 , we have C ( σ,f ) max &lt; ∞ as long as k ( σ,f ) ( · , · ) &lt; ∞ holds.

## C.15 Alternative for violation of (NS4)

In Theorem C.5, (NS4) may not hold. There may not be a corresponding functional norm space (e.g., Wasserstein for d ≥ 2 ) or the functional element may be unbounded, that is, sup θ ∈ Θ sup z,s,a G θ ( z | s, a ) = ∞ . For such examples, we should replace (NS3) and (NS4) of Section 3.2 with the following (MS3) and (MS4). We can maintain (NS1) and (NS2) from NS properties, renaming them (MS1) and (MS2).

Assumption C.10 (Metric-space association) . The functional Bregman divergence d and probability metric m over P form metric-space association (MS association, in short) if there exists an metric space ( G , m ) where any element µ ∈ P can be uniquely (up to equivalence under zero metric m ) represented by the corresponding element G ( ·| µ ) ∈ G such that

- (MS1) The Z t -conditioned modulus ∆ n ( θ | ˆ θ n,t -1 ) := { ˆ F n,t ( θ | ˆ θ n,t -1 ) -F n,t ( θ | ˆ θ n,t -1 ) } -{ ˆ F n,t ( θ ∗ ,t | ˆ θ n,t -1 ) -F n,t ( θ ∗ ,t | ˆ θ n,t -1 ) } is a mean-zero sub-gaussian process with respect to ¯ m n, 1 . That is, for any Z t ⊂ S × A and ˆ θ n,t -1 ∈ Θ , we have

<!-- formula-not-decoded -->

- (MS2) We have F n,t ( θ | ˆ θ n,t -1 ) -F n,t ( θ ∗ ,t | ˆ θ n,t -1 ) ≥ λ · ¯ m l n, 1 ( θ, θ ∗ ,t ) for some l ≥ 2 . Here, the value of λ &gt; 0 does not depend on Z t , θ , ˆ θ n,t -1 (or θ ∗ ,t such that Υ θ ∗ ,t = T π Υ ˆ θ n,t -1 ).

(MS3) ¯ m ρ, 1 ( ˆ θ n,t , θ ∗ ,t ) → P 0 converges in a rate that does not depend on ˆ θ n,t -1 (or θ ∗ ,t ).

<!-- formula-not-decoded -->

It is possible that there exists an element in G that does not represent any µ ∈ P .

Under these alternative conditions, we can obtain the following theorem, which is proved in Appendix C.15.1.

Theorem C.11. Suppose Assumptions 3.1, 3.2, C.4, and C 3 &lt; ∞ for Lemma C.12. Moreover, given a probability metric η that satisfies (S-L-C) with convexity parameter q ≥ 1 and scale-sensitivity parameter c &gt; 1 / (2 q ) (i.e., (15) of Appendix C.4), consider the FDE with a functional Bregman divergence d satisfying Assumption C.10 of Appendix C.15. Then, by letting T = ⌊ 1 c -1 2 q · δ ′ 2 β 1 ·

log 1 /γ N ⌋ with δ ′ = min { δ, 1 /q } , we have

<!-- formula-not-decoded -->

where C ′ B is specified in (35) . Note that we need q &gt; 1 / (2 c ) to ensure γ c -2 q ∈ (0 , 1)

The outline of the proof is very similar to that of Theorem C.5, which is shown in Appendix C.8.1. We first derive the convergence of single iteration, and then later combine multiple convergences altogether. However, the important difference is that we cannot employ the same technique that ensures the same convergence rate for ¯ m n, 1 and ¯ m ρ, 1 , since it can violate (23) that is required by Theorem C.5. That is, the functional representation may not be bounded or there may not be a corresponding functional norm space. Therefore, we resort to an alternative theoretical tool (Lemma C.12, with its proof in Appendix D.4) that can possibly lead to a slower convergence rate.

Lemma C.12. Assume Assumption 3.1 and C 3 &gt; 0 defined below has finite value. For any β &gt; 0 , we have some sequence δ n = max { δ (1) n , δ (2) n } such that δ n → 0 as n → ∞ and following ϕ ( · ) mentioned in Theorem D.1. Here, δ (1) n and δ (2) n are defined as follows:

<!-- formula-not-decoded -->

Here are the constants:

<!-- formula-not-decoded -->

C 3 &gt; 0 is the finite constant that uniformly bounds the modulus ˜ ∆ n ( θ | ˆ θ n,t -1 ) with F ( θ | ˆ θ n,t -1 ) := E S,A ∼ ρ,R,S ′ ∼ ˜ p ( ···| S,A ) ( d { Υ θ ( S, A ) , Ψ π ( R,S ′ , Υ ˆ θ n,t -1 ) } )

<!-- formula-not-decoded -->

## C.15.1 Proof for Theorem C.11

In (MS2), we take expectation on both sides. Then, the LHS becomes F ( θ | ˆ θ n,t -1 ) -F ( θ ∗ ,t | ˆ θ n,t -1 ) , where the unconditioned (not conditioned on Z t unlike (22)) objective function is defined as follows:

<!-- formula-not-decoded -->

For the RHS, we have following,

<!-- formula-not-decoded -->

Thus we have F ( θ | ˆ θ n,t -1 ) -F ( θ ∗ ,t | ˆ θ n,t -1 ) ≥ λ · ¯ m l ρ, 1 ( θ, θ ∗ ,t ) , by which we can replace (MS2).

Now, we shall combine this with (MS3) and Lemma C.12 to apply Theorem D.1. We should find δ n and β &gt; 0 that satisfies following with β ′ = min { β, 2 } ,

<!-- formula-not-decoded -->

The first condition of (34) can be rewritten as follows,

<!-- formula-not-decoded -->

Remember that β &gt; 0 is a free variable that we choose. There can be two cases: (i) l ≤ 2 + α 0 and (ii) l &gt; 2 + α 0 . We will select β differently for each case (based on the first two conditions of (34)), and we will hereafter denote its selected value as β 0 .

Let us consider the first case. To meet the first two conditions, we solve l = α 0 β ′ / 2 = β . Temporarily asuming β ≤ 2 (which also needs to be confirmed later), we obtain β 0 = l/ (1 + α 0 / 2) . We can confirm that l ≤ 2 + α 0 implies β 0 ≤ 2 , and can see α 0 β ′ = α 0 β 0 &lt; 2 l .

In the second case, we let β 0 = 2 . This leads to β ′ = 2 , and the first two conditions of (34) can be summarized as follows,

<!-- formula-not-decoded -->

We confirm that α 0 β ′ = 2 α 0 &lt; 2( l -2) &lt; 2 l holds. Incorporating Cases (i) and (ii), we let

<!-- formula-not-decoded -->

With the selected β 0 , the first two conditions of (34) become following,

<!-- formula-not-decoded -->

Since l ≤ 2 + α 0 is equivalent to l -α 0 ≤ l 1+ α 0 / 2 , we have

<!-- formula-not-decoded -->

Now we select δ n as follows,

<!-- formula-not-decoded -->

and we can confirm that third statement of (34) holds as follows. Note that the selected value of β is β 0 , not β 1 . Using some constant C LHS &gt; 0 and β 0 ≤ β 1 along with C 2 β 0 A ≥ 4 C 2 2 , we have

<!-- formula-not-decoded -->

Since α 0 ∈ (0 , 1) as specified in Lemma C.12, we can see that LHS is larger than RHS for sufficiently large n , satisfying the third statement of (34).

Now we can apply peeling argument (Theorem D.1) to obtain the following,

<!-- formula-not-decoded -->

We can copy the logic of Appendix C.8.3 that we used to prove Theorem C.5. We can apply Lemma C.7 (with satisfaction of its condition deferred in Appendix D.5). We can let

<!-- formula-not-decoded -->

and further let

<!-- formula-not-decoded -->

Ignoring the flooring effect, this leads to

<!-- formula-not-decoded -->

Then, applying the same logic as Appendix C.8.3 (extending towards multiple iterations), we finally obtain bound ¯ η d π ,q (Υ θ T , Υ π ) by following,

<!-- formula-not-decoded -->

where C ϕ , C 1 , C 2 are defined in Lemma C.12.

## C.15.2 Proof for Corollary B.8

For the case of ( d , m, η ) = (MMD 2 cou , MMD cou , W p ) , we have q = p , δ = 1 /p , l = 2 , c = 1 , C subg = C ( k ) max = C (1 + γ 1 -d/ 2 ) · sup θ ∈ Θ sup s,a E 1 / 2 cou (Υ θ ( s, a )) let = C ( cou ) max , C surr = C · D 1 -1 2 p · V 1 2 p d .

(MS1) and (MS2) are straightforward from Appendix C.9.2. Note that C ( k ) max = C ( cou ) max (29) can be derived as follows (for unbounded kernel k cou ). Based on ∥ κ µ ∥ 2 H = ⟨ κ µ , κ µ ⟩ H = E { k cou ( X , X ′ ) } with X , X ′ iid ∼ µ , we can use ∥ κ µ ∥ H = E 1 / 2 cou ( µ ) to obtain the following,

<!-- formula-not-decoded -->

Since we have the following based on definition of κ ( cou ) 0 ,

<!-- formula-not-decoded -->

This gives us C ( k ) max = C ( cou ) max .

(MS3) can be shown analogously with Section C.10. (MS4) regards the model complexity that we assume. C 3 &gt; 0 can be shown to be bounded by using sup θ ∈ Θ sup s,a E cou (Υ θ ( s, a )) &lt; ∞ , which is assumed in Corollary B.8.

We also obtain C surr = C · D 1 -1 2 p · V 1 2 p d where V d is the volume of a unit ball in R d , by Theorem 1.1 of [11] as follows,

<!-- formula-not-decoded -->

Then applying this into Theorem C.11 gives us following bound, since we have β 1 = 2 / ( 3 -1 α ) ,

2 4 W p,d π ,p (Υ θ T , Υ π ) ≤ D 1 -1 2 p · V 1 2 p d 1 -γ 1 -1 2 p · C 1 2 p cover · C ′ B · O P { 1 log(1 /γ ) · ( (log N ) 2 N ) 6 -α 16 p } , C ′ B := 1 1 -α/ 2 · ( max { C ( cou ) max · C 1 / 2 met , C 1 / 2 met · diam(Θ; W p, ∞ ) 2 -α/ 2 , diam(Θ; W p, ∞ ) 2 }) 6 -α 8 p where C ′ B is defined accordingly to (35).

<!-- formula-not-decoded -->

## C.15.3 Proof for Corollary B.3

For the case of ( d , m, η ) = (MMD 2 β =1 , W 1 , W 1 ) , we have q = 1 , δ = 1 , l = r (2 d +3) -1 r -1 , c = 1 , C subg = C , C surr = 1 , ϵ 0 = 0 . We need β &gt; 1 for q = 1 as we mentioned in (16).

(MS1) holds, as we have already shown in Appendix C.10. This can be copied since l 2 2 is a special case of MMD 2 β =1 for d = 1 . However, (MS2) should be changed as follows, based on Proposition 17 of [36]: when M r &lt; ∞ , we have

<!-- formula-not-decoded -->

Refer to their Proposition 17 for the definition of constant C ( d, r, β, M r ) &gt; 0 . This can be developed into

<!-- formula-not-decoded -->

Since we are limiting to β = 1 , (MS2) is satisfied with λ = C ( d, r, β, M r ) -2 /ρ ′ and l = 2 ρ ′ = r (2 d +3) -1 r -1 ≥ 2 d +3 .

(MS3) can be shown analogously to C.10. In the part of applying Hoeffding's inequality (generalized version for unbounded distributions, e.g., Theorem 2.6.2), we should assume sup θ ∈ Θ sup s,a E ∥ Z ( s, a ; θ ) ∥ &lt; ∞ and sup s,a ∥ R ( s, a ) ∥ ψ 2 &lt; ∞ .

(MS4) is what we assume. Finiteness of C 3 &gt; 0 can be shown by using the same trick that holds between energy distance and Wassertein-1 metric (as we mentioned in Appendix C.10),

<!-- formula-not-decoded -->

Then, we can apply Theorem C.11 to obtain the following bound, since β 1 = l -1 + α/ 2 .

<!-- formula-not-decoded -->

## C.16 Proof for Corollary B.9

We acknowledge that the following proof is based upon Chapter 7 of [20] and Section 6.4 of [51]. Throughout this proof, we will resort to the following trick shown by [51],

<!-- formula-not-decoded -->

Now, let us define the following functions. Define E s,a to be expectation taken with respect to s, a ∼ ρ , E r,s ′ to be expectation taken with respect to r, s ′ ∼ ˜ p ( · · · | s, a ) , E z ′ ∼ f ˆ θ n,t -1 ( ···| s ′ ,π ) to be expectation

taken with respect to z ′ ∼ f ˆ ( s ′ , π ) , where f ˆ ( ·| s ′ , π ) := ∫ f ˆ ( ·| s ′ , a ′ )d π ( a ′ | s ′ )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that it is more appropriate to write it as M ( θ | ˆ θ n,t -1 ) and ˆ M n ( θ | ˆ θ n,t -1 ) . However, we will take ˆ θ n,t -1 ∈ Θ (along with the corresponding θ ∗ ,t ) as given, and simplify it as M ( θ ) = M ( θ | ˆ θ n,t -1 ) and ˆ M n ( θ ) = ˆ M n ( θ | ˆ θ n,t -1 ) . Since the inner part of E s,a of M ( θ ) is equivalent to KLD ( f θ ∗ ,t ( ·| s, a ) ∥ f θ ( ·| s,a )+ f θ ∗ ,t ( ·| s,a ) 2 ) ≥ 0 where equality holds under θ = θ ∗ ,t , θ ∗ ,t becomes the minimizer of M ( θ ) with M ( θ ∗ ,t ) = 0 .

## C.16.1 Quadratic lower bound

For fixed s, a , denote p 0 ( · ) = f θ ∗ ,t ( ·| s, a ) and p ( · ) = f θ ( ·| s, a ) . Then, the terms inside E s,a is simplified as follows with q = 1 2 ( p + p 0 ) ,

<!-- formula-not-decoded -->

where h 2 0 ( µ 1 , µ 2 ) := 1 2 ∫ p 0 &gt; 0 ( √ f 1 ( z ) -√ f 2 ( z )) 2 d z where f 1 , f 2 are corresponding pdf's of µ 1 , µ 2 ∈ P . Allowing abuse of notation, we will allow putting densities instead of probabilty measures for the inputs.

Then, since M ( θ ∗ ,t ) = 0 , we have the following where ¯ h 0 ,ρ, 1 is defined accordingly to (8),

<!-- formula-not-decoded -->

Here we have ¯ h 2 0 ,ρ, 1 ( θ 1 , θ 2 ) := E s,a ∼ ρ { h 2 0 (Υ θ 1 ( s, a ) , Υ θ 2 ( s, a )) } with h 2 0 (Υ θ 1 ( s, a ) , Υ θ 2 ( s, a )) takes the integral over the support of Υ θ ∗ ,t ( s, a ) , or f θ ∗ ,t ( ·| s, a ) &gt; 0 .

## C.16.2 Convergence of modulus

We recall the following Theorem 6.13 of [51].

Theorem C.13. For any class F of measurable functions f : X → R with ∥ f ∥ P 0 ,B ≤ δ , with X i iid ∼ P 0 , we have following with ≲ indicating 'bounded by some multiplicative constant, '

<!-- formula-not-decoded -->

Let us define the following.

<!-- formula-not-decoded -->

We also define M Θ := { m θ ( · · · ) : θ ∈ Θ } = { m θ ( · · · ) -m θ ∗ ,t ( · · · ) : θ ∈ Θ } , since we have m θ ∗ ,t ( · · · ) = 0 . Since m θ ( s, a, r, s ′ ) ≤ log 2 and x &lt; log 2 implies exp( | x | ) -1 - | x | ≤ 4 · (exp( -x/ 2) -1) 2 , we have the following,

<!-- formula-not-decoded -->

Note that E r,s ′ ∼ ˜ p ( ···| s,a ) E z ′ ∼ f ˆ θ n,t -1 ( ·| s,a ) can be reduced into E z ∼ f θ ∗ ,t ( ·| s,a ) . Again, letting p 0 ( · ) = f θ ∗ ,t ( ·| s, a ) and p ( · ) = f θ ( ·| s, a ) for fixed s, a , along with q = ( p + p 0 ) / 2 , the term within E s,a can be rewritten as follows,

<!-- formula-not-decoded -->

Then, this leads to ∥ m θ ∥ P 0 ,B ≤ 2 √ 2 · ¯ h 0 ,ρ, 1 ( θ ∗ ,t , θ ) , meaning that ¯ h 0 ,ρ, 1 ( θ ∗ ,t , θ ) ≤ δ 2 √ 2 implies ∥ m θ ∥ P 0 ,B ≤ δ . Then, applying Theorem C.13 gives us the following with modulus constructed as ∆ n ( θ | ˆ θ n,t -1 ) = ( ˆ M n ( θ ) -M ( θ )) -( ˆ M n ( θ ∗ ,t ) -M ( θ ∗ ,t )) ,

<!-- formula-not-decoded -->

Now let us simplify the term J [ ] ( δ, M Θ , ∥ · ∥ P 0 ,B ) . Assume a bracketing pair m p ( · · · ) ≤ m q ( · · · ) , which may not be elements of M Θ . That is, we have p ( ·| · · · ) ≥ q ( ·| · · · ) (where we can assume q ( ·| · · · ) ≥ 0 but not necessarily conditional probability densities), and

<!-- formula-not-decoded -->

The term ∥ m p ( · · · ) -m q ( · · · ) ∥ P 0 ,B is defined as follows,

<!-- formula-not-decoded -->

Employing the same trick that we have used before, the term inside E s,a can be bounded as follows with p 0 ( z ) = f θ ∗ ,t ( ·| s, a ) for fixed s, a ,

<!-- formula-not-decoded -->

Then, we have ∥ m p ( · · · ) -m q ( · · · ) ∥ P 0 ,B ≤ 4 · ∥ √ p ( ·| · · · ) -√ q ( ·| · · · ) ∥ L 2 ,ρ . ∥ · ∥ L 2 ,ρ is defined accordingly to (24). Thus we have the following,

<!-- formula-not-decoded -->

Then, this leads to following by assumption,

<!-- formula-not-decoded -->

We will later select δ n such that C · C 1 / 2 brack · 1 1 -α/ 2 · δ 1 -α/ 2 n = √ n · δ 2 n and apply Theorem D.1.

## C.16.3 Consistence

Before we apply Theorem D.1, let us show ¯ h 0 ,ρ, 1 ( ˆ θ n,t , θ ∗ ,t ) → P 0 . By Lemma 4.2 of [20], we have following,

<!-- formula-not-decoded -->

The first inequality holds by following. Letting p be an arbitrary density and ¯ p = 1 2 ( p + p 0 ) , we have

<!-- formula-not-decoded -->

The second inequality above holds due to following. Note that ˆ θ n,t is defined as follows by (10) with d ( P, Q ) = KLD ( Q ∥ P ) and we have following based on concavity of logarithm function, i.e., log x + y 2 ≥ 1 2 log x + 1 2 log y for x, y &gt; 0 .

<!-- formula-not-decoded -->

Then this leads to following which justifies the inequality,

<!-- formula-not-decoded -->

The last line above holds, since the term E s,a,r,s ′ {-m ˆ θ n,t ( s, a, r, s ′ ) } can be bounded using the following fact,

<!-- formula-not-decoded -->

Using | M ( ˆ θ n,t ) -ˆ M n ( ˆ θ n,t ) | = ∆ n ( θ | ˆ θ n,t -1 ) since ˆ M n ( θ ∗ ,t ) = M ( θ ∗ ,t ) = 0 , we can show the following by Markov's inequality and (37), with probability bound not depending on ˆ θ n,t -1 ,

<!-- formula-not-decoded -->

as long as diam( ˜ F 1 / 2 Θ , ∥ · ∥ L 2 ,ρ ) &lt; ∞ .

## C.16.4 Finalization

As we mentioned at the end of Appendix C.16.2, we let

<!-- formula-not-decoded -->

Now let us bound further bound Wasserstein metric by h 0 . Considering two densities p and p 0 , we have

<!-- formula-not-decoded -->

This allows us to bound TVD metric as follows,

<!-- formula-not-decoded -->

Based on W p ( P, Q ) ≤ D · TVD 1 /p ( P, Q ) shown in Lemma C.6 of [63], this leads to W p ( p 0 , p ) ≤ C · D · h 1 /p 0 ( p 0 , p ) . This leads to following,

<!-- formula-not-decoded -->

Letting T = ⌊ 1 1 -1 2 p · 1 /p 2+ α · log 1 /γ N ⌋ , we apply n = N/T based on (6), and we obtain the following bound for W p,d π ,p (Υ θ T , Υ π ) ,

<!-- formula-not-decoded -->

Of course, we should show that condition of Lemma C.7 is satisfied before we apply (6). Its proof is analogous to Appendix D.5 that we used in proving Theorem C.11.

## C.17 Comparison and examples on Cramér FDE

## C.17.1 Comparison between Cramér FDE and FLE

Although FLE [63] did not suggest a probability bound for W p,d π ,p -inaccuracy, we can utilize their key results (e.g., their Lemma 4.9) to build such bound. With probability larger than 1 -δ , we have the following with in a single iteration based on D t with |D t | = n :

<!-- formula-not-decoded -->

Here, ∥ · ∥ can be any norm that can be defined on ˜ F Θ . For fair comparison, let us use

<!-- formula-not-decoded -->

which is what we already defined in (NS4) (or Appendix C.10). Skipping algebraic details, this leads to following with appropriate choice of T with respect to the size of whole data N = |D| ,

<!-- formula-not-decoded -->

Assuming the same condition, we can also bound log N [ ] ( F Θ , ∥ · ∥ L 1 , ∞ , ϵ ) , where F Θ represents the conditional cdf family, as we have mentioned in Appendix C.10. Letting M = log N [ ] ( ˜ F Θ , ∥ · ∥ L 1 , ∞ , ϵ ) , there exists pairs of functions { [ l i ( ·| · · · ) , u i ( ·| · · · )] } i M =1 that serve as ϵ -bracketing of ˜ F Θ . Then, let us define

<!-- formula-not-decoded -->

Now arbitrarily pick F θ ( ·| · · · ) ∈ F Θ , and let its corresponding conditional pdf as f θ ( ·| · · · ) . There exists a bracketing pair [ l k ( ·| · · · ) , u k ( ·| · · · )] such that l k ( ·| · · · ) ≤ f θ ( ·| · · · ) ≤ u k ( ·| · · · ) and

∥ l k ( ·| · · · ) -u k ( ·| · · · ) ∥ L 1 , ∞ ≤ ϵ . Then, not only do we have L k ( ·| · · · ) ≤ F θ ( ·| · · · ) ≤ U k ( ·| · · · ) , but also

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This further implies log N [ ] ( F Θ , ∥ · ∥ L 1 , ∞ , ( b -a ) · ϵ ) ≤ log N [ ] ( ˜ F Θ , ∥ · ∥ L 1 , ∞ , ϵ ) ≲ ϵ -α , thereby satisfying the condition of (NS4). This allows us to apply Corollary B.1, yielding W p,d π ,p (Υ θ T , Υ π ) = ˜ O P ( N -1 p (2+ α ) ) .

## C.17.2 Example 1: Linear MDP

Let us assume the following model with Θ = ( F ) K , where F is a collection of cdf's of random variables bounded in [ a, b ] . Here, we can see θ as an element H ( · ) = ( H 1 ( · ) , · · · , H K ( · )) ⊺ with each H k ∈ F .

<!-- formula-not-decoded -->

Note that tabular setting ( |S × A| &lt; ∞ ) is a special case of linear MDP model. This is the case with ϕ k ( s, a ) = 1 ( s, a = ( s, a ) k ) with k = 1 , · · · , |S × A| .

Assumption 3.1 is satisfied when the transition is expressed as a linear combination of features ˜ p ( r, s ′ | s, a ) = ⟨ ϕ ( s, a ) , θ ( r, s ′ ) ⟩ with θ ( r, s ′ ) = ( θ 1 ( r, s ′ ) , · · · , θ K ( r, s ′ )) ⊺ being valid densities over R ×S . Also assume Assumption C.4.

For (NS4), we can use Theorem 2.7.5 of [55] to bound log N [ ] ( F Θ , ∥ · ∥ ∞ ,L 1 , ϵ ) . Letting Q = Unif[ a, b ] , we have the following,

<!-- formula-not-decoded -->

where we have used ∥ F 1 -F 2 ∥ L 1 = ( b -a ) · ∥ F 1 -F 2 ∥ L 1 ( Q ) in the first equality.

Arbitrarily choose Υ H 1 , Υ H 2 ∈ M Θ and their corresponding cdf's F H 1 ( ·| s, a ) = ⟨ ϕ ( s, a ) , H 1 ( · ) ⟩ and F H 2 ( ·| s, a ) = ⟨ ϕ ( s, a ) , H 2 ( · ) ⟩ . Then, we can derive the following with F [ a, b ] being the cdf's of bounded random variables in [ a, b ] ,

<!-- formula-not-decoded -->

The first inequality can be verified as follows. Let the bracketing pairs of F be [ l 1 ( · ) , u 1 ( · )] , · · · [ l M ( · ) , u M ( · )] where M = log [ ] ( F , ∥ · ∥ 1 , ϵ ) . Make K copies for each pair, so that we have [ l ( k ) 1 ( · ) , u ( k ) 1 ( · )] , · · · [ l ( k ) M ( · ) , u ( k ) M ( · )] for each k ∈ [ K ] . This means that we can construct M K tuples of

<!-- formula-not-decoded -->

Pick an arbitrary F ( ·| · · · ) ∈ F Θ , and it has corresponding H = ( H 1 , · · · , H K ) such that F ( ·| s, a ) = ⟨ ϕ ( s, a ) , H ( · ) ⟩ . Since H 1 ( · ) ∈ [ l (1) j 1 , u (1) j 1 ] , · · · , H K ( · ) ∈ [ l ( K ) j K , u ( K ) j K ] for some j 1 , · · · , j K ∈ [ M ] . Then, letting F l ( ·| s, a ) = ∑ k ϕ k ( s, a ) l ( k ) j k ( · ) and F u ( ·| s, a ) = ∑ k ϕ k ( s, a ) u ( k ) j k ( · ) , we have F ( ·| · · · ) ∈ [ F l ( ·| · · · ) , F u ( ·| · · · )] . We eventually have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which leads to N [ ] ( F Θ , ∥ · ∥ L 1 , ∞ , ϵ ) ≤ {N [ ] ( F , ∥ · ∥ L 1 , ϵ ) } K . Note that we have sup θ ∈ Θ sup s,a sup z | F θ ( z | s, a ) | ≤ 1 and can let C brack = CK ( b -a ) , α = 1 in (NS4). Eventually this leads to ˜ O P ( N -1 / (3 p ) ) W p,d π ,p -convergence in Corollary B.1. However, FLE cannot suggest a meaningful bound since they assume the existence of conditional densities, which may not hold in our cdf-based model.

## C.17.3 Example 2: Linear Quadratic Regulator

We will denote state-action pairs as x, a ∈ S × A = R d x × R d a , its conditional reward as R ( x, a ) , and subsequent stater as x ′ = x ′ ( x, a ) . We assume the following setting

<!-- formula-not-decoded -->

Temporarily, we will assume bounded distribution for ϵ R (although this can be relaxed for Energy FDE). Letting ϵ ret := ∑ t ≥ 1 γ t -1 ϵ R,t with ϵ R,t being iid copies of ϵ R , we assume the following model,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Throughout the proof, we shall assume the analogous statements that [63] suggested in their Lemma B.3, bounding ∥ x ∥ , ∥ a ∥ , ∥ M 1 ∥ F , ∥ M 2 ∥ F , ∥ M 3 ∥ F .

Assumption 3.1 can be satisfied with appropriate modeling of Θ ⊆ R d x × d x × R d a × d x × R d a × d a . By following the same logic shown in D.9 of [63], we can present an example. Skipping all calculations, T π Z θ ( x, a ) ∼ T π Υ θ ( x, a ) follows a distribution T π Z θ ( x, a ) = E {T π Z θ ( x, a ) } + ϵ ret with,

<!-- formula-not-decoded -->

Let our model be Θ = { ( M 1 , M 2 , M 3 ) : ∥ M ∥ F , ∥ M 2 ∥ F , ∥ M 3 ∥ F ≤ m } , T π Υ θ ∈ M Θ holds if the following conditions are satisfied,

<!-- formula-not-decoded -->

Assumption C.4 can be simplified as follows. For X = µ x + ϵ ret and Y = µ y + ϵ ret , we have the following with P X , P Y being their probability measures and F X , F Y being their cdf's,

<!-- formula-not-decoded -->

Note that the second equality holds, since F Y is only a location shift of F X , i.e., F Y ( · ) = F X ( ·-( µ y -µ x )) . This leads to the following with θ A = ( M ( A ) 1 , M ( A ) 2 , M ( A ) 3 ) and θ B = ( M ( B ) 1 , M ( B ) 2 , M ( B ) 3 ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ϕ ( x, a ) := ( x ⊺ , a ⊺ ) ⊺ ⊗ ( x ⊺ , a ⊺ ) ⊺ and some θ ∈ R ( d x + d a ) 2 . Using the fact that W 1 { Υ θ A ( x, a ) , Υ θ B ( x, a ) } = θ ⊺ AB ϕ ( x, a ) ϕ ( x, a ) ⊺ θ AB , we obtain

̸

<!-- formula-not-decoded -->

This is the same result that [63] provided in their Lemma B.2.

Lastly, let us bound the term log N [ ] ( F Θ , ∥ · ∥ ∞ , 1 , ϵ ) for (NS4). First, we define

<!-- formula-not-decoded -->

We can see N [ ] ( F Θ , ∥ · ∥ L 1 , ∞ , ϵ ) ≤ N [ ] ( F µ Θ , ∥ · ∥ ∞ , ϵ ) for the following reason. Letting M = N [ ] ( F µ Θ , ∥ · ∥ ∞ , ϵ ) , there exist bracketing pairs { [ l i ( · ) , u i ( · )] } i M =1 such that

<!-- formula-not-decoded -->

Now consider the bracketing pairs [ F l i ( ·| · · · ) , F u i ( ·| · · · )] ( i = 1 , · · · , M ) . For an arbitrary F θ ( ·| · · · ) ∈ F Θ , its corresponding expectation function satisfies l i ( x, a ) ≤ µ θ ( x, a ) ≤ u i ( x, a ) for some i ∈ [ M ] . As we have previously mentioned, all F θ ( ·| x, a ) are results of location-shifts of the cdf of ϵ ret . This leads to F θ ( ·| · · · ) ∈ [ F l i ( ·| · · · ) , F u i ( ·| · · · )] for some i ∈ [ M ] , and it satisfies

<!-- formula-not-decoded -->

Thus, we have N [ ] ( F Θ , ∥ · ∥ L 1 , ∞ , ϵ ) ≤ N [ ] ( F µ Θ , ∥ · ∥ ∞ , ϵ ) . Further assuming ∥ x ∥ ≤ D x and ∥ a ∥ ≤ D a , we have

<!-- formula-not-decoded -->

where d 3 ( θ A , θ B ) := max {∥ M ( A ) 1 -M ( B ) 1 ∥ F , ∥ M ( A ) 2 -M ( B ) 2 ∥ F , ∥ M ( A ) 3 -M ( B ) 3 ∥ F } and D 0 := max { D x , D a } . Then, we can bound our target as follows,

<!-- formula-not-decoded -->

where Θ i := { M i : ∥ M i ∥ F ≤ m } . Since Θ 1 , Θ 2 , Θ 3 with ∥ · ∥ F can be perceived as subsets of Euclidean spaces R d x d x , R d a d x , R d a d a with Euclidean norms ∥ · ∥ , we can use volume comparison lemma

<!-- formula-not-decoded -->

This leads to the following,

<!-- formula-not-decoded -->

Since our term can be bounded by inverse of polynomial N [ ] ( F Θ , ∥ · ∥ L 1 , ∞ , ϵ ) ≲ ϵ -α with any α &gt; 0 , we can let α N = 1 / log N .

If we assume bounded distributions of ϵ rew , then we can apply Corollary B.1 with the same trick (32), Cramér FDE obtains ˜ O P ( N -1 / (2 p ) ) of W p,d π ,p -convergence. If we assume unbounded distributions, then Energy FDE obtain ˜ O P ( N -1 / 8 ) of W 1 ,d π , 1 -inaccuracy by Corollary B.3. For FLE [63], unless we have a standard parametric distribution for ϵ ret (e.g., truncated gaussian), we cannot obtain the bound, since we cannot relate ∥ f θ A ( ·| · · · ) -f θ B ( ·| · · · ) ∥ L 1 , ∞ and d 3 ( θ A , θ B ) .

## D Auxiliary proofs for Appendix C

## D.1

## Supporting theorem

Following is a standard convergence theorem in M-estimation. We slightly adapted it from Theorem 6.1 of [51], only modifying l = 2 into generalized l ≥ 2 . So we will skip the proof.

Theorem D.1. (Adapted version of Theorem 6.1 of [51]) Assume that F n,t ( · ) : Θ → R has a minimizer θ ∗ . Let ˜ δ n ≥ 0 be a sequence that satisfies the following for δ ≥ ˜ δ n :

<!-- formula-not-decoded -->

2. We have ϕ ( δ ) = C 0 · δ α 0 with some α 0 ∈ (0 , l ) such that following holds,

<!-- formula-not-decoded -->

Further assume that we have δ n and some estimator such that

<!-- formula-not-decoded -->

Then, we have d n ( ˆ θ n , θ ∗ ) = O P ( δ n ) . If ˜ δ n = 0 , then we have δ n = C 1 l -α 0 0 · n -1 2( l -α 0 ) .

## D.2 Proof of Lemma C.7

Without loss of generality, we can assume ˜ C = 1 . We let Z n,t := n 1 /b · X n,t , and C = lim sup n →∞ sup t ∈ N E | n 1 /b · X n,t | &lt; ∞ ,

<!-- formula-not-decoded -->

Then, letting ϵ = C 1 -ζ · ˜ ϵ , we have

<!-- formula-not-decoded -->

Thus we have ∑ T t =1 ζ T -t · X n,t = O P ( 1 1 -ζ · n -1 /b ) . More generally, we have ∑ T t =1 ζ T -t · X n,t = ˜ C · O P ( 1 1 -ζ · n -1 /b ) .

## D.3 Proof of Lemma C.8

Showing the upper bound is easy. Let X,X ′ iid ∼ Q and Y, Y ′ iid ∼ P . Based on technique shown in Appendix A.6 of [26], we can derive

<!-- formula-not-decoded -->

Due to Jensen's inequality, it is straightforward to see W p ( P, Q ) ≤ W r ( P, Q ) for p &lt; r .

Now let us show the lower bound. We assume that X ∼ P and Y ∼ Q have bounded support [ a, b ] , and we denote their cdf's as F P and F Q . Letting U ∼ Unif[ a, b ] , we have

<!-- formula-not-decoded -->

where l 2 2 ( P, Q ) := ∫ ∞ ∞ | F P ( x ) -F Q ( x ) | 2 d x denotes Cramér distance. Since Energy distance is equivalent to Cramér distance in 1D domain [7], that is 1 2 E ( P, Q ) = l 2 2 ( P, Q ) , this leads to

<!-- formula-not-decoded -->

Using W p p ( P, Q ) ≥ = ( b -a ) p -r · W r r ( P, Q ) that was shown by [44] (see their Section 2.3), we can show the lower bound.

## D.4 Proof of Lemma C.12

## D.4.1 Conditioned event

Let ˆ θ n,t -1 ∈ Θ be fixed. Then, we have a unique value of θ ∗ ,t = arg min θ ∈ Θ F ( θ | ˆ θ n,t -1 ) . Based on this, we define the following:

<!-- formula-not-decoded -->

With Ω being the probability space, we define the following event based on β ∈ (0 , 2] whose exact value will later be determined,

<!-- formula-not-decoded -->

Prior to calculating its probability P (Ω δ ) , let us first show its sub-gaussianity. Since ∥ · ∥ ψ 2 is a norm (Example 2.5.7 of [57]), we can use its convexity to derive

<!-- formula-not-decoded -->

Assume that θ 1 , θ 2 ∈ Θ δ . Then, we have

<!-- formula-not-decoded -->

where diam(Θ δ ; m ∞ ) := sup θ 1 ,θ 2 ∈ Θ δ m ∞ ( θ 1 , θ 2 ) . This leads to ∥ Y θ 1 1 -Y θ 2 1 ∥ ψ 2 ≤ C · diam(Θ δ ; m ∞ ) · m ∞ ( θ 1 , θ 2 ) . Then, by applying Theorem 8.1.6 of [57], we have the following bound for arbitrary u &gt; 0 ,

<!-- formula-not-decoded -->

Using the constants defined in Lemma C.12, we can let u = 1 2 · √ n · δ β /C 2 to show that the following holds with probability larger than 1 -2 · exp( -n · δ 2 β / 4 C 2 2 ) ,

<!-- formula-not-decoded -->

Now, we assume the following:

<!-- formula-not-decoded -->

Since we have 1 2 δ β ≥ C 1 √ n , we have

<!-- formula-not-decoded -->

Under Ω δ , we have the following. Suppose θ ∈ Θ δ . In other words, ¯ m ρ, 1 ( θ, θ ∗ ,t ) &lt; δ . Then, we have

<!-- formula-not-decoded -->

This leads to the following with β ′ = min { β, 2 } ,

<!-- formula-not-decoded -->

where we assumed δ ∈ (0 , 1) without loss of generality. Then, we have the following,

<!-- formula-not-decoded -->

## D.4.2 Concentration of modulus

Assume that Z t := { ( s i , a i ) } n i =1 is given (fixed). However, there still exists randomness in the transition r i , s ′ i ∼ ˜ p ( · · · | s i , a i ) . For this conditional probability, denote its corresponding conditional expectation as E Z t ( · · · ) := E ( · · · |Z t ) and its corresponding sub-gaussian norm as ∥ · ∥ ψ 2 ( n ) . Then, by Dudley's inequality (Theorem 8.1.3 of [57]) and (MS1), we obtain following,

<!-- formula-not-decoded -->

By Assumption of Lemma C.12, we have 0 &lt; C 3 &lt; ∞ . Then, we have the following,

<!-- formula-not-decoded -->

Then, we set another criterion as below,

<!-- formula-not-decoded -->

This leads to following bound, whose RHS does not depend on ˆ θ n,t -1 ∈ Θ ,

<!-- formula-not-decoded -->

## D.5 Satisfaction of the condition of Lemma C.7 in Theorem C.11

Before bounding ¯ η d π ,q (Υ θ T , Υ π ) based on (6), we first apply Lemma C.7 by letting X n,t = ¯ m δ ′ d π , 1 (Υ θ t , T π Υ ˆ θ n,t -1 ) . Proving lim sup n →∞ sup t ∈ N E | n 1 /b · X n,t | &lt; ∞ is the analogous to Appendix C.8.4. We do not need Stage 2 of Appendix C.8.4. Skipping the details, we can repeat Stage 1 of Appendix C.8.4 to obtain the following, with ˜ ∆ n ( θ | ˆ θ n,t -1 ) defined in Lemma C.12:

<!-- formula-not-decoded -->

Using (45) and letting β ′ = min { l 1+ α 0 / 2 , 2 } with α 0 = 1 -α/ 2 (as we chose within the proof of Theorem C.11 of Appendix C.15.1), we can take up the above inequality as

<!-- formula-not-decoded -->

After that, copying the remaining logic of Appendix C.8.4 will give us the desired result.

## E Experiment Details

## E.1 Details in Linear Quadratic Regulator

LQR is a parametric environment with deterministic transitions and Gaussian noise in the rewards. It can be verified that Assumption C.4 holds given the chosen behavior policy, and Assumptions 3.1 is satisfied under the appropriate model space, both of which we will present. We will compare the performance of the baseline method FLE [63] with our proposed FDE method using different functional Bregman divergences: Energy ( β = 1 ), Laplace (Matern with ν = 1 ), RBF, PDF-L2, KL. The evaluation is based on the W 1 ,d π , 1 -inaccuracy.

The proposed FDE methods using different functional Bregman divergences outperform FLE in terms of both mean inaccuracy and stability (as measured by standard deviation), as visualized in the inaccuracy graphs of Figure 2. This performance gap is expected, as FLE does not make use of the closed-form density of Ψ π ( r, s ′ , Υ ˆ θ n,t -1 ) , making it suffer from large Monte Carlo errors. See Table 4 in Appendix E.1.4 for detailed inaccuracy results.

## E.1.1 Data collection and model

Our offline data are collected as follows. States x = ( r · cos θ x , r · sin θ x ) ⊺ are generated with r ∼ Unif [0 , 1] , θ x ∼ Unif [0 , 2 π ] . Given the state, action is generated by behavior policy b ( a | x ) = 1 / 5 with a = Rot ( θ a ) x where Rot ( θ a ) is the (counter-clockwise) rotation matrix with angle θ a ∼ Unif { 0 , 2 π 5 , 4 π 5 , 6 π 5 , 8 π 5 } . Since the all x within the unit circle has positive density value and Rot ( θ a ) = K holds with positive probability, Assumption C.4 is satisfied.

We assume deterministic target policy (which, for the time being, can be regarded as π : S → A with abuse of notation). With Gaussian noise added to the reward ϵ R ∼ N (0 , σ 2 0 ) , our goal is to estimate Υ π ∈ P S×A . States x ∈ S and actions a ∈ A are both generated from a continuous subset of R 2 (i.e., S , A ⊆ R 2 ). Matrices A,B,Q,R ∈ R 2 × 2 that control the environment dynamics (corresponding to ˜ p ( r, s ′ | s, a ) ) are as follows (but unknown in the simulations).

<!-- formula-not-decoded -->

Letting K be the identity matrix, we assumed γ = 0 . 99 and σ 0 = 1 .

Return distribution model based on (40) of Appendix C.17.3 can be written as follows,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that this setup satisfies Assumption 3.1 (proof in Appendix C.17.3).

## E.1.2 Data splitting

Weapplied the same splitting rule of data D = ∪ T t =1 D t by selecting the following T . Weacknowledge that we did not strictly apply the rule of Theorem C.5 that will lead to asymptotic convergence with the suggested rate.

1. We used the rule of Theorem C.5 with the parameters of Energy FDE with inaccuracy measured with Wasserstein-1 metric ( l = 5 , δ = 1 , c = 1 , q = 1 ). We also used α = 0 since it is a parametric model.
2. We divided it with a constant C divide &gt; 0 and then choose the floored integer as follows. Since larger discount rate shall require bigger subdatasets (i.e., larger |D t | ), we let C divide = 5 for γ = 0 . 99 .

<!-- formula-not-decoded -->

3. Allocate the same number of samples into each of the T subdatasets, and then put all the remaining samples to the last one D T .

<!-- formula-not-decoded -->

## E.1.3 Closed form for gaussian distributions

Since we have gaussian conditional densities, we have used the following closed form. For MMD that we have used (Energy, Laplace, RBF), we have the following with k ( x, y ) = k 0 ( x -y ) with mutually independent X,X ′ ∼ P and Y, Y ′ ∼ Q ,

<!-- formula-not-decoded -->

Defining K 0 ( µ, σ 2 ) := E Z ∼ N ( µ,σ 2 ) { k 0 ( Z ) } , we have following for Energy distance which has k 0 ( y ) = -| y | ,

<!-- formula-not-decoded -->

RBF kernel with k 0 ( y ) = exp( -y 2 / 4 σ RBF 2 ) have the following.

<!-- formula-not-decoded -->

Laplace kernel with k 0 ( y ) = exp( -| y | /σ Lap ) has

<!-- formula-not-decoded -->

For PDF-L2, we used the following formula for ϕ ( ·| µ, σ 2 ) represents the density function of N ( µ, σ 2 ) ,

<!-- formula-not-decoded -->

For KL Divergence, we have

<!-- formula-not-decoded -->

Within our simulations, we have used fixed values of σ Lap = σ RBF = 1 .

## E.1.4 Simulation results

We have applied L-BFGS-B algorithm in solving the minimization problem of (5) at each iteration t ∈ [ T ] . For the initial value of L-BFGS-B algorithm, we always used the previous iterate ˆ θ n,t -1 . Simulation results are visualized in Figure 2, with details in Table 4.

Figure 2: W 1 ,d π , 1 -inaccuracy (Y-axis: logarithmic scale) for different sample sizes N (X-axis) through 50 simulations. Shaded areas are ( mean ± STD / √ 50) regions for each method, with thick lines being the means.

<!-- image -->

Table 4: Mean W 1 ,d π , 1 -inaccuracy over 50 simulations (standard deviation in parentheses)

|    N | Energy          | Laplace         | RBF             | PDF-L2          | KL              | FLE             |
|------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
|  300 | 0.627 (0.5522)  | 0.3361 (0.2142) | 0.3074 (0.1776) | 0.2643 (0.1493) | 0.6302 (0.5493) | 1.5668 (0.2838) |
|  350 | 0.3453 (0.2095) | 0.2648 (0.1396) | 0.2471 (0.1287) | 0.2105 (0.1096) | 0.3453 (0.2093) | 1.5537 (0.2734) |
|  400 | 0.3082 (0.1758) | 0.2557 (0.1294) | 0.2261 (0.1056) | 0.2128 (0.1025) | 0.3078 (0.1755) | 1.5348 (0.2568) |
|  450 | 0.2221 (0.1268) | 0.1771 (0.0782) | 0.1773 (0.0765) | 0.1710 (0.0744) | 0.2222 (0.1269) | 1.5293 (0.2485) |
|  500 | 0.2143 (0.1037) | 0.1962 (0.0946) | 0.1747 (0.0723) | 0.1480 (0.0574) | 0.2143 (0.1037) | 1.5597 (0.3352) |
|  550 | 0.2114 (0.1250) | 0.1821 (0.0951) | 0.1686 (0.0913) | 0.1577 (0.0841) | 0.2115 (0.1251) | 1.4932 (0.2865) |
|  600 | 0.1896 (0.0889) | 0.1766 (0.0890) | 0.1658 (0.0829) | 0.1466 (0.0759) | 0.1897 (0.0891) | 1.5288 (0.2571) |
|  650 | 0.1683 (0.0936) | 0.1518 (0.0809) | 0.1496 (0.0780) | 0.1392 (0.0749) | 0.1684 (0.0936) | 1.4921 (0.3214) |
|  700 | 0.1507 (0.0733) | 0.1459 (0.0628) | 0.1424 (0.0611) | 0.1358 (0.0517) | 0.1506 (0.0733) | 1.5143 (0.3056) |
|  750 | 0.1343 (0.0702) | 0.1313 (0.0682) | 0.1297 (0.0656) | 0.1215 (0.0585) | 0.1343 (0.0702) | 1.4949 (0.3226) |
|  800 | 0.1198 (0.0512) | 0.1177 (0.0553) | 0.1146 (0.0576) | 0.1091 (0.0517) | 0.1198 (0.0512) | 1.5040 (0.3862) |
|  850 | 0.1199 (0.0479) | 0.1150 (0.0470) | 0.1100 (0.0476) | 0.1047 (0.0484) | 0.1199 (0.0479) | 1.5714 (0.3766) |
|  900 | 0.1136 (0.0493) | 0.1098 (0.0450) | 0.1072 (0.0448) | 0.1059 (0.0429) | 0.1136 (0.0494) | 1.5689 (0.3886) |
|  950 | 0.1113 (0.0448) | 0.1063 (0.0405) | 0.1010 (0.0404) | 0.0940 (0.0381) | 0.1113 (0.0448) | 1.5502 (0.3319) |
| 1000 | 0.1003 (0.0438) | 0.0983 (0.0467) | 0.0947 (0.0452) | 0.0944 (0.0463) | 0.1003 (0.0438) | 1.4774 (0.2979) |

## E.2 Details in Atari games

## E.2.1 Deep Neural Network structure

Our model consists of two parts: (i) CNN layers that reduce 4 × 84 × 84 image into 512 features, (ii) multiple layers that transform 512 features into |A| × M × 3 . Part (i) is identical to the structure of [35] that first applied DQN in Atari games. After training the optimal model via DQN (with the same network structure with [35]) for ourselves, we copied this part to (i), and then freezed it, assuming that this part already performs good recognition of the image. Part (ii) contains the components that we trained in our methods. To make the model sufficiently complex, we included multiple hidden layers that reduce the 512 features into 512 → 450 → 400 → 350 → 300 → 250 → 200 → 150 → 128 →|A|× ( M × 3) , with each layer containing ReLU. For a given input image, the model outputs three parameters (weight, mean, variance) for each gaussian mixture component, for every action as follows,

<!-- formula-not-decoded -->

Here, w m represents gaussian component for the GMM model, which together sum up to 1, i.e., ∑ M m =1 w m ( s, a ; θ ) = 1 .

For quantile-based methods, we preserve the same layers, except that the output distribution has M quantiles instead of ( M × 3) parameters. That is, this consists of multiple hidden layers that reduce the 512 features into 512 → 450 → 400 → 350 → 300 → 250 → 200 → 150 → 128 →|A| × M , to form the following distributions based on dirac-delta's δ m .

<!-- formula-not-decoded -->

Note that GMM modeling can (asymptotically) accommodate dirac delta based modeling with sufficiantly small variance values σ 2 m ( s, a ; θ ) ≈ 0 and equal weights w m ( s, a ; θ ) = 1 /M .

## E.2.2 Algorithmic details

Our goal is to estimate Υ π ∈ P S×A . Here, the target policy is π = π ∗ ϵ tar , which is the epsilon-greedy variant (e.g., see Section 2.2 of [53]) of DQN-trained policy π ∗ [35] with ϵ = ϵ tar . We collected offline data ( N = 2 K, 5 K, 10 K ) through the trajectory of an agent following a behavior policy b = π ∗ ϵ beh with ϵ beh . For most cases, we let ϵ beh &gt; ϵ tar so as to satisfy Assumption C.4. In all simulations, we applied FDE methods based on Algorithm 1 with T = 50 . Minimization of (5) of Algorithm 1 is done stochastically by Adam. In each t -th iteration ( t = 1 , · · · , T ), we ran 1000 stochastic gradient updates, each based on a batch of 32 randomly selected samples. For practicality, unlike LQR simulations shown in Appendix E.1.2, we did not use data splitting for each iteration, but instead reused samples throughout multiple iterations.

When measuring the inaccuracy, instead of W 1 ,d π , 1 (Υ θ T , Υ π ) that requires heavy computation to approximate, we computed W 1 (Υ d π θ T , Υ d π π ) , where Υ d π := ∫ S×A Υ( s, a )d d π ( s, a ) ∈ P . Here, d π is approximated with 1000 pre-sampled observations of state-action pairs. These are sampled by Algorithm 2, which is a commonly used strategy of sampling s, a ∼ d π (e.g., see Algorithm 1 of [1]). Using the pre-sampled state-action pairs s ( m ) , a ( m ) ( m = 1 , · · · , 1000 ), we sample z ( m ) ∼ Υ θ T ( s ( m ) , a ( m ) ) for each m , by which we approximate Υ d π θ T . Based on the same s ( m ) , a ( m ) , we can also sample z ( m ) ∼ Υ π ( s ( m ) , a ( m ) ) by forming a long enough trajectory starting from initial state-action pair s ( m ) , a ( m ) and adding the consecutive rewards. This gives us approximation of Υ d π π .

Although our theorems (Theorems 3.4, C.11) give us bounds for ¯ η d π ,q (Υ θ T , Υ π ) , this can also lead to bound for η (Υ d π θ T , Υ d π π ) . This is due to the following inequality based on convexity of (15) and (16) in Appendix C.4,

<!-- formula-not-decoded -->

```
Algorithm 2 Sampling m state-action pairs from d π 1: Input: Initial state distribution µ , target policy π , discount rate γ , transition P ( ·| s, a ) 2: Output: State-action pairs { ( s ( m ) , a ( m ) ) } 1000 m =1 3: for m = 1 to 1000 do 4: Sample s cur ∼ µ and a cur ∼ π ( · | s cur ) 5: Accept ← False 6: while not Accept do 7: Sample U ∼ Unif(0 , 1) 8: if U < 1 -γ then 9: Accept ← True 10: else 11: Sample s ′ ∼ P ( · | s cur , a cur ) 12: Sample a ′ ∼ π ( · | s ′ ) 13: ( s cur , a cur ) ← ( s ′ , a ′ ) 14: end if 15: end while 16: ( s ( m ) , a ( m ) ) ← ( s cur , a cur ) 17: output ( s ( m ) , a ( m ) ) 18: end for
```

## E.2.3 Closed / approximated form

Based on the formula of Appendix E.1.3, we computed a single term in (10), i.e., d (Υ θ ( s, a ) , Ψ π ( r, s ′ , Υ ˆ θ n,t -1 )) , for MMD methods and PDF-L2. Since Laplace FDE had ill performance due to its numerical difficulties that we aforementioned, we excluded it. For other methods (i.e., KL, TVD, Hyvärinen divergences), we do not have a closed form objective function for GMM. Therefore, we approximated it with Monte Carlo approximation by z ′ j 2 ,b ∼ iid N ( r + γ · µ j 2 ( s ′ , a ′ ; θ -1 ) , γ 2 · σ 2 j 2 ( s ′ , a ′ ; θ -1 )) with j 2 = 1 , · · · , M and b = 1 , · · · , B , which represents a single mixture component of Ψ π ( r, s ′ , Υ ˆ θ n,t -1 ) .

For KL FDE, we can convert it to maximizing the expected log-likelihood, with individual term being as follows. Here, f ( ·| P ) is the density of the probability measure P and ϕ ( ·| µ, σ 2 ) means the density of N ( µ, σ 2 ) . When computing the density of Ψ π ( r, s ′ , Υ ˆ θ n,t -1 ) , we sampled a ′ ∼ π ( ·| s ′ ) instead of making use of π ( a ′ | s ′ ) for all a ′ ∈ A in every method, for the sake of computational convenience.

<!-- formula-not-decoded -->

For Hyvärinen divergence, we instead maximized Hyvärinen score with individual term being E Z ′ ∼ Ψ π ( r,s ′ , Υ ˆ θ n,t -1 ) { S H ( Z ′ , Υ θ ( s, a ) ) } as follows, which is computed by MC approximation in the same way (assuming GMM for P ),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For TVD, we approximated ∥ f ( ·| Υ θ ( s i , a i )) -f ( ·| Ψ π ( r i , s ′ i , Υ ˆ θ n,t -1 , a ′ i )) ∥ L 1 by using the density ratio as follows,

<!-- formula-not-decoded -->

We let B = 100 and B = 50 for deterministic transition and stochastic transition, respectively, which we will explain in the following subsection (Appendix E.2.4). We added a term ϵ var = 1 . 0 to the variance σ 2 j ( s, a ; θ ) in PDF-L2, KL, TVD, to prevent explosion of density values, thereby mitigating numerical instability, and used the same value for the tuning parameter of RBF FDE, i.e., σ RBF = 1 . 0 . For Hyvärinen, we put ϵ var = 10 . 0 since it was more prone to numerical instabilities. Even so, we could not run simulations for M = 200 due to numerical instabilities. The good thing about Energy FDE is that it does not require any tuning parameter, which makes it more user-friendly (and performance was good as well).

## E.2.4 Simulation settings

In our simulations, we always assumed γ = 0 . 95 . Every time a reward is observed, we clipped it between -10 and 10 , and then multiplied it by a fixed constant ( 20 in our case). Then, we tried two different settings: (i) deterministic transition ˜ p ( r, s ′ | s, a ) as the original Atari games, (ii) added small noise N (0 , 1) to every reward to make it more random (which makes it a better environment to apply distributional RL). We also tried two behavior policies for each setting, which leads to different coverage.

First, let us start with Setting-1, deterministic transition. Here, the target policy is π ∗ ϵ tar ( ϵ tar -greedy version of DQN-trained optimal policy π ∗ ) with ϵ tar = 0 . 3 . The behavior policy is ϵ beh = 0 . 4 , 0 . 8 (but ϵ beh = 0 . 4 , 0 . 5 for Enduro and Pong to prevent sparse observation of rewards). Of course, weak coverage (i.e., ϵ beh being a lot bigger than ϵ tar ) leads to worse performance. Results are visualized in the first two rows of Figures 3-5 for N = 10 K (Tables 6-19 for N = 2 K, 5 K, 10 K ). In the simulations, M refers to the number of gaussian mixtures in GMM model and the number of quantiles in quantile-based models (QRDQN, IQN).

Second, we ran simulations on Setting-2, random transition (or reward). Here, the target policy is π = π ∗ (deterministic policy learned by DQN) in all games except Breakout. In Breakout, we let π = π ∗ ϵ tar with ϵ tar = 0 . 3 to prevent the agent from being stuck at certain point (not proceeding with the game and repeating the same actions). The behavior policy has ϵ beh = 0 . 1 , 0 . 5 for all games ( ϵ beh = 0 . 1 , 0 . 3 for Pong only). Results are visualized in the last two rows of Figures 3-5 for N = 10 K (Tables 20-33 for N = 2 K, 5 K, 10 K ).

In our simulations, we have tried 7 games, 3 different sample sizes ( N = 2 K, 5 K, 10 K ), 3 different number of mixtures ( M = 10 , 100 , 200 ), 2 different environments (random / deterministic reward), 2 different behavior policies (good or bad coverage). These amount to 7 × 3 × 3 × 2 × 2 = 252 different settings, each leading to various shapes of return distributions. In each setting, we have applied different methodologies (FLE, QRDQN, IQM, KL, Energy, PDF-L2, RBF, TVD, Hyvärinen) under 5 different seeds. In Figures 3-5, we plotted ( mean ± STD ) area for each method.

## E.2.5 Simulation results

Simulation results for each of the aforementioned 252 settings are shown in Figures 3-5 and Tables 6-33. As we have stronger coverage (i.e., ϵ beh being closer to ϵ tar ), we achieve lower inaccuracy levels. In many cases, we could see that the inaccuracy levels generally became lower with larger sample sizes. We could also see that density modeling (based on gaussian mixture models) helps improving the accuracy, even when the reward is deterministic. This can be seen by comparing our methods (KL, Energy, PDF-L2, RBF) with quantile-based methods (QRDQN, IQN) that use the same number of M (number of gaussian components in GMM or number of quantiles). Moreover, TVD does not improve the accuracy throughout iterations. This corroborates our claim in Theorem 2.4, since TVD is not a functional Bregman divergence.

Since it is difficult to compare the performances for every single setting, we summarized the performances of each method in Table 5. We have grouped the 252 settings into four categories based on (i) deterministic / random reward and (ii) strong / weak coverage. In each category (which consists of 84 settings), we measured the rank values of nine methods (with 1 being the best and 9 being the worst) based on their mean inaccuracy in each setting, and then recorded the mean of rank values. Our FDE methods (KL, Energy, Hyvärinen) showed the highest three accuracies in most categories. Although KL FDE recorded the highest mean of rank in all four categories, it does not mean that we should always resort to KL FDE. There have been well-known criticism on KL divergence (e.g., unbounded divergence value, high sensitivity to the tails of distributions, necessity that one probability measure is absolutely continuous to the other, inability to consider closeness in outcome values) [e.g., 28, 50, 7].

Table 5: Mean of rank values of inaccuracy under all settings of each category. Three methods with best accuracies are boldfaced in each category.

| Category      | Category   | Method   | Method   | Method   | Method   | Method   | Method   | Method   | Method   | Method   |
|---------------|------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| Reward        | Cover      | FLE      | QRD      | IQN      | KL       | Energy   | PDF      | RBF      | TVD      | Hyv      |
| Deterministic | Strong     | 3.60     | 6.19     | 8.25     | 1.77     | 3.36     | 4.38     | 5.03     | 8.11     | 2.78     |
| Deterministic | Weak       | 3.00     | 6.39     | 8.30     | 1.84     | 3.14     | 4.69     | 5.47     | 8.09     | 2.42     |
| Random        | Strong     | 4.42     | 5.30     | 7.00     | 3.03     | 3.15     | 4.07     | 4.07     | 8.63     | 3.59     |
| Random        | Weak       | 3.76     | 6.09     | 7.31     | 2.34     | 3.61     | 4.46     | 4.23     | 8.61     | 2.40     |

10

Figure 3: (Mean ± STD) of W 1 (Υ d π θ , Υ dπ π ) -inaccuracy in each games (columns) for different methods with M = 10 , N = 10 K : (row1) deterministic transition with strong coverage ( ϵ beh = 0 . 4 ), (row2) deterministic transition with weak coverage ( ϵ beh &gt; 0 . 4 ), (row3) random transition with strong coverage ( ϵ beh = 0 . 1 ), (row4) random transition with weak coverage ( ϵ beh &gt; 0 . 1 ).

<!-- image -->

Figure 4: (Mean ± STD) of W 1 (Υ d π θ , Υ dπ π ) -inaccuracy in each games (columns) for different methods with M = 100 , N = 10 K : (row1) deterministic transition with strong coverage ( ϵ beh = 0 . 4 ), (row2) deterministic transition with weak coverage ( ϵ beh &gt; 0 . 4 ), (row3) random transition with strong coverage ( ϵ beh = 0 . 1 ), (row4) random transition with weak coverage ( ϵ beh &gt; 0 . 1 ).

<!-- image -->

Figure 5: (Mean ± STD) of W 1 (Υ d π θ , Υ dπ π ) -inaccuracy in each games (columns) for different methods with M = 200 , N = 10 K : (row1) deterministic transition with strong coverage ( ϵ beh = 0 . 4 ), (row2) deterministic transition with weak coverage ( ϵ beh &gt; 0 . 4 ), (row3) random transition with strong coverage ( ϵ beh = 0 . 1 ), (row4) random transition with weak coverage ( ϵ beh &gt; 0 . 1 ).

<!-- image -->

Table 6: Reward-variance=0, Eps=Small, Game=AtlantisNoFrameskip-v4

|   M | N    |   FLE |   QRD |   IQN |   KLD |   ENE |   PDF |   RBF | TVD       | HYV   |
|-----|------|-------|-------|-------|-------|-------|-------|-------|-----------|-------|
|  10 | 2 K  |  2.02 |  9.84 |  9.86 |  2.13 |  2.54 |  3.54 |  3.7  | 9.85 9.85 | 1.3   |
|  10 | 5 K  |  1.87 |  9.84 |  9.86 |  1.51 |  2.31 |  3.6  |  3.73 |           | 1.29  |
|  10 | 10 K |  1.73 |  9.84 |  9.86 |  1.8  |  2.54 |  3.59 |  3.72 | 9.86      | 1.34  |
| 100 | 2 K  |  2.22 |  4.13 |  9.85 |  1.71 |  2.28 |  3.54 |  3.77 | 9.87      | 1.77  |
| 100 | 5 K  |  1.88 |  3.87 |  9.85 |  1.67 |  2.29 |  3.65 |  3.44 | 9.85      | 1.77  |
| 100 | 10 K |  1.85 |  3.93 |  9.85 |  1.93 |  2.26 |  3.87 |  3.42 | 9.87      | 1.88  |
| 200 | 2 K  |  2.27 |  2.84 |  9.85 |  1.93 |  2.4  |  3.76 |  3.63 | 9.85      | NA    |
| 200 | 5 K  |  1.69 |  2.88 |  9.38 |  1.61 |  2.33 |  3.34 |  3.16 | 9.85      | NA    |
| 200 | 10 K |  2.39 |  2.83 |  9.85 |  1.68 |  2.35 |  3.56 |  3.24 | 9.86      | NA    |

Table 7: Reward-variance=0, Eps=Small, Game=BreakoutNoFrameskip-v4

|   M | N    | FLE       | QRD         | IQN        | KLD       | ENE       | PDF       | RBF       | TVD         | HYV   |
|-----|------|-----------|-------------|------------|-----------|-----------|-----------|-----------|-------------|-------|
|  10 | 2 K  | 3.63 3.21 | 10.87 10.87 | 10.89 10.9 | 6.68 2.78 | 3.37 3.22 | 4.48 4.39 | 4.81 4.37 | 10.89 10.89 | 2.81  |
|  10 | 5 K  |           |             |            |           |           |           |           |             | 2.58  |
|  10 | 10 K | 3.44      | 10.87       | 10.89      | 2.29      | 3.25      | 4.18      | 4.36      | 10.89       | 2.91  |
| 100 | 2 K  | 3.51      | 4.93        | 10.89      | 2.7       | 3.49      | 4.56      | 4.43      | 10.88       | 2.91  |
| 100 | 5 K  | 3.47      | 4.58        | 10.89      | 2.9       | 3.29      | 4.38      | 4.22      | 10.88       | 3.19  |
| 100 | 10 K | 3.24      | 4.24        | 10.88      | 2.37      | 3.1       | 4.28      | 4.29      | >100        | 2.32  |
| 200 | 2 K  | 3.28      | 3.88        | 10.88      | 3.15      | 3.41      | 4.49      | 4.21      | 10.9        | NA    |
| 200 | 5 K  | 3.39      | 3.65        | 10.9       | 3.02      | 3.26      | 4.11      | 3.92      | 10.9        | NA    |
| 200 | 10 K | 4.06      | 3.19        | 8.34       | 2.9       | 3.23      | 3.74      | 3.97      | 10.9        | NA    |

Table 8: Reward-variance=0, Eps=Small, Game=EnduroNoFrameskip-v4

|   M | N            | FLE               | QRD               | IQN               | KLD               | ENE               | PDF               | RBF               | TVD               | HYV               |
|-----|--------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
|  10 | 2 K 5 K 10 K | 12.69 >100 11.28  | 19.25 19.25 19.25 | 19.25 19.25 19.25 | 12.68 9.42 11.67  | 13.0 11.17 11.72  | 12.37 10.89 10.98 | 12.36 10.81 10.98 | 19.25 19.25 19.24 | 14.1 11.32 12.17  |
| 100 | 2 K 5 K 10 K | 12.52 11.51 10.95 | 15.33 13.51 12.15 | 19.26 19.25 19.25 | 11.08 10.64 11.46 | 12.83 10.91 11.61 | 12.31 10.84 10.99 | 12.4 10.93 11.0   | 19.25 19.24 19.25 | 12.99 10.65 12.43 |
| 200 | 2 K 5 K 10 K | 12.6 10.47 12.11  | 14.31 12.5 12.55  | 19.25 19.26 19.26 | 12.24 10.58 11.65 | 12.9 11.1 11.71   | 12.51 10.89 10.92 | 12.57 10.85 11.0  | 19.25 19.24 19.24 | NA NA NA          |

Table 9: Reward-variance=0, Eps=Small, Game=KungFuMasterNoFrameskip-v4

|   M | N            | FLE            | QRD            | IQN            | KLD            | ENE            | PDF            | RBF           | TVD            | HYV            |
|-----|--------------|----------------|----------------|----------------|----------------|----------------|----------------|---------------|----------------|----------------|
|  10 | 2 K 5 K 10 K | 2.83 2.82 3.1  | 9.04 9.09 9.1  | 9.25 9.25 9.26 | 1.84 2.85 2.57 | 2.66 3.38 3.36 | 3.61 4.3 4.09  | 4.12 6.7 7.8  | 9.23 9.24 9.26 | 1.91 2.6 2.72  |
| 100 | 2 K 5 K 10 K | 2.53 2.89 >100 | 4.1 4.8 4.26   | 9.22 9.23 9.23 | 1.42 2.85 2.29 | 2.52 3.38 3.02 | 3.3 4.05 3.94  | 3.62 4.52 4.7 | 9.26 9.25 9.23 | 2.17 3.65 3.03 |
| 200 | 2 K 5 K 10 K | 2.74 3.73 3.01 | 3.16 4.19 3.79 | 9.24 9.24 9.23 | 2.0 2.77 1.97  | 2.63 3.43 3.17 | 3.38 4.19 4.43 | 3.49 4.44 4.2 | 9.26 9.24 9.24 | NA NA NA       |

Table 10: Reward-variance=0, Eps=Small, Game=PongNoFrameskip-v4

|   M | N    | FLE   |   QRD |   IQN |   KLD |   ENE |   PDF |   RBF | TVD       | HYV     |
|-----|------|-------|-------|-------|-------|-------|-------|-------|-----------|---------|
|  10 | 2 K  | 1.79  |  4.99 |  4.99 |  1.66 |  1.79 |  2.42 |  3.93 | 4.98 4.98 | 3.2 2.8 |
|  10 | 5 K  | 1.57  |  4.99 |  4.99 | 92.04 |  1.64 |  2.53 |  3.92 |           |         |
|  10 | 10 K | 1.92  |  4.99 |  4.99 |  2.05 |  2.05 |  2.81 |  4.22 | 4.98      | 3.02    |
| 100 | 2 K  | 1.53  |  4.71 |  4.99 |  1.76 |  1.62 |  2.38 |  2.41 | 4.98      | 2.74    |
| 100 | 5 K  | 1.62  |  4.61 |  4.99 |  1.77 |  1.65 |  2.5  |  2.62 | 4.98      | 2.61    |
| 100 | 10 K | >100  |  4.59 |  4.99 |  1.91 |  2.12 |  2.81 |  2.96 | 4.98      | 2.92    |
| 200 | 2 K  | 1.62  |  2.75 |  4.99 |  1.87 |  1.6  |  2.32 |  2.39 | 4.98      | NA      |
| 200 | 5 K  | 4.84  |  2.62 |  4.99 |  1.76 |  1.63 |  2.42 |  2.52 | 4.99      | NA      |
| 200 | 10 K | >100  |  2.67 |  4.99 |  2.04 |  2.04 |  2.73 |  2.99 | 4.98      | NA      |

Table 11: Reward-variance=0, Eps=Small, Game=QbertNoFrameskip-v4

|   M | N            | FLE               | QRD               | IQN               | KLD              | ENE               | PDF               | RBF               | TVD               | HYV               |
|-----|--------------|-------------------|-------------------|-------------------|------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
|  10 | 2 K 5 K 10 K | >100 10.96 11.2   | 21.74 21.76 21.76 | 21.82 21.81 21.81 | 10.8 10.79 10.29 | 11.82 11.52 11.29 | 12.74 12.5 11.55  | 12.9 12.09 11.45  | 21.81 21.83 21.81 | 10.87 10.74 10.51 |
| 100 | 2 K 5 K 10 K | 23.68 10.71 10.47 | 12.79 11.94 11.65 | 21.8 18.76 21.8   | 10.69 10.34 9.83 | 11.47 11.26 10.65 | 12.0 10.7 10.46   | 12.29 11.42 10.95 | 21.8 21.81 21.79  | 10.08 10.63 10.49 |
| 200 | 2 K 5 K 10 K | 11.67 11.19 >100  | 11.88 11.13 10.9  | 21.81 21.81 21.8  | 10.74 10.22 9.98 | 11.37 10.93 10.59 | 11.76 10.68 10.21 | 11.79 10.81 10.29 | 21.81 21.81 21.8  | NA NA NA          |

Table 12: Reward-variance=0, Eps=Small, Game=SpaceInvadersNoFrameskip-v4

|   M | N            | FLE            | QRD               | IQN               | KLD            | ENE            | PDF            | RBF            | TVD               | HYV            |
|-----|--------------|----------------|-------------------|-------------------|----------------|----------------|----------------|----------------|-------------------|----------------|
|  10 | 2 K 5 K 10 K | 3.02 2.87 2.79 | 15.35 15.35 15.35 | 15.37 15.37 15.37 | 2.73 2.16 2.17 | 3.55 3.72 3.66 | 5.76 5.79 5.63 | 5.82 5.71 5.54 | 15.38 >100 15.36  | 2.73 2.2 2.88  |
| 100 | 2 K 5 K 10 K | 3.03 2.58 2.43 | 4.53 4.23 4.24    | 15.37 14.07 15.37 | 2.42 2.15 1.93 | 3.07 3.22 3.12 | 3.73 3.6 3.09  | 4.8 4.31 3.91  | 15.37 15.38 15.38 | 2.38 2.31 2.06 |
| 200 | 2 K 5 K 10 K | 3.39 2.92 2.43 | 3.5 3.04 3.14     | 15.36 11.07 15.37 | 2.54 2.58 2.07 | 3.1 3.07 2.67  | 3.58 3.22 2.82 | 4.06 3.74 2.88 | 15.36 15.38 15.37 | NA NA NA       |

Table 13: Reward-variance=0, Eps=Big, Game=AtlantisNoFrameskip-v4

|   M | N       |   FLE | QRD       |   IQN | KLD       | ENE       | PDF      | RBF       | TVD       | HYV   |
|-----|---------|-------|-----------|-------|-----------|-----------|----------|-----------|-----------|-------|
|  10 | 2 K 5 K |  2.31 | 9.84 9.84 |  9.86 | 2.34 1.67 | 3.09 2.53 | 3.82 3.5 | 3.91 3.56 | 9.86 9.87 | 1.5   |
|  10 |         |  1.87 |           |  9.86 |           |           |          |           |           | 1.88  |
|  10 | 10 K    |  2.55 | 9.84      |  9.86 | 1.98      | 2.87      | 3.82     | 3.78      | 9.85      | 1.58  |
| 100 | 2 K     |  2.25 | 4.5       |  9.85 | 2.18      | 2.81      | 3.9      | 3.91      | 9.87      | 1.46  |
| 100 | 5 K     |  2.42 | 4.17      |  9.86 | 1.97      | 2.36      | 3.22     | 3.46      | 9.85      | 1.49  |
| 100 | 10 K    |  2.02 | 4.54      |  9.85 | 2.14      | 2.51      | 3.88     | 3.92      | 9.85      | 1.74  |
| 200 | 2 K     |  2.65 | 3.32      |  9.86 | 2.2       | 2.77      | 4.21     | 3.92      | 9.84      | NA    |
| 200 | 5 K     |  1.8  | 3.06      |  9.86 | 1.7       | 2.39      | 3.78     | 3.78      | 9.85      | NA    |
| 200 | 10 K    |  1.81 | 3.26      |  9.85 | 2.12      | 2.7       | 4.27     | 3.84      | 9.86      | NA    |

Table 14: Reward-variance=0, Eps=Big, Game=BreakoutNoFrameskip-v4

|   M | N            | FLE            | QRD               | IQN               | KLD            | ENE            | PDF            | RBF            | TVD              | HYV            |
|-----|--------------|----------------|-------------------|-------------------|----------------|----------------|----------------|----------------|------------------|----------------|
|  10 | 2 K 5 K 10 K | 7.14 6.13 6.45 | 10.88 10.88 10.88 | 10.9 10.9 10.9    | 6.66 5.6 5.09  | 7.29 6.87 6.74 | 9.31 8.25 8.17 | 7.95 8.39 8.01 | 10.9 10.89 10.88 | 6.72 6.02 5.98 |
| 100 | 2 K 5 K 10 K | 6.66 6.8 6.56  | 8.97 8.24 8.37    | 10.89 10.89 10.89 | 6.73 6.13 6.08 | 7.08 6.6 6.55  | 7.13 7.15 6.82 | 7.37 7.09 6.83 | 10.89 10.9 >100  | 6.28 6.18 6.07 |
| 200 | 2 K 5 K 10 K | 7.25 6.67 6.63 | 7.98 7.51 7.57    | 10.9 10.85 10.9   | 6.75 6.42 6.17 | 7.03 6.82 6.54 | 7.21 7.09 6.94 | 7.45 6.92 6.99 | 10.88 10.9 10.87 | NA NA NA       |

Table 15: Reward-variance=0, Eps=Big, Game=EnduroNoFrameskip-v4

|   M | N            | FLE               | QRD               | IQN               | KLD               | ENE               | PDF               | RBF               | TVD               | HYV              |
|-----|--------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|------------------|
|  10 | 2 K 5 K 10 K | 16.85 15.46 15.68 | 19.26 19.26 19.26 | 19.25 19.26 19.26 | 16.72 15.33 15.85 | 16.65 15.29 15.7  | 16.97 15.34 17.29 | 17.78 16.22 17.59 | 19.24 19.24 19.25 | 17.6 16.8 17.12  |
| 100 | 2 K 5 K 10 K | 16.79 15.69 15.23 | 18.53 18.5 19.14  | 19.25 19.26 19.25 | 16.7 15.42 15.89  | 16.7 15.34 15.65  | 16.42 14.79 15.19 | 16.88 15.9 15.89  | 19.24 19.24 19.25 | 17.2 16.25 16.81 |
| 200 | 2 K 5 K 10 K | 16.99 15.6 15.5   | 17.94 17.13 17.53 | 19.25 19.25 19.26 | 16.58 15.28 15.81 | 16.73 15.27 15.65 | 16.48 14.89 14.88 | 16.75 15.8 15.39  | 19.25 19.25 19.25 | NA NA NA         |

Table 16: Reward-variance=0, Eps=Big, Game=KungFuMasterNoFrameskip-v4

|   M | N            | FLE            | QRD            | IQN            | KLD            | ENE            | PDF            | RBF            | TVD            | HYV            |
|-----|--------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
|  10 | 2 K 5 K 10 K | 8.37 8.07 7.86 | 9.24 9.24 9.24 | 9.24 9.25 9.24 | 8.02 7.73 7.6  | 8.19 8.03 7.95 | 9.12 9.15 9.13 | 9.21 9.22 9.21 | 9.24 9.25 9.24 | 8.23 7.75 7.71 |
| 100 | 2 K 5 K 10 K | 8.16 7.94 7.89 | 9.2 9.2 9.2    | 9.24 9.23 9.24 | 8.05 7.67 7.55 | 8.16 7.92 7.77 | 8.11 8.44 7.75 | 8.76 8.94 8.98 | 9.22 9.26 9.23 | 8.06 7.67 7.6  |
| 200 | 2 K 5 K 10 K | 8.29 8.0 7.71  | 9.09 9.0 8.93  | 9.23 9.24 9.24 | 8.05 7.64 7.52 | 8.1 7.94 7.73  | 8.2 8.25 8.07  | 8.86 8.75 8.88 | 9.25 9.23 9.23 | NA NA NA       |

Table 17: Reward-variance=0, Eps=Big, Game=PongNoFrameskip-v4

|   M | N            | FLE            | QRD            | IQN            | KLD            | ENE            | PDF            | RBF            | TVD            | HYV           |
|-----|--------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|---------------|
|  10 | 2 K 5 K 10 K | 2.36 >100 2.13 | 4.99 4.99 4.99 | 4.99 4.99 4.99 | 2.33 2.26 2.18 | 2.51 2.66 2.32 | 2.95 3.51 3.19 | 3.35 3.78 3.54 | 4.98 4.98 4.98 | 3.1 3.05 2.9  |
| 100 | 2 K 5 K 10 K | 2.16 2.23 2.24 | 3.86 3.78 4.11 | 4.6 4.99 4.99  | 2.41 2.46 2.25 | 2.4 2.76 2.34  | 3.57 3.71 3.22 | >100 3.71 3.49 | 4.98 4.98 4.98 | 2.57 3.16 3.0 |
| 200 | 2 K 5 K 10 K | 2.43 2.43 2.19 | 2.54 2.58 2.59 | 4.99 4.99 4.99 | 2.58 2.57 2.19 | 2.44 2.55 2.42 | 3.13 3.56 3.0  | 3.51 3.71 3.74 | 4.98 4.98 4.99 | NA NA NA      |

Table 18: Reward-variance=0, Eps=Big, Game=QbertNoFrameskip-v4

|   M | N            | FLE               | QRD               | IQN               | KLD               | ENE              | PDF               | RBF              | TVD              | HYV               |
|-----|--------------|-------------------|-------------------|-------------------|-------------------|------------------|-------------------|------------------|------------------|-------------------|
|  10 | 2 K 5 K 10 K | 14.35 14.27 14.75 | 21.8 21.8 21.8    | 21.81 21.82 21.81 | 12.08 12.62 13.96 | 14.59 14.72 15.1 | 15.56 15.8 15.91  | 15.6 15.63 15.89 | 21.81 21.82 21.8 | 13.04 12.88 13.06 |
| 100 | 2 K 5 K 10 K | >100 13.95 15.8   | 15.4 15.84 16.35  | 21.81 21.81 21.81 | 14.24 13.91 13.99 | 14.31 14.7 14.93 | 15.64 15.81 15.79 | 15.43 15.7 15.79 | >100 21.82 21.79 | 12.68 13.22 13.03 |
| 200 | 2 K 5 K 10 K | 14.03 13.71 14.79 | 14.72 14.69 15.22 | 20.51 21.82 21.13 | 12.97 13.69 13.9  | 14.14 14.7 14.87 | 15.45 16.06 16.16 | 15.57 15.9 16.17 | 21.81 21.8 21.82 | NA NA NA          |

Table 19: Reward-variance=0, Eps=Big, Game=SpaceInvadersNoFrameskip-v4

|   M | N            | FLE            | QRD               | IQN               | KLD            | ENE            | PDF            | RBF            | TVD               | HYV            |
|-----|--------------|----------------|-------------------|-------------------|----------------|----------------|----------------|----------------|-------------------|----------------|
|  10 | 2 K 5 K 10 K | 7.26 7.24 6.33 | 15.36 15.36 15.36 | 15.38 15.37 15.37 | 7.39 6.91 6.17 | 8.12 7.81 7.45 | 9.22 8.88 8.69 | 9.34 8.94 8.58 | >100 15.37 15.37  | 5.95 5.8 4.72  |
| 100 | 2 K 5 K 10 K | 7.87 7.26 6.64 | 9.29 9.09 8.06    | 14.73 15.37 15.38 | 7.45 6.78 6.19 | 7.85 7.59 6.8  | 8.95 9.1 8.79  | 8.91 8.65 8.48 | 15.39 >100 >100   | 5.89 5.47 5.04 |
| 200 | 2 K 5 K 10 K | 7.5 6.98 6.55  | 8.13 7.92 6.88    | 15.37 15.37 15.36 | 7.41 6.7 6.18  | 7.92 7.45 7.11 | 9.0 8.68 8.09  | 8.5 8.59 8.43  | 15.38 15.37 15.38 | NA NA NA       |

Table 20: Reward-variance=1, Eps=Small, Game=AtlantisNoFrameskip-v4

|   M | N            | FLE            | QRD               | IQN               | KLD            | ENE            | PDF             | RBF              | TVD               | HYV            |
|-----|--------------|----------------|-------------------|-------------------|----------------|----------------|-----------------|------------------|-------------------|----------------|
|  10 | 2 K 5 K 10 K | 9.3 7.43 8.01  | 15.58 15.34 15.47 | 16.0 16.37 16.14  | 8.67 6.61 7.46 | 9.79 8.23 8.78 | 10.64 9.8 9.98  | 10.86 9.69 10.01 | >100 16.96 16.95  | 8.66 6.88 7.09 |
| 100 | 2 K 5 K 10 K | 8.96 7.87 7.27 | 10.37 8.96 8.35   | 16.05 13.8 15.89  | 8.52 6.38 7.39 | 9.81 7.73 8.34 | 10.71 9.28 9.87 | 10.78 9.13 9.68  | 16.89 16.95 16.95 | 8.14 6.47 7.31 |
| 200 | 2 K 5 K 10 K | 8.04 7.51 7.67 | 9.62 8.09 7.87    | 12.89 16.08 14.63 | 8.71 6.91 7.41 | 9.66 7.73 8.44 | 10.34 8.47 9.3  | 10.34 8.4 9.33   | 16.89 16.95 16.95 | NA NA NA       |

Table 21: Reward-variance=1, Eps=Small, Game=BreakoutNoFrameskip-v4

| M   | N    |   FLE |   QRD |   IQN | KLD       |   ENE |   PDF |   RBF | TVD        | HYV   |
|-----|------|-------|-------|-------|-----------|-------|-------|-------|------------|-------|
| 10  | 2 K  |  7.94 |  7.45 |  8.91 | 7.96 6.41 |  6.21 |  4.47 |  4.66 | >100 10.89 | 8.2   |
| 10  | 5 K  |  7.75 |  7.23 |  8.26 |           |  6.88 |  4.97 |  4.88 |            | 8.56  |
| 10  | 10 K |  7.57 |  7.35 |  8.45 | 6.27      |  5.76 |  3.84 |  3.6  | 10.9       | 8.29  |
| 100 | 2 K  |  7.62 |  4.8  |  7.58 | 7.65      |  6.48 |  3.94 |  5.52 | >100       | 7.87  |
| 100 | 5 K  |  8.17 |  6.02 |  7.09 | 7.15      |  6.34 |  6.49 |  6.32 | 10.89      | 9.81  |
| 100 | 10 K |  6.8  |  4.72 |  6.58 | 6.52      |  6.88 |  5.72 |  5.66 | >100       | 7.92  |
| 200 | 2 K  |  7.14 |  5.58 |  9.33 | 7.51      |  6.38 |  5.43 |  5.9  | >100       | NA    |
|     | 5 K  |  8.37 |  6.77 |  6.52 | 7.14      |  7.37 |  6.28 |  6.35 | 10.88      | NA    |
|     | 10 K |  7.67 |  5.37 |  7.97 | 6.84      |  5.62 |  5.62 |  6.03 | 10.9       | NA    |

Table 22: Reward-variance=1, Eps=Small, Game=EnduroNoFrameskip-v4

|   M | N            | FLE               | QRD               | IQN               | KLD               | ENE               | PDF               | RBF               | TVD               | HYV               |
|-----|--------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
|  10 | 2 K 5 K 10 K | 16.47 12.88 12.94 | 30.61 28.69 28.97 | 28.15 27.79 34.49 | 16.11 14.16 13.01 | 15.65 13.99 15.23 | 19.27 17.56 17.04 | 20.04 17.3 17.01  | 35.43 35.44 35.42 | 16.65 13.75 13.94 |
| 100 | 2 K 5 K 10 K | 15.29 13.21 19.26 | 20.12 17.4 17.35  | 29.65 25.74 33.79 | 17.63 14.44 14.46 | 17.02 13.07 12.57 | 18.45 15.42 15.89 | 18.9 14.12 16.2   | 35.43 35.43 35.42 | 16.91 14.17 14.24 |
| 200 | 2 K 5 K 10 K | 16.47 13.48 23.39 | 18.94 16.46 14.49 | 28.05 32.63 33.97 | 17.57 12.04 14.49 | 16.25 14.91 13.34 | 18.12 14.96 15.79 | 18.14 15.53 15.63 | 35.43 35.43 35.42 | NA NA NA          |

Table 23: Reward-variance=1, Eps=Small, Game=KungFuMasterNoFrameskip-v4

|   M | N            | FLE            | QRD               | IQN               | KLD            | ENE            | PDF            | RBF            | TVD               | HYV            |
|-----|--------------|----------------|-------------------|-------------------|----------------|----------------|----------------|----------------|-------------------|----------------|
|  10 | 2 K 5 K 10 K | 5.28 5.18 5.47 | 13.84 13.79 13.67 | 14.28 15.45 11.28 | 4.71 4.53 4.08 | 5.38 6.39 5.7  | 6.15 6.74 6.41 | 5.67 6.99 6.78 | 16.35 16.21 16.26 | 4.47 4.91 4.91 |
| 100 | 2 K 5 K 10 K | 5.06 5.04 5.57 | 6.52 6.82 6.61    | 14.1 15.51 15.33  | 4.27 4.39 4.4  | 4.64 6.12 5.02 | 5.45 6.36 5.61 | 5.17 5.82 5.95 | 16.36 16.23 16.25 | 3.94 3.86 3.8  |
| 200 | 2 K 5 K 10 K | 4.43 5.29 7.53 | 5.73 5.44 5.85    | 15.26 14.88 15.38 | 3.51 5.34 4.3  | 4.7 5.33 4.99  | 5.17 6.4 5.71  | 5.2 6.19 5.54  | 16.37 16.23 16.24 | NA NA NA       |

Table 24: Reward-variance=1, Eps=Small, Game=PongNoFrameskip-v4

|   M | N            | FLE            | QRD            | IQN            | KLD            | ENE            | PDF            | RBF            | TVD            | HYV            |
|-----|--------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
|  10 | 2 K 5 K 10 K | 2.07 2.23 2.29 | 4.4 4.91 4.97  | 3.98 5.4 4.01  | 1.57 1.84 1.83 | 1.62 1.87 1.89 | 1.54 1.71 1.65 | 1.58 1.64 1.74 | 5.82 5.94 5.98 | 1.66 1.87 1.99 |
| 100 | 2 K 5 K 10 K | 1.85 2.11 2.74 | 2.4 2.52 2.8   | 3.32 3.16 3.48 | 1.63 1.87 1.97 | 1.59 1.8 1.84  | 1.3 1.38 1.46  | 1.36 1.4 1.56  | 5.82 5.93 5.98 | 1.67 1.94 1.74 |
| 200 | 2 K 5 K 10 K | 1.75 2.35 2.2  | 1.64 1.78 2.06 | 3.03 3.39 4.84 | 1.64 1.79 1.86 | 1.54 1.86 1.8  | 1.39 1.38 1.59 | 1.35 1.4 1.46  | 5.82 5.93 5.99 | NA NA NA       |

Table 25: Reward-variance=1, Eps=Small, Game=QbertNoFrameskip-v4

|   M | N            | FLE               | QRD               | IQN               | KLD               | ENE               | PDF               | RBF               | TVD               | HYV               |
|-----|--------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
|  10 | 2 K 5 K 10 K | 55.54 20.71 20.08 | 20.34 18.41 23.4  | 18.33 10.83 29.16 | 17.53 20.86 19.75 | 16.87 17.7 17.04  | 19.14 >100 19.03  | 19.07 19.07 18.54 | 30.08 30.23 30.31 | 21.31 22.11 20.86 |
| 100 | 2 K 5 K 10 K | >100 21.43 44.06  | 21.99 11.46 16.0  | 17.19 28.83 29.07 | 21.47 21.11 17.35 | 18.71 14.68 19.83 | 19.38 19.5 19.44  | 19.61 19.5 19.15  | >100 30.22 >100   | 21.05 21.13 20.9  |
| 200 | 2 K 5 K 10 K | 20.54 20.55 19.93 | 21.43 18.94 16.07 | 24.77 24.2 28.88  | 21.07 21.18 17.77 | 17.79 17.11 13.27 | 19.62 19.36 19.67 | 19.48 19.94 19.39 | 30.07 >100 30.3   | NA NA NA          |

Table 26: Reward-variance=1, Eps=Small, Game=SpaceInvadersNoFrameskip-v4

|   M | N            | FLE           | QRD              | IQN              | KLD            | ENE            | PDF            | RBF            | TVD              | HYV            |
|-----|--------------|---------------|------------------|------------------|----------------|----------------|----------------|----------------|------------------|----------------|
|  10 | 2 K 5 K 10 K | 2.45 2.7 2.63 | 15.31 15.73 15.6 | 7.41 10.72 11.07 | 1.79 2.65 2.02 | 2.75 3.91 2.98 | 4.52 4.95 4.44 | 4.17 5.39 4.6  | >100 19.05 >100  | 1.53 2.16 1.56 |
| 100 | 2 K 5 K 10 K | 6.26 3.1 3.89 | 3.75 4.07 3.94   | 13.67 8.68 12.68 | 1.88 2.39 1.77 | 2.38 2.81 2.93 | 2.79 3.91 3.05 | 2.98 4.18 3.2  | 18.93 19.05 19.2 | 1.54 1.68 1.5  |
| 200 | 2 K 5 K 10 K | 2.44 3.53 2.7 | 2.62 2.86 2.81   | 13.69 8.45 6.55  | 1.86 2.27 2.27 | 2.46 3.4 2.67  | 2.66 3.44 2.71 | 2.83 3.77 2.44 | 18.93 >100 19.2  | NA NA NA       |

Table 27: Reward-variance=1, Eps=Big, Game=AtlantisNoFrameskip-v4

|   M | N    |   FLE |   QRD |   IQN |   KLD |   ENE |   PDF |   RBF | TVD        | HYV   |
|-----|------|-------|-------|-------|-------|-------|-------|-------|------------|-------|
|  10 | 2 K  |  8.59 | 15.19 | 15.67 |  8.08 |  9.23 | 10.4  | 10.41 | 16.47 >100 | 7.87  |
|  10 | 5 K  |  8.62 | 15.5  | 14.08 |  8.41 |  9.38 | 10.5  | 10.42 |            | 7.95  |
|  10 | 10 K |  8.47 | 15.57 | 15.11 |  8.35 |  9.41 | 10.62 | 10.54 | 16.81      | 8.23  |
| 100 | 2 K  |  8.02 |  9.79 | 15.67 |  7.99 |  9.06 | 10.31 | 10.17 | 16.47      | 8.11  |
| 100 | 5 K  |  8.85 |  9.95 | 13.83 |  8.42 |  9.06 | 10.38 | 10.33 | 16.63      | 8.57  |
| 100 | 10 K |  8.49 | 10.13 | 15.36 |  8.24 |  9.06 | 10.49 | 10.64 | 16.8       | 7.76  |
| 200 | 2 K  |  8.29 |  9    | 14.87 |  8.04 |  8.86 |  9.82 | 10.08 | 16.47      | NA    |
| 200 | 5 K  |  8.99 |  8.29 | 15.71 |  7.82 |  9.09 |  9.95 | 10.09 | 16.63      | NA    |
| 200 | 10 K |  8.54 |  9.32 | 13.2  |  8.18 |  9.28 | 10.29 |  9.96 | 16.8       | NA    |

Table 28: Reward-variance=1, Eps=Big, Game=BreakoutNoFrameskip-v4

|   M | N    | FLE   |   QRD |   IQN | KLD       |   ENE | PDF       |   RBF | TVD        | HYV   |
|-----|------|-------|-------|-------|-----------|-------|-----------|-------|------------|-------|
|  10 | 2 K  | 3.22  |  9.3  |  7.43 | 2.58 2.72 |  3.24 | 4.47 4.87 |  4.47 | 10.52 >100 | 2.94  |
|  10 | 5 K  | 3.8   |  9.44 | 10.19 |           |  3.66 |           |  4.85 |            | 3.27  |
|  10 | 10 K | 3.57  |  9.64 | 10.28 | 3.0       |  3.93 | 4.8       |  4.79 | 10.81      | 3.07  |
| 100 | 2 K  | >100  |  4.23 |  7.17 | 2.44      |  3.39 | 4.13      |  4.15 | 10.52      | 2.79  |
| 100 | 5 K  | 3.49  |  4.67 |  8.32 | 3.32      |  3.68 | 4.27      |  4.71 | 10.73      | 3.07  |
| 100 | 10 K | 3.98  |  4.97 |  7.5  | 3.21      |  3.65 | 4.89      |  4.77 | 10.83      | 3.21  |
| 200 | 2 K  | 3.73  |  3.55 |  8.33 | 2.45      |  3.25 | 4.13      |  3.65 | 10.52      | NA    |
| 200 | 5 K  | 4.11  |  3.98 |  9.21 | 2.66      |  3.67 | 4.28      |  4.22 | 10.72      | NA    |
| 200 | 10 K | 3.58  |  4.27 | 10.09 | 3.22      |  3.75 | 4.54      |  4.44 | 10.82      | NA    |

Table 29: Reward-variance=1, Eps=Big, Game=EnduroNoFrameskip-v4

|   M | N    |   FLE | QRD         |   IQN |   KLD |   ENE | PDF         |   RBF | TVD         | HYV   |
|-----|------|-------|-------------|-------|-------|-------|-------------|-------|-------------|-------|
|  10 | 2 K  | 32.27 | 34.71 34.54 | 32.87 | 32.4  | 32.41 | 32.06 31.01 | 32.08 | 35.28 35.24 | 32.38 |
|  10 | 5 K  | 31.56 |             | 34.43 | 31.49 | 31.36 |             | 30.99 |             | 31.42 |
|  10 | 10 K | 31.51 | 34.7        | 34.37 | 31.46 | 31.41 | 31.18       | 31.1  | 35.39       | 31.46 |
| 100 | 2 K  | 32.27 | 33.53       | 32.97 | 32.33 | 32.29 | 31.98       | 32.01 | 35.26       | 32.32 |
| 100 | 5 K  | 31.47 | 33.19       | 32.92 | 31.32 | 31.37 | 30.72       | 30.86 | 35.24       | 31.38 |
| 100 | 10 K | 31.65 | 33.67       | 35.27 | 31.52 | 31.56 | 31.18       | 31.07 | 35.39       | 31.59 |
| 200 | 2 K  | 32.19 | 33.08       | 34.61 | 32.37 | 32.28 | 32.0        | 32.06 | 35.27       | NA    |
| 200 | 5 K  | 31.37 | 32.5        | 34.32 | 31.13 | 31.32 | 32.18       | 30.89 | 35.24       | NA    |
| 200 | 10 K | 31.52 | 32.86       | 35.26 | 31.46 | 31.47 | 31.12       | 31.01 | 35.38       | NA    |

Table 30: Reward-variance=1, Eps=Big, Game=KungFuMasterNoFrameskip-v4

|   M | N    |   FLE |   QRD |   IQN |   KLD |   ENE |   PDF |   RBF |   TVD | HYV         |
|-----|------|-------|-------|-------|-------|-------|-------|-------|-------|-------------|
|  10 | 2 K  | 12.89 | 15.41 | 14.16 | 12.92 | 13.21 | 13.4  | 13.69 | 16.17 | 12.75 12.63 |
|  10 | 5 K  | 12.76 | 15.31 | 13.43 | 12.53 | 11.82 | 13.05 | 13.13 | 16.08 |             |
|  10 | 10 K | 12.86 | 15.4  | 15.97 | 12.54 | 12.97 | 13.14 | 13.12 | 16.06 | 12.26       |
| 100 | 2 K  | 13.05 | 14.31 | 15.68 | 12.89 | 13.15 | 13.16 | 13.08 | 16.16 | 12.77       |
| 100 | 5 K  | 12.52 | 14.06 | 15.89 | 12.6  | 12.95 | 13    | 12.98 | 16.07 | 12.67       |
| 100 | 10 K | 12.92 | 14.05 | 15.99 | 12.67 | 12.82 | 12.92 | 12.9  | 16.06 | 12.35       |
| 200 | 2 K  | 12.8  | 13.64 | 15.63 | 12.85 | 13.06 | 13.16 | 13.21 | 16.17 | NA          |
| 200 | 5 K  | 12.94 | 13.37 | 15.89 | 12.55 | 12.96 | 13.06 | 12.93 | 16.07 | NA          |
| 200 | 10 K | 13.17 | 13.36 | 15.75 | 12.59 | 12.84 | 13.08 | 13    | 16.06 | NA          |

Table 31: Reward-variance=1, Eps=Big, Game=PongNoFrameskip-v4

|   M | N            | FLE            | QRD            | IQN             | KLD           | ENE            | PDF            | RBF            | TVD            | HYV            |
|-----|--------------|----------------|----------------|-----------------|---------------|----------------|----------------|----------------|----------------|----------------|
|  10 | 2 K 5 K 10 K | 3.26 3.98 4.25 | 4.99 5.17 5.24 | 3.71 5.95 12.39 | 3.08 3.9 4.03 | 3.07 3.72 3.9  | 2.8 3.7 3.83   | 2.86 3.66 3.82 | 5.94 5.92 5.97 | 3.15 3.82 4.01 |
| 100 | 2 K 5 K 10 K | 3.38 4.9 4.17  | 3.44 4.1 4.16  | 4.14 3.79 5.52  | 3.1 3.87 3.9  | 3.01 3.73 3.88 | 2.31 3.14 3.39 | 2.44 3.12 3.23 | 5.95 5.92 5.98 | 3.27 3.68 3.84 |
| 200 | 2 K 5 K 10 K | 3.42 3.91 4.29 | 2.91 3.67 3.68 | 1.73 4.7 5.33   | 3.06 3.64 4.0 | 2.98 3.74 3.85 | 2.36 3.1 3.34  | 2.37 3.19 3.36 | 5.94 5.92 5.98 | NA NA NA       |

Table 32: Reward-variance=1, Eps=Big, Game=QbertNoFrameskip-v4

|   M | N            | FLE               | QRD               | IQN               | KLD               | ENE               | PDF               | RBF               | TVD               | HYV               |
|-----|--------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
|  10 | 2 K 5 K 10 K | 20.91 20.15 19.81 | 28.59 28.36 28.37 | 29.96 29.58 29.8  | 19.67 18.3 19.19  | 21.92 20.7 20.87  | 22.96 21.67 21.84 | 23.01 21.93 21.81 | 30.16 30.11 30.24 | 20.76 19.52 18.97 |
| 100 | 2 K 5 K 10 K | 20.77 20.26 19.68 | 22.46 21.62 21.31 | 29.32 29.26 28.75 | 19.01 19.7 19.83  | 21.59 20.14 20.11 | 22.53 21.3 20.8   | 22.69 21.35 20.65 | 30.17 30.11 >100  | 20.81 19.38 18.92 |
| 200 | 2 K 5 K 10 K | 21.34 19.63 19.67 | 20.79 20.54 20.24 | 29.01 29.7 27.25  | 20.68 19.52 19.54 | 21.54 20.04 20.26 | 22.19 20.73 20.19 | 22.05 21.01 20.52 | 30.16 30.11 30.23 | NA NA NA          |

Table 33: Reward-variance=1, Eps=Big, Game=SpaceInvadersNoFrameskip-v4

|   M | N            | FLE            | QRD               | IQN               | KLD            | ENE            | PDF           | RBF            | TVD               | HYV            |
|-----|--------------|----------------|-------------------|-------------------|----------------|----------------|---------------|----------------|-------------------|----------------|
|  10 | 2 K 5 K 10 K | 6.15 6.76 5.1  | 16.76 16.75 16.81 | 11.54 9.79 16.36  | 5.91 5.61 5.81 | 7.54 7.38 6.92 | 9.86 9.4 9.34 | 9.78 9.62 9.21 | 18.89 18.91 19.06 | 5.33 5.02 5.41 |
| 100 | 2 K 5 K 10 K | 6.35 6.55 6.14 | 7.91 7.94 7.67    | 12.26 16.3 12.07  | 5.56 5.57 4.86 | 6.25 6.61 6.07 | 8.23 7.59 7.3 | 7.71 7.86 7.18 | 18.88 18.9 19.05  | 5.98 5.14 5.41 |
| 200 | 2 K 5 K 10 K | 6.21 6.2 5.83  | 6.8 6.87 6.43     | 10.59 13.29 10.75 | 5.4 5.31 4.86  | 6.95 6.38 6.33 | 7.87 7.1 6.95 | 7.43 7.14 6.78 | 18.88 18.91 19.05 | NA NA NA       |

## E.3 Computation resources

Our simulations are extensive. We used high performance computing system. For LQR simulation (Appendix E.1), we have used CPU, 2GB memory. A single run of a single method under a fixed setting takes approximately 20 seconds. For Atari games' simulation (Appendix E.2), we have used GPU, 10GB memory. A single run of a single method under a fixed setting takes approximately 700 seconds.