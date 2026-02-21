## Isotropic Noise in Stochastic and Quantum Convex Optimization

Annie Marsden ∗

Liam O'Carroll †

Aaron Sidford †

Chenyi Zhang †

## Abstract

We consider the problem of minimizing a d -dimensional Lipschitz convex function using a stochastic gradient oracle. We introduce and motivate a setting where the noise of the stochastic gradient is isotropic in that it is bounded in every direction with high probability. We then develop an algorithm for this setting which improves upon prior results by a factor of d in certain regimes, and as a corollary, achieves a new state-of-the-art complexity for sub-exponential noise. We give matching lower bounds (up to polylogarithmic factors) for both results. Additionally, we develop an efficient quantum isotropifier , a quantum algorithm which converts a variancebounded quantum sampling oracle into one that outputs an unbiased estimate with isotropic error. Combining our results, we obtain improved dimension-dependent rates for quantum stochastic convex optimization.

## 1 Introduction

Stochastic convex optimization (SCO) [51, 47, 48, 9] is a foundational problem in optimization and learning theory, with numerous applications in theoretical computer science [16], operations research [50], machine learning [11], and beyond. Algorithms for SCO, most notably stochastic gradient descent (SGD) and its many variants, are extensively studied and widely deployed in practice. In addition, there has been increasing interest recently in its quantum analog, quantum stochastic convex optimization , and new provably efficient quantum algorithms with have been developed [53, 67].

In one of its simplest forms which we focus on in this paper, SCO asks to compute an glyph[epsilon1] -optimal point of a differentiable 3 convex L -Lipschitz function f : R d → R , minimized at an unknown point x glyph[star] with ‖ x glyph[star] ‖ ≤ R for the Euclidean norm ‖·‖ , given access to a bounded stochastic gradient oracle, which we define formally in Definition 1 and Definition 2 below. Throughout, we assume that the randomness in calls to any oracle are independent of previous calls.

Definition 1 (SGO) . O ( · ) is a stochastic gradient oracle (SGO) for differentiable f : R d → R if when queried at x ∈ R d , it outputs O ( x ) ∈ R d such that E O ( x ) = ∇ f ( x ) .

Definition 2 (BSGO) . O B ( · ) is a σ B -bounded SGO ( σ B -BSGO) for differentiable f : R d → R if it is an SGO for f such that E ‖O B ( x ) ‖ 2 ≤ σ 2 B for any query x ∈ R d .

It is well known that SGD, in particular, iterating x t +1 ← x t -η O B ( x t ) with an appropriate choice of step size η , solves SCO with O ( R 2 σ 2 B /glyph[epsilon1] 2 ) queries, which is optimal even when d = 1 [2]. In this paper, we seek to bypass this fundamental limit by studying more fine-grained models of the stochastic gradient. In particular, we study the following question:

∗ Google Deepmind, anniemarsden@google.com

†

3

Stanford University, {ocarroll,sidford,chenyiz}@stanford.edu Preprint available at arXiv:2510.20745.

We assume all objective functions are differentiable. Following a similar argument in prior works (e.g., [17, 53]), our results extend to non-differentiable settings because convex functions are almost everywhere differentiable, and our algorithms are robust to polynomially small numerical errors.

Can we identify shape-dependent assumptions on the noise of the stochastic gradient which allow us to obtain new state-of-the-art algorithmic guarantees?

One of the most natural shape-dependent assumptions is to assume a bound on the variance of the SGO, which we capture in the following definition:

Definition 3 (VSGO) . O V ( · ) is a σ V -variance-bounded SGO ( σ V -VSGO) for differentiable f : R d → R if it is an SGO for f such that E ‖O V ( x ) -∇ f ( x ) ‖ 2 ≤ σ 2 V for any query x ∈ R d .

In the setting we consider where f is L -Lipschitz (see Problem 1 for a formal definition), SGD trivially achieves a rate of O ( R 2 ( L 2 + σ 2 V ) /glyph[epsilon1] 2 ) = O ( R 2 σ 2 V /glyph[epsilon1] 2 + R 2 L 2 /glyph[epsilon1] 2 ) under a σ V -VSGO due to the observation that a σ V -VSGO is a O ( L + σ V ) -BSGO. Inspecting the above rate, the R 2 σ 2 V /glyph[epsilon1] 2 term is unimprovable even when d = 1 [2]. 4 However, the same is not true for the R 2 L 2 /glyph[epsilon1] 2 term, which stems from the non-stochastic component of the problem. Indeed, when σ V = 0 , in which case the SGO is a gradient oracle, the O ( R 2 L 2 /glyph[epsilon1] 2 ) rate (which is achieved by gradient descent [49]) is optimal only for glyph[epsilon1] ≥ Ω( RL/ √ d ) [24]; cutting plane methods which achieve ˜ O ( d ) rates are better at smaller target precisions, where we use ˜ O ( · ) throughout the paper to hide polylogarithmic factors in 1 /glyph[epsilon1] , log(1 /δ ) , d , R , and L .

This leaves the door open for improved dimension-dependent rates for SCO under a VSGO, which have not been explicitly studied to the best of our knowledge. In particular, in light of the fact that a O ( R 2 σ 2 V /glyph[epsilon1] 2 + R 2 L 2 /glyph[epsilon1] 2 ) rate is achievable by SGD and it is possible to achieve min { O ( R 2 L 2 /glyph[epsilon1] 2 ) , ˜ O ( d ) } when σ V = 0 , this begs the natural open problem:

Open Problem 1. Is it possible to solve SCO with ˜ O ( R 2 σ 2 V /glyph[epsilon1] 2 + d ) queries to a σ V -VSGO?

While we do not solve this open problem, we nonetheless make substantial progress in characterizing the complexity of SCO under more fine-grained shape-dependent assumptions on the SGO. Our main conceptual contribution is to answer this open problem in the affirmative under a stronger noise model which we introduce in the next section, termed isotropic noise . We provide a dimension-dependent algorithm in this setting which achieves the conjectured ˜ O ( R 2 σ 2 /glyph[epsilon1] 2 + d ) rate (albeit σ 2 is not the variance but a different parameter associated with our noise model). We then instantiate this result to achieve the target ˜ O ( R 2 σ 2 /glyph[epsilon1] 2 + d ) rate for sub-exponential noise, a widely studied class of distributions in machine learning and statistics, as well as a ˜ O ( dR 2 σ 2 V /glyph[epsilon1] 2 + d ) rate for SCO under a VSGO, 5 which is worse than the conjectured rate by a d -factor. We then give lower bounds for isotropic noise and sub-exponential noise which show that in any parameter regime, applying the better of our algorithm and SGD is optimal (up to polylogarithmic factors).

Finally, we leverage our algorithm to obtain a new state-of-the-art guarantee for quantum SCO under a quantum analog of the VSGO oracle. In what may be of independent interest, our quantum result relies on a quantum subroutine which we call a quantum isotropifier , which converts a quantum analog of the VSGO oracle into one which outputs an unbiased estimate of the gradient with isotropic noise. In other words, this subroutine allows us to apply our improved guarantees for isotropic stochastic gradients to an even broader class of noise models in the quantum setting.

In the remainder of the introduction, we define our noise model and present our results in more detail (Section 1.1), discuss additional related work (Section 1.2), define notation (Section 1.3), and give a roadmap of the rest of the paper (Section 1.4).

## 1.1 Results

Isotropic and sub-exponential SGOs. As our main conceptual contribution, we define a new noise model for SCO which captures the impact of the shape of the noise, as opposed to only its moments. In particular, we consider an isotropic noise model , formalized in the following definition, where the magnitude of the noise in any fixed direction is bounded with probability at least 1 -δ .

4 This follows from the same lower bound construction which shows that Ω( R 2 σ 2 B /glyph[epsilon1] 2 ) queries are necessary for SCO under a σ B -BSGO. Indeed, in that construction (e.g., Section 5 in [23]), it is the case that σ 2 V = Θ( σ 2 B ) .

5 We emphasize that although they study a different setting, this rate can be recovered by modifying the techniques of [53]; see Sections 1.1 and 2 for further discussion.

Definition 4 (ISGO) . O I ( · ) is a ( σ I , δ ) -isotropic SGO (( σ I , δ )-ISGO) for differentiable f : R d → R if it is an SGO for f such that

<!-- formula-not-decoded -->

The 1 / √ d scaling factor in Eq. 1 is included to ensure σ I in Definition 4 and σ V in Definition 3 are comparable in scale. For example, a ( σ, 0) -ISGO O I ( · ) is also a σ -VSGO, since with e i as the i -th standard basis vector:

<!-- formula-not-decoded -->

Informally, an ISGO can be thought of as a VSGO with the stronger assumption that the noise is bounded evenly in all directions, thereby imposing additional structure on its shape. Formally, we show later in Lemma 9 that a (6 σ V √ d, δ ) -ISGO query can be implemented with logarithmically many queries to a σ V -VSGO, thereby allowing us to translate algorithmic guarantees for an ISGO to a VSGO at the cost of a √ d -factor.

ISGOs also naturally capture a variety of light-tailed distributions, for which we use sub-exponential distributions as a representative example throughout this paper due to their widespread prevalence in machine learning and statistics. In particular, we consider the following ESGO:

Definition 5 (ESGO) . O E ( · ) is a σ E -sub-exponential SGO ( σ E -ESGO) for differentiable f : R d → R if it is an SGO for f such that for any x, u ∈ R d such that ‖ u ‖ = 1 :

√

<!-- formula-not-decoded -->

The √ d factor in Definition 5 is present for the same reason it is present Definition 4, and we show for all δ ∈ (0 , 1) that any σ -ESGO is a ( σ log(2 /δ ) , δ ) -ISGO (see Lemma 8). We collect and review basic facts about sub-exponential distributions, as well as provide some additional discussion about the relationships between the various SGO oracles we define, in Appendix D. In particular, we show that a σ E -ESGO is a Cσ E -VSGO for some absolute constant C , and thus Definition 5 can also be interpreted as imposing structure on the shape of the noise beyond a variance bound.

The complexity of SCO with ISGOs, ESGOs, and VSGOs. We formalize the general setting we consider in Problem 1 and then present our results, including upper and lower bounds.

Problem 1 (SCO) . Given access to an SGO O ( · ) for a convex and L -Lipschitz (with respect to the glyph[lscript] 2 -norm) function f : R d → R such that there exists x glyph[star] ∈ argmin x ∈ R d f ( x ) with ‖ x glyph[star] ‖ ≤ R , the goal is to output an glyph[epsilon1] -optimal point.

In Theorem 1 and Corollary 2 (proven in Section A and Section B respectively), we achieve upper and lower bounds for Problem 1 given access to a ( σ I , δ ) -ISGO. We obtain the upper bound Theorem 1 via a stochastic cutting plane method by extending the techniques of [53], and the lower bound Corollary 2 by reducing a mean estimation problem to Problem 1; see Section 2 for a technical overview. In Corollary 2, we use ˜ Ω( · ) to hide logarithmic factors in d .

Theorem 1. Problem 1 can be solved with probability at least 2 / 3 in ˜ O ( R 2 σ 2 I /glyph[epsilon1] 2 + d ) queries to a ( σ I , δ ) -ISGO for any δ ≤ 1 Md ( R 2 σ 2 I /glyph[epsilon1] 2 + d ) , where M = ˜ O (1) .

Corollary 2. Any algorithm which solves Problem 1 with probability at least 2 / 3 using a ( σ I , δ ) -ISGO makes at least ˜ Ω( R 2 σ 2 I / ( glyph[epsilon1] 2 log 2 (1 /δ )) + min { R 2 L 2 /glyph[epsilon1] 2 , d } ) queries.

In other words, Theorem 1 says that an ˜ O ( R 2 σ 2 I /glyph[epsilon1] 2 + d ) rate is achievable as long as δ is inversepolynomially small, and Corollary 2 says this is tight up to polylog factors when the target precision is sufficiently small so that d ≤ R 2 L 2 /glyph[epsilon1] 2 or equivalently glyph[epsilon1] ≤ RL/ √ d . In particular, Theorem 1 demonstrates that shape-dependent assumptions on the noise can lead to speedups over the known rates for SGD. For example, for a ( σ, 0) -ISGO (which we saw above is also a σ -VSGO), the ˜ O ( R 2 σ 2 /glyph[epsilon1] 2 + d ) rate of Theorem 1 improves upon the O ( R 2 σ 2 /glyph[epsilon1] 2 + R 2 L 2 /glyph[epsilon1] 2 ) rate of SGD when L glyph[greatermuch] max { σ, glyph[epsilon1] √ d/R } , which captures a variety of high precision, low variance regimes.

Next, using the fact discussed above that ISGO queries can be implemented via ESGO and VSGO oracles, we obtain the following upper bounds (proven in Section A) as corollaries of Theorem 1:

Corollary 3. Problem 1 can be solved with probability at least 2 / 3 in ˜ O ( R 2 σ 2 E /glyph[epsilon1] 2 + d ) queries to a σ E -ESGO.

Corollary 4. Problem 1 can be solved with probability at least 2 / 3 in ˜ O ( dR 2 σ 2 V /glyph[epsilon1] 2 + d ) queries to a σ V -VSGO.

Thus, sub-exponential noise allows us to match the rate conjectured in Open Problem 1, whereas using a VSGO results in an additional d factor in the variance-dependent term. Nevertheless, Corollary 4 still outperforms the O ( R 2 σ 2 V /glyph[epsilon1] 2 + R 2 L 2 /glyph[epsilon1] 2 ) rate achieved by SGD for solving Problem 1 given a σ V -VSGO when L glyph[greatermuch] √ d · max { σ V , glyph[epsilon1]/R } . We emphasize that while they study a different setting, a modification of the techniques of [53] can achieve the same complexity as Corollary 4; see the technical overview in Section 2 for further discussion. However, we believe the fact that we are able to recover Corollary 4 as a simple consequence using our more general ( σ I , δ ) -ISGO primitive illustrates its potential broader applicability.

Finally, we prove in Section B that Corollary 3 is tight up to polylog factors when d ≥ R 2 L 2 /glyph[epsilon1] 2 :

Theorem 5. Any algorithm which solves Problem 1 with probability at least 2 / 3 using a σ E -ESGO makes at least ˜ Ω( R 2 σ 2 E /glyph[epsilon1] 2 +min { R 2 L 2 /glyph[epsilon1] 2 , d } ) queries.

This effectively solves Open Problem 1 for the special case of sub-exponential noise in light of the fact that SGD matches our lower bound in the other regime where d ≤ R 2 L 2 /glyph[epsilon1] 2 .

Quantum SCO under variance-bounded noise. As an additional application of Theorem 1, we consider stochastic convex optimization (SCO) with access to a quantum analog of a VSGO, which we refer to as a QVSGO. The QVSGO is a direct and natural extension of a VSGO, and slightly generalizes the notion of QSGO introduced in [53, Definition 3].

Definition 6 (QVSGO) . For f : R d → R and σ V &gt; 0 , its quantum bounded stochastic gradient oracle ( σ V -QVSGO) is defined as 6 7

<!-- formula-not-decoded -->

where p f,x ( · ) is the probability density function of the stochastic gradient that satisfies

<!-- formula-not-decoded -->

Given access to a QVSGO, we develop a quantum algorithm that achieves improved query complexity for the following quantum SCO problem. This problem slightly generalizes [53, Problem 1] by introducing an additional parameter σ V alongside the Lipschitzness L .

Problem 2 (Quantum SCO) . Given query access to a σ V -QVSGO O QV for a convex and L -Lipschitz (with respect to the glyph[lscript] 2 -norm) function f : R d → R such that there exists x glyph[star] ∈ argmin x ∈ R d f ( x ) with ‖ x glyph[star] ‖ ≤ R , the goal is to output an glyph[epsilon1] -optimal point.

To solve Problem 2, we develop an unbiased quantum multivariate mean estimation algorithm whose error is isotropic, which we refer to as quantum isotropifier . We use this algorithm to prepare an ISGO using a QVSGO:

Theorem 6. For any differentiable f : R d → R , a ( σ I , δ ) -ISGO of f can be implemented using ˜ O ( σ V √ d log 7 (1 /δ ) /σ I ) queries to a σ V -QVSGO.

Combining Theorem 1 and Theorem 6, we obtain the following improved query complexity for quantum SCO (proven in Section C):

Theorem 7. With success probability at least 2 / 3 , Problem 2 can be solved using ˜ O ( dRσ V /glyph[epsilon1] ) queries to a σ V -QVSGO.

6 Considering such quantum extensions of classical oracles is standard in the literature, see, e.g., [20, 65, 53]. Moreover, theoretically there are well-established techniques for implementing these quantum analogs of classical oracles. Specifically, if a classical oracle can be implemented via a classical circuit, its corresponding quantum oracle can be implemented using a quantum circuit of the same size.

7 A description of the quantum notation we use can be found in Section C.

To compare, the prior state-of-the-art for solving Problem 2 includes two quantum algorithms of [53] which achieve query complexities of ˜ O ( d 3 / 2 ( L + σ V ) R/glyph[epsilon1] ) and ˜ O ( d 5 / 8 (( L + σ V ) R/glyph[epsilon1] ) 3 / 2 ) . With minor modifications, the query complexity of the first algorithm can be reduced to ˜ O ( d 3 / 2 σ V R/glyph[epsilon1] ) . However, it remains unclear whether a similar reduction in query complexity can be achieved for the second algorithm. In comparison, our algorithm achieves a polynomial improvement in d .

## 1.2 Additional related work

Stochastic convex optimization. SCO is a foundational problem in optimization theory and has been extensively studied for decades. In particular, SCO with a VSGO has been studied under different assumptions on the objective function f than the L -Lipschitz assumption we focus on in this paper. Notably, if the gradient of f is β -Lipschitz, the seminal paper [39] achieved an optimal error of O ( β/T 2 + σ V / √ T ) after T queries. We note that our guarantees for Problem 1 can be extended to the setting where the gradient of f is β -Lipschitz at the cost of only polylogarithmic factors since our bounds have polylogarithmic dependence on the Lipschitz constant of f . This follows because a function with a β -Lipschitz gradient which is minimized in a ball of radius R is itself O ( βR ) -Lipschitz in the ball.

Beyond the bounded variance assumption, [62] examines SGOs with heavy-tailed or infinite variance noise and shows that stochastic mirror descent is optimal in such cases. As for cutting plane methods, there is a long line of work in deterministic settings which has sought to improve the query complexity and/or runtime; see e.g. [56, 40, 34].

Quantum mean estimation. Quantum mean estimation and the closely closely related problems of amplitude estimation and phase estimation have been extensively studied [1, 54, 32, 64, 14, 46, 31]. The seminal amplitude estimation algorithm [15] can be used to obtain a quadratic improvement on the query complexity for estimating the mean of any Bernoulli random variable. Building upon this result, [30] (whose query complexity is further improved by [38]) introduced a quantum sub-Gaussian mean estimator that achieves a quadratically better query complexity than the classical counterpart while providing a mean estimate with sub-Gaussian error, without requiring additional assumptions on the variance or tail behavior of the underlying distribution. Multilevel Monte Carlo methods have also been widely used in quantum algorithms [3, 42].

Quantum algorithms and lower bounds for optimization problems. There has been a rich study of quantum algorithms for classical optimization problems, including semidefinite programs [7, 13, 12, 37, 58, 60], convex optimization [6, 59, 19, 41], and non-convex optimization [65, 21, 28, 43, 41]. Despite these recent progress in quantum algorithms using quantum evaluation oracles and quantum gradient oracles, their limitations have also been explored in a series of works establishing quantum lower bounds for certain settings of convex optimization [25, 66] and non-convex optimization [66].

## 1.3 Notation

‖·‖ is the glyph[lscript] 2 -norm. We define B p ( r, x 0 ) := { x ∈ R d : ‖ x -x 0 ‖ p ≤ r } to be the glyph[lscript] p -ball of radius r centered at x 0 , and use B p ( r ) := B p ( r, 0) and B p := B p (1 , 0) . A halfspace is denoted by H ≥ ( a ∈ R d , b ∈ R ) := { x ∈ R d : a glyph[latticetop] x ≥ b } . For f : R d → R , we let f glyph[star] := inf x f ( x ) and call x ∈ R d glyph[epsilon1] -(sub)optimal if f ( x ) ≤ f glyph[star] + glyph[epsilon1] . A function f : R d → R is L -Lipschitz if f ( x ) -f ( y ) ≤ L ‖ x -y ‖ for all x, y ∈ R d . For two matrices A,B ∈ R d × d , we write A glyph[precedesequal] B if x glyph[latticetop] Ax ≤ x glyph[latticetop] Bx for all x ∈ R d . The standard basis in R d is denoted { e 1 , . . . , e d } . For random variables X,Y , we use H ( X ) := -∑ x p ( x ) log p ( x ) to denote the entropy of X , use H ( Y | X ) := H ( X,Y ) -H ( X ) to denote the conditional entropy of Y with respect to X , and use I ( X ; Y ) := H ( X )+ H ( Y ) -H ( X,Y ) to denote the mutual information between X and Y . The cardinality of a finite set S is denoted | S | . We use ˜ O ( · ) and ˜ Ω( · ) to denote bigO notation omitting polylogarithmic factors.

## 1.4 Paper organization

We give a technical overview of the paper in Section 2 and conclude in Section 3. We prove our upper bounds for Problem 1 under ISGO, ESGO, and VSGO oracles (resp. Theorem 1 and Corollaries 3 and 4) in Appendix A. We establish our lower bounds for Problem 1 given either an ISGO or ESGO

oracle in Appendix B. In Appendix C, we present our quantum isotropifier and then apply it to obtain an improved bound for quantum SCO under variance-bounded noise. In Appendix D, we review sub-exponential distributions and further discuss the relationships between the various stochastic gradient oracles we study. Finally, we collect miscellaneous technical lemmas in Appendix E.

## 2 Technical overview

In this section, we present high-level overviews of our techniques. In Section 2.1, we give an overview of our stochastic cutting plane method , based on an extension of the techniques of [53], which we use to obtain our upper bounds for Problem 1. In Section 2.2, we give a technical overview of our lower bounds for Problem 1. We conclude in Section 2.3 with a discussion of our application to quantum SCO; we believe our quantum isotropifier subroutine discussed in that section may be of independent interest. The formal proofs of the results discussed in Sections 2.1, 2.2, and 2.3 can be found in Sections A, B, and C respectively.

## 2.1 Overview of our upper bounds for SCO

Recall that our main non-quantum algorithmic result is Theorem 1, which says that ˜ O ( R 2 σ 2 I /glyph[epsilon1] 2 + d ) queries to a ( σ I , δ ) -ISGO suffice to solve Problem 1 when δ is inverse-polynomially small. We prove this upper bound using a stochastic cutting plane method via an extension of the techniques of [53] (see Section A for formal proofs). Cutting plane methods are a foundational class of algorithms in convex optimization which solve feasibility problems, for which we use the following formulation: (recall H ≥ ( a ∈ R d , b ∈ R ) denotes the halfspace { x ∈ R d : a glyph[latticetop] x ≥ b } ):

Definition 7 (Feasibility Problem) . For R ′ ≥ r &gt; 0 , we define the ( R ′ , r ) -feasibility problem as follows: We are given query access to a (potentially randomized) halfspace oracle which when queried at x ∈ B 2 ( R ′ ) , outputs a vector g x ∈ R d \{ 0 } . The goal is to query the oracle at a sequence of points x 1 , . . . , x T ∈ B 2 ( R ′ ) such that B 2 ( R ′ ) ⋂ t ∈ [ T ] H ≥ ( g x t , g glyph[latticetop] x t x t ) does not contain a ball of radius r (namely, any set of the form B 2 ( r, z ) for z ∈ R d ). We call an algorithm a T -algorithm for the ( R ′ , r ) -feasibility problem if it achieves this goal with at most T queries.

Avariety of cutting plane methods [56, 40, 34] are ˜ O ( d ) -algorithms for the ( R ′ , r ) -feasibility problem, where ˜ O ( · ) hides logarithmic factors in d , R ′ , and 1 /r . Furthermore, it is well known that solving the feasibility problem where the halfspace oracle is given by the negative gradient of f suffices to solve Problem 1 using ˜ O ( d ) queries to an exact gradient oracle, where ˜ O ( · ) hides logarithmic factors in R , d , L , and 1 /glyph[epsilon1] . (Technically, there is also an additional post-processing step needed which we discuss later.) This reduction follows from the fact that by the Lipschitzness of f , every point in the ball B 2 ( glyph[epsilon1]/L,x glyph[star] ) := { z ∈ R d : ‖ z -x glyph[star] ‖ ≤ glyph[epsilon1]/L } is glyph[epsilon1] -optimal. Therefore, setting r := glyph[epsilon1]/L and R ′ := 2 R so that B 2 ( glyph[epsilon1]/L,x glyph[star] ) ⊆ B 2 ( R ′ ) (we can assume r ≤ R without loss of generality since if glyph[epsilon1] &gt; RL , the origin is glyph[epsilon1] -optimal) and running one of the aforementioned ˜ O ( d ) -algorithms for the ( R ′ , r ) -feasibility problem with the halfspace oracle given by -∇ f ( · ) , we must have queried the oracle at an iterate x t such that -∇ f ( x t ) glyph[latticetop] z &lt; -∇ f ( x t ) glyph[latticetop] x t for some z ∈ B 2 ( glyph[epsilon1]/L,x glyph[star] ) . This implies x t is glyph[epsilon1] -optimal by convexity:

<!-- formula-not-decoded -->

Recently, [53], which broadly studies quantum algorithms for stochastic optimization in a variety of settings, observed the above analysis can be extended to the setting where we only have access to an approximate gradient, which they capture as follows (paraphrased from Definition 5 in [53]):

Definition 8 (AGO) . ˜ h ( · ) is a γ -approximate gradient oracle ( γ -AGO) if when queried at x ∈ R d , the (potentially random) output ˜ h ( x ) ∈ R d satisfies ‖ ˜ h ( x ) -∇ f ( x ) ‖ ≤ γ .

In other words, a γ -AGO ˜ h ( · ) gives an estimate of the gradient with at most γ error in glyph[lscript] 2 -norm. Then running an ˜ O ( d ) -algorithm for the ( R ′ := 2 R,r := glyph[epsilon1]/L ) -feasibility problem as before, except with -˜ h ( · ) as the halfspace oracle as opposed to -∇ f ( x ) , implies there exists an iterate x t and glyph[epsilon1] -optimal z ∈ B 2 ( glyph[epsilon1]/L,x glyph[star] ) such that -h glyph[latticetop] t z &lt; -h glyph[latticetop] t x t , where h t was the result of calling ˜ h ( · ) with input x t at

the t -th step. Thus, by convexity:

<!-- formula-not-decoded -->

where the last inequality follows because z is glyph[epsilon1] -optimal; 1 &lt; 0 by assumption; and 2 ≤ ‖∇ f ( x t ) -h t ‖‖ x t -z ‖ ≤ 4 Rγ by the Cauchy-Schwarz inequality, Definition 8, and the fact that x t , z ∈ B 2 ( R ′ ) = B 2 (2 R ) . In particular, if γ ≤ glyph[epsilon1]/ (4 R ) , then x t is 2 glyph[epsilon1] -optimal.

[53] used this observation to obtain then state-of-the-art quantum algorithms for Problem 1 using a quantum analog of the BSGO oracle (see Definition 2). In particular, they implement an glyph[epsilon1]/ (4 R ) -AGO at each step with high probability using a quantum variance reduction procedure. While it is not explicitly stated, this analysis can be extended to the non-quantum setting by mini-batching. One can show that an glyph[epsilon1]/ (4 R ) -AGO query can be implemented using ˜ O ( R 2 σ 2 V /glyph[epsilon1] 2 + 1) queries to a σ V -VSGO O V ( · ) with success probability at least 1 -ξ , where ˜ O ( · ) hides logarithmic factors in 1 /ξ . Then choosing ξ appropriately to union bound the failure probabilities over all iterations, an ˜ O ( d ) -algorithm for the feasibility problem with the halfspace oracle given by this mini-batching procedure yields an ˜ O ( dRσ 2 V /glyph[epsilon1] 2 + d ) rate for solving Problem 1 given a σ V -VSGO, matching the rate we recover in Corollary 4 as a result of our more general framework.

Our improvements over [53] are built on the key observation that it in fact suffices to only weakly control the error of the approximate gradient in glyph[lscript] 2 -norm (it can be polynomially large), as long as we tightly control the error of the approximate gradient in a particular fixed direction-namely, the direction to the optimum x glyph[star] . To start, we refine Definition 8 as follows by defining a marginal approximate gradient oracle :

Definition 9 (MAGO) . ˜ g ( · , · ) is a marginal ( η ≥ 0 , Γ ≥ 0) -approximate gradient oracle ( ( η, Γ) -MAGO ) if when queried at x, u ∈ R d , the (potentially random) output ˜ g ( x, u ) ∈ R d satisfies

<!-- formula-not-decoded -->

Next, consider running a cutting plane method for the ( R ′ := 2 R,r := min { glyph[epsilon1]/L, glyph[epsilon1]/ Γ } ) -feasibility problem, where the halfspace oracle at a query point x ∈ R d is given by -˜ g ( x, x -x glyph[star] ) . As before, there exists an iterate x t and glyph[epsilon1] -optimal z ∈ B 2 ( r, x glyph[star] ) such that -g glyph[latticetop] t z &lt; -g glyph[latticetop] t x t , where g t was the result of calling ˜ g ( x t , x t -x glyph[star] ) at the t -th step. Then by convexity:

<!-- formula-not-decoded -->

where the last inequality followed because z is glyph[epsilon1] -optimal; 3 &lt; 0 by assumption; by Definition 9 and a triangle inequality 4 ≤ η ‖ x t -x glyph[star] ‖ ≤ 4 Rη ; and finally 5 ≤ ‖∇ f ( x t ) -g t ‖‖ x glyph[star] -z ‖ ≤ Γ r ≤ glyph[epsilon1] by Cauchy-Schwarz, Definition 9, and the choice of r . Thus, if η ≤ glyph[epsilon1]/ (4 R ) , then x t is 3 glyph[epsilon1] -optimal. Crucially, because cutting plane methods for the feasibility problem have logarithmic dependence on 1 /r , a polynomial bound on Γ suffices to maintain an ˜ O ( d ) -query complexity for this problem.

Next, we show how to implement a MAGO using a ( σ I , δ ) -ISGO in the following lemma, proven in Section A:

Lemma 1. For any δ, ξ ∈ (0 , 1) ; x, u ∈ R d ; and η &gt; 0 , a query ˜ g ( x, u ) to an ( η, Γ := η √ d ) -MAGO can be implemented using K = O ( σ 2 I dη 2 log(2 d/ξ ) + 1) queries to a ( σ I , δ ) -ISGO O I ( · ) , with success probability at least 1 -ξ -δdK and without access to the input u .

With η ← glyph[epsilon1]/ (4 R ) , Lemma 1 implies that ˜ O ( R 2 σ 2 I dglyph[epsilon1] 2 + 1) queries are enough to implement an ( glyph[epsilon1]/ (4 R ) , glyph[epsilon1] √ d/ (4 R )) -MAGO, which suffices to ensure x t is 3 glyph[epsilon1] -optimal by the above analysis while also maintaining a polynomial bound on Γ . Critically, note that implementing each MAGO query does not require knowledge of the second argument u , which is important since the second argument to the MAGO depends on x glyph[star] in the above analysis. We also emphasize that the term 5 above cannot

be bounded by attempting to control the error of the MAGO in the direction x glyph[star] -z ; this is because z is not fixed in advance and indeed depends on g t .

Thus, leaving details regarding handling δ to Section A, combining Lemma 1 with an ˜ O ( d ) -algorithm for the feasibility problem suffices to ensure an iterate x t is 3 glyph[epsilon1] -optimal using ˜ O ( R 2 σ 2 I /glyph[epsilon1] 2 + d ) -queries to an ISGO. However, it is not a priori clear which of the ˜ O ( d ) iterates returned by this algorithm is 3 glyph[epsilon1] -optimal. [53] solve an analogous problem in their setting via a post-processing procedure which iteratively refines the output of the first stage via binary search to ultimately return an O ( glyph[epsilon1] ) -optimal point (see also [33, 18, 5] for related procedures). In Section A, we carefully adapt this procedure to a ( σ I , δ ) -ISGO, showing it can also be implemented with ˜ O ( R 2 σ 2 I /glyph[epsilon1] 2 + d ) queries and thereby achieving a d -factor savings over using a VSGO as in the first stage described above. Finally, at the end of Section A we instantiate Theorem 1 to achieve a new state-of-the-art ˜ O ( R 2 σ 2 E /glyph[epsilon1] 2 + d ) -complexity for Problem 1 given a σ E -ESGO (see Corollary 3), and also recover the aforementioned ˜ O ( dR 2 σ 2 V /glyph[epsilon1] 2 + d ) rate given a σ V -VSGO as a simple consequence (see Corollary 4).

## 2.2 Overview of our lower bounds for SCO

The starting point of our lower bounds for SCO with ISGOs and ESGOs is the connection between SCO and multivariate mean estimation problems, which has been widely used to derive lower bounds for SCO in various settings (see e.g., [23, 53]). Specifically, as detailed in Section B.2, for any random variable X , we consider an instance of Problem 1 where f is defined as the expectation of a collection of linear functions with an added regularizer term. The set of linear functions is parameterized by samples from X , which also determines the stochastic gradient, ensuring that it follows a noise model of the same form as X . Moreover, finding an glyph[epsilon1] -optimal point of f yields a ˜ O ( glyph[epsilon1]/R ) -estimate of E [ X ] , thus establishing a reduction from the mean estimation problem to SCO.

Building on this reduction, we establish a lower bound for SCO with ESGOs by introducing a variant of the mean estimation problem where the noise follows a sub-exponential distribution. Specifically, we construct a hard instance in which the random variable X is parameterized by another random variable V , as detailed in Section B.1. We show that any algorithm approximating E [ X ] must retain significant mutual information with V . We then upper bound this mutual information in terms of the number of samples, yielding the desired lower bound for the mean estimation problem (see Lemma 11) and, consequently, for SCO with ESGOs (see Theorem 5). As a corollary, we derive a lower bound for SCO with ISGOs (see Corollary 2). Regarding the mean estimation problem itself (namely, Problem 3), we note that similar lower bounds to Lemma 11 may be derived as corollaries of recent high-probability minimax lower bounds [45], but we provide our own analysis for completeness.

## 2.3 Quantum isotropifier and quantum SCO under variance-bounded noise

Our main technical ingredient of our improved bound for quantum SCO is an unbiased quantum multivariate mean estimation algorithm whose error is isotropic in the sense that it is small in every direction with high probability, which we refer to as quantum isotropifier . Adopting the notation of [53, Definition 1], we define having quantum access to a d -dimensional random variable X as the ability to query a quantum sampling oracle that produces a quantum superposition representing the probability distribution of X .

Definition 10 (Quantum sampling oracle) . For a d -dimensional random variable X , its quantum sampling oracle O X is defined as

<!-- formula-not-decoded -->

where p X ( · ) represents the probability density function of X .

Using ˜ O ( σ/glyph[epsilon1] ) queries to a quantum sampling oracle O X of any random variable X ∈ R d with variance σ 2 , our quantum isotropifier outputs an unbiased estimate ˆ µ of E [ X ] such that for any unit vector v ∈ R d , |〈 v, ˆ µ -E [ X ] 〉| is at most glyph[epsilon1] with high probability. This allows us to construct a ( σ I , δ ) -ISGO using ˜ O ( σ V poly log(1 /δ/σ I ) queries to a σ V -QVSGO. Combined with our stochastic cutting plane result in Theorem 1, and with appropriate choices of σ I and δ , this leads to our

improved ˜ O ( dRσ V /glyph[epsilon1] ) query complexity for quantum SCO in Theorem 7, whose proof can be found in Section C.5.

Next, we provide a technical overview of our quantum isotropifier, which builds on the quantum multivariate mean estimation framework of [22]. We begin by considering a simplified scenario where the random variable X is bounded and satisfies ‖ X ‖ ≤ 1 . In this case, [22] first implements a directional mean oracle that approximately maps | g 〉 to e i 〈 g, E [ X ] 〉 | g 〉 for a query g ∈ R d . This oracle requires only a constant number of queries to the quantum sampling oracle O X provided that | E 〈 g, X 〉| ≤ ‖ E [ X ] ‖ . To estimate E [ X ] using this directional mean oracle, [22] applies it to the superposition ∑ g ∈ G ⊗ d m | g 〉 , where G ⊗ d m is a d -dimensional grid defined as

<!-- formula-not-decoded -->

The resulting quantum state approximately decomposes as a tensor product

<!-- formula-not-decoded -->

Then, phase estimation can be performed independently in each dimension to estimate each coordinate of E [ X ] , similar to the quantum gradient estimation algorithm in [36].

The error in this procedure has two parts: the error in the quantum directional mean oracle and the error from quantum phase estimation. The first part of the error arises as the directional mean oracle is accurate only for grid points g ∈ G d m with |〈 g, E [ X ] 〉| ≤ ‖ E [ X ] ‖ . In Section C.2, we provide an improved error analysis, showing that only an exponentially small fraction of g ∈ G ⊗ d m do not satisfy this condition. Hence, in the analysis we can effectively replace the directional mean oracle by a perfect one that performs the map | g 〉 → e i 〈 g, E [ X j ] 〉 | g 〉 exactly for all g ∈ G d m , and the resulting quantum state only has an exponentially small change in trace distance.

glyph[negationslash]

As for the second part of the error, we note that while the phase estimation procedures in different coordinates are independent, i.e., for any j = k , the estimates ˆ µ j and ˆ µ k of E [ X j ] and E [ X k ] are sampled independently from two distributions, their biases may still contribute positively in certain directions. Consequently, even if each ˆ µ j has a bounded error satisfying | ˆ µ j -E [ X j ] | ≤ glyph[epsilon1] , there may exist a unit vector u ∈ R d such that |〈 ˆ µ -E [ X ] , u 〉| = Θ( glyph[epsilon1] √ d ) which is larger than the per-coordinate guarantee by a factor of √ d . In this work, we demonstrate that the additional √ d overhead can be avoided by replacing the original phase estimation with the boosted unbiased phase estimation algorithm from [57]. For each coordinate j , this algorithm produces an unbiased estimate ˆ µ j that has bounded error | ˆ µ j -E [ X j ] | ≤ glyph[epsilon1] with high probability. Consequently, for any unit vector u ∈ R d , the error 〈 ˆ µ -E [ X ] , u 〉 can be approximated as a weighted sum of zero-mean bounded random variables, which follows a sub-Gaussian distribution with variance Θ( glyph[epsilon1] 2 ) [61]. This implies that ˆ µ not only is unbiased but also has isotropic noise.

We then extend this result to unbounded random variables, as detailed in QUnbounded (Algorithm 3) in Section C.3. Following a similar approach to [22], we decompose X into a sequence of truncated bounded random variables and estimate each one independently. We show that the isotropic noise property is maintained throughout this process. Despite our use of the unbiased phase estimation subroutine [57], the output of QUnbounded may still be biased due to truncations. Notably, while [53] developed a general debiasing scheme for quantum mean estimation algorithms that treats them as black-box procedures, directly applying their scheme to QUnbounded would compromise the isotropic noise property. As detailed in Section C.4, we utilize the multi-level Monte Carlo (MLMC) technique [26] to address this issue. Specifically, we develop a modified version of the MLMC variants introduced in [10, 4, 53] that preserves isotropic noise while maintaining unbiasedness.

To illustrate our approach, suppose we want an unbiased estimate whose error is smaller than glyph[epsilon1] in any direction with high probability. Let ˆ µ ( j ) denote the estimate obtained by running QUnbounded to accuracy β j glyph[epsilon1] , i.e., |〈 ˆ µ ( j ) -E [ X ] , u 〉| ≤ β j glyph[epsilon1] with high probability, where β j is a chosen coefficient. Then, our new estimator ˆ µ is:

<!-- formula-not-decoded -->

This estimator is unbiased as long as lim j →∞ β j = 0 , given that

<!-- formula-not-decoded -->

Moreover, the expected query complexity of the new estimator ˆ µ is larger than that of ˆ µ (0) , which is of order ˜ O ( σ/glyph[epsilon1] ) , by a multiplicative factor of

<!-- formula-not-decoded -->

This factor is a constant if we choose β j = 2 -j + C · log j for any C &gt; 1 . Furthermore, we show later that ˆ µ also satisfies |〈 ˆ µ -E [ X ] , u 〉| ≤ O ( glyph[epsilon1] ) with high probability as long as C is a constant, preserving the isotropic noise property. In our full algorithm, Algorithm 4 in Section C.4, we choose C = 2 for simplicity.

## 3 Conclusion

We define a new gradient noise model for SCO, termed isotropic noise, for which we achieve tight upper and lower bounds (up to polylogarithmic factors). Our upper bound improves upon the state-ofthe-art (and in particular SGD) in certain regimes, and as a corollary we achieve a new state-of-the-art complexity for sub-exponential noise. We then develop a subroutine which may be of independent interest called a quantum isotropifier , which converts a variance-bounded quantum sampling oracle into an unbiased estimator with isotropic noise. By combining our results, we obtain improved dimension-dependent rates for quantum SCO.

One limitation of our work is we only resolve Problem 1 under stronger assumptions on the noise (e.g., sub-exponential noise in Corollary 3), whereas the rate we achieve under a VSGO (Corollary 4) is worse by roughly a d -factor. Another limitation is that our improved quantum rates are dimensiondependent, and have unclear long-term practical impact. Regarding the former, we note that a series of works have shown that dimensions-dependence is necessary for quantum algorithms to achieve an improved scaling in 1 /glyph[epsilon1] in a variety of settings [24, 25, 66].

We hope that by identifying the natural open problem Problem 1 and developing techniques to resolve it under stronger assumptions, we leave the door open to future work on resolving fundamental trade-offs between the variance, Lipschitz constant, and dimension in SCO. We believe charting these trade-offs is important since it would yield a sharp understanding of the precisions at which algorithms such as SGD are superseded by other methods. Such questions have long been understood in the deterministic setting, and we believe our work is an important step in resolving them under stochasticity and in quantum settings.

## Acknowledgments

Thank you to anonymous reviewers for their feedback. Aaron Sidford was funded in part by a Microsoft Research Faculty Fellowship, NSF CAREER Award CCF-1844855, NSF Grant CCF1955039, and a PayPal research award. Chenyi Zhang was supported in part by the Shoucheng Zhang graduate fellowship.

## References

- [1] Daniel S. Abrams and Colin P. Williams. Fast quantum algorithms for numerical integrals and stochastic processes. arXiv preprint quant-ph/9908083 , 1999.
- [2] Alekh Agarwal, Martin J Wainwright, Peter Bartlett, and Pradeep Ravikumar. Informationtheoretic lower bounds on the oracle complexity of convex optimization. Advances in Neural Information Processing Systems , 22, 2009.
- [3] Dong An, Noah Linden, Jin-Peng Liu, Ashley Montanaro, Changpeng Shao, and Jiasu Wang. Quantum-accelerated multilevel Monte Carlo methods for stochastic differential equations in mathematical finance. Quantum , 5:481, 2021.

- [4] Hilal Asi, Yair Carmon, Arun Jambulapati, Yujia Jin, and Aaron Sidford. Stochastic bias-reduced gradient methods. Advances in Neural Information Processing Systems , 34:10810-10822, 2021.
- [5] Hilal Asi, Daogao Liu, and Kevin Tian. Private stochastic convex optimization with heavy tails: Near-optimality from simple reductions, 2024.
- [6] Brandon Augustino, Dylan Herman, Enrico Fontana, Junhyung Lyle Kim, Jacob Watkins, Shouvanik Chakrabarti, and Marco Pistoia. Fast convex optimization with quantum gradient methods, 2025.
- [7] Brandon Augustino, Giacomo Nannicini, Tamás Terlaky, and Luis F. Zuluaga. Quantum interior point methods for semidefinite optimization. Quantum , 7:1110, 2023.
- [8] Aharon Ben-Tal and Arkadi Nemirovski. Lecture notes: Optimization iii (convex analysis, nonlinear programming theory, nonlinear programming algorithms). Lecture Notes, ISYE 6663, Georgia Institute of Technology &amp; Technion - Israel Institute of Technology, 2023.
- [9] Albert Benveniste, Michel Métivier, and Pierre Priouret. Adaptive algorithms and stochastic approximations , volume 22. Springer Science &amp; Business Media, 2012.
- [10] Jose H. Blanchet and Peter W. Glynn. Unbiased Monte Carlo for optimization and functions of expectations via multi-level randomization. In 2015 Winter Simulation Conference (WSC) , pages 3656-3667. IEEE, 2015.
- [11] Léon Bottou, Frank E Curtis, and Jorge Nocedal. Optimization methods for large-scale machine learning. SIAM review , 60(2):223-311, 2018.
- [12] Fernando G.S.L. Brandão, Amir Kalev, Tongyang Li, Cedric Yen-Yu Lin, Krysta M. Svore, and Xiaodi Wu. Quantum SDP solvers: Large speed-ups, optimality, and applications to quantum learning. In Proceedings of the 46th International Colloquium on Automata, Languages, and Programming , volume 132 of Leibniz International Proceedings in Informatics (LIPIcs) , pages 27:1-27:14. Schloss Dagstuhl-Leibniz-Zentrum fuer Informatik, 2019.
- [13] Fernando G.S.L. Brandão and Krysta M. Svore. Quantum speed-ups for solving semidefinite programs. In 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS) , pages 415-426. IEEE, 2017.
- [14] Gilles Brassard, Frederic Dupuis, Sebastien Gambs, and Alain Tapp. An optimal quantum algorithm to approximate the mean and its application for approximating the median of a set of points over an arbitrary distance. 2011.
- [15] Gilles Brassard, Peter Hoyer, Michele Mosca, and Alain Tapp. Quantum amplitude amplification and estimation. Contemporary Mathematics , 305:53-74, 2002.
- [16] Sébastien Bubeck et al. Convex optimization: Algorithms and complexity. Foundations and Trends® in Machine Learning , 8(3-4):231-357, 2015.
- [17] Sébastien Bubeck, Qijia Jiang, Yin-Tat Lee, Yuanzhi Li, and Aaron Sidford. Complexity of highly parallel non-smooth convex optimization. Advances in neural information processing systems , 32, 2019.
- [18] Yair Carmon and Oliver Hinder. The price of adaptivity in stochastic convex optimization. In Shipra Agrawal and Aaron Roth, editors, Proceedings of Thirty Seventh Conference on Learning Theory , volume 247 of Proceedings of Machine Learning Research , pages 772-774. PMLR, 30 Jun-03 Jul 2024.
- [19] Shouvanik Chakrabarti, Andrew M. Childs, Shih-Han Hung, Tongyang Li, Chunhao Wang, and Xiaodi Wu. Quantum algorithm for estimating volumes of convex bodies. ACM Transactions on Quantum Computing , 4(3):1-60, 2023.
- [20] Shouvanik Chakrabarti, Andrew M. Childs, Tongyang Li, and Ronald de Wolf. Quantum algorithms and lower bounds for convex optimization. Quantum , 4:221, 2020.
- [21] Andrew M. Childs, Jiaqi Leng, Tongyang Li, Jin-Peng Liu, and Chenyi Zhang. Quantum simulation of real-space dynamics. Quantum , 6:680, 2022.

- [22] Arjan Cornelissen, Yassine Hamoudi, and Sofiene Jerbi. Near-optimal quantum algorithms for multivariate mean estimation. In Proceedings of the 54th Annual ACM SIGACT Symposium on Theory of Computing , pages 33-43, 2022.
- [23] John C. Duchi. Introductory lectures on stochastic optimization. The mathematics of data , 25:99-186, 2018.
- [24] Ankit Garg, Robin Kothari, Praneeth Netrapalli, and Suhail Sherif. No quantum speedup over gradient descent for non-smooth convex optimization, 2020.
- [25] Ankit Garg, Robin Kothari, Praneeth Netrapalli, and Suhail Sherif. Near-optimal lower bounds for convex optimization for all orders of smoothness. Advances in Neural Information Processing Systems , 34:29874-29884, 2021.
- [26] Michael B. Giles. Multilevel Monte Carlo methods. Acta Numerica , 24:259-328, 2015.
- [27] András Gilyén and Tongyang Li. Distributional property testing in a quantum world. In 11th Innovations in Theoretical Computer Science Conference (ITCS 2020) . Schloss DagstuhlLeibniz-Zentrum für Informatik, 2020.
- [28] Weiyuan Gong, Chenyi Zhang, and Tongyang Li. Robustness of quantum algorithms for nonconvex optimization, 2022.
- [29] Branko Grünbaum. Partitions of mass-distributions and of convex bodies by hyperplanes. Pacific Journal of Mathematics , 10:1257-1261, 1960.
- [30] Yassine Hamoudi. Quantum sub-gaussian mean estimator. In 29th Annual European Symposium on Algorithms (ESA 2021) . Schloss Dagstuhl-Leibniz-Zentrum für Informatik, 2021.
- [31] Yassine Hamoudi and Frédéric Magniez. Quantum Chebyshev's inequality and applications. In 46th International Colloquium on Automata, Languages, and Programming (ICALP 2019) , 2019.
- [32] Stefan Heinrich. Quantum summation with an application to integration. Journal of Complexity , 18(1):1-50, 2002.
- [33] Arun Jambulapati, Aaron Sidford, and Kevin Tian. Closing the computational-query depth gap in parallel stochastic convex optimization. In Shipra Agrawal and Aaron Roth, editors, Proceedings of Thirty Seventh Conference on Learning Theory , volume 247 of Proceedings of Machine Learning Research , pages 2608-2643. PMLR, 30 Jun-03 Jul 2024.
- [34] Haotian Jiang, Yin Tat Lee, Zhao Song, and Sam Chiu-wai Wong. An improved cutting plane method for convex optimization, convex-concave games, and its applications. In Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing , pages 944-953, Chicago IL USA, June 2020. ACM.
- [35] Chi Jin, Praneeth Netrapalli, Rong Ge, Sham M. Kakade, and Michael I. Jordan. A short note on concentration inequalities for random vectors with subgaussian norm, 2019.
- [36] Stephen P Jordan. Fast quantum algorithm for numerical gradient estimation. Physical review letters , 95(5):050501, 2005.
- [37] Iordanis Kerenidis and Anupam Prakash. A quantum interior point method for LPs and SDPs, 2018.
- [38] Robin Kothari and Ryan O'Donnell. Mean estimation when you have the source code; or, quantum Monte Carlo methods. In Proceedings of the 2023 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA) , pages 1186-1215. SIAM, 2023.
- [39] Guanghui Lan. An optimal method for stochastic composite optimization. Mathematical Programming , 133(1-2):365-397, 2012.

- [40] Yin Tat Lee, Aaron Sidford, and Sam Chiu-Wai Wong. A faster cutting plane method and its implications for combinatorial and convex optimization. In 2015 IEEE 56th Annual Symposium on Foundations of Computer Science , pages 1049-1065, Berkeley, CA, USA, October 2015. IEEE.
- [41] Jiaqi Leng, Ethan Hickman, Joseph Li, and Xiaodi Wu. Quantum Hamiltonian descent. Bulletin of the American Physical Society , 68, 2023.
- [42] Xiantao Li. Enabling quantum speedup of Markov chains using a multi-level approach, 2022.
- [43] Yizhou Liu, Weijie J. Su, and Tongyang Li. On quantum speedups for nonconvex optimization via quantum tunneling walks. Quantum , 7:1030, 2023.
- [44] Gábor Lugosi and Shahar Mendelson. Mean estimation and regression under heavy-tailed distributions: A survey. Foundations of Computational Mathematics , 19(5):1145-1190, 2019.
- [45] Tianyi Ma, Kabir A. Verchand, and Richard J. Samworth. High-probability minimax lower bounds, 2024.
- [46] Ashley Montanaro. Quantum speedup of Monte Carlo methods. Proceedings of the Royal Society A , 471(2181):20150301, 2015.
- [47] Arkadi Nemirovski. Efficient methods in convex programming, 1994. Lecture notes.
- [48] Arkadi Nemirovski, Anatoli Juditsky, Guanghui Lan, and Alexander Shapiro. Robust stochastic approximation approach to stochastic programming. SIAM Journal on optimization , 19(4):15741609, 2009.
- [49] Jurij Evgen'evi v c Nesterov. Lectures on convex optimization . Springer optimization and its applications. Springer Nature, Cham, second edition edition, 2018.
- [50] Warren B Powell. A unified framework for stochastic optimization. European Journal of Operational Research , 275(3):795-821, 2019.
- [51] Herbert Robbins and Sutton Monro. A stochastic approximation method. The Annals of Mathematical Statistics , 22(3):400-407, 1951.
- [52] Aaron Sidford. Optimization algorithms, 2024. Lecture notes; compiled on March 18, 2024.
- [53] Aaron Sidford and Chenyi Zhang. Quantum speedups for stochastic optimization. Advances in Neural Information Processing Systems , 36, 2024.
- [54] Barbara Terhal. Quantum algorithms and quantum entanglement . PhD thesis, University of Amsterdam, 1999.
- [55] MTCAJ Thomas and A Thomas Joy. Elements of information theory . Wiley-Interscience, 2006.
- [56] P.M. Vaidya. A new algorithm for minimizing convex functions over convex sets. In 30th Annual Symposium on Foundations of Computer Science , pages 338-343, 1989.
- [57] Joran van Apeldoorn, Arjan Cornelissen, András Gilyén, and Giacomo Nannicini. Quantum tomography using state-preparation unitaries. In Proceedings of the 2023 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA) , pages 1265-1318. SIAM, 2023.
- [58] Joran van Apeldoorn and András Gilyén. Improvements in quantum SDP-solving with applications. In 46th International Colloquium on Automata, Languages, and Programming (ICALP 2019) . Schloss Dagstuhl-Leibniz-Zentrum fuer Informatik, 2019.
- [59] Joran van Apeldoorn, András Gilyén, Sander Gribling, and Ronald de Wolf. Convex optimization using quantum oracles. Quantum , 4:220, 2020.
- [60] Joran van Apeldoorn, András Gilyén, Sander Gribling, and Ronald de Wolf. Quantum SDPsolvers: Better upper and lower bounds. Quantum , 4:230, 2020.

- [61] Roman Vershynin. High-dimensional probability , volume 47 of Cambridge Series in Statistical and Probabilistic Mathematics . Cambridge University Press, Cambridge, 2018. An introduction with applications in data science, With a foreword by Sara van de Geer.
- [62] Nuri Mert Vural, Lu Yu, Krishna Balasubramanian, Stanislav Volgushev, and Murat A Erdogdu. Mirror descent strikes again: Optimal stochastic convex optimization under infinite noise variance. In Conference on Learning Theory , pages 65-102. PMLR, 2022.
- [63] Martin J. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint . Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, New York, NY, 1st ed edition, 2019.
- [64] Pawel Wocjan, Chen-Fu Chiang, Daniel Nagaj, and Anura Abeyesinghe. Quantum algorithm for approximating partition functions. Physical Review A-Atomic, Molecular, and Optical Physics , 80(2):022340, 2009.
- [65] Chenyi Zhang, Jiaqi Leng, and Tongyang Li. Quantum algorithms for escaping from saddle points. Quantum , 5:529, 2021.
- [66] Chenyi Zhang and Tongyang Li. Quantum lower bounds for finding stationary points of nonconvex functions, 2022.
- [67] Yexin Zhang, Chenyi Zhang, Cong Fang, Liwei Wang, and Tongyang Li. Quantum algorithms and lower bounds for finite-sum optimization, 2024.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: See the results section: Section 1.1.

## Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: See the conclusion: Section 3.

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

Justification: Full proofs are given in the appendices. See Section 1.4 for a roadmap of the appendices.

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

Justification: Our results are theoretical.

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

Justification: Our results are theoretical.

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

Justification: Our results are theoretical.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: Our results are theoretical.

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

Justification: Our results are theoretical.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: It conforms to the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This paper is largely theoretical and consequently we do not expect broad societal impacts (beyond further research and educational use) without further work.

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

Justification: This is a theoretical work and we do not foresee any such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: We do not use existing assets.

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

Justification: We do not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our paper does not involve crowdsourcing nor research with human subjects. Guidelines:

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

## A Upper bounds for SCO under ISGO, ESGO, and VSGO oracles

In this section, we prove an upper bound for Problem 1 given a ( σ I , δ ) -ISGO in Theorem 1, and then instantiate the result to achieve upper bounds for Problem 1 given either a σ E -ESGO or σ V -VSGO in Corollaries 3 and 4 respectively. Given an ISGO, our algorithm for Problem 1 proceeds in two stages. In Section A.1, we give a stochastic cutting plane method which utilizes a ( σ I , δ ) -ISGO to return a finite set of points with the guarantee that at least one of the points is glyph[epsilon1] -optimal. Then in Section A.2, we show how to use a ( σ I , δ ) -ISGO to obtain an approximate minimum among this finite set of points. We note that all results and discussion in Sections A.1 and A.2 are stated in the context of solving Problem 1 given a ( σ I , δ ) -ISGO, unless we explicitly specify otherwise. Finally, we put everything together and prove Theorem 1 in Section A.3.

## A.1 Finding candidate solutions via a stochastic cutting plane method

In this subsection, we show how to obtain a finite set of points with the guarantee that at least one of them is glyph[epsilon1] -optimal via a stochastic cutting plane method in the setting of Problem 1 given a ( σ I , δ ) -ISGO. Cutting plane methods solve feasibility problems, for which we will use the formulation of [52, Definition 8.1.2], except we translate from distance bounds in the glyph[lscript] ∞ -norm to distance bounds in the glyph[lscript] 2 -norm.

Definition 7 (Feasibility Problem) . For R ′ ≥ r &gt; 0 , we define the ( R ′ , r ) -feasibility problem as follows: We are given query access to a (potentially randomized) halfspace oracle which when queried at x ∈ B 2 ( R ′ ) , outputs a vector g x ∈ R d \{ 0 } . The goal is to query the oracle at a sequence of points x 1 , . . . , x T ∈ B 2 ( R ′ ) such that B 2 ( R ′ ) ⋂ t ∈ [ T ] H ≥ ( g x t , g glyph[latticetop] x t x t ) does not contain a ball of radius r (namely, any set of the form B 2 ( r, z ) for z ∈ R d ). We call an algorithm a T -algorithm for the ( R ′ , r ) -feasibility problem if it achieves this goal with at most T queries.

We give a standard query complexity for the ( R ′ , r ) -feasibility problem in the following proposition. For completeness, we also give a standard short proof using the center of gravity cutting plane method, but we emphasize that our results are agnostic to the choice of cutting plane method used to prove Proposition 1. See, e.g., [56, 40, 34] for other cutting plane methods with similar query complexities (and which may have improved runtimes).

Proposition 1. There exists an O ( d log( R ′ /r )) -algorithm for the ( R ′ , r ) -feasibility problem.

Proof. We apply the so-called center of gravity method, e.g., [8]. For iterations t = 1 , 2 , . . . , the center of gravity method queries the halfspace oracle at the origin for t = 1 (the center of gravity of B 2 ( R ′ ) ), and at the center of gravity of the set B 2 ( R ′ ) ⋂ k ∈ [ t -1] H ≥ ( g x k , g glyph[latticetop] x k x k ) for t &gt; 1 , where x k ∈ R d denotes the iterate at step k and g x k ∈ R d denotes the halfspace queried at x k during step k . Recall that the volume of an d -dimensional Euclidean ball (equivalently, glyph[lscript] 2 -ball) of radius β &gt; 0 is h ( d ) · β d , where h is some function of the dimension d which won't matter in the following. Then letting vol( · ) denote the volume of a set, we have vol( B 2 ( R ′ )) = h ( d ) · ( R ′ ) d , and the volume of any Euclidean ball of radius r is h ( d ) · r d . Letting S t := B 2 ( R ′ ) ⋂ k ∈ [ t ] H ≥ ( g x k , g glyph[latticetop] x k x k ) for t ∈ N , Grünbaum's theorem [29] yields vol( S t ) ≤ (1 -1 /e ) t · vol( B 2 ( R ′ )) , and the result follows.

It is well known that solving the feasibility problem where the halfspace oracle is given by the negative gradient of f suffices to obtain a finite set of points such that at least one is glyph[epsilon1] -optimal [47, 40]. [53] demonstrated that this is still possible with stochastic estimates of the gradient, provided that -∇ f is approximated to sufficiently high accuracy in the glyph[lscript] 2 -norm with high probability at each iteration by averaging over multiple samples. Our key observation is that the analysis of [53] mainly relies on controlling the error of the gradient estimator in a single direction, namely the one-dimensional subspace spanned by x t -x glyph[star] , where x t is the current iterate. A ( σ, δ ) -ISGO with δ sufficiently small allows us to control this error with fewer samples than, for example, a σ -VSGO (Definition 3), allowing for savings of up to a multiplicative d factor in the complexity of estimating the gradient at each step when compared to a σ -VSGO or σ -BSGO.

Toward formalizing this discussion, we first define a marginal version of the η -approximate gradient oracle specified in [53, Definition 5]. Note that this definition still allows us to control the error of the estimator in glyph[lscript] 2 -norm. This will be necessary to obtain our guarantee, but we will only need to weakly control this error which can be polynomial in d .

Definition 9 (MAGO) . ˜ g ( · , · ) is a marginal ( η ≥ 0 , Γ ≥ 0) -approximate gradient oracle ( ( η, Γ) -MAGO ) if when queried at x, u ∈ R d , the (potentially random) output ˜ g ( x, u ) ∈ R d satisfies

<!-- formula-not-decoded -->

Next, we show that an ( η, η √ d ) -MAGO can be implemented with high probability using a ( σ I , δ ) -ISGO without knowledge of the second argument, the directional input u .

Lemma 1. For any δ, ξ ∈ (0 , 1) ; x, u ∈ R d ; and η &gt; 0 , a query ˜ g ( x, u ) to an ( η, Γ := η √ d ) -MAGO can be implemented using K = O ( σ 2 I dη 2 log(2 d/ξ ) + 1) queries to a ( σ I , δ ) -ISGO O I ( · ) , with success probability at least 1 -ξ -δdK and without access to the input u .

Proof. For K ∈ N to be chosen later, consider the estimator Z := 1 K ∑ k ∈ [ K ] Z k for Z 1 , . . . , Z K iid ∼ O I ( x ) , which does not require knowledge of u . Let { u glyph[lscript] } glyph[lscript] ∈ [ d ] denote any orthonormal basis of R d such that u 1 = u/ ‖ u ‖ , and define for all glyph[lscript] ∈ [ d ] :

<!-- formula-not-decoded -->

Note that E Y ( glyph[lscript] ) k = 0 , and | Y ( glyph[lscript] ) k | ≤ σ I / √ d for all ( k, glyph[lscript] ) ∈ [ K ] × [ d ] with probability at least 1 -δdK by Definition 4 and a union bound. Then conditioning on the latter event E , Hoeffding's inequality [63, Proposition 2.5] yields for all glyph[lscript] ∈ [ d ] :

<!-- formula-not-decoded -->

by a union bound. Finally, note that the event E ′ implies the first condition in (4) since then

<!-- formula-not-decoded -->

and the second condition in (4) follows because recall we chose u 1 = u/ ‖ u ‖ .

Next, we show how to obtain a finite set of points such that at least one is glyph[epsilon1] -optimal using an appropriate MAGO. Critically, the second argument to the MAGO must be allowed to depend on x glyph[star] . However, this does not pose an issue when we ultimately instantiate the MAGO queries via ISGO queries since doing so does not require any knowledge of the second argument to the MAGO, per Lemma 1.

Lemma 2 (Obtaining candidate solutions via the feasibility problem) . Suppose R ′ , r &gt; 0 are such that B 2 ( r, x glyph[star] ) ⊆ B 2 ( R ′ ) and f ( z ) ≤ f ( x glyph[star] ) + glyph[epsilon1]/ 2 for all z ∈ B 2 ( r, x glyph[star] ) . Then given an ( η := glyph[epsilon1] 8 R ′ , Γ := η √ d ) -MAGO ˜ g ( · , · ) where only the second argument can depend on x glyph[star] and a T -algorithm for the ( R ′ , r ) -feasibility problem, and additionally supposing r ≤ glyph[epsilon1]/ (4Γ) , we can compute a finite set of points S ⊆ B 2 ( R ′ ) with | S | = T such that min x ∈ S f ( x ) ≤ f ( x glyph[star] ) + glyph[epsilon1] .

Proof. We run the T -algorithm for the ( R ′ , r ) -feasibility problem with the halfspace oracle at a point x ∈ R d given by -˜ g ( x, x -x glyph[star] ) . By definition, this produces a sequence of points S := { x t ∈ B 2 ( R ′ ) } t ∈ [ T ] such that B 2 ( R ′ ) ⋂ t ∈ [ T ] H ≥ ( -g t , -g glyph[latticetop] t x t ) does not contain a ball of radius r , where -g t denotes the output of the halfspace oracle at the t -th step (i.e., g t was the result of the oracle call ˜ g ( x t , x t -x glyph[star] ) ). (If the T -algorithm terminates with less than T queries, we can pad S with

duplicates.) Note that we can terminate immediately if the halfspace oracle outputs g t = 0 at some step t since then

<!-- formula-not-decoded -->

glyph[negationslash]

by convexity, the fact that x t , x glyph[star] ∈ B 2 ( R ′ ) , and the definition of ˜ g ( x t , x t -x glyph[star] ) . Then supposing g t = 0 for all t ∈ [ T ] , it must be the case that at some iteration t ∈ [ T ] , there exists z ∈ B 2 ( r, x glyph[star] ) such that -g glyph[latticetop] t z &lt; -g glyph[latticetop] t x t . Then by convexity

<!-- formula-not-decoded -->

where the last inequality followed because 1 &lt; 0 by the definition of z ; we have 2 ≤ η ‖ x t -x glyph[star] ‖ ≤ 2 R ′ η ≤ glyph[epsilon1]/ 4 by the definition of ˜ g ( x t , x t -x glyph[star] ) ; we have 3 ≤ Γ r ≤ glyph[epsilon1]/ 4 by Cauchy-Schwarz, the definition of ˜ g ( x t , x t -x glyph[star] ) , and the additional assumption on r ; and finally z is glyph[epsilon1]/ 2 -optimal.

Finally, we give our main result for this subsection:

Lemma 3 (Obtaining candidate solutions via an ISGO) . For T = O ( d log( d + RL/glyph[epsilon1] )) , there exists an algorithm which returns a finite set of points S ⊆ B 2 (2 R ) with | S | = T and min x ∈ S f ( x ) ≤ f ( x glyph[star] ) + glyph[epsilon1] using

<!-- formula-not-decoded -->

and with success probability at least 1 -ξ -δdM .

Proof. We first apply Lemma 2 with R ′ := 2 R and r := min { glyph[epsilon1] 2 L , glyph[epsilon1] 4Γ } (with η := glyph[epsilon1] 8 R ′ and Γ := η √ d as in Lemma 2). Note that we can assume glyph[epsilon1] ∈ (0 , RL ) without loss of generality as otherwise the origin is glyph[epsilon1] -optimal, in which case R ≥ r , implying B 2 ( r, x glyph[star] ) ⊆ B 2 ( R ′ ) . Furthermore, every point in B 2 ( r, x glyph[star] ) is glyph[epsilon1]/ 2 -optimal given that f is L -Lipschitz. Thus, combining Lemma 2 with Proposition 1, we can obtain S with

<!-- formula-not-decoded -->

√

queries to a ( η = glyph[epsilon1] 8 R ′ , Γ = η d ) -MAGO. The result then follows by instantiating each MAGO query per Lemma 1, and the final success probability follows from a union bound.

## A.2 Approximate minimum finding among a finite set of points

Here we show how to use an ISGO to obtain an approximate minimum among a finite set of points S ⊆ R d . (As a reminder, Section A.2 is stated in the context of Problem 1 given access to a ( σ I , δ ) -ISGO, unless we specify otherwise.) To do so, we adapt the techniques of [53, Section 4.2], which solves the same problem except with a (quantum) BSGO (recall Definition 2). 8 At a high level, the procedure involves iteratively replacing pairs of points ( x, x ′ ) in S with a point ¯ x on the line segment between x and x ′ , with the guarantee that the objective value of ¯ x is (approximately) at least as good as that of both x and x ′ . This is done in multiple levels, eventually comparing pairs that were the result of previous comparisons, until only a single point remains. For a pair ( x, x ′ ) , the point ¯ x is computed via a binary search procedure on the line segment between them using the ISGO. Critically, analogously to the main result of Section A.1, a ( σ, δ ) -ISGO allows for up to a d -factor savings when implementing this binary search over a σ -BSGO or σ -VSGO.

To start, Algorithm 1, which is stated independently of the context of Problem 1, gives a procedure for computing an approximate minimum over [0 , 1] of a one-dimensional Lipschitz convex function, given access to a sequence of approximate first-order derivatives (see Line 4). We will ultimately use Algorithm 1 to obtain ¯ x for a pair ( x, x ′ ) as discussed above. Algorithm 1 and its analysis are based on Algorithm 4 in [53].

8 An alternate procedure for solving this problem via a BSGO was given in [33, Proposition 3]. However, unlike the procedure of [53, Section 4.2], it is not clear how to adapt their techniques to an ISGO (or even a VSGO). See also [18, 5] for related procedures.

```
′ )
```

## Algorithm 1: InexactLineSearch ( h, glyph[epsilon1]

```
Input: Convex G -Lipschitz h : R → R , target accuracy glyph[epsilon1] ′ > 0 Output: ¯ z ∈ [0 , 1] s.t. h (¯ z ) ≤ min z ∈ [0 , 1] h ( z ) + glyph[epsilon1] ′ 1 z glyph[lscript] ← 0 , z r ← 1 2 while z r -z glyph[lscript] > glyph[epsilon1] ′ /G do 3 z m ← ( z r + z glyph[lscript] ) / 2 4 Let ˜ u z m be s.t. | ˜ u z m -h ′ ( z m ) | ≤ glyph[epsilon1] ′ / 4 5 if | ˜ u z m | ≤ glyph[epsilon1] ′ / 4 then return ¯ z ← z m 6 if ˜ u z m > 0 then z r ← z m else z glyph[lscript] ← z m 7 return ¯ z ← z glyph[lscript]
```

Lemma 4 (Algorithm 1 guarantee) . Given differentiable, convex, G -Lipschitz h : R → R and glyph[epsilon1] ′ &gt; 0 , Algorithm 1 terminates after O (log( G/glyph[epsilon1] ′ )) iterations 9 and returns ¯ z ∈ [0 , 1] such that h (¯ z ) ≤ min z ∈ [0 , 1] h ( z ) + glyph[epsilon1] ′ .

Proof. The iteration bound is immediate from the fact that each iteration halves the length of [ z glyph[lscript] , z r ] . As for correctness, if Algorithm 1 terminates on Line 6, then | h ′ ( z m ) | ≤ glyph[epsilon1] ′ / 2 by a triangle inequality, in which case convexity implies for all z ∈ [0 , 1] :

<!-- formula-not-decoded -->

On the other hand, suppose Algorithm 1 terminates on Line 7. Pick any z glyph[star] ∈ argmin z ∈ [0 , 1] h ( z ) , and we claim Algorithm 1 maintains the invariant z glyph[star] ∈ [ z glyph[lscript] , z r ] . This follows by induction because the condition ˜ u z m &gt; 0 in Line 6 implies ˜ u z m &gt; glyph[epsilon1] ′ / 4 due to the failure of the termination condition in Line 5, in which case h ′ ( z m ) &gt; 0 by a reverse triangle inequality. Then

<!-- formula-not-decoded -->

The case where ˜ u z m &lt; 0 in Line 6 is analogous. To conclude, the termination condition of Line 2 implies the following in Line 7 by the invariant and Lipschitzness:

<!-- formula-not-decoded -->

Before proceeding, we provide a version of Lemma 1 where we don't control the error of the MAGO estimator in glyph[lscript] 2 -norm, allowing for a potential logarithmic-factor improvement in the ISGO query complexity needed to implement the MAGO as well as a higher success probability. Technically Lemma 5 is not necessary to prove our ultimate guarantee Theorem 1, where we do not explicitly state polylogarithmic factors for brevity, but we include it for completeness and to make it clear what aspects of the MAGO we actually need for individual lemmas.

Lemma 5. For any δ, ξ ∈ (0 , 1) ; x, u ∈ R d ; and η &gt; 0 , a query ˜ g ( x, u ) to an ( η, ∞ ) -MAGO can be implemented using K = O ( σ 2 I dη 2 log(2 /ξ ) + 1 ) queries to a ( σ I , δ ) -ISGO O I ( · ) , with success probability at least 1 -ξ -δK and without access to the input u .

Proof. Consider the estimator Z := 1 K ∑ k ∈ [ K ] Z k for Z 1 , . . . , Z K iid ∼ O I ( x ) , which doesn't require knowledge of u . Define

<!-- formula-not-decoded -->

Note E Y k = 0 and | Y k | ≤ σ I / √ d for all k ∈ [ K ] with probability at least 1 -δK by Definition 4 and a union bound. Letting E denote the latter event, an application of Hoeffding's inequality implies P [ | Y | ≥ η | E ] ≤ ξ , in which case

<!-- formula-not-decoded -->

9 An iteration is a single execution of Lines 3-6.

We now use Algorithm 1 as a subroutine to obtain ¯ x for a pair ( x, x ′ ) .

Lemma 6. For glyph[epsilon1] ′ &gt; 0 ; ξ ∈ (0 , 1) ; and x, x ′ ∈ R d with D := ‖ x ′ -x ‖ , there exists an algorithm which computes ¯ x , a convex combination of x and x ′ such that f (¯ x ) ≤ min λ ∈ [0 , 1] f ( x + λ ( x ′ -x ))+ glyph[epsilon1] ′ using at most

<!-- formula-not-decoded -->

queries to a ( σ I , δ ) -ISGO O I ( · ) , with success probability at least 1 -ξ -δM .

Proof. If D = 0 , we return x . Otherwise, define x λ := x + λ ( x ′ -x ) for λ ∈ R , and let h ( λ ) := f ( x λ ) . Note that h is DL -Lipschitz since

<!-- formula-not-decoded -->

Note that for a given λ ∈ R and γ &gt; 0 , we can obtain ˜ u λ ∈ R such that | ˜ u λ -h ′ ( λ ) | ≤ γ using a single query to a ( γ/D, ∞ ) -MAGO ˜ g ( · , · ) . Indeed, with Z ∼ ˜ g ( x λ , x ′ -x ) and ˜ u λ ←〈 Z, x ′ -x 〉 ,

<!-- formula-not-decoded -->

Furthermore, for ξ ′ ∈ (0 , 1) , each query to ˜ g ( · , · ) can be implemented with K = O ( D 2 σ 2 I dγ 2 log(2 /ξ ′ ) + 1 ) queries to a ( σ I , δ ) -ISGO and success probability 1 -ξ ′ -δK by Lemma 5.

Putting this together, we can obtain ˜ u λ such that | ˜ u λ -h ′ ( λ ) | ≤ γ with K = O ( D 2 σ 2 I dγ 2 log(2 /ξ ′ ) + 1 ) queries to a ( σ I , δ ) -ISGO and success probability 1 -ξ ′ -δK . Combining this with the fact that h is DL -Lipschitz, it is clear that ¯ λ ← InexactLineSearch ( h, glyph[epsilon1] ′ ) can be implemented with

<!-- formula-not-decoded -->

queries to a ( σ I , δ ) -ISGO, with, by a union bound, success probability at least 1 -ξ -δM . To conclude, note

<!-- formula-not-decoded -->

Finally, we give our main result for this subsection:

Lemma 7 (Finding the best among a finite set of points) . For T ∈ N , suppose S := { x t ∈ R d } t ∈ [ T ] is such that ‖ x i -x j ‖ ≤ D for all i, j ∈ [ T ] . Then for glyph[epsilon1] &gt; 0 and ξ ∈ (0 , 1) , there is an algorithm which returns a point x in the convex hull of S such that f ( x ) ≤ min x ′ ∈ S f ( x ′ ) + glyph[epsilon1] using at most M ( T -1) queries to a ( σ I , δ ) -ISGO O I ( · ) , for

<!-- formula-not-decoded -->

and it succeeds with probability at least 1 -ξ -δM log T

Proof. For a pair x, x ′ in the convex hull of S , we can apply Lemma 6 to obtain ¯ x , a convex combination of x and x ′ such that f (¯ x ) ≤ min { f ( x ) , f ( x ′ ) } + glyph[epsilon1]/ log T using

<!-- formula-not-decoded -->

queries to a ( σ I , δ ) -ISGO, with success probability 1 -ξ/ log T -δM .

Next, we assume without loss of generality that | T | is a power of 2, as x 1 can be duplicated if this is not the case. Then consider a complete binary tree with T leaves, and define the glyph[lscript] -th layer for glyph[lscript] ∈ { 0 } ∪ [log 2 T ] to denote those vertices which are distance glyph[lscript] from a leaf node. (In other words, the 0-th layer contains the leaf nodes, and the final (log 2 T ) -th layer is the root.) We assign x 1 , . . . , x T to the leaf nodes, and iteratively populate the glyph[lscript] -th layer of the tree for glyph[lscript] = 1 , 2 , . . . , log 2 T

via the following process: For each node in the glyph[lscript] -th layer with children x and x ′ in the ( glyph[lscript] + 1) -th layer, we assign to that node the estimator described above with inputs x and x ′ . Assuming x 1 ∈ argmin x ∈ S f ( x ) without loss of generality, note that conditioned on all of the estimates along the path from the leaf x 1 to the root succeeding, which happens with probability at least 1 -ξ -δM log T by a union bound, the value of f at the root x r (which we output) can be bounded as f ( x r ) ≤ f ( x 1 ) + glyph[epsilon1] . The final query bound follows from the fact that we call the estimator T -1 times, since there are T -1 non-leaf nodes in the tree.

## A.3 Putting it all together

We now restate and prove our main guarantee for Problem 1 given a ( σ I , δ ) -ISGO:

Theorem 1. Problem 1 can be solved with probability at least 2 / 3 in ˜ O ( R 2 σ 2 I /glyph[epsilon1] 2 + d ) queries to a ( σ I , δ ) -ISGO for any δ ≤ 1 Md ( R 2 σ 2 I /glyph[epsilon1] 2 + d ) , where M = ˜ O (1) .

Proof. By Lemma 3, we can obtain a finite set S ⊆ B 2 (2 R ) with | S | = ˜ O ( d ) and min x ∈ S f ( x ) ≤ f ( x glyph[star] ) + glyph[epsilon1]/ 2 using K = ˜ O ( R 2 σ 2 I /glyph[epsilon1] 2 + d ) queries to the ( σ I , δ ) -ISGO, with success probability at least 9 / 10 -δdK . Then giving this set S as input to Lemma 7, we can obtain x ∈ B 2 (2 R ) such that f ( x ) ≤ min x ′ ∈ S f ( x ′ ) + glyph[epsilon1]/ 2 using K ′ = ˜ O ( R 2 σ 2 I /glyph[epsilon1] 2 + d ) queries to the ( σ I , δ ) -ISGO, with success probability at least 9 / 10 -δ · ˜ O ( R 2 σ 2 I dglyph[epsilon1] 2 +1 ) . Then f ( x ) ≤ f ( x glyph[star] ) + glyph[epsilon1] with probability at least 2 / 3 by the valid range of δ and a union bound, and the total number of queries is K + K ′ = ˜ O ( R 2 σ 2 I /glyph[epsilon1] 2 + d ) .

Next, we instantiate Theorem 1 to obtain upper bounds for solving Problem 1 with a σ E -ESGO or σ V -VSGO. First, we show in the following lemmas that ISGOs can be implemented using ESGOs and VSGOs respectively. In particular, Lemma 8 says that an ISGO query can be implemented with an ESGO at the cost of only a log factor, whereas Lemma 9 says that doing so with a VSGO additionally picks up a √ d factor. Indeed, this √ d -factor is the reason our rate for solving Problem 1 with a VSGO is a d -factor worse than solving Problem 1 with an ISGO or ESGO. We note that the proof of Lemma 9 is based on ideas from [53, Lemma 7].

Lemma 8. For any σ E &gt; 0 and δ ∈ (0 , 1) , a σ E -ESGO is a ( σ E log(2 /δ ) , δ ) -ISGO.

Proof. For any δ ∈ (0 , 1) and any unit vector u ∈ R d , by Eq. 2 we have

<!-- formula-not-decoded -->

Lemma9. For any σ V &gt; 0 and δ ∈ (0 , 1) , a query to a (6 σ V √ d, δ ) -ISGO O I ( · ) can be implemented using O (log(2 /δ )) queries to a σ V -VSGO O V ( · ) .

Proof. Supposing we wish to query O I ( · ) at x ∈ R d , let Z 1 , . . . , Z K iid ∼ O V ( x ) for K ∈ N to be chosen later. Chebyshev's inequality yields P [ ‖ Z k -∇ f ( x ) ‖ ≥ 2 σ V ] ≤ 1 / 4 for all k ∈ [ K ] . Define J := { k ∈ [ K ] : ‖ Z k -∇ f ( x ) ‖ ≤ 2 σ V } , in which case a Chernoff bound gives P [ | J | ≤ 2 K/ 3] ≤ 2 e -cK for some absolute constant c ∈ (0 , 1) . Thus, choosing K = O (log(2 /δ )) yields P [ | J | ≤ 2 K/ 3] ≤ δ . Then conditioning on the event where | J | ≥ 2 K/ 3 , which happens with probability at least 1 -δ , observe that if a sample Z k for some k ∈ [ K ] is such that

<!-- formula-not-decoded -->

namely Z k is 4 σ V -close to at least 2 K/ 3 of Z 1 , . . . , Z K , then there must exist some j ∈ J such that ‖ Z k -Z j ‖ ≤ 2 σ V , implying ‖ Z k -∇ f ( x ) ‖ ≤ 6 σ V by a triangle inequality. Furthermore, such a sample Z k is guaranteed to exist when conditioning on | J | ≥ 2 K/ 3 since any sample corresponding to an index in J satisfies this property in particular by a triangle inequality. Finally, we conclude by noting therefore that when | J | ≥ 2 K/ 3 , we have for any u ∈ R d with ‖ u ‖ = 1 :

<!-- formula-not-decoded -->

Finally, we give our upper bounds for Problem 1 with a σ E -ESGO or σ V -VSGO in the following corollaries, which we restate here for convenience.

Corollary 3. Problem 1 can be solved with probability at least 2 / 3 in ˜ O ( R 2 σ 2 E /glyph[epsilon1] 2 + d ) queries to a σ E -ESGO.

Proof. By Lemma 8, O E ( · ) is a ( σ E log(2 /δ ) , δ ) -ISGO per Definition 4. The result then follows by setting δ ← 1 Md ( R 2 σ 2 E /glyph[epsilon1] 2 + d ) where M = ˜ O (1) and applying Theorem 1.

Corollary 4. Problem 1 can be solved with probability at least 2 / 3 in ˜ O ( dR 2 σ 2 V /glyph[epsilon1] 2 + d ) queries to a σ V -VSGO.

Proof. By Lemma 9, we can implement a query to a (6 σ V √ d, δ ) -ISGO with O (log(1 /δ )) queries to O V ( · ) . We conclude by setting δ ← 1 Md 2 ( R 2 σ 2 V /glyph[epsilon1] 2 + d ) where M = ˜ O (1) and applying Theorem 1.

## B Lower bounds for SCO under ISGO and ESGO oracles

In this section, we establish lower bounds for solving Problem 1 with either a ( σ I , δ ) -ISGO or σ E -ESGO. We begin by defining a mean estimation problem for random variables with sub-exponential noise and provide a lower bound for this problem in Section B.1. Then in Section B.2, we establish our lower bound for solving Problem 1 given a σ E -ESGO by reducing this isotropic mean estimation problem to the former. Finally, we give our lower bound for Problem 1 given a ( σ I , δ ) -ISGO at the end of Section B.2 as a simple corollary by noting that a σ -ESGO is a ˜ O ( σ log(2 /δ ) , δ ) -ISGO for any δ ∈ (0 , 1) .

## B.1 Lower bound for mean estimation with sub-exponential noise

Problem 3 (Mean estimation with sub-exponential noise) . Given sample access to a d -dimensional random variable X that satisfies

<!-- formula-not-decoded -->

for any t &gt; 0 and any unit vector u , the goal is to output an estimate ˆ µ of µ := E [ X ] satisfying ‖ ˆ µ -µ ‖ 2 ≤ ˜ glyph[epsilon1] .

Let v ∈ {± 1 } d be a fixed vector in the d -dimensional hypercube. We establish our lower bound for Problem 3 by considering the following d -dimensional random variable X , where each component X i are sampled independently at random

<!-- formula-not-decoded -->

for all i ∈ [ d ] . Let P v ( · ) denote this probability distribution.

Lemma 10. For any fixed v ∈ {± 1 } d and any unit vector u ∈ R d , the random variable X defined in Eq. 6 satisfies

<!-- formula-not-decoded -->

Proof. For any unit vector u ∈ R d , using the fact that each coordinate X j is sampled independently and satisfies | X j -E [ X j ] | ≤ σ E / √ d , by Lemma 21 we know that 〈 u, X -E [ X ] 〉 is a sub-Gaussian random variable with variance

<!-- formula-not-decoded -->

Hence, for any t &gt; 0 we have

<!-- formula-not-decoded -->

Lemma 11. Any algorithm that solves Problem 3 with success probability at least 2 / 3 must have observed at least i.i.d. samples of X .

Lemma 12 (Chain rule of mutual information, Theorem 2.5.2 of [55]) . For any n &gt; 0 and any random variables V, X (1) , . . . , X ( n ) , we have

<!-- formula-not-decoded -->

Proof of Lemma 11. Consider the random variable X ∼ P v defined in Eq. 6 where v is chosen uniformly at random from {± 1 } d . For convenience, let p = 8˜ glyph[epsilon1] √ log d/σ E and ˜ σ E = σ E 2 √ d . Observe that V → ( X (1) , . . . , X ( n ) ) → ˆ X is a Markov chain, regardless of the algorithm used to construct ˆ X . Therefore, by the data processing inequality,

<!-- formula-not-decoded -->

where I ( V ; ˆ X ) is the mutual information between V and ˆ X , as defined in Section 1.3. We will prove that if an algorithm outputs ˆ X such that ‖ ˆ X -E P v [ X ] ‖ ≤ ˜ glyph[epsilon1] with success probability at least 2 / 3 then,

<!-- formula-not-decoded -->

On the other hand, we will prove that

<!-- formula-not-decoded -->

Assuming Eq. 8 and Eq. 9 for now and combining them with Eq. 7 we have d/ 6 ≤ 28 ndp 2 . Therefore,

<!-- formula-not-decoded -->

Proof of Eq. 8: Suppose an algorithm outputs ˆ X satisfying ‖ ˆ X -E P v [ X ] ‖ ≤ ˜ glyph[epsilon1] with success probability at least 2 / 3 . Let E be the event that ‖ ˆ X -E P v [ X ] ‖ ≤ ˜ glyph[epsilon1] . We have

<!-- formula-not-decoded -->

Observe that p ( E ) ≥ 2 / 3 and H ( V ) = d . To compute H ( V | ˆ X,E ) we consider the set

<!-- formula-not-decoded -->

and note that H ( V | ˆ X,E ) ≤ log | S | . To bound the size of this set S , fix some v 0 ∈ S and consider,

<!-- formula-not-decoded -->

Using the fact that E P v [ X ] = p ˜ σ E v and the triangle inequality, we have S ⊆ S ′ . To bound the cardinality of S ′ note that for v ∈ S ′ , v cannot differ from v 0 on more than N = glyph[ceilingleft] ˜ glyph[epsilon1] 2 / ( p 2 ˜ σ 2 E ) glyph[ceilingright] = glyph[ceilingleft] d/ (16 log d ) glyph[ceilingright] coordinates. Since N ≤ d/ 2 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus for d ≥ 2 ,

<!-- formula-not-decoded -->

We conclude that

<!-- formula-not-decoded -->

This concludes the proof of Eq. 8.

Proof of Eq. 9: Consider I ( V ; X (1) , . . . , X ( n ) ) . By Lemma 12 and the fact that X ( i ) and X ( j ) are identically and independently distributed conditioned on V , we have

<!-- formula-not-decoded -->

Similarly, since the coordinates of X are independent we have

<!-- formula-not-decoded -->

Next we give an expression for I ( V ; X i ) in terms of p :

<!-- formula-not-decoded -->

where h 2 ( t ) is the binary entropy: h 2 ( t ) := -t log( t ) -(1 -t ) log(1 -t ) . We claim

<!-- formula-not-decoded -->

Assuming Eq. 10 we have,

<!-- formula-not-decoded -->

It remains to show that h 2 ( 1 2 + p ) ≥ 1 -28 p 2 . Observe the following equality:

<!-- formula-not-decoded -->

So it is equivalent to upper bound the following by 28 p 2 :

<!-- formula-not-decoded -->

Since p ≥ 0 , 1 2 log ( 1 -4 p 2 ) ≤ 0 . Hence, it suffices to show that log ((1 + 2 p ) / (1 -2 p )) ≤ 28 p . To do this we use the facts that 1 / (1 -x ) ≤ 1+2 x for x ≤ 1 / 4 , and log(1 + x ) ≤ 2 x for any x ≥ 0 , and finally 0 ≤ p ≤ 1 / 4 :

<!-- formula-not-decoded -->

## B.2 Proving Theorem 5 and Corollary 2

We establish our lower bound for SCO with sub-exponential noise by establishing a correspondence between Problem 3 and solving Problem 1 given a σ E -ESGO. Specifically, for any random variable X in Problem 3 with ˜ glyph[epsilon1] = 48 glyph[epsilon1] √ log d/R , we design the following convex function whose optimal point is related to E [ X ] ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 13. Denote w := E [ X ] / ‖ E [ X ] ‖ . Given that E [ X ] ≤ L , the function ¯ f defined in (11) has the following properties:

1. ¯ f is convex.
2. ¯ f is minimized at x ∗ = R 2 w .
3. Every glyph[epsilon1] -optimum x of ¯ f satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

5. For any t &gt; 0 and unit vector u ∈ R d , we have

<!-- formula-not-decoded -->

Proof. Proof of (1) : By linearity of expectation,

<!-- formula-not-decoded -->

This is a convex function of x since it is the sum of a linear function and the max of two convex functions.

Proof of (2) : Suppose ‖ x ‖ = 1 . We claim that for any c &gt; R/ 2 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

4. For we have

given that

Since ‖ E [ X ] ‖ ≤ L , the last line holds. Recalling w = E [ X ] / ‖ E [ X ] ‖ , notice that if x satisfies x glyph[latticetop] w ≥ 0 , we can decrease the function value by increasing the norm of x up to the value of R/ 2 . That is, for any c &lt; R/ 2 we have

<!-- formula-not-decoded -->

Therefore, since the optimal x satisfies that x glyph[latticetop] w ≥ 0 , we must have x ∗ = ( R/ 2) w .

Proof of (3) : Given any glyph[epsilon1] -optimum x , since we must have that x glyph[latticetop] w ≥ 0 , it follows by Eq. 15 that R 2 · x ‖ x ‖ is also an glyph[epsilon1] -optimum. Therefore,

<!-- formula-not-decoded -->

Since x ∗ = ( R/ 2) w we equivalently have

<!-- formula-not-decoded -->

Using that ‖ w ‖ 2 = 1 and rearranging terms we get

<!-- formula-not-decoded -->

Proof of (4) : By Eq. 14 and linearity of expectation we have

<!-- formula-not-decoded -->

Proof of (5) : For any x , ˆ g X ( x ) takes the form

<!-- formula-not-decoded -->

where c 1 = -1 3 and y = 2 L 3 · 1 { ‖ x ‖ -R 2 &gt; 0 } · x ‖ x ‖ . Then by Lemma 10 we can conclude that

<!-- formula-not-decoded -->

for any t &gt; 0 and any unit vector u ∈ R d .

The next lemma establishes a lower bound for Problem 1 with access to actual gradients, or equivalently a σ E -ESGO with σ E = 0 .

Lemma 14 (Theorem 3 of [24]) . Any algorithm that solves Problem 1 with success probability at least 2 / 3 must make at least Ω(min { RL/glyph[epsilon1] 2 , d } ) queries.

Theorem 5. Any algorithm which solves Problem 1 with probability at least 2 / 3 using a σ E -ESGO makes at least ˜ Ω( R 2 σ 2 E /glyph[epsilon1] 2 +min { R 2 L 2 /glyph[epsilon1] 2 , d } ) queries.

Proof. Consider the function ¯ f ( · ) defined in Eq. 11 with the random variable X ∼ P v defined in Eq. 6, where v is chosen uniformly at random from {± 1 } d . Let x denote the output of the algorithm after making n queries. If x is an glyph[epsilon1] -optimal point, then by Lemma 13,

<!-- formula-not-decoded -->

Using the fact that, for any unit-norm vectors u and v , ‖ u -v ‖ 2 = √ 2 (1 -u glyph[latticetop] v ) , we have

<!-- formula-not-decoded -->

For X defined in Eq. 6 we know that ‖ E [ X ] ‖ = 4˜ glyph[epsilon1] √ log d , regardless of the value of v . Therefore, defining ˆ X := ‖ E [ X ] ‖ ‖ x ‖ · x , and set glyph[epsilon1] = ˜ glyph[epsilon1]R 48 √ log d , we have

<!-- formula-not-decoded -->

Next, we observe that we can simulate the responses of the subgradient oracle defined in Eq. 13 using only n i.i.d. samples of X ∼ P v , which is a σ E -ESGO by Lemma 13. Therefore, any algorithm which can output an glyph[epsilon1] -optimal point x with probability at least 2 / 3 can be used to construct a meanestimation algorithm which outputs an estimate ˆ X that satisfies ∥ ∥ ˆ X -E [ X ] ∥ ∥ ≤ ˜ glyph[epsilon1] with probability at least 2 / 3 . Therefore by the hardness of mean estimation with isotropic noise, established in Lemma 11, we must have that

<!-- formula-not-decoded -->

Combined with Lemma 14, we obtain the desired result.

Finally, we obtain our lower bound for Problem 1 given a ( σ I , δ ) -ISGO as a simple corollary.

Corollary 2. Any algorithm which solves Problem 1 with probability at least 2 / 3 using a ( σ I , δ ) -ISGO makes at least ˜ Ω( R 2 σ 2 I / ( glyph[epsilon1] 2 log 2 (1 /δ )) + min { R 2 L 2 /glyph[epsilon1] 2 , d } ) queries.

Proof. By Lemma 8, any σ E -ESGO is a ( σ E log(2 /δ ) , δ ) -ISGO. The result then follows by setting σ E ← σ I / log(2 /δ ) and applying Theorem 5.

## C Quantum isotropifier and an improved bound for quantum SCO

## C.1 Qubit notation and conventions.

We use the notation |·〉 to denote input or output registers composed of qubits that can exist in superpositions . Specifically, given m points x 1 , . . . , x m ∈ R d and a coefficient vector c ∈ C m such that ∑ i ∈ [ m ] | c i | 2 = 1 , the quantum register could be in the state | ψ 〉 = ∑ i ∈ [ m ] c i | x i 〉 , which represents a superposition over all m points simultaneously. Upon measuring this state, the outcome will be x i with probability | c i | 2 . Moreover, to characterize a classical probability distribution p over R d in a quantum framework, we can prepare the quantum state ∫ x ∈ R d √ p ( x )d x | x 〉 , which we denote as the state over R d with wave function √ p ( x ) . Measuring this state will yield outcomes according to the probability density function p . When applicable, we use | garbage( · ) 〉 to denote possible garbage states. 10

Throughout this paper, we assume that any quantum oracle O is a unitary operation, and we can also access its inverse O -1 satisfying O -1 O = OO -1 = I . This is a standard assumption in prior works on quantum algorithms, see e.g. [22, 53].

## C.2 Bounded random variables

In this subsection, we introduce our quantum multivariate mean estimation algorithm for bounded random variables whose error is small in any direction with high probability. This algorithm is a variant of [22, Algorithm 2]. We begin by presenting some useful algorithmic components.

Lemma15 (Directional mean oracle, Proposition 3.2 of [22]) . Suppose we have access to the quantum sampling oracle O X of a bounded random variable X satisfying ‖ X ‖ ≤ 1 . Then for any ν &gt; 0 , there exists two procedures, QDirectionalMean1 ( X,m,α,ν ) and QDirectionalMean2 ( X,m,α,ν ) ,

10 The garbage state is the quantum counterpart of classical garbage information generated when preparing a classical random sample or a classical stochastic gradient, which, in general, cannot be erased or uncomputed. In this work, we consider a general model without any assumptions about the garbage state. See e.g., [27, 53] for a similar discussion on the standard use of garbage quantum states.

Throughout this paper, whenever we query a quantum oracle that contains a garbage state, we do not assume we know its identity. Nevertheless, our algorithm requires that the garbage state be maintained coherently as part of the system to perform the inverse operation.

<!-- formula-not-decoded -->

that respectively uses ˜ O ( m √ E | X | log 2 (1 /ν )) and ˜ O ( m log 2 (1 /ν )) queries, and output quantum states

<!-- formula-not-decoded -->

where and

<!-- formula-not-decoded -->

Using QDirectionalMean β ( · ) as a subroutine, [22] develops a quantum multivariate mean estimation algorithms for bounded random variables. In this work, we provide an improved error analysis of this algorithm, based on which we apply the boosted unbiased phase estimation technique introduced in [57] to suppress the bias in multivariate mean estimation.

Lemma 16 (Boosted Unbiased Phase Estimation, Theorem 28 of [57]) . Given k copies of a quantum state | ψ 〉 = 1 √ m ∑ g ∈ G m e imφg | g 〉 for some unknown phase φ satisfying -2 π 3 ≤ φ ≤ 2 π 3 , there is a procedure BoostedUnbiasedPhaseEstimation ( | ψ 〉 ⊗ k ) that returns an unbiased estimate ˆ φ satisfying

<!-- formula-not-decoded -->

Proposition 2. For any bounded random variable X satisfying ‖ X ‖ ≤ 1 and min X =0 ‖ X ‖ ≥ glyph[epsilon1] , Algorithm 2 and its output ˆ µ satisfy the following:

1. ‖ ˆ µ ‖ ≤ √ d .
2. ˆ µ is almost unbiased, i.e., ‖ E [ˆ µ ] -E [ X ] ‖ ≤ ˜ O ( δ √ d ) .
3. For any coordinate j ∈ [ d ] , we have

<!-- formula-not-decoded -->

4. For any unit vector u ∈ R d , we have

<!-- formula-not-decoded -->

5. Algorithm 2 uses ˜ O (√ E ‖ X ‖ log 3 (1 /δ ) /glyph[epsilon1] ) queries to O X when β = 1 and ˜ O ( log 3 (1 /δ ) /glyph[epsilon1] ) queries when β = 2 .

Proof. By Lemma 15, the quantum state | ψ out 〉 in Line 4 is defined on G d m . Hence, we have ‖ ˆ µ ‖ ∞ ≤ 1 and ‖ ˆ µ ‖ ≤ √ d . Moreover, it satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We first analyze the algorithm with | ψ out 〉 replaced by | ψ prod 〉 in Line 4. Note that | ψ prod 〉 is a product state, so the outcomes ˆ φ 1 , . . . , ˆ φ d of BoostedUnbiasedPhaseEstimation follows a where

<!-- formula-not-decoded -->

glyph[negationslash]

Algorithm 2: Bounded quantum mean estimation with boosted unbiased phase estimation ( Debiased-QBounded )

Input: Random variable X , target accuracy 0 &lt; glyph[epsilon1] ≤ 1 , failure probability δ ≤ 1 2 , choice of subroutines β = 1 or 2

Output: An estimate ˆ µ of E [ X ]

- 1 Set glyph[epsilon1] ′ ← glyph[epsilon1]/ 8 , n ← √ E ‖ X ‖ log( d/δ ) /glyph[epsilon1] ′

<!-- formula-not-decoded -->

- 3 for k = 1 , . . . , glyph[ceilingleft] 18 log(1 /δ ) glyph[ceilingright] do
- 4 | ψ ( k ) out 〉 ← QDirectionalMean β ( X,m,α,ν )
- 5 Run BoostedUnbiasedPhaseEstimation on all the d coordinates independently and denote ˆ φ 1 , . . . , ˆ φ d to be the outcomes
- 6 return ˆ µ ← 2 π α ( ˆ φ 1 , . . . , ˆ φ d ) glyph[latticetop]

product distribution. Hence, in this ideal case ˆ µ also follows a product distribution, which we denote as

<!-- formula-not-decoded -->

Given that the phase in | ψ prod 〉 satisfies | α E [ X ] | ∞ ≤ 2 π/ 3 , by Lemma 16, we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Thus, for any unit vector u ∈ R d , the random variable

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

by Lemma 22 where we set k = d .

Next, we discuss the error caused by the difference between | ψ prod 〉 and the actual state | ψ out 〉 , i.e., the |⊥ β 〉 term in Eq. 19. By Lemma 15, we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

given the choice of parameters of α, m, γ in Algorithm 2. By Corollary 9, we have

<!-- formula-not-decoded -->

satisfies

and and

<!-- formula-not-decoded -->

By Lemma 15, when β = 1 the number of queries to O X is

<!-- formula-not-decoded -->

For β = 2 , the number of queries is

<!-- formula-not-decoded -->

## C.3 Unbounded random variables with bounded expectation

In this subsection, we introduce our quantum multivariate mean estimation algorithm for unbounded random variable with bounded expectation, a variant of [22, Algorithm 2], obtained by applying Algorithm 2 to a series of truncated bounded random variables.

## Algorithm 3: Unbounded quantum mean estimation ( QUnbounded )

Input: Random variable X , target accuracy 0 &lt; glyph[epsilon1] ≤ 1 , failure probability δ Output: An estimate ˆ µ of E [ X ] with glyph[lscript] ∞ error at most glyph[epsilon1] √

- 1 Set σ ′ ← σ/ log( σ/glyph[epsilon1] ) , glyph[epsilon1] ′ ← glyph[epsilon1]/ ( σ ′ log(1 /δ )) , K ← ⌈ 2 log ( 2 2 /glyph[epsilon1] ′ )⌉
- 2 Take glyph[ceilingleft] 64 log 2 (1 /glyph[epsilon1] ) log( d/δ ) glyph[ceilingright] classical random samples X 1 , . . . , X glyph[ceilingleft] 64 log 2 (1 /glyph[epsilon1] ) log( d/δ ) glyph[ceilingright] , use η to denote their coordinate median
- 3 Define a new random variable Y ← X -η

```
4 a -1 ← 0 5 for k = 0 , . . . , K do 6 a k ← 2 k σ ′ 7 Define the bounded random variable Y k := Y a k · I { a k -1 ≤ ‖ Y ‖ < a k } 8 if k = 0 then ˆ µ ′ k ← Debiased-QBounded2 ( Y k , 2 -k -1 glyph[epsilon1] ′ /K,δ/ ( Kd )) 9 else ˆ µ ′ k ← Debiased-QBounded1 ( Y k , 2 -k -1 glyph[epsilon1] ′ /K,δ/ ( Kd )) 10 if ‖ ˆ µ ′ k ‖ ≤ 2 -2 k +2 then ˆ µ k ← ˆ µ ′ k 11 else ˆ µ k ← 0
```

<!-- formula-not-decoded -->

Lemma 17 (Theorem 2 of [44]) . For any n independent samples X 1 , . . . , X n of a random variable X ∈ R d , any δ &gt; 0 , and any n ≥ glyph[ceilingleft] 32 log( d/δ ) glyph[ceilingright] , their coordinate-wise median η satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which leads to ‖ |⊥〉 1 ‖ , ‖ |⊥〉 2 ‖ ≤ δ/ 2 . Hence, the actual probability distribution ˆ p of ˆ µ satisfies

<!-- formula-not-decoded -->

Then, we can derive that

<!-- formula-not-decoded -->

given that ‖ ˆ µ ‖ ≤ √ d and ‖ X ‖ ≤ 1 . Moreover, we have

<!-- formula-not-decoded -->

Proposition 3. For any glyph[epsilon1],δ &gt; 0 and any random variable X ∈ R d with variance Var[ X ] ≤ σ 2 , Algorithm 3 outputs an estimate ˆ µ satisfying E [ˆ µ ] -E [ X ] ≤ σ/ log(1 /glyph[epsilon1] ) and

<!-- formula-not-decoded -->

for any unit vector u ∈ R d . Moreover, Algorithm 3 makes ˜ O ( σ log 5 (1 /δ ) /glyph[epsilon1] ) queries to O X .

Proof. Denote ˆ µ Y = ∑ K k =0 a k ˆ µ k . We first consider the case that ‖ X -η ‖ ≤ σ ′ , which happens with probability at least 1 -δ by Lemma 17. Under this condition, we have

<!-- formula-not-decoded -->

by Proposition 2. Additionally, we have

<!-- formula-not-decoded -->

and

Therefore,

<!-- formula-not-decoded -->

From this, we deduce that if

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

holds for all unit vectors u ∈ R d , we have | ˆ µ k | ≤ 2 -2 k +2 and ˆ µ ′ k = ˆ µ k . Consequently,

<!-- formula-not-decoded -->

by union bound. Furthermore, we have

<!-- formula-not-decoded -->

which leads to

<!-- formula-not-decoded -->

Note that ‖ η ‖ ≤ ‖ E [ X ] ‖ + σ ≤ L + σ in this case, we have ˆ µ -E [ X ] = ˆ µ Y -E [ Y ] . Thus,

<!-- formula-not-decoded -->

Counting in the error probability when ‖ η -E [ X ] ‖ ≥ σ ′ , we have

<!-- formula-not-decoded -->

As for the bias of ˆ µ , we have

<!-- formula-not-decoded -->

Next, we discuss the query complexity of Algorithm 3. The number of queries in the iteration k = 0 is

<!-- formula-not-decoded -->

and the number of queries in the k -th iteration for k &gt; 0 is

<!-- formula-not-decoded -->

Combining with the number of classical samples to obtain µ , we can conclude that the total number of queries equals

<!-- formula-not-decoded -->

## C.4 Removing the bias

In this subsection, we combine Algorithm 3 with the multi-level Monte Carlo (MLMC) technique to obtain an unbiased estimate of an unbounded random variable whose error is small in any direction with high probability.

## Algorithm 4: Quantum Isotropifier

Input: Random variable X , target accuracy glyph[epsilon1] , failure probability δ

Output: An unbiased estimate ˆ µ of E [ X ]

- 1 Define β j := 2 -j j 2 , ∀ j ∈ N 2 Set ˆ µ (0) ← QUnbounded ( X,glyph[epsilon1]/ 6 , δ ) 3 Randomly sample j ∼ Geom ( 1 2 ) ∈ N 4 ˆ µ ( j ) ← QUnbounded ( X,β j glyph[epsilon1]/ 6 , δ ) 5 ˆ µ ( j -1) ← QUnbounded ( X,β j -1 glyph[epsilon1]/ 6 , δ ) 6 ˆ µ ← ˆ µ (0) +2 j (ˆ µ ( j ) -ˆ µ ( j -1) ) 7 return ˆ µ

Theorem 8. For any glyph[epsilon1],δ &gt; 0 and any random variable X ∈ R d with variance Var[ X ] ≤ σ 2 , the output ˆ µ of Algorithm 4 satisfies E [ˆ µ ] -E [ X ] = 0 and

<!-- formula-not-decoded -->

for any unit vector u ∈ R d . Moreover, Algorithm 4 makes ˜ O ( σ log 5 (1 /δ ) /glyph[epsilon1] ) queries to O X in expectation.

Proof. The structure of our proof is similar to the proof of Theorem 4 of [53]. Note that the output ˆ µ of Algorithm 4 can be written as

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

For each j ∈ N and ˆ µ ( j ) , we denote ξ ( j ) = ˆ µ ( j ) -E [ X ] . Then,

<!-- formula-not-decoded -->

where by Proposition 3 we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Similarly, we have

<!-- formula-not-decoded -->

which gives P [ |〈 u, ˆ µ -E [ X ] 〉| ≥ glyph[epsilon1] log 2 (8 /δ ) ] ≤ ˜ O ( δ ) . Moreover, the number of queries of Algorithm 4 equals

<!-- formula-not-decoded -->

by Proposition 3.

## C.5 An improved bound for quantum SCO

In this subsection, we apply Algorithm 4 to obtain an ISGO using queries to a QVSGO, and then solve SCO using the stochastic cutting plane method developed in Section A.

Theorem 6. For any differentiable f : R d → R , a ( σ I , δ ) -ISGO of f can be implemented using ˜ O ( σ V √ d log 7 (1 /δ ) /σ I ) queries to a σ V -QVSGO.

Proof. For any x ∈ R d , it suffices to apply QuantumIsotropifier (Algorithm 4) to the QVSGO at x . In particular, by Theorem 8, there exists some ˆ δ = ˜ O ( δ ) such that the output of QuantumIsotropifier ( σ I d -1 / 2 / log 2 (8 / ˆ δ ) , ˆ δ ) is an ( σ I , δ ) -ISGO, and the number of queries equals ˜ O ( σ V √ d log 7 (1 / ˆ δ ) /σ I ) = ˜ O ( σ V √ d log 7 (1 /δ ) /σ I ) in expectation.

Theorem 7. With success probability at least 2 / 3 , Problem 2 can be solved using ˜ O ( dRσ V /glyph[epsilon1] ) queries to a σ V -QVSGO.

Proof. The proof is established by combining Theorem 1 and Theorem 6, where we set σ I = glyph[epsilon1] √ d .

## D Sub-exponential distributions and additional discussion of SGO oracles

In this section, we review sub-exponential distributions and also further discuss the relationships between the various SGOs we define.

Review of sub-exponential distributions. As there are several equivalent ways to define a subexponential random variable [61, Proposition 2.7.1], we will use the following 'tail-inequality' version which suits our purposes:

Definition 11. A random variable X ∈ R is σ -sub-exponential if

<!-- formula-not-decoded -->

Analogously to the definition of a sub-Gaussian random vector (see, e.g., Definition 3.4.1 in [61] or Definition 2 in [35]), we say a random vector X ∈ R d is sub-exponential if all of the one-dimensional marginals are sub-exponential random variables:

Definition 12. A random vector X ∈ R d is σ -sub-exponential if for any unit vector u ∈ R d , we have that 〈 X,u 〉 is σ -sub-exponential, namely:

<!-- formula-not-decoded -->

It is well known that sub-exponential distributions generalize sub-Gaussian distributions and therefore bounded random variables in particular. Finally, we prove a short lemma which we will reference below:

Lemma 18. If X ∈ R d is σ -sub-exponential, then E ‖ X -E X ‖ 2 ≤ Cdσ 2 for some absolute constant C .

Proof. Letting e i denote the i -th standard basis vector, we have

<!-- formula-not-decoded -->

where the last equality follows because if a random variable Z ∈ R is σ -sub-exponential, then E ( Z -E Z ) 2 ≤ Cσ 2 [61, Proposition 2.7.1].

Additional discussion of SGO oracles. Note that a σ E -ESGO as defined in Definition 5 is σ E / √ d -sub-exponential per Definition 5 as opposed to σ E -sub-exponential. We perform this scaling so that, per Lemma 18, a σ E -ESGO is also a Cσ E -VSGO for some absolute constant C . In other words, this scaling makes it so that Definition 5 is truly a restriction of Definition 3, and thus the rates of Corollaries 3 and 4 are comparable.

## E Technical lemmas

In this section, we collect some miscellaneous technical lemmas. The following lemma from [22] shows that for any fixed vector x ∈ R d , most of the vectors g ∼ G d m have a relatively small inner product with x .

Lemma 19 (Lemma 3.1 of [22]) . Let α &gt; 0 . For any vector x ∈ R d we have

<!-- formula-not-decoded -->

We extend this result and show that this exponentially small probability bound still holds when x is a random variable instead of a fixed vector.

Lemma 20. Let α &gt; 0 . Consider a d -dimensional random variable Y ∈ R d that satisfies

<!-- formula-not-decoded -->

for some p ( · ) that is a function of α , then for any random variable X ∈ R d , we have

<!-- formula-not-decoded -->

glyph[negationslash]

and

<!-- formula-not-decoded -->

Proof. We first prove Eq. 29 by contradiction. Assume the contrary of Eq. 29, we have

<!-- formula-not-decoded -->

However, by Eq. 28, we have

<!-- formula-not-decoded -->

contradiction.

Then, glyph[negationslash]

E ‖ X j ‖ = P [ X j = 0] · E ‖ ˜ X j ‖ ≥ P [ X j = 0] · max ‖ ˜ X j ‖ 2 , E |〈 Y, X j 〉| = P [ X j = 0] · E |〈 Y, ˜ X j 〉| .

By Eq. 29, we have which leads to

and

<!-- formula-not-decoded -->

by union bound. Combining Eq. 31, we can conclude that

<!-- formula-not-decoded -->

Set we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

glyph[negationslash]

Since the above inequality holds for any α ≥ 0 , we can rescale α by a factor of 8 and conclude that glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

Next, we prove Eq. 30 by applying Eq. 29. Denote ζ := min X =0 ‖ X ‖ . For any ξ &gt; 0 , let k = ⌈ log ( √ E [ ‖ X ‖ 2 ] ζ √ ξ )⌉ and define a j := ζ 2 j for each j ∈ [ k ] . We then define and

which leads to and

<!-- formula-not-decoded -->

given that

<!-- formula-not-decoded -->

For any j = 1 , . . . , k , we define a new random variable ˜ X j that satisfies glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

glyph[negationslash]

and

<!-- formula-not-decoded -->

Proof. The results are obtained by combining Lemma 19 and Lemma 20.

The following lemma establishes that the sum of independent zero-mean sub-Gaussian random variables also follows a sub-Gaussian distribution.

Lemma 21 (Proposition 2.6.1 of [61]) . For any k independent zero-mean sub-Gaussian random variables Y 1 , . . . , Y k with variances σ 2 1 , . . . , σ 2 k , their sum ∑ k j Y j is also sub-Gaussian with variance 8 ∑ k j σ 2 j .

Lemma 22. For any glyph[epsilon1],δ &gt; 0 , u ∈ R k , and k independent random variables Y 1 , . . . , Y k satisfying

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

Proof. Denote Z j := Y j -E [ Y j ] . By Eq. 32, each random variable Z j follows a probability distribution p j that is δ -close to a probability distribution ˜ p j such that max Z j ∼ ˜ p j ‖ Z j ‖ ≤ 2 glyph[epsilon1] log(1 /δ ) . Then, the random variable Z j ∼ ˜ p j follows a sub-Gaussian distribution with variance at most glyph[epsilon1] 2 , and the random variable

<!-- formula-not-decoded -->

is also a sub-Gaussian distribution with variance at most

<!-- formula-not-decoded -->

where C is the absolute constant in Lemma 21, which leads to

<!-- formula-not-decoded -->

Counting in the difference between ˜ p j and the actual distribution p j , we have

<!-- formula-not-decoded -->

glyph[negationslash]

Corollary 9. Let α &gt; 0 . For any random variable X ∈ R d with ‖ X ‖ ≤ 1 and min X =0 ‖ X ‖ ≥ glyph[epsilon1] , we have

<!-- formula-not-decoded -->