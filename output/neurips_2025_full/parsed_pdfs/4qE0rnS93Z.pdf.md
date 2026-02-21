## Preference Optimization on Pareto Sets: On a Theory of Multi-Objective Optimization

Abhishek Roy ∗

Texas A&amp;M University abhishekroy@tamu.edu

Geelon So ∗ UC San Diego geelon@ucsd.edu

## Abstract

In multi-objective optimization, a single decision vector must balance the trade-offs across many objectives. Pareto-optimal solutions are those achieving optimal tradeoffs, where improving any objective comes at a cost to another. As many different decisions can be Pareto optimal, this raises the question of which solution to pick and how. We formulate this problem as one of optimizing a preference function over the set of Pareto-optimal solutions, or Pareto-constrained optimization for short. It poses significant challenges: not only is the constraint set defined implicitly, but it is also generally non-convex and non-smooth, even when the objectives are strongly convex. We propose an equivalent formulation of the problem where the constraint set is the simplex, leading to clearer notions of optimality and stationarity that improve upon existing definitions in literature. We give an algorithm with a last-iterate convergence rate of O ( K -1 / 2 ) to stationarity when the preference function is Lipschitz smooth and when the objective functions are strongly convex and Lipschitz smooth. Motivated by applications like Reinforcement Learning with Human Feedback (RLHF), we also extend this algorithm to the case where access to the preference function is only available through dueling feedback.

## 1 Introduction

Modern, large-scale machine learning often draws data from diverse sources, are simultaneously deployed across many settings to perform many tasks, or need to perform well across many different metrics. These learning settings can naturally be formulated as multi-objective optimization (MOO) problems, including multi-task, multi-distribution, meta-learning; multi-calibration, multi-group learning; learning from crowds, or from heterogeneous/multi-fidelity sources; personalization, preference learning; hyperparameter optimization, model fine-tuning and model fusion (Jin, 2007; Sener and Koltun, 2018; Huang et al., 2015; Haghtalab et al., 2022; Ye et al., 2021; Reed et al., 2022; Martinez et al., 2020; La Cava, 2023; Kamani et al., 2021; Globus-Harris et al., 2022; Lee et al., 2022; Tosh and Hsu, 2022; Raykar et al., 2010; Chen et al., 2025a). As a result, MOO has increasingly attracted the interest of the learning community. But, in contrast to the single-objective case, far less has been established theoretically and algorithmically for the multi-objective setting-even very basic questions may not have rigorous definitions or solutions. We formalize and tackle such a problem.

To give an intuitive problem description, consider the concrete example of large language model (LLM) alignment (Houlsby et al., 2019; Liu et al., 2022; Ding et al., 2023). In this problem, we aim to finetune LLM outputs to achieve a number of desiderata: positivity of the tone, succinctness of the answer, consistency with lexical conventions, and so on. While no model is generally optimal under all criteria, it is Pareto optimal if it makes an optimal trade-off between the objectives. We would like our decision to at least be Pareto optimal, as this means no other decision was strictly better by all metrics. Many different trade-offs can usually be made; we call the set of all such decisions the

∗ Equal contribution.

Yi-An Ma UC San Diego yianma@ucsd.edu

Pareto set . Eventually, as we must single out one from this set to deploy, we ask how to select the most preferred Pareto-optimal solution. That is, out of the many different possible trade-offs that can be made, which one do we choose and how do we discover it?

There are two main frameworks to making this selection (Hwang and Masud, 2012). The first is through scalarization , where multiple objectives are aggregated into one. Thus, it reduces the problem back to the familiar single-objective setting (Mahapatra and Rajan, 2020; Lin et al., 2024). The second, indirect approach is to find a representative subsample of the Pareto set; this helps a decision maker by paring down the number of solutions that need to be inspected (Lin et al., 2019; Liu et al., 2021; Kobayashi et al., 2019; Guerreiro et al., 2021).

Neither approach quite fully answers to how to select the final trade-off. It might not be clear how to meaningfully scalarize a multi-objective problem, especially when the objectives are incomparable. And while subsampling can reduce the complexity of the problem, the actual decision is left open. For classical settings, these gaps may not be crucial. But modern, large-scale decision-making settings can be high-dimensional, incorporate many objectives, or have high-throughput and need to be automatic. It may be impossible to scalarize the problem by hand, and even a representative subsample of the Pareto set can become untenably large, scaling exponentially with the number of objectives (Papadimitriou and Yannakakis, 2000). There is a need for a more principled selection, which motivates the question we ask:

Given a set of objectives ( f 1 , . . . , f n ) and a preference f 0 , how do we define a suitable notion of the most preferred Pareto solution, and how do we efficiently approximate it?

The preference-based formulation of multi-objective optimization, with the inclusion of a preference function f 0 , refines more common versions of MOO given in standard references (e.g., Ehrgott (2005)). While the objectives f 1 , . . . , f n and the preference f 0 are mathematically the same type of objects, the latter can conceptually represent the more ineffable desiderata that human decision-makers or users may have, as in the earlier example about LLM alignment. For this reason, we also consider learning from dueling preference feedback in this paper. In that setting, instead of direct access to f 0 , we can only ask users to rank different options x according to their preferences f 0 ( x ) .

Besides LLM finetuning, several standard problems in machine learning also fit this preference-based framework of MOO. In fairness-aware learning, the objectives can represent subgroup utilities, while the preference f 0 is a social welfare function. In neural architecture search, objectives can be accuracy, latency, and energy consumption while f 0 can be user-specific preferences reflecting their priorities. In portfolio optimization, objectives could be return, risk, and liquidity, while f 0 might capture the risk aversion tendency of an investor. We now formalize the problem mathematically.

## 1.1 Pareto-Constrained Optimization

Let F ≡ ( f 1 , . . . , f n ) : R d → R n be a set of n objectives that are jointly minimized over a shared decision space R d and let Pareto( F ) be the set Pareto-optimal solutions, consisting of decision vectors x ∈ R d that make an optimal trade-off between objectives. As it is often not clear a priori which trade-off to make, we consider the Pareto-constrained optimization problem , where the aim is to optimize a preference function f 0 : R d → R constrained to the Pareto set of F :

<!-- formula-not-decoded -->

and we call a solution x of this optimization problem preference optimal . This problem is also called semivectorial bilevel optimization or optimization on efficient sets , and it can be considered as an instantiation of bi-level optimization (Bolintinéanu, 1993b; Yamamoto, 2002; Bonnel and Morgan, 2006; Dempe, 2018; Ye and Liu, 2022). It has applications for economics, portfolio management, manufacturing planning, and machine learning (Thach et al., 1996; Yamamoto, 2002; Ye and Liu, 2022). While heuristics have been proposed, little more is known. In fact, even notions such as stationarity have not been formalized, which is generally needed to study convergence.

Main Challenges There are major challenges to this problem.

1. Theoretical Challenges. The Pareto set is generally non-smooth and non-convex. It can have 'needle-like extensions' and 'knees' (Kulkarni et al., 2022), or 'singularities' (Sheftel et al., 2013). These are possible even when the objectives are quadratics (see Figure 1). Moreover,

in the setting where both objectives and preferences are linear, the problem is known to be NP-hard (Fülöp, 1993). Thus, there is a need for appropriate relaxations of the problem that are algorithmically attainable. But even defining a reasonable notion of stationarity is not straightforward, given the complicated nature of the constraint set (Zhang et al., 2020; Kornowski and Shamir, 2021; Li et al., 2020; Jordan et al., 2023; Kornowski et al., 2024).

2. Algorithmic Challenges. The Pareto set is non-smooth, non-convex, and an implicitly defined object. It is not clear how to analytically parameterize Pareto( F ) or to specify Pareto( F ) as a feasible set of a system of inequalities. Pareto-constrained optimization (1) becomes even more challenging when we do not have access to the preference function f 0 , but only preference comparisons between two decision vectors, as is the case of RLHF for LLM alignment.

## 1.2 Main Results

We consider Pareto-constrained optimization with strongly convex and smooth objectives f 1 , . . . , f n , and smooth, but potentially nonconvex preference function f 0 . We list our main contributions below.

1. We introduce the Pareto manifold of F (Definition 3), a ' lifting ' of the Pareto set, that recovers the Pareto set when projected down to R d . We show that the Pareto manifold is a smooth manifold diffeomorphic to the ( n -1) -simplex. This leads to a clear notion of stationarity.
2. We use the connection with the simplex to introduce an (approximate) stationarity condition (Definitions 3 and 5) for the Pareto-constrained optimization. We show that any non-trivial, local stationarity condition requires more than local first-order information about F .
3. We propose the Pareto Majorization-Minimization algorithm (Algorithm 1), which converges to an ( ε 0 , ε ) -approximate preference stationary point of Pareto-constrained optimization with iteration complexity ˜ O ( ε -2 0 ) , ignoring logarithmic factors, under first-order feedback ∇ f 0 (Theorem 10), and noisy dueling preference feedback (Theorem 11).

## 1.3 Related Work

Selecting the most-preferred decision vector out of the Pareto set is a classical problem in MOO (see Bolintinéanu (1993b) and related works therein for classical motivation), and it has also gained renewed interest from the machine learning community. However, it is well-established to be challenging (e.g. Fülöp (1993)), and most prior work studying the Pareto-constrained optimization problem have largely focused on (i) linear preferences (Philip, 1972; Benson, 1984; Liu and Ehrgott, 2018), (ii) linear objectives (Dauer, 1991; Bolintinéanu, 1993a; Tao et al., 1996; Yamamoto, 2002), or (iii) specific choices of preference functions (Steuer, 1989; Mahapatra and Rajan, 2020).

To our knowledge, there are only two prior works that have studied the problem more generally in the nonlinear setting. The first work is Bolintinéanu (1993b), which considers a regularized version of the problem: they balance the preference f 0 with a penalty term capturing the Pareto set of F . While regularized solutions are suboptimal for the original problem, they show that these solutions asymptotically become optimal as the weight on the penalty term goes to infinity. They also describe a necessary condition for the regularized solution, which can be approximated via nonlinear programming. The second work is Ye and Liu (2022), which provides a heuristic for the same problem based on a similar but distinct penalty function. They also propose a stationary condition-stationarity with respect to their optimization dynamics. But, this turns out to not be a necessary condition, meaning that their dynamics can actively avoid optimal points (see Appendix D).

These two conditions in both papers entangle independent aspects of an ideal solution: (a) being close to preference optimal and (b) being close to the Pareto set. As a result, these conditions can seem somewhat opaque. In this work, we first clarify the manifold structure of the Pareto set under strong convexity of the objectives. This enables us to keep these two aspects disentangled, and to derive standard, necessary relaxations of preference optimality. While existing work have studied the smoothness structure of the Pareto set (Hillermeier, 2001a,b; Hamada et al., 2020), the prior focus has been on extrinsic smoothness within the decision space. We instead leverage the intrinsic geometry of the Pareto manifold, which is diffeomorphic to the simplex, to simplify optimization.

The algorithm we propose along with its analysis draw on majorization-minimization and trust-region ideas to handle the implicit nature of the problem (Lange et al., 2000; Marumo et al., 2023). This algorithm can make use of both first-order preference information or dueling feedback. For dueling

feedback, we work under a standard preference learning model (see Section 6.2) from psychology, statistics, and also more recent learning literature (Bradley and Terry, 1952; Agresti, 2012; Wang et al., 2023). For this setting, we also make use of ideas from derivative-free or zeroth-order optimization (Jamieson et al., 2012; Saha et al., 2025; Cai et al., 2022). In terms of guarantees, we provide the first finite-time convergence result to approximate preference stationarity. Additionally, we provide non-asymptotic guarantees on the suboptimality of approximately preference-optimal solutions, which parallel the asymptotic guarantee of Bolintinéanu (1993b).

Following an earlier version of this article shared on arXiv, 2 Chen et al. (2025b) developed results for the Pareto-constrained optimization problem that goes beyond strictly-convex objectives.

Organization In Section 2 we discuss some preliminaries on Pareto set. In Section 3, we introduce the Pareto manifold. In Section 4, we define preference stationarity. In Section 5, define approximate preference stationarity and formalize our assumptions. In Section 6, we present Algorithm 1, which solves Pareto-constrained optimization under two feedback models: (a) access to a first-order feedback, and (b) access to preference comparisons. In Section 7, we provide rates of convergence. For a detailed glossary, see Appendix A. Full proofs are provided in the appendix.

## 2 The Pareto Set

From multi-objective optimization, recall that a decision x is Pareto optimal if there is no way to improve any one f i without also worsening some other f j . Formally:

Definition 1 (Pareto optimality) . Given objectives f 1 , . . . , f n , we say that a decision vector x ∈ R d is Pareto optimal if for all x ′ ∈ R d :

<!-- formula-not-decoded -->

The set of Pareto optimal decision vectors of f 1 , . . . , f n forms the Pareto set , denoted Pareto( F ) .

For smooth objectives, a related, first-order condition called Pareto stationarity is necessary for Pareto optimality Maru¸ sciac (1982). In the following, let ∆ n -1 denote the ( n -1) -simplex, and for all β ∈ ∆ n -1 , let f β denote the linear scalarization :

<!-- formula-not-decoded -->

Definition 2 (Pareto stationarity) . Given objectives f 1 , . . . , f n , we say that a decision x ∈ R d is Pareto stationary if ∇ f β ( x ) = 0 for some β ∈ ∆ n -1 .

When the objectives f 1 , . . . , f n are twice-differentiable and strictly convex, Pareto stationarity is also necessary for Pareto optimality Fliege et al. (2009). We shall assume throughout that the objectives are smooth and strongly convex (Assumption A), so these two notions coincide. Given these definitions, optimization constrained to the Pareto set as defined in (1) is highly non-trivial: the Pareto set remains implicit, and it is generally non-smooth and non-convex. The following gives the example in Figure 1.

Example 1 (A singular Pareto set arising from strongly convex objectives) . Define three positivedefinite quadratic objectives f i : R 2 → R of the form f i ( x ) = 1 2 ( x -x i ) ⊤ A i ( x -x i ) ,

<!-- formula-not-decoded -->

Even in this highly-structured setting where all objectives are strongly convex quadratics, the Pareto set is non-convex and has a singularity. This example is visualized in Figure 1.

## 3 Lifting to the Pareto Manifold

We can overcome the issue of non-smoothness of the Pareto set by lifting the problem into a higherdimensional space R d × ∆ n -1 , which contains what we call the Pareto manifold . This is the set of all tuples ( x, β ) such that x is a Pareto stationary point of the linearly scalarized objective f β .

Definition 3 (Pareto manifold) . The Pareto manifold P ( F ) ⊂ R d × ∆ n -1 is the zero set:

<!-- formula-not-decoded -->

2 Earlier title: Optimization on Pareto Sets: On a Theory of Multi-Objective Optimization (Roy et al., 2023).

Figure 1: The Pareto set of the three positive-definite quadratic objectives given by Example 1. (Left) The 2-simplex parametrizes all scalarizations f β . (Right) The Pareto set in R 2 . For each β ∈ ∆ 2 , the minimizer x ∗ ( β ) of f β is Pareto optimal; the colors preserve this correspondence.

<!-- image -->

In the case of strictly convex objectives, the projection of the Pareto manifold P ( F ) to its first component in the decision space R d precisely yields the Pareto set, as shown by Lemma 1. As this projection can also collapse any smoothness structure that P ( F ) has, it becomes clear how non-smoothness arises on the Pareto set. On the other hand, Proposition 2 shows that it is a smooth submanifold of R d × ∆ n -1 diffeomorphic to the ( n -1) -simplex.

Lemma 1. Let F ≡ ( f 1 , . . . , f n ) be a collection of smooth and strictly convex objectives. Then:

<!-- formula-not-decoded -->

Proposition 2 (Characterization of the Pareto manifold) . Let F ≡ ( f 1 , . . . , f n ) be a collection of smooth and strictly convex objectives. Define x ∗ : ∆ n -1 → Pareto( F ) by:

<!-- formula-not-decoded -->

Let ∇ F ( x ) ∈ R n × d be the Jacobian. Then, the map x ∗ has derivative:

<!-- formula-not-decoded -->

so that the map β ↦→ ( x β , β ) is a diffeomorphism of ∆ n -1 with the Pareto manifold P ( F ) .

The main tool used is the implicit function theorem; we defer the full proofs to Appendix B.

## 4 Solution Concept: Preference Stationarity

The tight connection between the Pareto set and the Pareto manifold given by Lemma 1 allows us to smoothly lift the preference optimization problem (1) onto the Pareto manifold, as follows:

<!-- formula-not-decoded -->

And as Proposition 2 shows, the Pareto manifold is diffeomorphic to the simplex, so (5) further reduces to the following smooth optimization over the convex set ∆ n -1 .

<!-- formula-not-decoded -->

This is an equivalent formulation: if β solves (6), then x ∗ ( β ) solves (1) and (5). Furthermore, as the constraint set is the simplex, we can appeal to convex optimization for the following notion of preference stationarity , which is necessary for preference optimality (Nesterov, 2003).

Definition 3 (Preference stationarity) . We say that a point x ∈ Pareto( F ) is weakly preference stationary if there exists some β ∈ ∆ n -1 where ( x, β ) ∈ P ( F ) such that:

<!-- formula-not-decoded -->

with ∇ x ∗ as in (4). 3 If (7) holds for all β where ( x, β ) ∈ P ( F ) , then x is preference stationary .

<!-- formula-not-decoded -->

Proposition 4 (Necessary condition (Lemma 3.1.19 of Nesterov (2003))) . Preference optimality implies (weak) preference stationarity.

It turns out that developing reasonable relaxations of preference optimality without lifting the problem to Pareto manifold is surprisingly subtle. For example, a prior notion of stationarity (e.g., Ye and Liu (2022)) is not necessary for preference optimality (Appendix D.1). Any optimization dynamics that achieve stationarity conditions that are not necessary can actively avoid optimal decisions. We show that any local first-order condition suffers from this limitation (Appendix D).

## 5 Algorithmic Solution Concept: Approximate Preference Stationarity

In first-order optimization, notions of approximate stationarity allow us to provide meaningful, finitetime guarantees about the solutions that are reached by optimization algorithms in practice; likewise, we introduce the following ε -approximate version of preference stationarity. Again, we appeal to a standard notion of approximate solutions Nesterov (2013); Marumo et al. (2023).

Definition 5 (Approximate preference stationarity) . Let ε 0 , ε ≥ 0 . A point ( x, β ) ∈ R d × ∆ n -1 is ( ε 0 , ε ) -preference stationary if:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Intuitively, if (ˆ x, ˆ β ) is approximately preference stationary, then (a) there is a ball around ˆ β within which f 0 ◦ x ∗ decreases at most at an O ( ε 0 ) -rate when moving away from ˆ β , and (b) the point ˆ x is O ( ε ) -close to x ˆ β . We will formalize this geometric interpretation in Proposition 6. To do so, we need to formalize the assumptions, which are standard and used in the rest of the paper.

Assumption A. Let the objectives f 1 , . . . , f n : R d → R be twice differentiable, µ -strongly convex, and have L -Lipschitz continuous gradient. That is, µ I ⪯ ∇ 2 f i ( x ) ⪯ L I for all i = 1 , . . . , n . We also define κ := L/µ and r := max i,j ∈ [ n ] ∥ arg min f i ( x ) -arg min f j ( x ) ∥ 2 .

Assumption B. Let the objectives f 1 , . . . , f n : R d → R have L H -Lipschitz continuous Hessian. That is, for all x, y ∈ R d and i = 1 , . . . , n , we have ∥ ∥ ∇ 2 f i ( x ) -∇ 2 f i ( y ) ∥ ∥ 2 ≤ L H ∥ x -y ∥ 2 .

Assumption C. Let the preference function f 0 : R d → R have L 0 -Lipschitz continuous gradient. That is, for all x, y ∈ R d , we have ∥∇ f 0 ( x ) -∇ f 0 ( y ) ∥ 2 ≤ L 0 ∥ x -y ∥ 2 .

Assumption A allows us to bound the diameter of the Pareto set (Lemma E.1); Assumption B bounds the curvature of the Pareto manifold (Lemma E.2); and all three lead to bounds on approximation errors (Lemma E.5 and Lemma E.6) needed as the constraint set is implicit. They also lead to the following geometric intuition for approximate preference stationarity (see Appendix E for proof).

Proposition 6 (Geometric meaning of approximate stationarity) . Let (ˆ x, ˆ β ) be ( ε 0 , ε ) -preference stationary. With Assumptions A,B,C, and R := diam ( Pareto( F ) ) and s := 2 µ 2 ε 0 / ( L 0 L 2 R 2 ) , then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 5.1 Overcoming Implicitness: Approximating Local Information

The final issue is that these notions of (approximate) preference stationarity are implicit . They depend on ∇ x ∗ ( β ) and x ∗ , which come from solving the optimization problem:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since x β does not generally have a closed form, we cannot exactly compute ∇ x ∗ ( β ) , which is needed to directly check whether the pair (ˆ x, ˆ β ) is (approximately) preference stationary. Since we cannot compute ∇ x ∗ ( β ) exactly, an obvious estimator is the following, which uses local information ∇ 2 f β ( x ) and ∇ F ( x ) at x as a proxy for the corresponding local information at x β :

<!-- formula-not-decoded -->

Algorithm 1 Pareto majorization-minimization (PMM)

Input: objectives F ≡ ( f 1 , . . . , f n ) ; preference function f 0 ; black-box optimizer ̂ arg min ; a family of majorizing surrogates { g ( · ; x, β ) : ( x, β ) ∈ R d × ∆ n -1 } , exact (13) or approximate (16) Initialize: ( x 0 , β 0 ) ∈ R d × ∆ n -1

- 1: for k = 1 , . . . , K do
- 2: Select the (approximate) majorizing surrogate g k ( · ) ← g ( · ; x k , β k )
- 3: Compute approximate minimizers

<!-- formula-not-decoded -->

- 4: end for
- 5: return ( x K +1 , β K +1 )

With the above assumptions, approximate information is enough to verify approximate stationarity. Again, we defer proofs to Appendix E.

Lemma 7 (Verifiability of approximate stationarity) . Under Assumptions A,B,C, the point (ˆ x, ˆ β ) is ( ε 0 , ε ) -preference stationary if ∥∇ f ˆ β (ˆ x ) ∥ 2 ≤ ε , and for some x ∈ R d and α ∈ (0 , 1) ,

- (a) an α · ε 0 -approximate stationary condition holds for all β ′ ∈ ∆ n -1 :

<!-- formula-not-decoded -->

(b) an error bound holds: err ∇ f 0 ( ˆ β, x ) ≤ (1 -α ) · ε 0 , where:

<!-- formula-not-decoded -->

and M 0 = κR , M 1 = 2 κ 2 R (1 + L H R/µ ) .

## 6 Algorithm Design: Pareto Majorization-Minimization

In this section, we present our algorithm to solve (6). While this is conceptually the optimization of a smooth function f ◦ x ∗ over the convex set ∆ n -1 , the main challenge is that the objective is implicit. Methods like gradient descent require both x ∗ and ∇ x ∗ , to compute the gradient via chain rule:

<!-- formula-not-decoded -->

But, if we can estimate x ∗ ( β ) to arbitrary precision by solving the optimization in (3), then ∇ x ∗ ( β ) can also be approximated arbitrarily well: both of its terms, ∇ F ( x ) and ∇ 2 f β ( x ) -1 , are continuous in x and β by assumption. Thus, the estimator ̂ ∇ x ∗ ( x, β ) defined in (10) approaches ∇ x ∗ ( β ) as x approaches x β . We will quantify the validity of the estimator by using it to construct a majorizing surrogate function, a function that upper bounds f 0 ◦ x ∗ , where better estimators yield tighter bounds.

Definition 4 (Majorizing surrogate) . A function g : ∆ n -1 → R majorizes f 0 ◦ x ∗ if:

<!-- formula-not-decoded -->

for all β ′ ∈ ∆ n -1 . We say that g is a (majorizing) surrogate of f .

The following uses the estimator ̂ ∇ x ∗ ( x, β ) to construct a family of quadratic surrogates for f 0 ◦ x ∗ . An error term err ∇ f 0 ( x, β ) appears as a constant in (13), which Lemma E.6 shows goes to zero as ( x, β ) goes to ( x β , β ) . Thus, the estimator becomes more 'valid' with better estimates of x ∗ ( β ) .

Proposition 8 (A family of majorizing surrogates) . Let F and f 0 satisfy Assumptions A,B,C. Let err ∇ f 0 ( x, β ) be as defined in Lemma E.6. Define the family indexed by ( x, β ) ∈ R d × ∆ n -1 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This result is proved in Appendix F. Technically, we still cannot explicitly compute g ( β ′ ; x, β ) because it contains the term f 0 ( x β ) . However, to minimize g using any gradient-based method, we only need ∇ g ( β ′ ; x, β ) , which does not require the knowledge of f 0 ( x β ) nor err ∇ f 0 ( x, β ) .

## 6.1 Minimizing the Majorizing Surrogate with First-Order Preference Information

The first algorithm we describe assumes access to a first-order feedback to f 0 , which allows us to compute the majorizing surrogate g ( β ′ ; x, β ) from Proposition 8. We then directly minimize the surrogate. The idealized Pareto majorization-minimization (PMM) algorithm proceeds in rounds:

1. majorization: query ̂ ∇ x ∗ ( x k , β k ) to construct majorizing surrogate g k ( β ) ≡ g ( β ; x k , β k ) ,
2. minimization: make updates β k +1 ← arg min ∆ n -1 g k ( β ) and x k +1 ← arg min x ∈ R d f β k +1 ( x ) .

The majorizing property of the surrogate ensures that f 0 ( x β k ) improves every iteration. In fact, there is no need to fully optimize g k and f β k +1 ( x ) in the second step. By relaxing this step, we obtain Algorithm 1 when it uses the exact surrogates (13) and any black-box optimizer. In Section 7, we provide a last-iterate convergence rate for Algorithm 1 to an ( ε 0 , ε ) -preference stationary point.

## 6.2 A Model of Dueling Feedback

In the next section, we will extend Algorithm 1 to the dueling feedback setting. Especially in cases of human preferences, exact forms of f 0 or ∇ f 0 are unknown, but we can ask users to rank different options by preference. Such a setting arises, for example, in supervised fine-tuning or RLHF (e.g. Zhu et al. (2023); Ziegler et al. (2019)). We assume a comparison oracle that provides noisy, binary responses to the queries of the form: do you prefer x 1 over x 2 ?

Definition 5 (Preference feedback model) . Given two options x 1 and x 2 , the comparison oracle returns a binary random variable Y ( x 1 , x 2 ) ∈ { 0 , 1 } where:

<!-- formula-not-decoded -->

and σ ( · ) is a link function. A response Y ( x 1 , x 2 ) = 1 indicates that users prefer x 1 over x 2 .

This is a standard model from preference learning, psychology, and RLHF (e.g. Bradley and Terry (1952); Wang et al. (2023); Chen et al. (2025a)), and it aims to capture the phenomenon where preference judgments tend to be noisier when it is difficult to discriminate a clear winner (Nunnally and Bernstein, 1994). The link function σ formalizes the relationship between the 'random component' in the observed responses to the 'systematic component' of the underlying preference values f 0 ( x ) (Nelder and Wedderburn, 1972; Agresti, 2012). For analysis, we make standard assumptions, which are satisfied by all commonly-used link functions including the logistic, softplus, and tanh functions.

Assumption D (Link function) . Let B satisfy B &gt; sup | f 0 ◦ x ∗ | . Let σ ( · ) : [ -B,B ] → (0 , 1) be a known link function that is a smooth, L 1 -Lipschitz, and monotonically increasing function. We assume that σ (0) = 1 2 , and the inverse σ -1 ( p ) is locally L σ (1 + ( p (1 -p )) -1 ) -Lipschitz continuous.

## 6.3 Optimizing Preferences with Dueling Feedback

In Section 6.1, we described a version of PMM using first-order preference information, where the gradient ∇ f 0 ( x ) gives rise to the majorizing surrogate function g ( β ; x k , β k ) from Proposition 8. Under dueling feedback, the gradient is not directly available, but we construct an estimator ˆ ∇ f 0 ( x ) , as described in Algorithm 2. The basic idea is to estimate ∇ f 0 ( x ) by querying preferences between the pair x 1 = x + γU and x 2 = x -γU , where U is a uniform-at-random unit vector and γ is a precision parameter. A single comparison can be seen as an estimate of p U = σ (2 γ ∇ f 0 ( x ) ⊤ U ) . Since we assume knowledge of σ , we can invert this to measure ∇ f 0 ( x ) . This leads to the following family of approximate majorizing surrogate, indexed by ( x, β ) ∈ R d × ∆ n -1 :

<!-- formula-not-decoded -->

This family of approximate surrogates can also be used in Algorithm 1. Of course, it performance depends on the quality of the estimator ˆ ∇ f 0 ( x ) . It also requires a separate analysis, which will use the following bounds on the bias and the variance of ˆ ∇ f 0 ( x ) , proved in Appendix H.1.

Lemma 9 (Bias and variance of dueling gradient estimator) . Under Assumptions A to D, let ˆ ∇ f 0 ( x ) be defined as in (14) with α = 1 / 8 , m ≍ d 4 log( d/ε 0 ) /ε 4 0 , b ≍ d/ε 2 0 , and γ ≍ ε 0 /d . Then:

<!-- formula-not-decoded -->

In the next section, we provide the results on the iteration complexity of Algorithm 1 with first-order feedback (Theorem 10) and dueling feedback (Theorem 11).

Algorithm 2 Approximation of preference gradient with dueling feedback

Input: decision vector x , weights β , clipping threshold α , precision γ , number of batches b , batch size m , and a comparison oracle Y ( x 1 , x 2 ) satisfying (15)

- 1: Sample a unit vector in R d uniformly at random, U i ∼ Unif( S d -1 ) for each i ∈ [ b ]
- 2: Query comparison oracle for Y ij := Y ( x + γU i , x -γU i ) for each ( i, j ) ∈ [ b ] × [ m ]
- 3: Compute clipped Bernoulli parameter estimators for each i ∈ [ b ] and the gradient estimator:

<!-- formula-not-decoded -->

where CLIP [ α, 1 -α ] ( p ) is the projection of p into the interval [ α, 1 -α ] .

- 4: return ˆ ∇ f 0 ( x )

## 7 Convergence Analysis

We establish the convergence rate of Algorithm 1 in terms of the convergence guarantees of its two black-box optimizers: one for the surrogate g ( · ; x, β ) and another for the scalarized objective f β ( · ) . For our convergence results, we assume that the optimizers achieve the guarantees:

Assumption E (Black-box optimizers) . Let ˆ β and ˆ x ˆ β be the approximate solutions that are returned by the black-box optimizers for g ( · ; x, β ) and f ( · ) in (9) of Algorithm 1:

ˆ β

<!-- formula-not-decoded -->

We assume that there are constants c 1 , c 2 &gt; 0 such that:

1. the approximate minimizer ˆ β is O ( ε 0 ) -approximately stationary:

<!-- formula-not-decoded -->

2. the approximate minimizer ˆ x ˆ β is an O ( ε 2 0 ) -approximate solution: ∥∇ f ˆ β (ˆ x ˆ β ) ∥ ≤ c 2 · ε.

Theorem 10 (Convergence of PMM with first-order feedback) . Suppose that F and f 0 satisfy Assumptions A to C, and that the black-box optimizers satisfy Assumption E. Fix 0 &lt; ε 1 / 2 ≤ ε 0 ≤ 1 . Let ( x k , β k ) k be the iterates of Algorithm 1 using the family of surrogates (13) . There exist c 1 ( f 0 , F ) and c 2 ( f 0 , F ) bounded away from zero and a stopping time K such that ( f 0 ◦ x ∗ )( β k ) is monotonically decreasing for k ∈ [ K ] and ( x K , β K ) is an ( ε 0 , ε ) -preference stationary point. Furthermore:

<!-- formula-not-decoded -->

where f ∗ := max f 0 ( x ) and f ∗ = min f 0 ( x ) are optimized over the compact set Pareto( F ) .

Proof sketch. We use a standard approach and show that every iteration, either we have converged to an ε 0 -preference stationary point and can stop, or we can find a way to improve the preference by Ω( ε 2 0 ) . To do so, we require the optimizer for the surrogate g to also achieve O ( ε 0 ) -approximate stationarity, which is enough to achieve an O ( ε 2 0 ) -approximately optimal point (Lemmas G.4 and G.5):

<!-- formula-not-decoded -->

where β ∗ minimizes the surrogate. The surrogate contains an approximation error err ∇ f 0 ( β, x ) . If this error term is Ω( ε 2 0 ) , then it is possible for the surrogate to fail to either (i) decide that the current iterate β is ε 0 -preference stationary or (ii) make progress by finding some ˆ β that certifiably improves on f 0 . We preclude this by requiring the optimizer for f β to achieve O ( ε 2 0 ) -optimality.

The full proof of Theorem 10 is in Appendix G.

Figure 2: Visualization of learning dynamics for two Pareto-constrained optimization problems; ours Algorithm 1 (PMM) is in orange, and the existing method (PNG) introduced by Ye and Liu (2022) is in blue. In both cases, the dynamics begin at the black dot. The ground truth preference-optimal solution, found by exhaustive search, is marked by a black square. The boundary of the Pareto set of F is colored white. The preference function f 0 is visualized by the heatmap and contour lines. All objectives and preferences are strongly-convex quadratics. See Appendix I for details.

<!-- image -->

Remark 1. Algorithm 1 makes calls to sub-routines at each iteration to solve two sub-problems. As the problems are strongly-convex and Lipschitz-smooth, they can be solved using (projected) gradient descent with iteration complexity O (log(1 /ε 0 )) . And so, taking the computational cost of the sub-problems into account only increases the rate obtained in Theorem 10 by logarithmic factors.

Theorem 11 (Convergence of PMM with dueling feedback) . In addition to the assumptions of Theorem 10, suppose that the link function σ satisfies Assumption D. Let the family of approximate surrogates (16) be constructed by Algorithm 2 with input parameters ( α, γ, b, m ) that are specified by Lemma 9. Let ( x k , β k ) k be the iterates of Algorithm 1 using this family of surrogates. There is a stopping time K such that E [( f 0 ◦ x ∗ )( β k )] is monotonically decreasing for k ∈ [ K ] and ( x K , β K ) is an ( ε 0 , ε ) -preference stationary point in expectation, i.e., ∥∇ f β K ( x K ) ∥ 2 ≤ ε , and conditions (a) and (b) of Lemma 7 hold in expectation for ( x K , β K ) . Moreover, K = O ( ε -2 0 ) .

The complete theorem statement and its proof are given in Appendix H.2.

Remark 2. The rate obtained in Theorem 10 is dimension-independent ensuring its applicability to large-scale problems. However, under dueling feedback, to achieve the rate in Theorem 11, m and b needs to be dimension-dependent, which is unavoidable (Wang et al., 2023; Chen et al., 2025a; Saha et al., 2021). One might expect a rate dependent on n ≪ d under more informative dueling feedback, e.g., preferences between β vectors rather than x . Exploring such strategies is left for future work.

## 8 Conclusion

In this work, we provide a principled and efficient way to select a decision vector from the Pareto set of a set of objectives f 1 , . . . , f n given an additional preference function f 0 . The primary motivation is to seek the most preferred solution from a large model like LLM that is pretrained to satisfy a number of desiderata. A main contribution of this work is to provide a geometrically-meaningful notion of (approximate) preference stationarity. This is non-trivial due to the non-smoothness and non-convexity of the Pareto set. We achieve this by reformulating the constraint set as the Pareto manifold instead of the Pareto set. We also provide a simple algorithm that achieves ε 0 -approximate stationarity with iteration complexity of O ( ε -2 0 ) , under both first-order and dueling feedback.

There are several promising directions for future research. For example, extending this work to nonconvex F is significantly more challenging. Another impactful direction is incorporating deterministic dueling feedback to enhance practical applicability. A high-dimensional setup, where not only the decision vector is high-dimensional, but the number of objectives f 1 , f 2 , · · · , f n are allowed to increase with the problem scale, will also be quite interesting to explore.

## Acknowledgments and Disclosure of Funding

The research is partially supported by the NSF award CCF-2112665 (TILOS). It is also supported in part by the U.S. Department of Energy, the Office of Science, DARPA AIE FoundSci, as well as CDC-RFA-FT-23-0069 from the CDC's Center for Forecasting and Outbreak Analytics.

## References

- A. Agarwal, O. Dekel, and L. Xiao. Optimal algorithms for online convex optimization with multi-point bandit feedback. In Colt , pages 28-40. Citeseer, 2010.
- A. Agresti. Categorical data analysis , volume 792. John Wiley &amp; Sons, 2012.
- F. Bach and V. Perchet. Highly-smooth zero-th order online optimization. In Conference on Learning Theory , pages 257-283. PMLR, 2016.
- H. P. Benson. Optimization over the efficient set. Journal of Mathematical Analysis and Applications , 98(2): 562-580, 1984.
- S. Bolintinéanu. Minimization of a quasi-concave function over an efficient set. Mathematical Programming , 61:89-110, 1993a.
- S. Bolintinéanu. Necessary conditions for nonlinear suboptimization over the weakly-efficient set. Journal of optimization Theory and Applications , 78(3):579-598, 1993b.
- H. Bonnel and J. Morgan. Semivectorial bilevel optimization problem: penalty approach. Journal of Optimization Theory and Applications , 131:365-382, 2006.
- R. A. Bradley and M. E. Terry. Rank analysis of incomplete block designs: The method of paired comparisons. Biometrika , 39(3/4):324-345, 1952.
- H. Cai, D. McKenzie, W. Yin, and Z. Zhang. A one-bit, comparison-based gradient estimator. Applied and Computational Harmonic Analysis , 60:242-266, 2022.
- D. Chen, Y. Chen, A. Rege, Z. Wang, and R. K. Vinayak. Pal: Sample-efficient personalized reward modeling for pluralistic alignment. The 13th International Conference on Learning Representations , 2025a.
- L. Chen, Q. Xiao, E. H. Fukuda, X. Chen, K. Yuan, and T. Chen. Efficient first-order optimization on the Pareto set for multi-objective learning under preference guidance. In International Conference on Machine Learning . PMLR, 2025b.
- J. P. Dauer. Optimization over the efficient set using an active constraint approach. Zeitschrift für Operations Research , 35:185-195, 1991.
- S. Dempe. Bilevel optimization: theory, algorithms and applications , volume 3. TU Bergakademie Freiberg, Fakultät für Mathematik und Informatik, 2018.
- N. Ding, Y. Qin, G. Yang, F. Wei, Z. Yang, Y. Su, S. Hu, Y. Chen, C.-M. Chan, W. Chen, J. Yi, W. Zhao, X. Wang, Z. Liu, H.-T. Zheng, J. Chen, Y. Liu, J. Tang, J. Li, and M. Sun. Parameter-efficient fine-tuning of large-scale pre-trained language models. Nature Machine Intelligence , 5(3):220-235, 2023.
- A. L. Dontchev and R. T. Rockafellar. Implicit functions and solution mappings , volume 543. Springer, 2009.
- M. Ehrgott. Multicriteria optimization , volume 491. Springer Science &amp; Business Media, 2005.
- J. Fliege, L. G. Drummond, and B. F. Svaiter. Newton's method for multiobjective optimization. SIAM Journal on Optimization , 20(2):602-626, 2009.
- J. Fülöp. On the equivalence between a linear bilevel programming problem and linear optimization over the efficient set. Techn. Rep. WP , pages 93-1, 1993.
- I. Globus-Harris, M. Kearns, and A. Roth. An algorithmic framework for bias bounties. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency , pages 1106-1124, 2022.
- A. P. Guerreiro, C. M. Fonseca, and L. Paquete. The hypervolume indicator: Computational problems and algorithms. ACM Comput. Surv. , 54(6), 2021.
- N. Haghtalab, M. Jordan, and E. Zhao. On-demand sampling: Learning optimally from multiple distributions. Advances in Neural Information Processing Systems , 35:406-419, 2022.
- N. Hamada, K. Hayano, S. Ichiki, Y. Kabata, and H. Teramoto. Topology of Pareto sets of strongly convex problems. SIAM Journal on Optimization , 30(3):2659-2686, 2020.
- C. Hillermeier. Generalized homotopy approach to multiobjective optimization. Journal of Optimization Theory and Applications , 110(3):557-583, 2001a.
- C. Hillermeier. The Manifold of Stationary Points , pages 65-86. Birkhäuser Basel, Basel, 2001b. ISBN 978-3-0348-8280-4. doi: 10.1007/978-3-0348-8280-4\_5.

- N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. De Laroussilhe, A. Gesmundo, M. Attariyan, and S. Gelly. Parameter-efficient transfer learning for NLP. In International Conference on Machine Learning , pages 2790-2799. PMLR, 2019.
- Z. Huang, J. Li, S. M. Siniscalchi, I.-F. Chen, J. Wu, and C.-H. Lee. Rapid adaptation for deep neural networks through multi-task learning. In Sixteenth Annual Conference of the International Speech Communication Association , 2015.
3. C.-L. Hwang and A. S. M. Masud. Multiple objective decision making-methods and applications: a state-ofthe-art survey , volume 164. Springer Science &amp; Business Media, 2012.
- K. G. Jamieson, R. Nowak, and B. Recht. Query complexity of derivative-free optimization. Advances in Neural Information Processing Systems , 25, 2012.
- Y. Jin. Multi-objective machine learning , volume 16. Springer Science &amp; Business Media, 2007.
- M. Jordan, G. Kornowski, T. Lin, O. Shamir, and M. Zampetakis. Deterministic nonsmooth nonconvex optimization. In The Thirty Sixth Annual Conference on Learning Theory , pages 4570-4597. PMLR, 2023.
- M. M. Kamani, R. Forsati, J. Z. Wang, and M. Mahdavi. Pareto efficient fairness in supervised learning: From extraction to tracing. arXiv preprint arXiv:2104.01634 , 2021.
- K. Kobayashi, N. Hamada, A. Sannai, A. Tanaka, K. Bannai, and M. Sugiyama. Bézier simplex fitting: Describing Pareto fronts of simplicial problems with small samples in multi-objective optimization. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 33, pages 2304-2313, 2019.
- G. Kornowski and O. Shamir. Oracle complexity in nonsmooth nonconvex optimization. Advances in Neural Information Processing Systems , 34:324-334, 2021.
- G. Kornowski, S. Padmanabhan, and O. Shamir. On the hardness of meaningful local guarantees in nonsmooth nonconvex optimization. arXiv preprint arXiv:2409.10323 , 2024.
- A. Kulkarni, M. Kohns, M. Bortz, K.-H. Küfer, and H. Hasse. Regularities of pareto sets in low-dimensional practical multi-criteria optimisation problems: analysis, explanation, and exploitation. Optimization and Engineering , pages 1-22, 2022.
- W. G. La Cava. Optimizing fairness tradeoffs in machine learning with multiobjective meta-models. arXiv preprint arXiv:2304.12190 , 2023.
- K. Lange, D. R. Hunter, and I. Yang. Optimization transfer using surrogate objective functions. Journal of computational and graphical statistics , 9(1):1-20, 2000.
- D. Lee, G. Noarov, M. Pai, and A. Roth. Online minimax multiobjective optimization: Multicalibeating and other applications. Advances in Neural Information Processing Systems , 35:29051-29063, 2022.
- J. Li, A. M.-C. So, and W.-K. Ma. Understanding notions of stationarity in nonsmooth optimization: A guided tour of various constructions of subdifferential for nonsmooth functions. IEEE Signal Processing Magazine , 37(5):18-31, 2020.
- X. Lin, H.-L. Zhen, Z. Li, Q.-F. Zhang, and S. Kwong. Pareto multi-task learning. In Advances in Neural Information Processing Systems , volume 32, 2019.
- X. Lin, X. Zhang, Z. Yang, F. Liu, Z. Wang, and Q. Zhang. Smooth tchebycheff scalarization for multi-objective optimization. arXiv preprint arXiv:2402.19078 , 2024.
- H. Liu, D. Tam, M. Muqeeth, J. Mohta, T. Huang, M. Bansal, and C. A. Raffel. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. Advances in Neural Information Processing Systems , 35:1950-1965, 2022.
- X. Liu, X. Tong, and Q. Liu. Profiling Pareto front with multi-objective Stein variational gradient descent. In Advances in Neural Information Processing Systems , volume 34, pages 14721-14733, 2021.
- Z. Liu and M. Ehrgott. Primal and dual algorithms for optimization over the efficient set. Optimization , 67(10): 1661-1686, 2018.
- D. Mahapatra and V. Rajan. Multi-task learning with user preferences: Gradient descent with controlled ascent in Pareto optimization. In International Conference on Machine Learning , pages 6597-6607. PMLR, 2020.
- N. Martinez, M. Bertran, and G. Sapiro. Minimax Pareto fairness: A multi objective perspective. In International Conference on Machine Learning , pages 6755-6764. PMLR, 2020.
- N. Marumo, T. Okuno, and A. Takeda. Majorization-minimization-based Levenberg-Marquardt method for constrained nonlinear least squares. Computational Optimization and Applications , pages 1-42, 2023.
- I. Maru¸ sciac. On Fritz John type optimality criterion in multi-objective optimization. Mathematica-Revue d'analyse numérique et de théorie de l'approximation. L'analyse numérique et la théorie de l'approximation , pages 109-114, 1982.
- J. A. Nelder and R. W. Wedderburn. Generalized linear models. Journal of the Royal Statistical Society Series A: Statistics in Society , 135(3):370-384, 1972.

- Y. Nesterov. Introductory lectures on convex optimization: A basic course , volume 87. Springer Science &amp; Business Media, 2003.
- Y. Nesterov. Gradient methods for minimizing composite functions. Mathematical programming , 140(1): 125-161, 2013.
- J. Nunnally and I. Bernstein. Psychometric Theory (Third Edition) . McGraw Hill, Inc., New York, 1994.
- C. H. Papadimitriou and M. Yannakakis. On the approximability of trade-offs and optimal access of web sources. In Proceedings 41st annual symposium on foundations of computer science , pages 86-92. IEEE, 2000.
- J. Philip. Algorithms for the vector maximization problem. Mathematical programming , 2:207-229, 1972.
- V. C. Raykar, S. Yu, L. H. Zhao, G. H. Valadez, C. Florin, L. Bogoni, and L. Moy. Learning from crowds. Journal of machine learning research , 11(4), 2010.
- S. Reed, K. Zolna, E. Parisotto, S. G. Colmenarejo, A. Novikov, G. Barth-Maron, M. Gimenez, Y. Sulsky, J. Kay, J. T. Springenberg, et al. A generalist agent. arXiv preprint arXiv:2205.06175 , 2022.
- A. Roy, G. So, and Y.-A. Ma. Optimization on Pareto sets: On a theory of multi-objective optimization. arXiv preprint arXiv:2308.02145v1 , 2023.
- A. Saha, T. Koren, and Y. Mansour. Dueling convex optimization. In International Conference on Machine Learning , pages 9245-9254. PMLR, 2021.
- A. Saha, T. Koren, and Y. Mansour. Dueling convex optimization for general preferences: An unified framework for optimal convergence rates. International Conference on Machine Learning , 2025.
- O. Sener and V. Koltun. Multi-task learning as multi-objective optimization. Advances in neural information processing systems , 31, 2018.
- H. Sheftel, O. Shoval, A. Mayo, and U. Alon. The geometry of the Pareto front in biological phenotype space. Ecology and evolution , 3(6):1471-1483, 2013.
- S. Smale. Global analysis and economics I: Pareto optimum and a generalization of Morse theory. In Dynamical systems , pages 531-544. Elsevier, 1973.
- M. Spivak. Calculus on manifolds: a modern approach to classical theorems of advanced calculus . CRC press, 2018.
- R. E. Steuer. The Tchebycheff procedure of interactive multiple objective programming. In Multiple criteria decision making and risk analysis using microcomputers , pages 235-249. Springer, 1989.
- P. D. Tao et al. Numerical solution for optimization over the efficient set by dc optimization algorithms. Operations Research Letters , 19(3):117-128, 1996.
- P. Thach, H. Konno, and D. Yokota. Dual approach to minimization on the set of Pareto-optimal solutions. Journal of optimization theory and applications , 88:689-707, 1996.
- C. J. Tosh and D. Hsu. Simple and near-optimal algorithms for hidden stratification and multi-group learning. In International Conference on Machine Learning , pages 21633-21657. PMLR, 2022.
- Y. Wang, Q. Liu, and C. Jin. Is RLHF more difficult than standard RL? A theoretical perspective. Advances in Neural Information Processing Systems , 36:76006-76032, 2023.
- Y. Yamamoto. Optimization over the efficient set: overview. Journal of Global Optimization , 22:285-317, 2002.
- F. Ye, B. Lin, Z. Yue, P. Guo, Q. Xiao, and Y. Zhang. Multi-objective meta learning. Advances in Neural Information Processing Systems , 34:21338-21351, 2021.
- M. Ye and Q. Liu. Pareto navigation gradient descent: a first-order algorithm for optimization in Pareto set. In Uncertainty in Artificial Intelligence , pages 2246-2255. PMLR, 2022.
- J. Zhang, H. Lin, S. Jegelka, S. Sra, and A. Jadbabaie. Complexity of finding stationary points of nonconvex nonsmooth functions. In International Conference on Machine Learning , pages 11173-11182. PMLR, 2020.
- B. Zhu, M. Jordan, and J. Jiao. Principled reinforcement learning with human feedback from pairwise or k -wise comparisons. In International Conference on Machine Learning , pages 43037-43067. PMLR, 2023.
- D. M. Ziegler, N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593 , 2019.

## A Notation

| Symbol                                        | Usage                                                                                                                     |
|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| ∆ n - 1                                       | the ( n - 1) -simplex equipped with the ℓ 1 -metric ∆ n - 1 = { β ∈ R n : ∑ i ∈ [ n ] β i = 1 and ∀ i ∈ [ n ] , β i ≥ 0 } |
| ∆ n - 1 ( x ) ∇ x ∗ , ̂ ∇ x ∗ err ∇ f 0 ( x,β | the set of β satisfying ∇ f β ( x ) = 0 , (17) derivative of the map x ∗ and its approximation, (4), (10)                 |
| )                                             | bound on the approximation error of ∇ ( f 0 ◦ x ∗ ) , Lemma E.6                                                           |
| F , ( f 1 , . . . , f n )                     | the set of objective functions                                                                                            |
| f 0                                           | the preference function                                                                                                   |
| f β ( x )                                     | the scalarized objective ∑ i β i f i ( x ) , (2)                                                                          |
| g ( β ′ ; x,β )                               | majorizing surrogate for f ( x β ′ ) , (12)                                                                               |
| κ                                             | condition number κ := L/µ for ∇ 2 f i , Assumption A                                                                      |
| L , L H , L 0                                 | Lipschitz parameters for ∇ f i , ∇ 2 f i , and ∇ f 0 , Assumptions A, B, C                                                |
| M 0 , M 1                                     | Lipschitz parameters for x ∗ and ∇ x ∗ , Lemma E.2                                                                        |
| µ                                             | strong convexity parameter for f i , see Assumption A                                                                     |
| µ g                                           | strong convexity parameter nL 0 M 1 for the surrogate g , (12)                                                            |
| Pareto( F )                                   | the set of Pareto optimal solutions of F , Definition 1                                                                   |
| r                                             | distance between the minimizers of f 1 , . . . , f n , Assumption A                                                       |
| P ( F )                                       | the Pareto manifold, Definition 3                                                                                         |
| R                                             | diam ( Pareto( F ) ) := sup { ∥ x - x ′ ∥ 2 : x,x ′ ∈ Pareto( F ) } , Lemma E.1                                           |
| x ∗ ( β ) , x β                               | stationary point for f β , (3)                                                                                            |
| σ                                             | link function for comparison oracle, (15)                                                                                 |

## B Proofs for Section 3

Lemma 1. Let F ≡ ( f 1 , . . . , f n ) be a collection of smooth and strictly convex objectives. Then:

<!-- formula-not-decoded -->

Proof. The condition on the left states that x is Pareto optimal. The one on the right states that x is Pareto stationary. When the objectives are smooth and strongly convex, Pareto optimality and Pareto stationarity are equivalent (Theorem 3.1, Fliege et al. (2009)).

Proposition 2 (Characterization of the Pareto manifold) . Let F ≡ ( f 1 , . . . , f n ) be a collection of smooth and strictly convex objectives. Define x ∗ : ∆ n -1 → Pareto( F ) by:

<!-- formula-not-decoded -->

Let ∇ F ( x ) ∈ R n × d be the Jacobian. Then, the map x ∗ has derivative:

<!-- formula-not-decoded -->

so that the map β ↦→ ( x β , β ) is a diffeomorphism of ∆ n -1 with the Pareto manifold P ( F ) .

Proof. The map x ∗ is well-defined because f β is strongly convex-it is the convex combination of strongly convex objectives, so it has a unique minimizer. And as the objectives are smooth, the stationarity condition ∇ f β ( x ) = 0 uniquely holds at x ∗ ( β ) :

<!-- formula-not-decoded -->

Define the map ζ ( x, β ) = ∇ f β ( x ) . Then, the Pareto manifold is precisely the zero set P ( F ) = ζ -1 (0) , and which can be parametrized by simplex ∆ n -1 via the map β ↦→ ( x β , β ) .

In fact, it is a smooth parametrization. To see this, we apply the implicit function theorem (Theorem B.1), which states that the map x ∗ is smooth at β when ∇ x ζ ( x β , β ) is invertible. Indeed, we have that ζ is continuously differentiable, with:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As f β is strictly convex, it has positive definite Hessian, so that det ∇ x ζ ( x β , β ) = 0 . Furthermore, Theorem B.1 also implies that the derivative of ∇ x ∗ is given by Equation (4). It follows that the map β ↦→ ( x β , β ) is smooth. It also has a smooth inverse. Namely, the projection onto the second component ( x β , β ) ↦→ β . Thus, P ( F ) is diffeomorphic with ∆ n -1 .

̸

Theorem B.1 (Implicit function theorem, Spivak (2018)) . Let f : R d × R n → R d be continuously differentiable on an open set containing ( a, b ) and let f ( a, b ) = 0 . Let ∇ u f ( u, v ) be the d × d matrix:

<!-- formula-not-decoded -->

If det ∇ u f ( a, b ) = 0 , there are open sets U ⊂ R d and V ⊂ R n containing a and b respectively with the following property: for each v ∈ V there is a unique g ( v ) ∈ U such that f ( g ( v ) , v ) = 0 . Furthermore, the map g is differentiable with derivative given by:

<!-- formula-not-decoded -->

̸

Figure 3: An embedding of the Pareto manifold from (1) into three dimensions from two camera angles. The singularity seen in (1) is an artifact of the projection of the manifold into the decision space, rather than an intrinsic irregularity in the manifold.

<!-- image -->

## C A comparison of solution concepts

In this section, we elaborate on the relationship between different solution concepts (optimality, stationarity, approximate stationarity) for the Pareto-constrained optimization problem:

<!-- formula-not-decoded -->

The solution concepts are related by:

| preference optimality   | ⊂   | preference stationarity   | ⊂ weak preference stationarity   | ⊂ approximate preference stationarity   |
|-------------------------|-----|---------------------------|----------------------------------|-----------------------------------------|

It is fairly clear that the first and last inequalities are strict. We discuss the inner inequality.

It turns out that a point x can be weakly preference stationary without being preference stationary. However, this can only happen if x is also a point of singularity in Pareto( F ) . Geometrically, if we consider Pareto( F ) as the projection of P ( F ) onto its first component in R d , the this means that multiple points are collapsed onto x . Algebraically, this means that the set of gradients ∇ f 1 ( x ) , . . . , ∇ f n ( x ) fails to have full (Pareto) rank (Smale, 1973; Hamada et al., 2020).

To elaborate, define the set:

<!-- formula-not-decoded -->

Then, x is Pareto stationary if there is some β in ∆ n -1 ( x ) , so that:

<!-- formula-not-decoded -->

and the rank of this set of gradients is at most n -1 . Since ∆ n -1 does not contain any collinear vectors, if ∆ n -1 ( x ) contains more than a single point, then the rank of the set of gradients must be strictly less than n -1 . This leads us to the definition:

Definition C.1 (Pareto genericity) . Let { v 1 , . . . , v n } ⊂ R d . This set is Pareto generic if:

<!-- formula-not-decoded -->

and the non-degeneracy condition holds: rank( v 1 , . . . , v n ) = n -1 .

If ∇ F ( x ) is Pareto generic, then ∆ n -1 ( x ) contains a unique β , so we immediately have:

Proposition C.1 (Weak and strong preference stationarity) . When the set of gradients ∇ F ( x ) is Pareto generic and x is weakly preference stationary, then x is preference stationary.

However, when the gradients ∇ F ( x ) are not Pareto generic, then weak preference stationarity can be strictly weaker. Let ( x, β ) where β ∈ ∆ n -1 ( x ) be weakly preference stationary, so that:

<!-- formula-not-decoded -->

We can simplify this by using the fact that ∇ f β ( x ) = ∇ F ( x ) ⊤ β = 0 . Then, one way for the stationary condition to be fulfilled is for the underlined term to be normal to ∇ f 0 ( x ) :

<!-- formula-not-decoded -->

This statement has the following geometric interpretation. These vectors are contained in the Clarke tangent cone of Pareto( F ) at x . If these are the only vectors in the tangent cone, then this condition states that -∇ f 0 ( x ) is contained in the normal cone of Pareto( F ) at x .

But, in general, the tangent cone contains the union of subspaces:

<!-- formula-not-decoded -->

And so, when ∆ n -1 ( x ) does not contain a unique vector, the tangent cone can contain more vectors. By selecting different β 's, we recover different slices of the tangent cone. This also means that even if the above normality condition holds for one β , it may fail to hold for another ˜ β ∈ ∆ n -1 ( x ) . Then, ( x, β ) is weakly preference stationary while ( x, ˜ β ) may not be.

## D Insufficiency of first-order information

The notion of preference stationarity uses second-order information in F , for the term ∇ 2 f β ( x ) -1 . A natural question is whether there are reasonable notions of stationarity that only use first-order information. It turns out there are none, if we require the stationarity condition to be (i) non-trivial, (ii) necessary for preference optimality and (iii) decidable from local information at a single point x .

The reason is that from ∇ F ( x ) alone, we can only determine whether the point x is contained in the Pareto manifold. It is not enough to determine locally how the manifold curves. For example, (4) shows two different Pareto sets that share the same gradients at a point x . The local optimality of x with respect to f 0 depends on its neighboring Pareto points. It turns out that to attain a non-trivial and necessary condition, we would either need to look at higher-order information, or first-order information at more than a one point.

To formalize this, we define stationarity conditions as decision functions . These are functions that return Boolean outcomes true or false (we interpret these functions as ways of classifying points as stationary or not). In particular, we consider first-order stationary conditions, which makes decisions given only the gradients ∇ f 0 and ∇ f 1 , . . . , ∇ f n at a single point:

Definition D.1 (Stationarity function) . A first-order stationary condition is a decision function whose input is a tuple of n +1 vectors in R d :

<!-- formula-not-decoded -->

Let f 0 be a smooth preference function and f 1 , . . . , f n be smooth, strongly convex objectives. We say that a first-order condition is necessary if the following holds:

<!-- formula-not-decoded -->

̸

There are two specific cases in which it is possible to determine that a decision x is not preference optimal from first-order information. First, if x is not even Pareto optimal, then it cannot be preference optimal. This occurs whenever ∇ f β ( x ) = 0 for all β ∈ ∆ n -1 . The second case occurs if x is Pareto optimal, but ∇ f 0 ( x ) is in the convex cone spanned by the columns of ∇ F ( x ) ,

<!-- formula-not-decoded -->

̸

In particular, let x = x β . Then, let γ : [0 , 1] → ∆ n -1 be the curve parametrized by γ ( t ) = tλ +(1 -t ) β . Then whenever ∇ f 0 ( x β ) = 0 :

<!-- formula-not-decoded -->

Intuitively, these solutions are not preference optimal because we can find more preferable Pareto solutions by weighting those objectives that align with f 0 more. To exclude these specific cases, we introduce the notion of preference genericity . Then, Proposition D.1 shows that any necessary, first-order stationary condition must accept all preference generic sets.

Definition D.2 (Preference genericity) . Let { v 0 , v 1 , . . . , v n } ⊂ R d where 1 &lt; n ≤ d . This set is preference generic if there exists some β ∈ ∆ n -1 such that β 1 v 1 + · · · β n v n = 0 , and:

<!-- formula-not-decoded -->

Proposition D.1 (Necessary first-order conditions are trivial) . If Stationary is a necessary, first-order stationary condition, then it is trivial in the following sense:

<!-- formula-not-decoded -->

for any preference generic set of v 0 , . . . , v n ∈ R d .

Proof. It suffices to show that there exist f 0 , F , and x ⋆ such that x ⋆ is preference optimal and for i = 0 , . . . , n :

<!-- formula-not-decoded -->

Figure 4: Two instances of Pareto( f 1 , f 2 ) are shown (thick gray lines), where f 1 and f 2 are positive-definite quadratic objectives in R 2 (visualized by contour lines). At x (the black dot), the two instances share the same local information -∇ f 1 ( x ) and -∇ f 2 ( x ) (orange arrows); they cross the contour lines at right angles. When n = 2 , the Pareto set contains all z such that ∇ f 1 ( z ) = -λ ∇ f 2 ( z ) for λ ≥ 0 . Notice that if f 0 is strictly convex and x does not minimize f 0 over R 2 , then x cannot be stationary for both instances.

<!-- image -->

And since x ⋆ is preference optimal, any necessary stationary condition must accept:

<!-- formula-not-decoded -->

Without loss of generality, let x ⋆ = 0 by an affine transformation. To construct f 0 and F , we can simply consider a family of positive-definite quadratics:

- Let the preference function f 0 be:

<!-- formula-not-decoded -->

Notice that ∇ f 0 ( x ∗ ) = v 0 .

- Let the objectives f 1 , . . . , f n share the same Hessian:

<!-- formula-not-decoded -->

where A ∈ R d × d is full rank and z i ∈ R d . Let H = A ⊤ A for short.

We show that we can set A and the z i 's so that x ⋆ is preference optimal while (18) holds.

By Lemma D.1, the Pareto set is the convex hull C := conv( z 1 , . . . , z n ) . Notice that the choice of H and v i 's determines the z i 's, since we require ∇ f i ( x ∗ ) = v i , which expands to:

<!-- formula-not-decoded -->

From convex optimization, x ⋆ = 0 is preference optimal if (i) x ⋆ ∈ C and (ii) C is normal to ∇ f 0 . Indeed, these two conditions can be fulfilled:

- (i) Because v 1 , . . . , v n is assumed to be Pareto generic, zero is a convex combination of the v i 's. As the z i 's are related to the v i 's by a linear transformation, this also implies that zero is a convex combination of the z i 's (with the same set of convex weights).
- (ii) We need to show that the subspace span( v 1 , . . . , v n ) can be mapped into span( v 0 ) ⊥ by the map v ↦→-H -1 v where H is positive definite. Lemma D.2 shows that such a map H exists as long as v 0 / ∈ span( v 1 , . . . , v n ) , which is assumed from preference genericity.

Thus, there exists f 0 and F that is preference optimal at x ⋆ with matching first-order information. A necessary stationary condition must therefore be accepted.

Remark D.1. Suppose that Stationary is not necessary, but that we can design some optimization method that provably converges to a stationary point in { x : Stationary( x ) = true } . Then, there are settings in which the method provably avoids preference optimal points.

In the remainder of this section, we prove Lemma D.1 and Lemma D.2 used above.

Lemma D.1. Let f 1 , . . . , f n : R d → R be positive-definite quadratics with a shared Hessian:

<!-- formula-not-decoded -->

where A ∈ R d × d is full rank and z i ∈ R d . Then, the Pareto set is the convex hull:

<!-- formula-not-decoded -->

Proof. As the objectives f 1 , . . . , f n are strongly convex, optimality is equivalent to stationarity. Thus, x ∈ Pareto( f 1 , . . . , f n ) if and only if there exists some β ∈ ∆ n -1 such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

But as A is invertible, this is equivalent to:

<!-- formula-not-decoded -->

which, when expanded, states that:

which is to say that x ∈ conv( z 1 , . . . , z n ) .

Lemma D.2. Let U and V be linear subspaces of R d such that U ∩ V ⊥ = { 0 } . Then, there exists some positive definite map H : R d → R d such that H ( U ) ⊂ V .

Proof. If S ⊂ R d is a subspace, let Π S : R d → R d be the projection onto S . Define the map:

<!-- formula-not-decoded -->

̸

Then H satisfies the following:

- H is positive definite. To see this, let 0 = x ∈ R d have decomposition x = x 1 + x 2 , where x 1 ∈ U and x 2 ∈ U ⊥ . Then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

where the last inequality is strict because x = 0 and U ∩ V ⊥ = { 0 } .

- H ( U ) ⊂ V . If x ∈ U , then by definition Π U ⊥ x = 0 so that Hx = Π V x ∈ V .

## D.1 An example of a first-order stationarity condition avoiding optimality

In this section, we discuss the first-order stationarity condition of Ye and Liu (2022), defined to as stationarity with respect to their optimization dynamics, Pareto navigating gradient descent (PNG). We show that it fails to be a necessary condition for preference optimality.

Despite that, their condition and dynamics have appealing properties since (i) they do not require second-order information, which is computationally more expensive, and (ii) their dynamics largely satisfies what they call the Pareto improvement property , which ensures that each objective enjoys monotonic improvement during optimization:

<!-- formula-not-decoded -->

As Pareto improvement can be at odds with preference optimality, this leads to an open question: when and how should we balance Pareto improvement with preference optimality?

Definition D.3 (PNG stationarity, Ye and Liu (2022)) . Let c &gt; 0 . The PNG vector v c ( x ) is defined:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let ε &gt; 0 . A vector x ∈ R d is ( c, ε ) -PNG stationary if v c ( x ) = λ ∇ f 0 ( x ) for some λ ≤ 0 and:

<!-- formula-not-decoded -->

In the following, we consider an example in R 2 with n = 2 objectives. Let the standard basis be denoted e 1 , e 2 ∈ R 2 , and let the objective functions f 1 , f 2 : R 2 → R be defined:

<!-- formula-not-decoded -->

where A ∈ R 2 × 2 is full-rank. Lemma D.1 shows that the Pareto set of the objectives Pareto( f 1 , f 2 ) is the line segment from -e 1 to e 2 . Thus, the Pareto set is invariant under change in A , while PNG stationarity is not, as the constraint set changes with A :

<!-- formula-not-decoded -->

and H = A ⊤ A . Due to this discrepancy, PNG stationary points can be preference suboptimal.

Example D.1. Let the preference function be: f 0 ( x ) = 1 2 ∥ x -e 2 ∥ 2 , and let the objectives f 1 , f 2 be defined as in the above (19) with:

<!-- formula-not-decoded -->

Then, the unique preference optimal point is the origin 0 . However, the ( c, ε ) -PNG stationary point is bounded away from 0. It even converges to e 1 as the error tolerance ε goes to zero.

Proof. Consider the PNG vector v c ( x ) when x is in the region:

<!-- formula-not-decoded -->

Here, both constraints ∇ f i ( x ) ⊤ v ≥ c are active in the constrained optimization problem that defines the PNG vector; and so, v c ( x ) is the vertex point of the constraint set, satisfying:

<!-- formula-not-decoded -->

Expanding out the gradients, we obtain:

<!-- formula-not-decoded -->

This implies that e ⊤ 1 Hv c ( x ) = 0 . Now suppose that x PNG ∈ C is PNG stationary. Then, by definition, it must satisfy ∇ f 0 ( x PNG ) ∈ span ( v c ( x PNG ) ) , so it has the form:

<!-- formula-not-decoded -->

Whenever the standard basis vectors are not eigenvectors of H , the line e 2 + λu intersects Pareto( f 1 , f 2 ) away from 0. For example, let A satisfy (20).

Then, the line e 2 + λu runs through e 1 and e 2 . We can verify that C contains all points on this line between its two endpoints. When x = e 2 + λ ( e 1 -e 2 ) and λ ∈ (0 , 1) , we have:

<!-- formula-not-decoded -->

and similarly, we have:

<!-- formula-not-decoded -->

This implies that for all c &gt; 0 and ε &gt; 0 , the ( c, ε ) -PNG stationary point is bounded away from 0, converging to e 1 as ε goes to zero.

This example is visualized in Figure 2.

## E Proofs for Section 5

In this section, we derive the following implications from the assumptions from Section 5:

- Assumption A allows us to bound the size of the Pareto set (Lemma E.1).
- Assumption B additionally bounds the curvature of the Pareto manifold: we show that ∇ x ∗ is well-behaved (Lemma E.2) and is well-approximated by ̂ ∇ x ∗ (Lemma E.5).
- Assumption C further leads to error bounds when approximating gradient of f 0 ◦ x ∗ (Lemma E.6). It also allows us to define a notion of approximate preference stationarity that is geometrically meaningful (Proposition 6) and can be verified using approximate information (Lemma 7).

Let's recall from Assumption A that the condition number of the objectives ∇ 2 f i for i ∈ [ n ] is upper bounded by κ := L/µ . We also defined the scale parameter r as the maximum distance between any of the minimizers of the objectives:

<!-- formula-not-decoded -->

Lemma E.1 (Size of Pareto set) . Let F satisfy Assumption A. Then R ≤ √ κr , where:

<!-- formula-not-decoded -->

Proof. Because each f i is µ -strongly convex and L -Lipschitz smooth, so is the convex combination f β . This implies the upper and lower bounds:

<!-- formula-not-decoded -->

It follows that the minimizer of f β is bounded:

<!-- formula-not-decoded -->

On the other hand, if a point ∥ x -x i ∥ &gt; 2 s for some i ∈ [ n ] , then by reverse triangle inequality, ∥ x -x j ∥ &gt; s for all j ∈ [ n ] . This implies that:

<!-- formula-not-decoded -->

It follows that if ∥ x -x i ∥ &gt; 2 √ L/µ for some i , then x is not a Pareto optimal point.

Lemma E.2 (Smoothness of x ∗ ) . Let F satisfy Assumptions A,B. Then, x ∗ : ∆ n -1 → R d is M 0 -Lipschitz continuous and has M 1 -Lipschitz continuous gradients, where:

<!-- formula-not-decoded -->

Proof of Lemma E.2 That x ∗ is Lipschitz continuous with Lipschitz continuous gradients follows from the following two lemmas:

Lemma E.3. Let F ≡ ( f 1 , . . . , f n ) be a set of twice-differentiable objectives and let f 0 be a smooth preference. Suppose the objectives are L -Lipschitz smooth and µ -strongly convex:

<!-- formula-not-decoded -->

Let R := diam ( Pareto( F ) ) . Then, the map x ∗ : (∆ n -1 , ℓ 1 ) → ( R d , ℓ 2 ) is LR/µ -Lipschitz.

Proof. Recall from (4) that ∇ x ∗ ( β ) = -∇ 2 f β ( x β ) -1 ∇ F ( x β ) ⊤ . Then:

<!-- formula-not-decoded -->

where (i) is a property of the ∥ · ∥ 1 , 2 -norm, (ii) uses µ I ⪯ ∇ 2 f β ( x β ) and Lemma G.1.

Lemma E.4. Let β, β ′ ∈ ∆ n -1 . Then,

<!-- formula-not-decoded -->

Proof. By definition, we have:

<!-- formula-not-decoded -->

We can add and subtract ∇ 2 f β ( x β ) -1 ∇ F ( x β ′ ) ⊤ inside the right-hand side (RHS):

<!-- formula-not-decoded -->

We can bound the two terms in the norm separately. For the first:

<!-- formula-not-decoded -->

where (i) follows the same argument as Lemma E.5, and (ii) applies Lemma E.3. For the second term, we can add and subtract ∇ 2 f β ′ ( x β ) -1 ∇ F ( x β ′ ) ⊤ to obtain:

<!-- formula-not-decoded -->

where ∇ 2 f β ( x ) -1 -∇ 2 f β ′ ( x ) -1 is bounded by Lemma G.3; ∇ 2 f β ( x ) -1 -∇ 2 f β ( x ′ ) -1 is bounded by Lemma G.2 and Lemma E.3; ∥∇ F ( x β ′ ) ⊤ ∥ 1 , 2 is bounded by Lemma G.1.

The result follows by substituting in the definitions of M 0 and M 1

. ■

Lemma E.5 (Approximability of ∇ x ∗ ) . If F satisfies Assumptions A,B. Then:

<!-- formula-not-decoded -->

Proof. Recall that x β := x ∗ ( β ) . Then, by definition, we have:

<!-- formula-not-decoded -->

We can add and subtract ∇ 2 f β ( x ) -1 ∇ F ( x β ) ⊤ inside the right-hand side (RHS) to get:

<!-- formula-not-decoded -->

where (i) the first blue term uses µ I ⪯ ∇ 2 f β and the L -Lipschitz smoothness of the objectives, while the bracket orange term follows from Lemma G.2 and the final purple term follows from Lemma G.1, and (ii) uses the µ -strong convexity of f β .

Lemma E.6 (Approximability of ∇ ( f 0 ◦ x ∗ ) ) . If F , f 0 satisfy Assumptions A,B,C. Then:

<!-- formula-not-decoded -->

where err ∇ f 0 ( x, β ) := 1 µ ( M 1 2 M 0 ∥∇ f 0 ( x ) ∥ 2 + L 0 M 0 ) ∥∇ f β ( x ) ∥ 2 .

Proof. Add and subtract ∇ f 0 ( x ) ⊤ ∇ x ∗ ( β ) within the norm on the right-hand side:

<!-- formula-not-decoded -->

where we use the fact that f 0 is L 0 -Lipschitz smooth by Assumption C, that x ∗ is M 0 -Lipschitz continuous by Lemma E.2, and that ∥∇ x ∗ ( β ) -̂ ∇ x ∗ ( β ) ∥ 1 , 2 is bounded by Lemma E.5. The result follows from upper bounding ∥ x β -x ∥ by µ -strong convexity of f β :

<!-- formula-not-decoded -->

Proposition 6 (Geometric meaning of approximate stationarity) . Let (ˆ x, ˆ β ) be ( ε 0 , ε ) -preference stationary. With Assumptions A,B,C, and R := diam ( Pareto( F ) ) and s := 2 µ 2 ε 0 / ( L 0 L 2 R 2 ) , then:

<!-- formula-not-decoded -->

- (b) ∥ ˆ x -x ˆ β ∥ 2 ≤ ε/µ .
- Proof. (a) Recall that x β is the minimizer of f β , by definition. Because f β is µ -strongly convex, we can bound the distance between x and x β by:

<!-- formula-not-decoded -->

where the second inequality follows from condition (8a).

- (b) Let β s := (1 -s ) β + sβ ′ parametrize the line connecting β and β ′ . Let γ : [0 , 1] → Pareto( F ) be the path γ ( s ) := x ∗ ( β s ) , so that:

<!-- formula-not-decoded -->

We can now upper bound the difference:

<!-- formula-not-decoded -->

Let's bound the integrals separately. Since x β s = ∫ s 0 dγ ( s )( β ′ -β ) , we have by Lemma E.3:

<!-- formula-not-decoded -->

We also have | dγ ( s ) | ≤ µ -1 LR ∥ β -β ′ ∥ 1 , by Lemma E.3. The first integral is bounded by:

<!-- formula-not-decoded -->

For the second integral, first note that condition (8b) implies:

<!-- formula-not-decoded -->

yielding the other bound:

<!-- formula-not-decoded -->

Putting these two together, we obtain:

<!-- formula-not-decoded -->

It follows that if we restrict ∥ β -β ′ ∥ 1 ≤ 2 µ 2 ε 0 L 0 L 2 R 2 , one of the factors of ∥ β -β ′ ∥ 1 in the first term can be absorbed into the constant, proving the result:

<!-- formula-not-decoded -->

Lemma 7 (Verifiability of approximate stationarity) . Under Assumptions A,B,C, the point (ˆ x, ˆ β ) is ( ε 0 , ε ) -preference stationary if ∥∇ f ˆ β (ˆ x ) ∥ 2 ≤ ε , and for some x ∈ R d and α ∈ (0 , 1) ,

(a) an α · ε 0 -approximate stationary condition holds for all β ′ ∈ ∆ n -1 :

<!-- formula-not-decoded -->

(b) an error bound holds: err ∇ f 0 ( ˆ β, x ) ≤ (1 -α ) · ε 0 , where:

<!-- formula-not-decoded -->

and M 0 = κR , M 1 = 2 κ 2 R (1 + L H R/µ ) .

Proof. For ( ε, ε 0 ) -preference stationarity, we require ∥∇ f ˆ β (ˆ x ) ∥ 2 ≤ ε and:

<!-- formula-not-decoded -->

Then by Lemma E.6, the left-hand side is lower bounded:

<!-- formula-not-decoded -->

for α ∈ (0 , 1) . The two terms are lower bounded by zero by conditions (1) and (2).

## F Proofs for Section 6

Proposition 8 (A family of majorizing surrogates) . Let F and f 0 satisfy Assumptions A,B,C. Let err ∇ f 0 ( x, β ) be as defined in Lemma E.6. Define the family indexed by ( x, β ) ∈ R d × ∆ n -1 :

<!-- formula-not-decoded -->

where µ := nL M . Then g ( β ; x, β ) majorizes f ◦ x , satisfying (12) .

<!-- formula-not-decoded -->

Proof. As f 0 and x ∗ are respectively L 0 - and M 1 -Lipschitz smooth (Assumption C and Lemma E.5), their composition is L 0 M 1 -Lipschitz smooth, and so it admits the standard quadratic upper bound (e.g., Lemma 1.2.3, Nesterov (2003)):

<!-- formula-not-decoded -->

where we have used ∥ β ′ -β ∥ 2 1 ≤ n ∥ β ′ -β ∥ 2 2 . This could be an easy choice for our majorizing surrogate g but the gradient ∇ ( f 0 ◦ x ∗ )( β ) is implicit. Substituting in the approximation ∇ f 0 ( x ) ⊤ ̂ ∇ x ∗ ( x, β ) yields the result, where the error term comes from Lemma E.6.

## G Convergence for Pareto majorization-minimization

Theorem 10 (Convergence of PMM with first-order feedback) . Suppose that F and f 0 satisfy Assumptions A to C, and that the black-box optimizers satisfy Assumption E. Fix 0 &lt; ε 1 / 2 ≤ ε 0 ≤ 1 . Let ( x k , β k ) k be the iterates of Algorithm 1 using the family of surrogates (13) . There exist c 1 ( f 0 , F ) and c 2 ( f 0 , F ) bounded away from zero and a stopping time K such that ( f 0 ◦ x ∗ )( β k ) is monotonically decreasing for k ∈ [ K ] and ( x K , β K ) is an ( ε 0 , ε ) -preference stationary point. Furthermore:

<!-- formula-not-decoded -->

where f ∗ := max f 0 ( x ) and f ∗ = min f 0 ( x ) are optimized over the compact set Pareto( F ) .

Proof. Fix k &gt; 1 . For short, we let:

<!-- formula-not-decoded -->

Claim. At each iteration, either (i) the preference improves by at least a constant:

<!-- formula-not-decoded -->

or (ii) the point (ˆ x, ˆ β ) is ( ε 0 , ε ) -preference stationary.

Assuming the claim holds, the theorem immediately follows: if the algorithm in K steps has not found an ( ε 0 , ε ) -preference stationary point, then the value f 0 ( x β k ) must decrease every iteration by a constant. But because f 0 ◦ x ∗ is lower bounded over ∆ n -1 by f ∗ , this can happen at most:

<!-- formula-not-decoded -->

Proof of the claim. Let β ∗ := arg min β ′ ∈ ∆ n -1 g ( β ′ ; x, β ) . Lemma G.4 shows that an approximate stationary point ˆ β of a strongly convex function is close to the exact point β ∗ :

<!-- formula-not-decoded -->

where we let δ denote this constant for short.

We can analyze ˆ β through β ∗ . There are two cases, leading to either (1) O ( ε 0 ) -preference stationarity or (2) O ( ε 2 0 ) -constant descent. The two cases depend on the suboptimality of β .

Case 1: ∥ β ∗ -β ∥ 2 &lt; 2 δ . Here, β is fairly close to the optimum β ∗ of the surrogate. We show that the approximate stationarity of ˆ β with respect to the surrogate implies approximate preference stationarity. We do so via Lemma 7, which states that (ˆ x, ˆ β ) is ( ε 0 , ε ) -preference stationary provided:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

While (22) is immediate from our choice of c 2 , defined in the last section of the proof, the others do not follow automatically from approximate stationarity with respect to the surrogate: the surrogate is derived from local information at ( x, β ) , while we would like guarantees at ( x, ˆ β ) . But because β ∗ is close to both β and ˆ β , we can control all of these. By triangle inequality:

<!-- formula-not-decoded -->

combining (21) and the assumption that ∥ β -β ∥ &lt; 2 δ .

We now show (23). We have for all β ′ ∈ ∆ n -1

∗ 2 ,

<!-- formula-not-decoded -->

where (i) adds and subtracts ∇ f 0 ( x ) ⊤ ̂ ∇ x ∗ ( x, β )( β ′ -ˆ β ) and applies Hölder's inequality, (ii) substitutes in Condition 1 for the first term and bounds the second via Lemma G.3, and (iii) bounds ∥ β -ˆ β ∥ 2 using (25), and (iv) applies the definition of c 1 , set in the last section of the proof.

To show (24), we have:

<!-- formula-not-decoded -->

where (i) expands out err ∇ f 0 , (ii) uses the fact that β ↦→∥∇ F ( x ) ⊤ β ∥ 2 is ∥∇ F ( x ) ⊤ ∥ 2 -Lipschitz in β with respect to the ℓ 2 -norm, (iii) applies the definition of err ∇ f 0 and the inequality (25), and (iv) follows by definition of c 1 and c 2 , set in the last section of the proof.

As (22, 23, 24) hold, Lemma 7 shows that (ˆ x, ˆ β ) is ( ε 0 , ε ) -preference stationary.

Case 2: ∥ β ∗ -β ∥ 2 ≥ 2 δ . Here β is suboptimal and β ∗ achieves a large descent:

<!-- formula-not-decoded -->

where (i) uses the majorizing property of g , (ii) follows from Lemma G.5, (iii) applies the definition of err ∇ f 0 ( x, β ) and the assumption that ε ≤ ε 2 0 , and (iv) uses the definition of c 2 .

The large descent also carries over to ˆ β because it is approximately stationary:

<!-- formula-not-decoded -->

where (i) uses the majorizing property of g , (ii) adds and subtracts g ( β ∗ ; x, β ) and (iii) applies (29) and Lemma G.4.

Thus, the preference improves by at least a constant. To finish proving the claim, we need to verify that it is indeed possible to set c 1 and c 2 appropriately.

Setting c 1 and c 2 : we tabled a few inequalities above. Recall:

For (22), we need:

For (26), we need:

For (27), we need:

<!-- formula-not-decoded -->

For (28), we need:

<!-- formula-not-decoded -->

It is unenlightening but straightforward to verify that it suffices to set:

<!-- formula-not-decoded -->

where a ∨ b := max { a, b } .

A concerned reader may wonder whether c 1 and c 2 may be bounded away from zero, as claimed in the theorem statement: we need to ensure that ∥∇ f 0 ( x ) ∥ 2 and ∥∇ F ( x ) ⊤ ∥ 2 do not blow up. Indeed, this holds because the iterates x k remain within a constant distance of the Pareto set. In particular, since c 2 ≤ 1 , by Condition 2, we have that the k th iterate satisfies:

<!-- formula-not-decoded -->

which follows from µ -strong convexity of f β k . Thus, all iterates of the algorithm are within ε/µ of the Pareto set and also satisfy for all k, k ′ ∈ N :

<!-- formula-not-decoded -->

Then, by L 0 -Lipschitz smoothness, we can bound:

<!-- formula-not-decoded -->

Similarly, by L -Lipschitz smoothness, we also have:

<!-- formula-not-decoded -->

## G.1 A bound on the gradient

Lemma G.1. Let R := diam ( Pareto( F ) ) . Then for any x β = x ∗ ( β ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. By definition, we have:

<!-- formula-not-decoded -->

where (i) follows from Jensen's inequality, (ii) holds because the max is no smaller than the average, (iii) applies L -Lipschitz smoothness. In particular, let x i = arg min f i ( x ) , so that ∇ f i ( x i ) = 0 . Then:

<!-- formula-not-decoded -->

The result holds because x β and all x i 's are contained in Pareto( F ) .

## G.1.1 Lemmas: matrix inverses

Lemma G.2. Let M : R d → R d × d be L -Lipschitz satisfying M ( x ) ⪰ µ I where R d has the ℓ 2 -norm and R d × d the operator norm. Then, the map x ↦→ M ( x ) -1 is L/µ 2 -Lipschitz.

Proof. For short, let us denote M ( x ) by M x . Note that I = ( M ′ x + M x -M ′ x ) M -1 x , so that:

<!-- formula-not-decoded -->

which is series of unenlightening algebraic manipulations. But now, we may apply L -Lipschitz continuity to obtain ∥ M x -M x ′ ∥ ≤ L ∥ x -x ′ ∥ and the µ -lower bound to obtain ∥ M -1 x ∥ , ∥ M -1 x ′ ∥ ≤ µ -1 . Together, we obtain L/µ 2 -Lipschitz continuity:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma G.3. Let M 1 , . . . , M n be positive-definite matrices in R d × d equipped with the operator norm, and let ∆ n -1 be equipped with the ℓ 1 norm. Suppose the following holds:

<!-- formula-not-decoded -->

The map β ↦→ M -1 β where M β := ∑ i ∈ [ n ] β i M i has bounded derivative ∥∇ β M -1 β ∥ 1 , 2 ≤ L/µ 2 .

Proof. We can compute the derivative of the above map:

<!-- formula-not-decoded -->

where ∇ β M β dβ = M dβ . The upper bound on the M i 's implies that ∥∇ β M β ∥ 1 , 2 ≤ L . On the other hand, the lower bound implies that ∥ M -1 β ∥ 2 ≤ µ -1 .

## G.1.2 Lemmas: constrained optimization of strongly convex functions

Lemma G.4. Let f : R n → R be smooth and convex and let C ⊂ R n be a convex constraint set. Suppose that β ∗ , ˆ β ∈ C are stationary and ε -approximately stationary, respectively:

<!-- formula-not-decoded -->

Then, f ( ˆ β ) -f ( β ∗ ) ≤ ε ∥ ˆ β -β ∗ ∥ . Furthermore, if f is µ -strongly convex, then ∥ ˆ β -β ∗ ∥ ≤ ε/µ .

Proof. For the first part, we apply the mean value theorem, which states that there exists some β that is a convex combination of ˆ β and β ∗ such that:

<!-- formula-not-decoded -->

where (i) applies the mean value theorem, (ii) uses the monotonicity of gradients of convex functions:

<!-- formula-not-decoded -->

and that ˆ β -β = λ ( ˆ β -β ∗ ) for some λ ∈ [0 , 1] , and (iii) applies the ε -stationarity condition.

For the second part, by strong convexity, we have on the one hand:

<!-- formula-not-decoded -->

And on the other, by stationarity and ε -stationarity, we have that:

<!-- formula-not-decoded -->

Dividing through by ∥ ˆ β -β ∗ ∥ yields the result.

Lemma G.5. Let C ⊂ R n be a convex constraint set with β ∈ C , and let Q : C → R be a quadratic:

<!-- formula-not-decoded -->

where c ∈ R , v ∈ R n , and C &gt; 0 . Let β ∗ ∈ C minimize Q . If ∥ β ∗ -β ∥ ≥ ε &gt; 0 , then:

<!-- formula-not-decoded -->

Proof. Define the quadratic function q : R → R by:

<!-- formula-not-decoded -->

where λ ∗ = -v ⊤ ( β ∗ -β ) C ∥ β ∗ -β ∥ 2 minimizes q . Restricting Q to the line between β and β ∗ , we get:

<!-- formula-not-decoded -->

for λ ∈ [0 , 1] . This follows by expanding the definition of Q .

Notice that q monotonically decreases on the interval 0 ≤ λ ≤ λ ∗ , and also that q monotonically increases for λ &gt; λ ∗ . Because Q ( β ∗ ) = q (1) minimizes Q on the convex set C , q must be descending on λ ∈ [0 , 1] . Thus, 1 ≤ λ ∗ . It follows that 1 -2 λ ∗ ≤ -1 . Plugging in into (33), we have:

<!-- formula-not-decoded -->

Applying Q ( β 0 ) = c and ∥ β ∗ -β ∥ ≥ ε yields the result.

## H Proofs for Dueling Feedback

## H.1 Proof of Lemma 9

Lemma 9 (Bias and variance of dueling gradient estimator) . Under Assumptions A to D, let ˆ ∇ f 0 ( x ) be defined as in (14) with α = 1 / 8 , m ≍ d 4 log( d/ε 0 ) /ε 4 0 , b ≍ d/ε 2 0 , and γ ≍ ε 0 /d . Then:

<!-- formula-not-decoded -->

Before proving Lemma 9, we first need the following lemma.

Lemma H.1. Let Assumption D be true. Then, we have,

<!-- formula-not-decoded -->

Proof. We will omit the subscripts and write ˆ p i , ˜ p i = 1 m ∑ j ∈ [ m ] Y ij , p x + γu i ,x as ˆ p, ˜ p, p when there is no confusion. Consider the following three events.

1. For some constants u, l &gt; 1 , define event E 1 := { α &lt; ˜ p &lt; 1 -α | 1 /u &lt; p &lt; 1 /l } such that α &gt; 1 /u , and 1 -α &lt; 1 /l .
2. E 2 := { ˜ p ≥ 1 -α | 1 /u &lt; p &lt; 1 /l } .
3. E 3 := { ˜ p ≤ α | 1 /u &lt; p &lt; 1 /l } .

By Hoeffding's inequality, we have,

<!-- formula-not-decoded -->

Let ν := ˜ p -p . Let the event E 1 be true. Then we have ˜ p = ˆ p , and

<!-- formula-not-decoded -->

where C α = ( 1 + 1 α (1 -α ) + ul ( l -1) ) . For sufficiently small max( p -α, 1 -α -p ) &gt; β &gt; 0 , consider the event, E 4 = {| ν | &gt; β | E 1 } .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, since | ν | ≤ 2 ,

Under E 2 ,

Under E 3 ,

<!-- formula-not-decoded -->

Combining (34), (35), (36), and (37), we obtain,

<!-- formula-not-decoded -->

Proof of Lemma 9. Since Algorithm 1 ensures that x k are contained in a compact set, Assumption C implies that ∥∇ f 0 ∥ 2 ≤ G for some constant G &gt; 0 as observed in (30). 4

<!-- formula-not-decoded -->

By Lipschitz continuity of σ ( · ) , using σ (0) = 1 / 2 , and γ ≤ 1 / (4 L 1 G ) , we have, ( l, u ) = (4 / 3 , 4) . First we look at the bias of ˆ ∇ f 0 ( x ) .

<!-- formula-not-decoded -->

where c 3 &gt; 0 is a constant. The first inequality follows by triangle inequality, the second inequality follows by Lemma H.1, Section 3 of Agarwal et al. (2010), and choosing α = 1 8 , and the third inequality follows by choosing m = max ( 1 /β 2 , 64 ) log(1 /β ) , the fourth inequality follows by choosing β = γ 2 , and the fifth inequality follows by choosing γ = ε 0 /d . Note that with these choice of α, u, l , one has C α is bounded by a constant 27 .

Now consider the variance term.

<!-- formula-not-decoded -->

4 To be precise, in the dueling feedback case, since we do not know ∇ f 0 ( x ) , we need to make a mild assumption that there is at least one point x b ∈ Ω where Ω is some compact set containing Pareto( F ) such that ∇ f 0 ( x b ) ≤ B where B &gt; 0 is a constant. Then the algorithm can always be initiated at x = x b and the ∇ f 0 ( x ) remain uniformly bounded as per (30). This is a mild assumption because if the gradient is large everywhere, then the problem itself becomes meaningless to study. For all practical purposes, we can assume ∥∇ f 0 ∥ 2 ≤ G .

where the first inequality follows by Young's inequality, the second inequality follows by Section 3 of Agarwal et al. (2010), the third inequality follows by Young's and Cauchy Schwarz inequality, the fourth inequality follows by Lemma H.1, Lemma 2 of Bach and Perchet (2016), and choosing α = 1 8 , the fifth inequality follows by choosing m = max ( 1 /β 2 , 64 ) log(1 /β ) , and the seventh inequality follows by choosing b = d/ε 2 0 , β = γ 2 , and γ = ε 0 /d .

## H.2 Proof of Theorem 11

We expand Theorem 11 here.

Theorem H.1 (Convergence of PMM with Dueling Feedback) . Let F , f 0 , and σ satisfy Assumptions A, B, C, and D. Fix 0 &lt; ε 1 / 2 ≤ ε 0 ≤ 1 . Let ˆ x β and ˆ β be the approximate solutions that are returned by the black-box optimizer for ˆ g ( · ; x, β ) and f β ( · ) , defined in (13) and (2) , respectively:

<!-- formula-not-decoded -->

Given constants c 1 , c 2 &gt; 0 , suppose the black-box optimizer achieves the following guarantees:

1. the approximate minimizer ˆ β is O ( ε 0 ) -approximately stationary:

<!-- formula-not-decoded -->

2. the approximate minimizer ˆ x β is an O ( ε 2 0 ) -approximate solution:

<!-- formula-not-decoded -->

Let ( x k , β k ) k be the iterates of Algorithm 1 with dueling feedback. Then, choosing α = 1 / 8 , m ≍ d 4 log( d/ε 0 ) /ε 4 0 , b ≍ d/ε 2 0 , and γ ≍ ε 0 /d in (14) , there exist c 1 ( f 0 , F ) and c 2 ( f 0 , F ) bounded away from zero and some K such that E [( f 0 ◦ x ∗ )( β k )] is monotonically decreasing for k ∈ [ K ] and ( x K , β K ) is an ( ε 0 , ε ) -preference stationary point in expectation, i.e., ∥∇ f β K ( x K ) ∥ 2 ≤ ε , and conditions (a) and (b) of Lemma 7 hold in expectation for ( x K , β K ) . In particular, we have the following.

<!-- formula-not-decoded -->

Also,

<!-- formula-not-decoded -->

Proof of Theorem 11 Fix k &gt; 1 . For short, we let:

<!-- formula-not-decoded -->

Claim. At each iteration, either (i) the preference improves by at least a constant on expectation:

<!-- formula-not-decoded -->

or (ii) the point (ˆ x, ˆ β ) is ( ε 0 , ε ) -preference stationary on expectation.

Assuming the claim holds, the theorem immediately follows: if the algorithm in K steps has not found an ( ε 0 , ε ) -preference stationary point, then E [ f 0 ( x β k )] must decrease every iteration by a constant. But as f 0 ◦ x ∗ is lower bounded over ∆ n -1 by f ∗ , on expectation, this can happen at most:

<!-- formula-not-decoded -->

Proof of the claim. Let β ∗ := arg min β ′ ∈ ∆ n -1 g ( β ′ ; x, β ) , and ˆ β ∗ := arg min β ′ ∈ ∆ n -1 ˆ g ( β ′ ; x, β ) . Lemma G.4 shows that an approximate stationary point ˆ β of a strongly convex function is close to the exact point ˆ β ∗ :

<!-- formula-not-decoded -->

where we let δ denote this constant for short. Since g ( β ′ ; x, β ) is a strongly convex function and simplex is a compact convex set, by tilt stability of solution mapping of strongly convex functions (see Proposition 2G.4 of Dontchev and Rockafellar (2009)), we have,

<!-- formula-not-decoded -->

The last inequality follows by Lemma E.2.

There are two cases, leading to either (1) O ( ε 0 ) -preference stationarity or (2) O ( ε 2 0 ) -constant descent. The two cases depend on the suboptimality of β .

Case 1: ∥ β -β ∗ ∥ 2 &lt; 2 δ . Here, β is fairly close to the optimum β ∗ of the true surrogate. We show that the approximate stationarity of ˆ β with respect to the noisy surrogate implies approximate expected preference stationarity. Similar to Theorem 10, (ˆ x, ˆ β ) is expected ( ε 0 , ε ) -preference stationary provided:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Observe that,

<!-- formula-not-decoded -->

combining (39), (40), and the condition that ∥ β ∗ -β ∥ 2 &lt; 2 δ .

Note that (41) follows trivially by Condition 2 of Theorem H.1. One can efficiently find such a ˆ β with log(1 /ε 0 ) iteration complexity as it just requires optimizing a strongly-convex objective over a compact convex set.

Now we show (42). We have the following bound.

<!-- formula-not-decoded -->

The equality follows by the definition of ˆ g . The first inequality follows by Condition 1 of Theorem 11, triangle inequality, and Holder's inequality. The second inequality follows by (39), (40), and by the condition ∥ β ∗ -β ∥ 2 &lt; 2 δ in Case 1.

Now, for all β ′ ∈ ∆ n -1 we have,

<!-- formula-not-decoded -->

where (i) adds and subtracts ∇ f 0 ( x ) ⊤ ̂ ∇ x ∗ ( x, β )( β ′ -ˆ β ) and applies Hölder's inequality, (ii) follows by adding and subtracting -ˆ ∇ f 0 ( x ) ⊤ ̂ ∇ x ∗ ( x, β )( β ′ -ˆ β ) , Lemma E.2, Holder's inequality, and the fact that ∥ x ∥ ∞ ≤ ∥ x ∥ 2 , (iii) follows by (45), Lemma E.2, and Lemma G.3, and (iv) bounds ∥ β -ˆ β ∥ 2 using (25), and (iv) follows by (44). Taking conditional expectation on both sides, using Lemma 9, and Cauchy-Schwarz inequality, we get,

<!-- formula-not-decoded -->

Taking expectation on both sides and by the definition of c 1 , and observing ∥ ∥ ∥ β ′ -ˆ β ∥ ∥ ∥ 1 ≤ 2 , we have,

<!-- formula-not-decoded -->

To show (43), we have:

<!-- formula-not-decoded -->

where (i) expands out err ∇ f 0 , (ii) uses the fact that β ↦→∥∇ F ( x ) ⊤ β ∥ 2 is ∥∇ F ( x ) ⊤ ∥ 2 -Lipschitz in β with respect to the ℓ 2 -norm, and (iii) applies the definition of err ∇ f 0 and (44). Taking conditional expectation on both sides, and by Lemma 9,

<!-- formula-not-decoded -->

Taking expectation on both sides, and by definitions of c 1 and c 2 , we have,

<!-- formula-not-decoded -->

Case 2: ∥ β ∗ -β ∥ 2 ≥ 2 δ . Here β is suboptimal and β ∗ achieves a large descent. Akin to (29), we have,

<!-- formula-not-decoded -->

Now we have,

<!-- formula-not-decoded -->

where (i) uses the majorizing property of g , (ii) applies (48), and strong convexity of g ( · ; x, β ) , (iii) follows from the definition of ˆ g ( · ; x, β ) , (iv) follows from Hölder's inequality, and choice of c 1 , (v) follows by (39), (40), and Lemma E.2.

Taking conditional expectation on both sides, using Lemma 9

<!-- formula-not-decoded -->

Taking expectation on both sides,

<!-- formula-not-decoded -->

Observe that, choosing parameters properly in Lemma 9, we can always ensure

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, in expectation, the preference improves by at least a constant. The choices of c 1 and c 2 follows similarly as in Theorem 10.

■

Then, we have,

## I Details to Visualization of Toy Examples

Figure 2 visualizes learning dynamics for two different Pareto-constrained optimization problems. The code can be found at https://github.com/geelon/preference-pareto .

Preference optimization on a non-smooth Pareto set The left sub-figure is an example of optimization on the Pareto set described in Example 1. In particular, the preference function is:

<!-- formula-not-decoded -->

Hyperparameters used using the PNG algorithm by Ye and Liu (2022) include the Pareto-set approximation threshold e = 0 . 001 , learning rate ξ = 0 . 01 , and regularization parameter α t = 0 . 01 . These were not particularly tuned. The two different phases in the algorithm can be clearly observed in the dynamics. In particular, when the iterates are close to the Pareto set, the PNG dynamics minimizes f 0 . Otherwise, it aims to move toward the Pareto set in a direction that also optimizes the preference. This leads to the jagged appearance of the PNG dynamics. We ran 40k iterations.

The PMM algorithm visualize in this example alternates between updating x k and β k . Each update step for β k is implemented by an approximate gradient descent, making use of ˆ ∇ x ∗ , followed by a projection back onto the simplex. The learning rate for β chosen here was 0 . 1 . Then, the update step for x k consists of a single step of gradient descent on f β k with learning rate of 0 . 1 . A total of 1500 iterations was run. These hyperparameters were not specifically tuned.

Preference optimization where first-order information is insufficient The right sub-figure visualizes the counterexample given in Example D.1.

The PNG algorithm in this example used parameters e = 0 . 001 , ξ = 0 . 1 , and α = 0 . 01 . We ran the algorithm for 26k iterations. The PMM algorithm in this example is the same as the previous example. The learning rate chosen here was 0 . 1 . The inner loop used 1 step of gradient descent with learning rate of 0 . 1 . We ran 1500 iterations and did not attempt to tune hyperparameters.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Abstract and introduction clearly states our main result with explicit rate. As this is a theoretical paper, this summarizes our main contribution.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We mention after Theorem 10 that dimension dpeendence in the dueling feedbak setting is unavoidable. Also our work is for strongly convex objectives.

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

Justification: All the assumptions are in the main body. Proofs are mostly in the Appendix.

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

Justification: The code is contained in the supplementary materials and at https://github. com/geelon/preference-pareto .

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

Justification: NA

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

Justification: No experiment.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: NO experiment.

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

Justification: No esperiment.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We abide by all the rules.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Even though this is a theoretical work, we mention the introduction about the possible applications nad their implications of our framework.

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

Justification: No data or model used.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Yes.

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

Justification: NA.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: NA.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [No]

Justification: No such participant.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [No]

Justification: No.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.