## Non-stationary Bandit Convex Optimization: A Comprehensive Study

Xiaoqi Liu ∗ Dorian Baudry ∗ † Julian Zimmert ‡ Patrick Rebeschini ∗ Arya Akhavan ∗ §

## Abstract

Bandit Convex Optimization is a fundamental class of sequential decision-making problems, where the learner selects actions from a continuous domain and observes a loss (but not its gradient) at only one point per round. We study this problem in non-stationary environments, and aim to minimize the regret under three standard measures of non-stationarity: the number of switches S in the comparator sequence, the total variation ∆ of the loss functions, and the path-length P of the comparator sequence. We propose a polynomial-time algorithm, Tilted Exponentially Weighted Average with Sleeping Experts (TEWA-SE), which adapts the sleeping experts framework from online convex optimization to the bandit setting. For strongly convex losses, we prove that TEWA-SE is minimax-optimal with respect to known S and ∆ by establishing matching upper and lower bounds. By equipping TEWA-SE with the Bandit-over-Bandit framework, we extend our analysis to environments with unknown non-stationarity measures. For general convex losses, we introduce a second algorithm, clipped Exploration by Optimization (cExO), based on exponential weights over a discretized action space. While not polynomial-time computable, this method achieves minimax-optimal regret with respect to known S and ∆ , and improves on the best existing bounds with respect to P .

## 1 Introduction

Many real-world decision-making problems, such as resource allocation, experimental design, or hyperparameter tuning require repeatedly selecting an action from a continuous space under uncertainty and limited feedback. These settings are naturally modeled as Bandit Convex Optimization (see [1] for an introduction), in which an adversary fixes a sequence of T loss functions f 1 , f 2 , . . . , f T : R d → R beforehand, and a learner sequentially interacts with the adversary for T rounds. At each round t , the learner selects an action z t from a continuous arm set Θ ⊆ R d , assumed to be convex and compact. The learner then incurs a loss f t ( z t ) and observes a noisy feedback:

<!-- formula-not-decoded -->

where ξ t is a sub-Gaussian noise variable (Definition 1). The goal is to minimize the learner's regret with respect to (w.r.t.) a performance benchmark. In the online learning literature [2, 3], the benchmark is typically the best static action in hindsight, with cumulative loss min z ∈ Θ ∑ T t =1 f t ( z ) .

However, non-stationarity arises in many applications where different actions may work well during different time intervals. Hence, a line of works [4-8] propose to compare the learner's actions against

∗ University of Oxford. Correspondence to: {shirley.liu, arya.akhavan}@stats.ox.ac.uk .

† Univ. Grenoble Alpes, Inria, CNRS, Grenoble INP, LIG, 38000 Grenoble, France.

‡ Google Research.

§ École Polytechnique de Paris, IP Paris.

a sequence of comparators u 1 , . . . , u T ∈ Θ , leading to a regret defined as

<!-- formula-not-decoded -->

where u 1: T denotes ( u t ) T t =1 , and the expectation is taken w.r.t. the randomness in the learner's actions z t 's and the randomness of the noise variables ξ t 's, similarly to the standard notion of pseudo-regret in the bandit literature, see e.g., [9, Section 4.8]. Choosing the regret-maximizing comparator in (2) gives rise to the notion of dynamic regret , defined as

<!-- formula-not-decoded -->

While addressing non-stationarity through dynamic regret has been extensively studied in multiarmed bandits (e.g., [10-12]), it remains relatively underexplored in continuum bandits [7, 13, 14]. This work aims to bridge this gap by proposing algorithms for Bandit Convex Optimization that achieve sublinear dynamic regret. Such a rate is generally unattainable without imposing structural constraints on the environment, i.e., the comparator sequence and the loss function sequence [6]. For the comparator sequence, two commonly studied constraints are the number of switches [15] and the path-length [4], defined respectively as

/negationslash

<!-- formula-not-decoded -->

For the loss function sequence, a popular constraint is the total variation [7], defined as

<!-- formula-not-decoded -->

The constraints that the upper bounds S , P and ∆ respectively impose on the comparators or on the loss functions lead to different notions of regret. We call the regret for environments constrained by S the switching regret , which we define as

<!-- formula-not-decoded -->

Similarly, we call the regret for environments constrained by P the path-length regret , denoted by R path ( T, P ) . We also use R dyn ( T, ∆ ) and R dyn ( T, ∆ , S ) to denote the dynamic regret, where the arguments after T specify environment constraints. See (8) for rigorous definitions.

We detail in Section 1.3 conversion results between the different regret definitions that we introduced: a sublinear switching regret R swi ( T, S ) implies sublinear R dyn ( T, ∆ ) , R dyn ( T, ∆ , S ) and R path ( T, P ) , as illustrated in Figure 1 (see also [16, 17]). Furthermore, the upper bounds on the switching regret presented in this work are derived from upper bounds on the adaptive regret [18, 19], which is defined using an interval length B ∈ [ T ] as follows,

<!-- formula-not-decoded -->

With an appropriate tuning of B depending on S , an adaptive regret sublinear in B implies a switching regret sublinear in T through a simple reduction, see e.g., discussions in [19]. We illustrate the relations between these regret notions in Figure 1.

<!-- formula-not-decoded -->

Figure 1: Conversions between regrets: R 1 R 2 means that if regret R 1 is sublinear in T (or B ), then regret R 2 is also sublinear in T , see Proposition 1 for precise mathematical statements.

We conclude this section by detailing the main notation and assumptions used throughout the paper.

Notation For k ∈ N + , we denote by [ k ] the set of positive integers ≤ k . We denote by ‖ · ‖ the Euclidean norm, B d = { u ∈ R d : ‖ u ‖ ≤ 1 } the unit Euclidean ball, and Π Θ ( x ) = arg min w ∈ Θ ‖ w -x ‖ the Euclidean projection of x to Θ . We use a ∨ b ≡ max( a, b ) and a ∧ b ≡ min( a, b ) . If A,B depend on T , we use A = O ( B ) (resp. A = Ω ( B ) ) when there exists c &gt; 0 s.t. A ≤ cB (resp. ≥ ) with c independent of T, d, S, ∆ and P . To hide polylogarithmic factors in T , we use interchangeably A = ˜ O ( B ) and A /lessorsimilar B (resp. A = ˜ Ω ( B ) and A /greaterorsimilar B ), e.g., A ≤ T log T = ⇒ A = ˜ O ( T ) . Moreover, A = o ( B ) means A/B → 0 as T →∞ .

Assumptions For simplicity, we assume that the time horizon T is known in advance; the case of unknown T can be handled using the standard doubling trick [20]. For some σ &gt; 0 , the noise variables ( ξ t ) T t =1 are σ -sub-Gaussian. For all t ∈ [ T ] , max x ∈ Θ | f t ( x ) | ≤ 1 . We consider two cases: (i) general convex losses f t , where we assume Lipschitz continuity with constant K , and (ii) the special stronglyconvex case, where we assume β -smoothness. The domain Θ is assumed to contain a ball of radius r for some constant r &gt; 0 , and has a bounded diameter diam ( Θ ) := sup { ‖ x -w ‖ : x , w ∈ Θ } ≤ D for some constant D &gt; 0 .

## 1.1 Main contributions

Existing works on non-stationary Bandit Convex Optimization study different aspects of the problem in isolation: [7, 14] focus on dynamic regret R dyn ( T, ∆ ) , while [13, 21] address path-length regret R path ( T, P ) . The present work aims to systematically unify and extend previously scattered results, establishing a complete picture of the state-of-the-art regret bounds w.r.t. all three non-stationarity measures S, ∆ and P .

Our first contribution is a polynomial-time algorithm called Tilted Exponentially Weighted Average with Sleeping Experts (TEWA-SE), which we design by adapting a series of works from online convex optimization [22-24] to the bandit setting with zeroth-order feedback. It addresses the absence of gradient information by employing the randomized perturbation technique from [25, 26] to estimate gradients, combined with the design of quadratic surrogate loss functions depending on a uniform upper bound on the norm of the gradient estimates.

Following [22-24], TEWA-SE runs multiple expert algorithms with different learning rates in parallel, and combines them using a tilted exponentially weighted average. This allows TEWA-SE to adapt to the curvature of the loss function f t 's without prior knowledge of parameters such as the strong-convexity parameter. For a given interval length B , an appropriately tuned TEWA-SE simultaneously achieves an adaptive regret of the order √ d B 3 4 for general convex losses and d √ B for strongly-convex losses (Theorem 1). Consequently, for a known S , we prove that an optimal tuning of TEWA-SE yields a switching regret upper bound of order √ dS 1 4 T 3 4 for general convex losses (Corollary 1). In the same result, we further prove that if the losses are strongly convex, and that ∆ is known and incorporated in the tuning of TEWA-SE, the algorithm simultaneously satisfies a min { d √ ST,d 2 3 ∆ 1 3 T 2 3 } dynamic regret bound. Importantly, TEWA-SE does not need to know the actual strong-convexity parameter, inheriting the adaptivity properties of the framework developed in [22-24]. We prove that this dynamic regret upper bound is minimax-optimal in T, d, S and ∆ by establishing a matching lower bound (Theorem 2). Finally, still for strongly-convex losses, we prove that TEWA-SE can also achieve a path-length regret of the order d 2 3 P 1 3 T 2 3 when P is known. We summarize these results in Table 1. To overcome the restriction of knowing S, ∆ and P to optimally tune TEWA-SE, we also analyze a variant equipped with the Bandit-over-Bandit framework [27].

Table 1: Regret bounds we obtain for R swi ( T, S ) , R dyn ( T, ∆ ) and R path ( T, P ) , respectively, for algorithms tuned with known S, ∆ and P (polylogarithmic factors omitted). Straight underlines indicate minimax-optimal rates. A wavy underline indicates the result is either new to the literature (strongly-convex case) or improves on the best-known P 1 4 T 3 4 rate [13] (general convex case).

|                        | TEWA-SE (Algorithm 1)                                                                                        | cExO (Algorithm 3)                                     |
|------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| Convex Strongly convex | √ dS 1 4 T 3 4 , d 2 5 ∆ 1 5 T 4 5 , d 2 5 P 1 5 T 4 5 d √ ST, d 2 3 ∆ 1 3 T 2 3 , d 2 3 P 1 3 T 2 3 ✿✿✿✿✿✿✿ | d 5 2 √ ST, d 5 3 ∆ 1 3 T 2 3 ,d 5 3 P 1 3 T 2 3 ✿✿✿✿✿ |

For general convex losses with known S, ∆ and P , TEWA-SE achieves a suboptimal T 3 4 rate (Corollary 1), matching the rates in similar analysis for the static regret [25, 26]. Thus, the second contribution of this work is the clipped Exploration by Optimization (cExO) algorithm with improved guarantees for this setting, which uses exponential weights on a discretized action space Θ with clipping [28]. For a given interval length B , this algorithm with an optimally tuned learning rate w.r.t. B attains an order d 5 2 √ B adaptive regret (Theorem 3). When S, ∆ and P are known beforehand, this algorithm with an optimally tuned learning rate achieves the minimax-optimal dynamic regret w.r.t. S and ∆ simultaneously, and attains a P 1 3 T 2 3 path-length regret (Corollary 2), improving on the previous best P 1 4 T 3 4 [13]. While this algorithm is not polynomial-time computable and has suboptimal rates w.r.t. the problem dimension d , it provides insights that may guide future research toward developing efficient algorithms with optimal guarantees for the convex case.

## 1.2 Related work

The literature on Bandit Convex Optimization (BCO) has traditionally focused on minimizing the static regret, see the recent monograph [1] for a comprehensive historical overview. Both convex and strongly convex objective functions have attracted significant attention, beginning with the foundational work of [25] and further developed in subsequent studies such as [29-36]. Minimizing regret in non-stationary environments has only received attention more recently [7, 13, 14, 21], see also [1, Section 2.4] for an overview for this topic. Among these works, [7, 14] study R dyn ( T, ∆ ) , whereas [13, 21] focus on R path ( T, P ) . As we explained above (and formalize in Section 1.3), the switching regret R swi ( T, S ) can induce guarantees on both R dyn ( T, ∆ ) and R path ( T, P ) , but the reverse does not necessarily hold. Therefore, the results in these works cannot be readily extended to provide regret guarantees w.r.t. all three measures S, ∆ and P .

Minimizing regret in environments with non-stationarity measures such as S, ∆ and P have been addressed with greater depth in Online Convex Optimization (OCO), where the learner has direct access to gradient information and can query the gradient or function value at multiple points of the loss function per round. The state-of-the-art algorithm with optimal adaptive regret guarantees is MetaGrad with sleeping experts [24], which queries only one gradient per round, and adapts to curvature information of the loss function such as strong-convexity when available. Our polynomialtime algorithm TEWA-SE builds upon [24] and its precursors [22, 23], adapting this approach to BCO by replacing the exact gradient per round with an approximate gradient estimate, and by designing a quadratic surrogate loss. The approach in [24] follows a long line of successive developments in OCO from expert tracking methods [15, 20, 37-41] to the study of adaptive regret [18, 19, 42-46], with recent advances [24, 47-50] reducing the query complexity from O (log T ) to O (1) per round, while achieving optimal adaptive regret or dynamic regret. The adaptivity of [24] directly inherits from MetaGrad [22] and its extension [23], which themselves build on earlier adaptive methods [51, 52].

For general convex functions, the approach of substituting a one-point gradient estimate for the exact gradient in each round of an OCO algorithm often yields suboptimal T 3 4 rates, both in static regret [25, 26] and dynamic regret [13, 21]; see also our Corollary 1. A series of breakthroughs [28, 32, 53-56] indicate that √ T rates (up to logarithms) are attainable for static regret, at the cost of a higher dependency on d . Our cExO algorithm follows this line of work, using exponential weights on a discretized action space [28]. By playing inside a clipped domain, we transform the algorithm from one with √ T static regret into one with √ B adaptive regret (modulo logarithms) for intervals of length ≤ B , which in turn leads to regret guarantees w.r.t. S, ∆ and P .

Finally, we mention that non-stationarity has been widely studied in the Multi-Armed Bandit (MAB) literature. A substantial body of work has focused on adapting standard policies-such as UCB [57, 58], EXP3 [59], and Thompson Sampling [60-62]-to perform effectively under non-stationarity. These adaptations often employ mechanisms to discard outdated information, either actively (e.g., change-detection methods [12, 63-67]), or passively (e.g., discounted rewards [10, 68], sliding windows [69, 70], or scheduled restarts [11]), but are not straightforward to adapt to BCO.

## 1.3 Conversions between different regret definitions

We present the key conversions between different regret notions, illustrated in Figure 1 above. Using the definition of R dyn ( T ) in (3), we overload notation slightly to define

<!-- formula-not-decoded -->

/negationslash and R dyn ( T, ∆ , S ) additionally constrains 1 + ∑ T t =2 min ( z ∗ t , z ∗ t -1 ) ∈ ( Z ∗ t , Z ∗ t -1 ) 1 ( z ∗ t = z ∗ t -1 ) ≤ S where Z ∗ t := arg min z ∈ Θ f t ( z ) for all t ∈ [ T ] . In Proposition 1, we show how the adaptive regret R ada ( B , T ) can be used to bound the switching regret R swi ( T, S ) , which in turn can be used to bound the dynamic regret R dyn ( T, ∆ , S ) and path-length regret R path ( T, P ) . Consequently, R ada ( B , T ) and R swi ( T, S ) are the primary objects to analyze.

Proposition 1. Suppose that an algorithm can be calibrated to satisfy R ada ( B , T ) ≤ C B κ , for any interval length B ∈ [ T ] , for some factor C &gt; 0 that is at most polynomial in d and log( T ) , and κ ∈ [0 , 1) .

Then, for any S, S ∆ , S P ∈ [ T ] , an appropriate choice of B yields the following regret guarantees:

<!-- formula-not-decoded -->

The proof is provided in Appendix B. We note that the reduction from R path ( T, P ) to R swi ( T, S ) in Proposition 1 is new and employs simple geometric arguments (see Lemma 2 in Appendix B). This reduction simplifies the analysis of R path ( T, P ) , though it can yield slightly looser bounds on R path ( T, P ) than a direct analysis, as discussed in [17].

## 2 The TEWA-SE algorithm

In this section, we develop a polynomial-time algorithm called Tilted Exponentially Weighted Average with Sleeping Experts (TEWA-SE, Algorithm 1), building on the two-layer structure of previous experts-based algorithms [18, 19, 43]. Each expert in TEWA-SE is uniquely defined by its lifetime and learning rate. We denote the active experts at time t by E 1 , E 2 , . . . , E n t , where E i operates over interval I i with learning rate η i . In each round t , the active experts each propose an action, denoted by x η i t,I i , and a meta-algorithm aggregates them into a single meta-action x t by computing their tilted exponentially weighted average [22, 24], see line 7 in the pseudo-code. Then the algorithm receives a noisy evaluation of f t at x t and constructs an approximate gradient estimate g t ∈ R d of f t at x t . Both x t and g t are shared with all experts, who update their actions via online gradient descent on their surrogate loss functions defined using x t and g t .

TEWA-SE employs the Geometric Covering scheme from [19, 24] to schedule experts across different time intervals, and the exponential grid from [22, 24] to assign varied learning rates to the experts. These deterministic schemes ensure that only a logarithmic number of experts are active per round, maintaining computational efficiency. Intuitively, the meta-algorithm achieves low adaptive regret on the original loss function because, for each subinterval of times, there exists at least one individual expert with low static regret on their surrogate loss functions on this subinterval. This is guaranteed by the careful design of the exponential grid of learning rates. While full details of TEWA-SE is deferred to Appendix C.1, we highlight below the distinctions between this paper and prior works.

/negationslash

Construction of one-point gradient estimate For a fixed parameter h ∈ (0 , r ) , we define the clipped domain ˜ Θ = { u ∈ Θ : u + h B d ⊂ Θ } , where h &lt; r ensures ˜ Θ = ∅ . In each round t , we select a meta-action x t ∈ ˜ Θ and query the function at a perturbed point x t + h ζ t , receiving noisy feedback y t = f t ( x t + h ζ t ) + ξ t , where ζ t ∈ R d is sampled uniformly from the unit sphere ∂ B d . This allows us to construct the gradient estimate g t = ( d/h ) y t ζ t . As implied by [25, Lemma 1], the

Algorithm 1 Tilted Exponentially Weighted Average with Sleeping Experts (TEWA-SE)

```
Input: d, T, B , h = min ( √ d B -1 4 , r ) , ˜ Θ = { u ∈ Θ : u + h B d ⊂ Θ } , G as in (10), expert algorithm E ( I, η ) defined in Algorithm 2, and ( n t ) t ∈ [ T ] and ( I i , η i ) i ∈ [ n t ] ∀ t ∈ [ T ] 1: for t = 1 , 2 , . . . , T do 2: for E i ≡ E i ( I i , η i ) ∈ { E 1 , E 2 , . . . , E n t } do /triangleright n t experts active at t 3: Receive action x η i t,I i from expert E i 4: if min { τ : τ ∈ I i } = t then initialize L η i t -1 ,I i = 0 , clipped domain ˜ Θ and parameter G 5: end if 6: end for 7: Set meta-action as x t = ∑ n t i =1 η i exp( -L η i t -1 ,I i ) x η i t,I i / ∑ n t j =1 η j exp( -L η j t -1 ,I j ) 8: Sample ζ t uniformly from ∂ B d 9: Query point z t = x t + h ζ t to obtain y t = f t ( z t ) + ξ t 10: Construct gradient estimate g t = ( d/h ) y t ζ t 11: for i = 1 , 2 , . . . , n t do 12: Send meta-action x t and g t to E i 13: Increment cumulative loss L η i t,I i = L η i t -1 ,I i + /lscript η i t ( x η i t,I i ) /triangleright /lscript η i t ( · ) depends on x t and g t 14: end for 15: end for
```

vector g t is an unbiased gradient estimate of a spherically smoothed version of f t at x t , satisfying

<!-- formula-not-decoded -->

In our setting, under the high probability event Λ T = { | ξ t | ≤ 2 σ √ log( T +1) , ∀ t ∈ [ T ] } , we have with ˜ ζ distributed uniformly on the unit ball B d . Importantly, ˆ f t inherits the convexity properties of f t [71, Lemmas A.2-A.3]. Our approach differs from related works in OCO [22-24, 47, 48] that use exact gradients in two key ways: i) in each round, we query the perturbed point z t = x t + h ζ t rather than x t , accumulating regret at the perturbed point, and ii) we constrain x t inside the clipped domain ˜ Θ to ensure all perturbed z t remain feasible.

<!-- formula-not-decoded -->

This implies a fundamental tradeoff in selecting the smoothing (and clipping) parameter h : larger values reduce G (and the variance of g t ), but increase both the approximation error between ˆ f t and f t and the error due to clipping, while smaller values reduce bias at the cost of a higher variance in g t . In Theorem 1 and Corollary 1, we establish the optimal h and the resulting regret guarantees.

Algorithm 2 Expert algorithm E ( I, η ) : projected online gradient descent (OGD)

Input: I = [ r, s ] , η , G , clipped domain ˜ Θ , and surrogate loss /lscript η t ( · ) defined in (11) ∀ t ∈ N + Initialize: x η r,I be any point in ˜ Θ 1: for t = r, r +1 , . . . , s do 2: Send action x η t,I to Algorithm 1 3: Receive meta-action x t and g t from Algorithm 1 4: Update x η t +1 ,I = Π ˜ Θ ( x η t,I -µ t ∇ /lscript η t ( x η t,I ) ) , where µ t = 1 / (2 η 2 G 2 ( t -r +1)) 5: end for

Design of expert algorithms and surrogate losses We choose projected online gradient descent (OGD) as the expert algorithms (Algorithm 2), i.e., each expert E ( I, η ) runs OGD during its lifetime I . In the full-information setting, where experts observe f t and gradients are evaluated at all of their actions, each expert could simply run OGD on the true loss functions. In contrast, for the bandit setting, with only one gradient estimate g t of the smoothed loss ˆ f t per round, we need to construct surrogate losses for the experts. The simplest option is the linear surrogate loss /lscript t ( x ) = -g /latticetop t ( x t -x ) , but this fails to leverage curvature information and leads to a large ˜ O ( √ | I | ) static regret for each expert, ultimately yielding linear adaptive regret.

To address these limitations, inspired by [22-24], we design the following strongly-convex surrogate loss /lscript η t : R d → R :

<!-- formula-not-decoded -->

where G is the upper bound (10) on ‖ g t ‖ , and η is the learning rate of the expert. We highlight that our choice of the quadratic term in (11) differs from the η 2 ‖ g t ‖ 2 ‖ x t -x ‖ 2 and η 2 ( g /latticetop t ( x t -x )) 2 in [24] and [22]. The latter necessitates an additional condition relating E [ ‖ g t ‖ ] and E [ ‖ g t ‖ 2 ] (or E [ g t g /latticetop t ] ) to be satisfied in the analysis, see e.g., [22, Theorem 2], and may yield suboptimal rates in dimension d for strongly-convex losses, similar to [22]. Our choice of the quadratic term, similar to [23], eliminates these limitations and simplifies the proof.

For a comparator u ∈ Θ , (11) implies that the linearized regret associated with ˆ f t on interval I can be bounded as:

<!-- formula-not-decoded -->

Due to the strong-convexity of /lscript η t , each expert attains only an O (log | I | ) static regret under OGD with an optimally tuned step size µ t (see line 4 of Algorithm 2, and Lemma 6 in Appendix C.4). This ensures term A above is also of O (log | I | ) . By the convexity of ˆ f t we have

<!-- formula-not-decoded -->

where α = 0 for general convex ˆ f t (and f t ), and α &gt; 0 for strongly-convex. Since both α and ∑ t ∈ I ‖ x t -u ‖ 2 are unknown a priori, we use a deterministic exponential grid of η values [19, 24], ensuring at least one expert covering I effectively minimize the RHS of (13), ultimately yielding a sublinear adaptive regret w.r.t. f t . We present this result in the following theorem.

Theorem 1. For any T ∈ N + and B ∈ [ T ] , Algorithm 1 with h = min( √ d B -1 4 , r ) satisfies

<!-- formula-not-decoded -->

and if f t is α -strongly-convex with arg min x ∈ R d f t ( x ) ∈ Θ for all t ∈ [ T ] , 1 it furthermore holds that

<!-- formula-not-decoded -->

where /lessorsimilar conceals polylogarithmic terms in B and T , independent of d and α .

The proof of Theorem 1 can be found in Appendix C.2. We emphasize that TEWA-SE does not require knowledge of the strong-convexity parameter α . This parameter is only used in the analysis and appear in the upper bound (15). Compared to the O ( √ B log T ) and O ( 1 α log T log B ) adaptive regrets in [24] for general convex and strongly-convex losses respectively, our bounds in Theorem 1 reflect the separation between online first-order and zeroth-order optimization. This mirrors the established gap in static regret analyses, see e.g. [74] vs. [72]. We further note that our bound for the strongly-convex case has a 1 α dependency, which is suboptimal compared to the 1 √ α dependency in [33, 73] for static regret in BCO for α /lessorsimilar 1 .

Applying Proposition 1, the adaptive regret bounds in Theorem 1 lead to the following bounds for R swi ( T, S ) , R dyn ( T, ∆ , S ) and R path ( T, P ) . In Corollary 1, for clarity we drop the /ceilingleft · /ceilingright operators from the expressions for B and assume without loss of generality B is an integer (proof in Appendix C.5).

Corollary 1. Consider any horizon T ∈ N + and assume that, for all t ∈ [ T ] , the loss f t is convex, or strongly-convex with arg min x ∈ R d f t ( x ) ∈ Θ . We refer to the second scenario as the strongly-convex (SC) case. Then, Algorithm 1 tuned with parameter B satisfies the following regret guarantees:

1 The assumption that loss minimizers lie inside Θ is common in zeroth-order optimization, see e.g., [7, 72, 73]. Without it, our upper bound analysis would have an extra term depending on the gradients at the minimizers.

<!-- formula-not-decoded -->

## 2.1 Lower bound for strongly-convex loss functions

In this section, we derive a minimax lower bound on the dynamic regret and path-length regret, and discuss the optimality of TEWA-SE. To derive the lower bound for the dynamic regret, we adopt a standard minimax approach by constructing a class of hard functions, following [71, Theorem 6.1]. We assume that the adversary either (i) partitions the time horizon into S segments and assigns a different function from this class to each segment, or (ii) selects a sequence of functions with total variation bounded by ∆ .

Theorem 2. Let Θ = B d . For α &gt; 0 denote by F α the class of α -strongly convex and smooth functions. Let π = { z t } T t =1 be any randomized algorithm (see Appendix D for a definition). Then there exists T 0 &gt; 0 such that for all T ≥ T 0 it holds that

<!-- formula-not-decoded -->

where c 1 &gt; 0 is a constant independent of d, T , S and ∆ .

We detail the proof in Appendix D. This lower bound establishes that TEWA-SE achieves the minimax-optimal dynamic regret (up to logarithms) for strongly convex and smooth functions w.r.t. d , T , S and ∆ . We note that [7] derives a lower bound only in terms of T and ∆ , matching (16), but it does not explicitly capture the dependence on d nor does it address the interplay between S and ∆ . In the special case where S = 1 , Theorem 2 recovers the classical minimax static regret of order d √ T [71, 72]. Interestingly, for d = 1 the scaling of the lower bound as function of T, S and ∆ is the same as standard lower bounds in the non-stationary MAB literature [10, 11]. The proof of Theorem 2 can be readily adapted to consider only the measure S with the switching regret, yielding a rate of d √ ST and thereby establishing the minimax optimality of TEWA-SE's switching regret bound.

We also derive a lower bound for path-length regret analogously to that for dynamic regret. In Theorem 4 in Appendix D we show that under the same assumptions as in the statement of Theorem 2,

<!-- formula-not-decoded -->

where c 2 &gt; 0 is a constant independent of d , T and P . Hence, TEWA-SE may not achieve the optimal regret rate for path-length. Additionally, Eq. (17) improves upon the only existing d √ PT lower bound from [13] in terms of the horizon T , by leveraging a different construction of a hard instance. This improvement comes from assuming P = o ( T ) , which is necessary for sublinear regret.

## 2.2 Parameter-free guarantees

In Corollary 1, we showed that the knowledge of the non-stationarity measures S, ∆ and P allows optimal tuning of TEWA-SE's parameter B . However, these measures can be hard to estimate. To obtain guarantees without such knowledge, we further analyze TEWA-SE under the Bandit-overBandit (BoB) framework from [27] (see Appendix C.6 for details), which divides the time horizon into epochs of suitable length L and uses an adversarial bandit algorithm (e.g., EXP3) to select B for TEWA-SE in each epoch from the set B = { 2 i : i = 0 , 1 , . . . , /floorleft log 2 T /floorright } . In Corollary 3 in Appendix C.6, we adapt all the upper bounds in Corollary 1 to this framework, and show that this procedure costs an additional d 1 3 T 5 6 term for the general convex case and d 1 2 T 3 4 for the stronglyconvex case. Our parameter-free path-length regret bound P 1 5 T 4 5 + T 5 6 for the general convex case improves on the P 1 2 T 3 4 bound in [13] when P = Ω ( T 1 6 ) .

Recent works on MAB [65-67, 75, 76] have proposed algorithms that achieve optimal dynamic regret without prior knowledge of S and ∆ . However, they use procedures that crucially rely on the finiteness of the arm set, and are thus ill-suited for BCO. It remains open to determine if the minimax regret rate can be attained without such knowledge in the settings considered in this paper.

## 3 Clipped Exploration by Optimization

In this section, we propose a second algorithm (Algorithm 3) to improve upon the suboptimal rates for R dyn ( T, ∆ , S ) and R path ( T, P ) that TEWA-SE achieves for general convex loss functions. For ease of presentation, we assume in this section that the problem is noiseless, i.e., ξ t = 0 for t ∈ [ T ] . We call this algorithm clipped Exploration by Optimization (cExO), which is built on Algorithm 8.3 (ExO) in [1]. The high level idea of ExO is to run exponential weights over a finite discretization of the feasible set, denoted by C ⊂ Θ . We assume the discretization C admits a worst-case error of ε := sup f ∈ F 0 min q ∈ ∆ ( C ) E z ′ ∼ q f ( z ′ ) -min z ∈ Θ f ( z ) , where F 0 denotes the class of convex and Lipschitz functions, and ∆ ( C ) denotes the ( |C| -1) -dimensional simplex.

With q 0 initialized as the uniform distribution, in each round t , given a loss estimate ̂ s t ∈ R |C| , ExO (in its mirror descent formulation) computes q t = arg min q ∈ ∆ ( C ) 〈 q , ̂ s t -1 〉 + 1 η KL ( q || q t -1 ) , where KL is the Kullback-Leibler divergence KL ( q || p ) = ∑ |C| i =1 q i log( q i /p i ) for q , p ∈ ∆ ( C ) . The update rule in cExO departs from the vanilla ExO in this single step, by taking the minimum over the clipped simplex ˜ ∆ = ∆ ( C ) ∩ [ γ , 1] |C| where γ ∈ (0 , 1 |C| ) is a constant to be tuned, see line 2 of Algorithm 3. Clipping is a standard technique in mirror descent to ensure the algorithm does not commit too hard to any single action, and therefore detect changes in the environments more easily, yielding regret guarantees w.r.t. non-stationary measures [9, Chapter 31.1].

Given the reference distribution q t , cExO selects a playing distribution p t ∈ ∆ ( C ) and an estimator function E t ∈ E which returns an updated loss estimate for each action in C , where E denotes the set of functions that map C × [ -1 , 1] to R |C| . It does so by solving an intractable optimization problem: 2

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where, with S q ( η ˆ s ) = max q ′ ∈ ∆ ( C ) 〈 q -q ′ , η ˆ s 〉 -KL ( q ′ || q ) , the objective function is defined by

This optimization problem is intractable due to the large size of E and F 0 . 3 The role of this optimization problem is to tradeoff the worst-case cost of deviating from the desired distribution q t versus the gain of improved exploration (hence the name Exploration by Optimization). Finally, cExO samples an action z t according to p t , observes the feedback f ( z t ) and constructs a loss estimate ̂ s t = E t ( z t , f ( z t )) to be used in the subsequent round. cExO achieves the adaptive regret guarantee stated in Theorem 3 below.

<!-- formula-not-decoded -->

We then use Proposition 1 to convert the bound of Theorem 3 into the following regret guarantees w.r.t. S, ∆ and P . Like in Corollary 1, we omit /ceilingleft · /ceilingright from the expressions for B for clarity.

Corollary 2. For any horizon T ∈ N + , Algorithm 3 calibrated as in Theorem 3 and tuned with interval size B (which determines η ) satisfies the following regret guarantees:

$$Switching: T S ⇒ R T, S /lessorsimilar d 2 ST , Dynamic: B = T S ∨ ( d 5 2 T/ ∆ ) 2 3 = ⇒ R dyn ( T, ∆ , S ) /lessorsimilar R swi ( T, S ) ∧ d 5 3 ∆ 1 3 T 2 3 , Path-length: B = ( rd 5 2 T/P ) 2 3 = ⇒ R path ( T, P ) /lessorsimilar r - 1 3 d 5 3 P 1 3 T 2 3 .$$

B = = swi ( ) 5 √

2 For detailed discussions about these functions, we refer the reader to [28].

3 One can in theory bound the domain of E and discretize E , F 0 and ∆ ( C ) . The optimization problem is hence computable, though not in polynomial time.

## Algorithm 3 clipped Exploration by Optimization (cExO)

Input: d, T, B , feasible set Θ , a finite covering set C ⊂ Θ of Θ , discretization error ε , learning rate η , clipping parameter γ ∈ (0 , 1 |C| ) , and ˜ ∆ = ∆ ( C ) ∩ [ γ , 1] |C| Initialize: q 0 ,i = 1 |C| ∀ i ∈ [ |C| ] . 1: for t = 1 , . . . , T do 2: Compute q t = arg min q ∈ ˜ ∆ 〈 q , ̂ s t -1 〉 + 1 η KL ( q || q t -1 ) 3: Find distribution p t ∈ ∆ ( C ) and E t ∈ E s.t. Λ η ( q t , p t , E t ) ≤ inf p ∈ ∆ ( C ) , E ∈ E Λ η ( q t , p , E ) + η d 4: Sample z t ∼ p t and observe f t ( z t ) 5: Compute ̂ s t = E t ( z t , f t ( z t )) 6: end for

The proofs of Theorem 3 and Corollary 2 are presented in Appendix E. By comparing these results to the lower bounds in Section 2.1, we obtain that for known S, ∆ and P , cExO achieves minimaxoptimal rates in T, S and ∆ , but remains suboptimal in d (for all measures), and potentially for the path-length bound (see Eq. (17)). This suboptimal dependence on d is unsurprising since even the best known static regret bounds of [31] and [34] suffer from similar dimensional dependence. Moreover, the gap between cExO's path-length regret bound and our minimax lower bound of order d 4 5 P 2 5 T 3 5 may stem from either (i) looseness in the lower bound, or (ii) sub-optimality of cExO, which runs OMD in distribution space rather than directly on the action set. The latter may allow us to bound path-length regret more directly and sharply.

To adapt to unknown non-stationarity measures, cExO equipped with the BoB framework yields the upper bounds in Corollary 2 with an additional d 5 4 T 3 4 term (see Corollary 4 in Appendix E). Our path-length regret of P 1 3 T 2 3 and P 1 3 T 2 3 + T 3 4 for known and unknown P , respectively, improves on the P 1 4 T 3 4 and P 1 2 T 3 4 rates in [13] in terms of T .

## 4 Conclusion

In this work, we develop and analyze two approaches for non-stationary Bandit Convex Optimization. For strongly convex losses, our polynomial-time TEWA-SE algorithm achieves minimax-optimal dynamic regret w.r.t. S and ∆ without knowing the strong-convexity parameter, but incurs a suboptimal T 3 4 rate for general convex losses. To address this, we propose a second algorithm, cExO, which achieves minimax-optimality for S and ∆ . However, this algorithm is not polynomial-time computable and has an increased dimension dependence. Our matching lower bounds confirm the optimality results, but also reveal potentially suboptimal guarantees w.r.t. the path-length P . This work highlights a central open challenge: designing algorithms that are simultaneously minimax-optimal and computationally efficient for general convex losses in non-stationary environments. A promising stepstone towards this goal is to incorporate second-order information, akin to the online Newton methods from [31, 34] that achieve state-of-the-art static regret guarantees for adversarial convex bandits. In particular, a restart criterion, similar to the one in line 15 of [35, Algorithm 1] or line 11 of [31, Algorithm 1], may enable tracking capabilities and lead to improved regret bounds.

## Acknowledgments and Disclosure of Funding

X. Liu, D. Baudry, P. Rebeschini and A. Akhavan were funded by UK Research and Innovation (UKRI) under the UK government's Horizon Europe funding guarantee [grant number EP/Y028333/1].

## References

- [1] Tor Lattimore. Bandit convex optimisation. arXiv:2402.06535 , 2024.
- [2] Elad Hazan. Introduction to online convex optimization. Foundations and Trends® in Optimization , 2(3-4):157-325, 2016.
- [3] Francesco Orabona. A modern introduction to online learning. arXiv:1912.13213 , 2019.

- [4] Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In International Conference on Machine Learning , pages 928-936, 2003.
- [5] Aryan Mokhtari, Shahin Shahrampour, Ali Jadbabaie, and Alejandro Ribeiro. Online optimization in dynamic environments: Improved regret rates for strongly convex problems. In Conference on Decision and Control , pages 7195-7201. IEEE, 2016.
- [6] A. Jadbabaie, A. Rakhlin, S. Shahrampour, and K. Sridharan. Online optimization: Competing with dynamic comparators. In Artificial Intelligence and Statistics , pages 398-406. PMLR, 2015.
- [7] Omar Besbes, Yonatan Gur, and Assaf Zeevi. Non-stationary stochastic optimization. Operations Research , 63(5):1227-1244, 2015.
- [8] Eric C Hall and Rebecca M Willett. Online convex optimization in dynamic environments. IEEE Journal of Selected Topics in Signal Processing , 9(4):647-662, 2015.
- [9] Tor Lattimore and Csaba Szepesvári. Bandit algorithms . Cambridge University Press, 2020.
- [10] Aurélien Garivier and Eric Moulines. On upper-confidence bound policies for switching bandit problems. In Proceedings of the 22nd International Conference on Algorithmic Learning Theory , 2011.
- [11] Omar Besbes, Yonatan Gur, and Assaf Zeevi. Stochastic multi-armed-bandit problem with non-stationary rewards. Advances in Neural Information Processing Systems , 27, 2014.
- [12] Lilian Besson, Emilie Kaufmann, Odalric-Ambrym Maillard, and Julien Seznec. Efficient change-point detection for tackling piecewise-stationary bandits. Journal of Machine Learning Research , 23(77):1-40, 2022.
- [13] Peng Zhao, Guanghui Wang, Lijun Zhang, and Zhi-Hua Zhou. Bandit convex optimization in non-stationary environments. Journal of Machine Learning Research , 22(125):1-45, 2021.
- [14] Yining Wang. On adaptivity in nonstationary stochastic optimization with bandit feedback. Operations Research , 73(2):819-828, 2025.
- [15] Mark Herbster and Manfred K. Warmuth. Tracking the Best Expert. Machine Learning , 32(2): 151-178, August 1998.
- [16] Lijun Zhang, Tianbao Yang, Zhi-Hua Zhou, et al. Dynamic regret of strongly adaptive methods. In International conference on machine learning , pages 5882-5891. PMLR, 2018.
- [17] Lijun Zhang, Shiyin Lu, and Tianbao Yang. Minimizing dynamic regret and adaptive regret simultaneously. In International Conference on Artificial Intelligence and Statistics , pages 309-319. PMLR, 2020.
- [18] Elad Hazan and C. Seshadhri. Efficient learning algorithms for changing environments. In International Conference on Machine Learning , volume 382 of ACM International Conference Proceeding Series , pages 393-400. ACM, 2009.
- [19] Amit Daniely, Alon Gonen, and Shai Shalev-Shwartz. Strongly adaptive online learning. In International Conference on Machine Learning , pages 1405-1411. PMLR, 2015.
- [20] Nicolò Cesa-Bianchi, Yoav Freund, David Haussler, David P. Helmbold, Robert E. Schapire, and Manfred K. Warmuth. How to use expert advice. Journal of the ACM , 44(3):427-485, May 1997.
- [21] Tianyi Chen and Georgios B Giannakis. Bandit convex optimization for scalable and dynamic IoT management. IEEE Internet of Things Journal , 6(1):1276-1286, 2018.
- [22] Tim van Erven, Wouter M. Koolen, and Dirk van der Hoeven. Metagrad: Adaptation using multiple learning rates in online learning. Journal of Machine Learning Research , 22(161): 1-61, 2021.

- [23] Guanghui Wang, Shiyin Lu, and Lijun Zhang. Adaptivity and optimality: A universal algorithm for online convex optimization. In Proceedings of The 35th Uncertainty in Artificial Intelligence Conference , volume 115 of Proceedings of Machine Learning Research , pages 659-668. PMLR, 2020.
- [24] Lijun Zhang, Guanghui Wang, Wei-Wei Tu, Wei Jiang, and Zhi-Hua Zhou. Dual adaptivity: a universal algorithm for minimizing the adaptive regret of convex functions. In International Conference on Neural Information Processing Systems . Curran Associates Inc., 2021.
- [25] Abraham Flaxman, Adam Tauman Kalai, and H. Brendan McMahan. Online convex optimization in the bandit setting: gradient descent without a gradient. In Proceedings of the Sixteenth Annual ACM-SIAM Symposium on Discrete Algorithms , pages 385-394. SIAM, 2005.
- [26] Robert Kleinberg. Nearly tight bounds for the continuum-armed bandit problem. In International Conference on Neural Information Processing Systems , page 697-704. MIT Press, 2004.
- [27] Wang Chi Cheung, David Simchi-Levi, and Ruihao Zhu. Learning to optimize under nonstationarity. In International Conference on Artificial Intelligence and Statistics , volume 89 of Proceedings of Machine Learning Research , pages 1079-1087. PMLR, 2019.
- [28] Tor Lattimore and Andras Gyorgy. Mirror descent and the information ratio. In Conference on Learning Theory , pages 2965-2992. PMLR, 2021.
- [29] Alekh Agarwal, Dean P Foster, Daniel J Hsu, Sham M Kakade, and Alexander Rakhlin. Stochastic convex optimization with bandit feedback. Advances in Neural Information Processing Systems , 24, 2011.
- [30] Ankan Saha and Ambuj Tewari. Improved regret guarantees for online smooth convex optimization with bandit feedback. In International conference on artificial intelligence and statistics , pages 636-642. JMLR Workshop and Conference Proceedings, 2011.
- [31] Hidde Fokkema, Dirk van der Hoeven, Tor Lattimore, and Jack J Mayo. Online newton method for bandit convex optimisation. In Conference on Learning Theory , volume 247 of Proceedings of Machine Learning Research , pages 1713-1714. PMLR, 2024.
- [32] Sébastien Bubeck, Ronen Eldan, and Yin Tat Lee. Kernel-based methods for bandit convex optimization. Journal of the ACM , 68(4):1-35, 2021.
- [33] Elad Hazan and Kfir Levy. Bandit convex optimization: Towards tight bounds. In Advances in Neural Information Processing Systems , volume 27. Curran Associates, Inc., 2014.
- [34] Arun Suggala, Y Jennifer Sun, Praneeth Netrapalli, and Elad Hazan. Second order methods for bandit optimization and control. In The Thirty Seventh Annual Conference on Learning Theory , pages 4691-4763. PMLR, 2024.
- [35] Arun Sai Suggala, Pradeep Ravikumar, and Praneeth Netrapalli. Efficient bandit convex optimization: Beyond linear losses. In Proceedings of Thirty Fourth Conference on Learning Theory , volume 134, pages 4008-4067. PMLR, 2021.
- [36] Tor Lattimore and András György. A second-order method for stochastic bandit convex optimisation. In Conference on Learning Theory , pages 2067-2094. PMLR, 2023.
- [37] Olivier Bousquet and Manfred K Warmuth. Tracking a small set of experts by mixing past posteriors. Journal of Machine Learning Research , 3(Nov):363-396, 2002.
- [38] N. Littlestone and M.K. Warmuth. The weighted majority algorithm. Information and Computation , 108(2):212-261, 1994.
- [39] V Vovk. A game of prediction with expert advice. Journal of Computer and System Sciences , 56(2):153-173, 1998.
- [40] Yoav Freund, Robert Schapire, Yoram Singer, and Manfred Warmuth. Using and combining predictors that specialize. Conference Proceedings of the Annual ACM Symposium on Theory of Computing , 01 1997.

- [41] Wouter M Koolen and Tim Van Erven. Second-order quantile methods for experts and combinatorial games. In Conference on Learning Theory , pages 1155-1175. PMLR, 2015.
- [42] Dmitry Adamskiy, Wouter M. Koolen, Alexey Chernov, and Vladimir Vovk. A closer look at adaptive regret. Journal of Machine Learning Research , 17(23):1-21, 2016.
- [43] Kwang-Sung Jun, Francesco Orabona, Stephen Wright, and Rebecca Willett. Online learning for changing environments using coin betting. arXiv preprint arXiv:1711.02545 , 2017.
- [44] Ashok Cutkosky. Parameter-free, dynamic, and strongly-adaptive online learning. In International Conference on Machine Learning , volume 119, pages 2250-2259. PMLR, 2020.
- [45] Zhou Lu, Wenhan Xia, Sanjeev Arora, and Elad Hazan. Adaptive gradient methods with local guarantees. arXiv preprint arXiv:2203.01400 , 2022.
- [46] Dheeraj Baby and Yu-Xiang Wang. Optimal dynamic regret in proper online learning with strongly convex losses and beyond. In International Conference on Artificial Intelligence and Statistics , pages 1805-1845. PMLR, 2022.
- [47] Guanghui Wang, Dakuan Zhao, and Lijun Zhang. Minimizing adaptive regret with one gradient per iteration. In International Joint Conference on Artificial Intelligence , IJCAI'18, page 2762-2768. AAAI Press, 2018.
- [48] Peng Zhao, Yan-Feng Xie, Lijun Zhang, and Zhi-Hua Zhou. Efficient methods for non-stationary online learning. Advances in Neural Information Processing Systems , 35:11573-11585, 2022.
- [49] Lijun Zhang, Shiyin Lu, and Zhi-Hua Zhou. Adaptive online learning in dynamic environments. In Proceedings of the 32nd International Conference on Neural Information Processing Systems , page 1330-1340. Curran Associates Inc., 2018.
- [50] Wenhao Yang, Yibo Wang, Peng Zhao, and Lijun Zhang. Universal online convex optimization with 1 projection per round. In Advances in Neural Information Processing Systems , volume 37, pages 31438-31472. Curran Associates, Inc., 2024.
- [51] Peter L. Bartlett, Elad Hazan, and Alexander Rakhlin. Adaptive online gradient descent. In Proceedings of the 21st International Conference on Neural Information Processing Systems , page 65-72. Curran Associates Inc., 2007.
- [52] Chuong B. Do, Quoc V. Le, and Chuan-Sheng Foo. Proximal regularization for online and batch learning. In International Conference on Machine Learning , page 257-264. Association for Computing Machinery, 2009.
- [53] Daniel Russo and Benjamin Van Roy. Learning to optimize via information-directed sampling. In Advances in Neural Information Processing Systems , volume 27. Curran Associates, Inc., 2014.
- [54] Sébastien Bubeck, Ofer Dekel, Tomer Koren, and Yuval Peres. Bandit convex optimization: √ T regret in one dimension. In Conference on Learning Theory , pages 266-278. PMLR, 2015.
- [55] Sébastien Bubeck and Ronen Eldan. Exploratory distributions for convex functions. Mathematical Statistics and Learning , 1(1):73-100, 2018.
- [56] Tor Lattimore. Improved regret for zeroth-order adversarial bandit convex optimisation. Mathematical Statistics and Learning , 2(3):311-334, 2020.
- [57] Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning , 47:235-256, 2002.
- [58] Olivier Cappé, Aurélien Garivier, Odalric-Ambrym Maillard, Rémi Munos, and Gilles Stoltz. Kullback-Leibler upper confidence bounds for optimal sequential allocation. Annals of Statistics , 41(3):1516-1541, 2013.
- [59] Peter Auer, Nicolo Cesa-Bianchi, Yoav Freund, and Robert E Schapire. The nonstochastic multiarmed bandit problem. SIAM journal on computing , 32(1):48-77, 2002.

- [60] William R Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika , 25(3/4):285-294, 1933.
- [61] Shipra Agrawal and Navin Goyal. Analysis of Thompson sampling for the multi-armed bandit problem. In Conference on Learning Theory , 2012.
- [62] Emilie Kaufmann, Nathaniel Korda, and Rémi Munos. Thompson sampling: An asymptotically optimal finite-time analysis. In International conference on algorithmic learning theory , pages 199-213. Springer, 2012.
- [63] Fang Liu, Joohyun Lee, and Ness B. Shroff. A change-detection based framework for piecewisestationary multi-armed bandit problem. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 32, 2018.
- [64] Yang Cao, Zheng Wen, Branislav Kveton, and Yao Xie. Nearly optimal adaptive procedure with change detection for piecewise-stationary bandit. In International Conference on Artificial Intelligence and Statistics , pages 418-427. PMLR, 2019.
- [65] Peter Auer, Pratik Gajane, and Ronald Ortner. Adaptively tracking the best bandit arm with an unknown number of distribution changes. In Conference on Learning Theory , volume 99, pages 138-158. PMLR, 2019.
- [66] Chen-Yu Wei and Haipeng Luo. Non-stationary reinforcement learning without prior knowledge: An optimal black-box approach. In Conference on learning theory , pages 4300-4354. PMLR, 2021.
- [67] Joe Suk and Samory Kpotufe. Tracking most significant arm switches in bandits. In Conference on Learning Theory , volume 178 of Proceedings of Machine Learning Research , pages 21602182, 2022.
- [68] Yoan Russac, Claire Vernade, and Olivier Cappé. Weighted linear bandits for non-stationary environments. Advances in Neural Information Processing Systems , 32, 2019.
- [69] Francesco Trovò, Marcello Restelli, and Nicola Gatti. Sliding-window thompson sampling for non-stationary settings. Journal of Artificial Intelligence Research , 68:311-364, 2020.
- [70] Dorian Baudry, Yoan Russac, and Olivier Cappé. On limited-memory subsampling strategies for bandits. In International Conference on Machine Learning , pages 727-737. PMLR, 2021.
- [71] Arya Akhavan, Massimiliano Pontil, and Alexandre Tsybakov. Exploiting higher order smoothness in derivative-free optimization and continuous bandits. Advances in Neural Information Processing Systems , 33:9017-9027, 2020.
- [72] Ohad Shamir. On the complexity of bandit and derivative-free stochastic convex optimization. In Conference on Learning Theory , pages 3-24. PMLR, 2013.
- [73] Shinji Ito. An optimal algorithm for bandit convex optimization with strongly-convex and smooth loss. In International Conference on Artificial Intelligence and Statistics , pages 22292239. PMLR, 2020.
- [74] Elad Hazan, Amit Agarwal, and Satyen Kale. Logarithmic regret algorithms for online convex optimization. Machine Learning , 69(2):169-192, December 2007.
- [75] Haipeng Luo, Chen-Yu Wei, Alekh Agarwal, and John Langford. Efficient contextual bandits in non-stationary worlds. In Conference On Learning Theory , pages 1739-1776. PMLR, 2018.
- [76] Yifang Chen, Chung-Wei Lee, Haipeng Luo, and Chen-Yu Wei. A new algorithm for nonstationary contextual bandits: Efficient, optimal and parameter-free. In Conference on Learning Theory , pages 696-726. PMLR, 2019.
- [77] Gilles Stoltz. Incomplete information and internal regret in prediction of individual sequences . Theses, Université Paris Sud - Paris XI, May 2005.
- [78] Arya Akhavan, Karim Lounici, Massimiliano Pontil, and Alexandre B Tsybakov. Contextual continuum bandits: Static versus dynamic regret. arXiv preprint arXiv:2406.05714 , 2024.

- [79] Alexandre B. Tsybakov. Introduction to nonparametric estimation . Springer Series in Statistics. Springer, New York, 2009.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: In Section 1, we define our problem setting, goals of the paper, and assumptions we make throughout the paper. In Section 1.1 we describe our main contributions and results, while in Section 1.2 we contextualize our work by discussing related work. The abstract summarizes these.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: In the main contributions section (Section 1.1), we describe the limitations of the work: the polynomial-time algorithm TEWA-SE we develop attains suboptimal regret bounds for general convex losses, and cExO, the second algorithm we propose, achieves optimal rates w.r.t. T, S and ∆ , but is not polynomial-time computable and has suboptimal rates w.r.t. the problem dimension d . These limitations are discussed in detail after we state each theoretical result, and restated in the conclusion (Section 4). We also discuss the potential suboptimality of both algorithms w.r.t. the path-length P in our main results section and in the conclusion.

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

Justification: We state all assumptions we make in Section 1. Each theoretical result is followed by a reference to its detailed proof in the appendices. We provide all proofs in the appendices: definitions (Appendix A), proof of Proposition 1 (Appendix B), proofs for TEWA-SE (Appendix C), proofs of lower bounds (Appendix D), and proofs for cExO (Appendix E).

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

Justification: This paper is theoretical in nature and does not include experimental results. Guidelines:

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

Justification: This paper does not include experiments.

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

Justification: This paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: This paper does not include experiments.

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

Justification: This paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: This work has been conducted in a way that fully conforms with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This work is theoretical in nature and is not expected to have significant social impacts, positive or negative.

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

Justification: This paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: This paper does not use such assets.

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

Justification: This paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

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
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/ LLM ) for what should or should not be described.

## A Definitions

Definition 1. Let σ &gt; 0 . A random variable ξ is σ -sub-Gaussian if for any λ &gt; 0 we have E [exp( λξ )] ≤ exp( σ 2 λ 2 / 2) .

Definition 2. Let α &gt; 0 . A differentiable function f : R d → R is called α -strongly convex, if for x , z ∈ R d , f ( z ) ≥ f ( x ) + ∇ f ( x ) /latticetop ( z -x ) + α 2 ‖ z -x ‖ 2 .

Definition 3. Let β &gt; 0 . Function f : R d → R is called β -smooth, if it is continuously differentiable and for any x , z ∈ R d , ‖∇ f ( x ) -∇ f ( z ) ‖ ≤ β ‖ x -z ‖ .

Definition 4. Let K &gt; 0 . Function f : R d → R is called K -Lipschitz if for any x , z ∈ R d , | f ( x ) -f ( z ) | ≤ K ‖ x -z ‖ .

## B Proof of Proposition 1

We start this section by restating the proposition, before detailing its proof.

Proposition 1. Suppose that an algorithm can be calibrated to satisfy R ada ( B , T ) ≤ C B κ , for any interval length B ∈ [ T ] , for some factor C &gt; 0 that is at most polynomial in d and log( T ) , and κ ∈ [0 , 1) .

Then, for any S, S ∆ , S P ∈ [ T ] , an appropriate choice of B yields the following regret guarantees:

<!-- formula-not-decoded -->

Switching: B = ⌈ T S ⌉ guarantees that R swi ( T, S ) ≤ 2 1+ κ CS 1 -κ T κ .

<!-- formula-not-decoded -->

Proof of Proposition 1 . The proof follows two steps. First, we state in Lemma 1 the conversion between adaptive regret and switching regret. A similar conversion can be found in [19], but we detail the proof for completeness. Next, we prove in Lemma 2 that switching regret guarantees for appropriate number of switches convert into dynamic and path-length regret guarantees.

In the remainder of this section, we detail the two supporting lemmas and their proof.

Lemma 1. Consider an algorithm that satisfies the adaptive regret guarantees of Proposition 1, then this algorithm calibrated with interval size B = ⌈ T S ⌉ satisfies

<!-- formula-not-decoded -->

/negationslash

<!-- formula-not-decoded -->

Proof of Lemma 1. Consider B = /ceilingleft T S /ceilingright . Let u 1: T ∈ Θ T be a sequence of arbitrary comparators with at most S switches. We divide the horizon into intervals of length B (the last interval may be shorter than B ), and further divide the intervals at the rounds where u t = u t -1 . This ensures each of these intervals is associated with a constant comparator. By construction, these intervals are of length ≤ B and the number of intervals is bounded by 2 S . Hence, we can apply the adaptive regret bound to each interval to obtain

We now prove the conversion between switching regret and dynamic and path-length regrets.

Lemma 2. Consider any fictitious number of switches S ′ ∈ [ T ] . Then the dynamic regret of environments constrained by ∆ satisfies and the path-length regret satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma 2. For both upper bounds, the switching regret term comes from dividing the horizon [ T ] into S ′ intervals, denoted by ( I s ) s ∈ [ S ′ ] , each of length at most ⌈ T S ′ ⌉ (defining them precisely is not important for the following arguments). Recall the definition of R ( T, u 1: T ) from (2). For any sequence of actions z 1: T ∈ Θ T chosen by the given algorithm, and for any arbitrary comparator sequences u 1: T ∈ Θ T and v 1: S ′ ∈ Θ S ′ , it holds that

<!-- formula-not-decoded -->

where the last step holds by the definition of the switching regret. It thus remains to choose a suitable v s ∈ Θ and upper bound the term V s for s ∈ [ S ′ ] . We choose a different v s for the proof of the dynamic regret bound vs. that of the path-length regret bound.

Dynamic regret. Consider the interval I s for s ∈ [ S ′ ] . Let L s be its length and ∆ s = ∑ t ∈ I s max z ∈ Θ | f t ( z ) -f t -1 ( z ) | be the total variation over this interval. Then, for any two time steps t and t ′ in I s and any z ∈ Θ , it holds that f t ( z ) -f t ′ ( z ) ≤ ∆ s by definition of total variation. Let ¯ f s denote the average of the functions over the interval I s and define v s ∈ arg min z ∈ Θ ¯ f s ( z ) , then we have

Taking the sum over all intervals and using L s ≤ ⌈ T S ′ ⌉ completes the proof of (20).

<!-- formula-not-decoded -->

Path-length regret. This proof proceeds similarly as that for the dynamic regret. Consider the interval I s for some s ∈ [ S ′ ] , and denote by L s its length and P s = ∑ t ∈ I s ‖ u t -u t -1 ‖ the pathlength of the comparator sequence on this interval. For the proof, we construct v s ∈ Θ differently from that in the proof of the dynamic regret. Before detailing the construction of v s , we first define a set of comparators ( u ′ t ) t ∈ I s ∈ Θ L s as follows: for some α 0 ∈ [0 , 1] and any time t ∈ I s , we define u ′ t to satisfy v s = α 0 u t +(1 -α 0 ) u ′ t . Using this and by the convexity and boundedness of f t , we can bound that

<!-- formula-not-decoded -->

We then proceed by choosing a suitable v s and α 0 to make this bound depend on the path-length. Since the path-length is P s , there exists an /lscript 2 -ball of radius P s 2 that contains all the comparators ( u t ) t ∈ I s , and its center c u lies in the feasible domain Θ . By assumption (as we stated in Section 1), there also exists a ball with radius r and center c r within the domain. We can thus construct v s to satisfy

<!-- formula-not-decoded -->

where v s ∈ Θ due to the convexity of the domain. Our goal is then to choose α 0 as large as possible (to make 1 -α 0 small) such that all the comparators ( u ′ t ) t ∈ I s belong to Θ . Eq. (23) implies that

<!-- formula-not-decoded -->

which by definition of the r -ball guarantees that u ′ t ∈ Θ as long as α 0 P s 2(1 -α 0 ) ≤ r . To satisfy this condition, we can thus pick α 0 = 2 r P s +2 r , which guarantees by construction that

<!-- formula-not-decoded -->

The desired bound on V s in (22) directly follows. The final result (21) then comes by summation over all intervals ( I s ) s ∈ [ S ′ ] .

## C Details and proofs for TEWA-SE

In this appendix, we provide additional details on TEWA-SE in Section C.1 and establish its theoretical guarantees in Sections C.4-C.6. We present the proof of Theorem 1 in Section C.2, followed by the supporting lemmas in Sections C.3 and C.4. We then provide the proof of Corollary 1 in Section C.5, and the parameter-free guarantees in Section C.6.

## C.1 Additional details on TEWA-SE

As we described in Section 2, TEWA-SE handles non-stationary environments by employing the Geometric Covering (GC) scheme from [19] to schedule experts across different time intervals. Additionally, TEWA-SE assigns an exponential grid of learning rates to the multiple experts covering each GC interval, to adapt to the curvature of the loss functions. We first invoke the definition of GC intervals from [19].

Definition 5 (Geometric Covering (GC) intervals [19]) . For k ∈ N , define the set of intervals that is, I k is a partition of N + \ [2 k -1] into intervals of length 2 k . Then we call I = ⋃ k ∈ N I k the set of Geometric Covering (GC) intervals.

<!-- formula-not-decoded -->

For any interval length L ∈ N + , we also define the exponential grid of learning rates as where G is the uniform upper bound (10) on ‖ g t ‖ , and D is the diameter of the feasible set Θ . For each given GC interval I = [ r, s ] ∈ I , TEWA-SE instantiates multiple experts in round r , each assigned a distinct learning rate η ∈ S ( | I | ) and surrogate loss /lscript η t as defined in (11). It removes these experts after round s . This scheduling scheme ensures at least one expert covering I effectively minimizes the linearized regret ∑ t ∈ I 〈 E [ g t | x t , Λ T ] , x t -u 〉 associated with ˆ f t on the interval I (Lemma 5), ultimately yielding the regret guarantees in Theorem 1 and Corollary 1.

<!-- formula-not-decoded -->

Polylogarithmic computational complexity For t ∈ N + , we use C t = { I ∈ I : t ∈ I } to denote the set of GC intervals covering time t . From Definition 5 it is easy to verify that |C t | = 1 + /floorleft log 2 t /floorright . The longest interval in C t has length at most t , which is associated with at most |S ( t ) | = 1+ ⌈ 1 2 log 2 t ⌉ experts. With A t = { E ( I, η ) : t ∈ I } representing the set of experts active in round t , the number of active experts in round t , denoted by n t = |A t | in Algorithm 1, satisfies

This ensures that the computational complexity of TEWA-SE is only O (log 2 T ) per round.

<!-- formula-not-decoded -->

Tilted Exponentially Weighted Average In each round t , TEWA-SE aggregates the actions proposed by the active experts E ( I, η ) ∈ A t using exponential weights, tilted by their respective learning rates, by computing

<!-- formula-not-decoded -->

where for I = [ r, s ] and t ∈ [ r +1 , s ] , L η t -1 ,I = ∑ t -1 τ = r /lscript η τ ( x η τ ,I ) represents the cumulative surrogate loss accrued by expert E ( I, η ) over the interval [ r, t -1] . Note that (27) is equivalent to line 7 of Algorithm 1, rewritten with notation better suited for our proof.

In what follows, we prove some theoretical guarantees for TEWA-SE.

## C.2 Proof of Theorem 1

In this section, we first restate Theorem 1 and provide its complete proof, which relies on several supporting lemmas. For clarity of exposition, we defer the statements and proofs of these supporting lemmas to the following sections.

Theorem 1. For any T N + and [ T ] , Algorithm 1 with h = min( √ d -1 4 , r ) satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and if f t is α -strongly-convex with arg min x ∈ R d f t ( x ) ∈ Θ for all t ∈ [ T ] , 4 it furthermore holds that

<!-- formula-not-decoded -->

where /lessorsimilar conceals polylogarithmic terms in B and T , independent of d and α .

Proof of Theorem 1. We prove (14) for the general convex case and (15) for the strongly-convex case similarly. To bound R ada ( B , T ) , we will uniformly bound ∑ q t = p E [ f t ( z t ) -f t ( u )] across all comparators u ∈ Θ and intervals [ p, q ] shorter than B .

Common setup: Invoking the event Λ T = { | ξ t | ≤ 2 σ √ log( T +1) , ∀ t ∈ [ T ] } defined above (10), since { ξ t } T t =1 are σ -sub-Gaussian, we have

<!-- formula-not-decoded -->

By the law of total expectation we can write for any u ∈ Θ ,

<!-- formula-not-decoded -->

To bound the first term in the last display, we consider the following decomposition

<!-- formula-not-decoded -->

Since f t is convex, by Jensen's inequality we obtain that term II is negative (c.f. [71, Lemma A.2 (ii)]). In what follows, we bound terms I, III and IV in this decomposition separately for the general convex case and the strongly-convex case.

General convex and Lipschitz case: Recall that ( ζ t ) T t =1 denote uniform samples from the unit sphere ∂ B d , and ˜ ζ denotes a uniform sample from the unit ball B d , while ˆ f t ( x ) = E [ f t ( x + h ˜ ζ )] ∀ x ∈ ˜ Θ . Since ( f t ) T t =1 are K -Lipschitz, ‖ ζ t ‖ = 1 , and E [ ‖ ˜ ζ ‖ ] ≤ 1 , we can bound term I and term IV by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

4 The assumption that loss minimizers lie inside Θ is common in zeroth-order optimization, see e.g., [7, 72, 73]. Without it, our upper bound analysis would have an extra term depending on the gradients at the minimizers.

To bound term III, recall that g t denotes the gradient estimate of ˆ f t at x t . We use the convexity of ˆ f t and apply Lemma 3 to obtain that for any u ∈ Θ , where all constants are explicit in the statement of the lemma. By combining the bounds for all four terms in (30) with (29), and using h = min ( √ d B -1 4 , r ) , G = d h (1 + 2 σ √ log( T +1)) , a p,q = 1 2 +2log(2 q ) + 1 2 log( q -p +1) ≤ 6 log( T +1) , and b p,q = 2 /ceilingleft log 2 ( q -p +2) /ceilingright ≤ 6 log( B +1) , we establish that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where C, C 1 , C 2 &gt; 0 are polylogarithmic in T , independent of d , defined with M T = 1 + 2 σ √ log( T +1) and N T, B = log( T +1)log( B +1) as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This concludes the proof of (14).

Strongly-convex and smooth case: Due to the β -smoothness of f t and the fact that E [ ζ t ] = E [ ˜ ζ ] = 0 and E [ ‖ ˜ ζ ‖ 2 ] ≤ E [ ‖ ζ t ‖ 2 ] = 1 , we can bound term I and term IV each by β 2 ( q -p +1) h 2 . When the f t 's are strongly-convex, we can derive a tighter bound on term III than that in (33) by restricting the comparator u to the clipped domain ˜ Θ and using the fact that when f t is α -strongly convex on Θ , ˆ f t is α -strongly convex on ˜ Θ (c.f. [71, Lemma A.3]). That is, we have for any u ∈ ˜ Θ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining (38) with (29)-(30) and simplifying yields for any u ∈ ˜ Θ ,

<!-- formula-not-decoded -->

The final step is to handle the case where the comparator u ∈ Θ \ ˜ Θ . Consider the worst case when the comparator is u ∗ ∈ arg min u ∈ Θ ∑ q t = p f t ( u ) with u ∗ ∈ Θ \ ˜ Θ . Let ˜ u ∗ = Π ˜ Θ ( u ∗ ) . If

arg min x ∈ R d f t ( x ) ∈ Θ ∀ t ∈ [ T ] , then by the β -smoothness of the f t 's we have

<!-- formula-not-decoded -->

Combining (39) with (40) yields

<!-- formula-not-decoded -->

where C ′ , C ′ 1 , C ′ 2 &gt; 0 are polylogarithmic in T and B , independent of d , defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This concludes the proof of (15).

The proof above crucially relies on Lemma 3, which we state and prove in the following section.

## C.3 Upper bounds on linearized regret

Lemma 3 establishes an upper bound on the linearized regret associated with the smoothed loss ˆ f t for any arbitrary interval I = [ p, q ] ⊆ [1 , T ] . This result builds on two key components: Lemma 4, which characterizes how a given arbitrary interval is covered by a sequence of GC intervals, and Lemma 5, which provides an upper bound on the linearized regret for each GC interval I ∈ I . For clarity, we first present and prove Lemma 3, then proceed to detail the supporting Lemmas 4 and 5.

Lemma 3 (Linearized regret on an arbitrary interval) . For an arbitrary interval I = [ p, q ] ⊆ [1 , T ] , Algorithm 1 satisfies for all u ∈ Θ ,

<!-- formula-not-decoded -->

where a p,q = 1 2 +2log(2 q ) + 1 2 log( q -p +1) and b p,q = 2 /ceilingleft log 2 ( q -p +2) /ceilingright .

Proof of Lemma 3. This proof follows similar arguments to those used in proving the first part of [24, Theorem 2]. To begin, according to Lemma 4, any arbitrary interval I = [ p, q ] ⊆ [1 , T ] can be covered by two sequences of consecutive and disjoint GC intervals, denoted by I -m , . . . , I 0 ∈ I and I 1 , . . . , I n ∈ I , where n, m ∈ N + with n ≤ /ceilingleft log 2 ( q -p +2) /ceilingright and m + 1 ≤ /ceilingleft log 2 ( q -p +2) /ceilingright . Note that negative indices correspond to GC intervals that precede I 0 , while positive indices correspond to intervals that follow it. The indices indicate temporal ordering and are unrelated to the length of the intervals.

By applying the linearized regret bound from Lemma 5 to each GC interval, and noticing that a r,s ≤ a p,q for any subinterval [ r, s ] ⊆ [ p, q ] (as evident from the definition of a p,q in (45)), we

establish for all u ∈ Θ ,

<!-- formula-not-decoded -->

where the last step uses n + m +1 ≤ 2 /ceilingleft log 2 ( q -p +2) /ceilingright =: b p,q .

We now present Lemmas 4 and 5 which we used to prove Lemma 3 above.

Lemma 4 (Covering property of GC intervals) . Any arbitrary interval I = [ p, q ] ⊆ N + can be partitioned into two finite sequences of consecutive and disjoint GC intervals, denoted by I -m , . . . , I 0 ∈ I and I 1 , . . . , I n ∈ I , where I = ⋃ n i = -m I i , such that

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

Proof of Lemma 4. Eq. (47) directly comes from [19, Lemma 1.2]. To prove (48), suppose for contradiction n &gt; /ceilingleft log 2 ( q -p +2) /ceilingright , then we have

<!-- formula-not-decoded -->

contradicting the fact that ⋃ n i = -m I i = I . By the same reasoning, we have m + 1 ≤ /ceilingleft log 2 ( q -p +2) /ceilingright .

Lemma 5 (Linearized regret on a GC interval) . For any GC interval I = [ r, s ] ∈ I , Algorithm 1 satisfies for all u ∈ Θ ,

<!-- formula-not-decoded -->

where a r,s = 1 2 +2log(2 s ) + 1 2 log( s -r +1) .

Proof of Lemma 5. This proof is similar to that of [24, Lemma 12]. For any GC interval I = [ r, s ] ∈ I and learning rate η ∈ S ( s -r +1) , we can apply the definition of surrogate loss /lscript η t from (11),

noticing that /lscript η t ( x t ) = 0 , to obtain for all u ∈ Θ ,

<!-- formula-not-decoded -->

where the last step applies the upper bound on the expert-regret established in Lemma 6 and the upper bound on the meta-regret in Lemma 7, both of which we defer to Section C.4. Eq. (51) can be rearranged into

<!-- formula-not-decoded -->

The optimal value of η that minimizes the RHS of (52) is

<!-- formula-not-decoded -->

Note that since a r,s ≥ 1 2 , η ∗ ≥ 1 GD √ 2( s -r +1) for all x ∈ Θ . The next step is to select a value η from the set S ( s -r +1) = { 2 -i 5 GD : i ∈ { 0 , 1 , . . . , ⌈ 1 2 log 2 ( s -r +1) ⌉ } } that best approximates η ∗ . Two cases arises:

- i) If η ∗ ≤ 1 5 GD , there must exist an η ∈ S ( s -r +1) such that η ∗ 2 ≤ η ≤ η ∗ . Substituting this choice of η into (52) gives

<!-- formula-not-decoded -->

- ii) If η ∗ &gt; 1 , then the best choice of η ( s r +1) is η = 1 , which leads to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining (54)-(55) concludes the proof.

The proof of Lemma 5 above relied on the upper bounds on the expert-regret and meta-regret from Lemmas 6 and 7. We present and prove these lemmas in the following section.

## C.4 Upper bounds on expert-regret and meta-regret

Lemma 6 (Expert-regret) . For any GC interval I = [ r, s ] ∈ I and learning rate η ∈ S ( s -r +1) , Algorithm 1 satisfies for all u ∈ ˜ Θ ,

<!-- formula-not-decoded -->

Proof of Lemma 6. The proof follows standard convergence analysis of projected online gradient descent for strongly convex objective functions, see e.g., [74, Theorem 1]. For any time step t ∈ I , the surrogate loss /lscript η t associated with the expert with learning rate η and lifetime I = [ r, s ] serves as our strongly-convex objective function. By applying the definition of /lscript η t , we have for all x ∈ Θ , where we introduced G ′ = η G + 2 η 2 G 2 D . By the update rule of our projected online gradient descent with step size µ t (line 4 of Algorithm 2), we have for all u ∈ ˜ Θ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which can be rearranged into

<!-- formula-not-decoded -->

Define shorthand λ ≡ 2 η 2 G 2 and recall µ t = 1 / ( λ ( t -r +1)) , then Eq. (58) implies that

<!-- formula-not-decoded -->

Noticing that with any given x t ∈ R d , /lscript η t is λ -strongly-convex, we apply (59) to obtain that for all u ∈ ˜ Θ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (i) is a result of (57), (ii) uses the bound ∑ n k =1 1 k ≤ 1 + log n for any n ∈ N + , and (iii) uses the fact that given η ≤ 1 5 GD it holds that

Lemma 7 (Meta-regret) . For any GC interval I = [ r, s ] ∈ I and learning rate η ∈ S ( s -r +1) , Algorithm 1 satisfies

<!-- formula-not-decoded -->

Proof of Lemma 7. The proof is similar to that of [24, Lemma 6]. By Jensen's inequality and the convexity of norms, we have for all x ∈ Θ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which, given η ≤ 1 5 GD , implies that

<!-- formula-not-decoded -->

Using (62)-(63) and applying the inequality ln(1 + z ) ≥ z -z 2 for any z ≥ -2 3 with z = η 〈 E [ g t | x t , Λ T ] , x t -x 〉 , we obtain for all x ∈ Θ ,

<!-- formula-not-decoded -->

Define shorthand F η t,I = { x t , x η t,I , Λ T } , and H η t,I = ∪ τ ∈ [ t ] F η τ ,I for t ∈ [ T ] . Using (64), we can write for every t ∈ [ T ] ,

<!-- formula-not-decoded -->

The second term on the RHS can be bounded as follows:

where (i) applies Jensen's inequality, and (ii) is due to the update rule of x t in (27). Combining (65)-(66) yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By summing both sides of (67) over t = 1 , . . . , s and rewriting, we obtain

<!-- formula-not-decoded -->

Canceling the equivalent last terms on both sides of (68) and noting that L η τ ,I = 0 for τ = min { t : t ∈ I } -1 by construction (see line 4 of Algorithm 1), we obtain for s ≥ 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (i) applies (26), and (ii) is due to 1 + log 2 t ≤ 2 √ t ∀ t ≥ 1 . Since exp( x ) &gt; 0 for x ∈ R , Eq. (69) implies that for any GC interval I = [ r, s ] ∈ I and learning rate η ∈ S ( | I | ) ,

Taking the logarithm of both sides completes the proof.

## C.5 Proof of Corollary 1

We first restate Corollary 1 and then provide the proof. Recall that for clarity we drop the /ceilingleft · /ceilingright operators from the expressions for B and assume without loss of generality the expressions take integer values.

Corollary 1. Consider any horizon T ∈ N + and assume that, for all t ∈ [ T ] , the loss f t is convex, or strongly-convex with arg min x ∈ R d f t ( x ) ∈ Θ . We refer to the second scenario as the strongly-convex (SC) case. Then, Algorithm 1 tuned with parameter B satisfies the following regret guarantees:

<!-- formula-not-decoded -->

Proof of Corollary 1. We begin by applying the first result in Proposition 1 with the adaptive regret guarantees in Theorem 1 to obtain switching regret guarantees. For known S , Algorithm 1 with parameter B = T S achieves in the general convex case,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and in the case where f t is α -strongly-convex and arg min x ∈ R d f t ( x ) ∈ Θ for all t ∈ [ T ] ,

<!-- formula-not-decoded -->

where C, C 1 , C 2 , C ′ , C ′ 1 , C ′ 2 &gt; 0 are the terms defined in (35)-(37) and (42)-(44) which are polylogarithmic in T and B . When S and ∆ are both known, we use (71)-(72) and apply the second result in Proposition 1 to bound R dyn ( T, ∆ , S ) . Specifically, for general convex losses, Algorithm 1 with B = T S ∨ ( √ dT ∆ ) 4 5 yields

<!-- formula-not-decoded -->

where F dyn ( T, ∆ ) := (2 C +1) d 2 5 ∆ 1 5 T 4 5 +2( C 1 + C 2 d + 4 d 2 ) d 8 5 ∆ 4 5 T 1 5 . For strongly-convex losses with minimizers inside Θ , Algorithm 1 with B = T S ∨ ( dT ∆ ) 2 3 gives

<!-- formula-not-decoded -->

where F dyn sc ( T, ∆ ) := (2 C ′ + 1) d 2 3 ∆ 1 3 T 2 3 + 2( C ′ 1 + C ′ 2 d + 4 d 2 ) d 4 3 ∆ 2 3 T 1 3 . Finally, for known P we use (71)-(72) and apply the third result in Proposition 1 to bound R path ( T, P ) . For the general convex case, taking B = ( r √ dT P ) 4 5 gives

<!-- formula-not-decoded -->

For strongly-convex losses with minimizers inside Θ , taking B = ( rdT P ) 2 3 yields R path ( T, P ) ≤ (2 C ′ +1) r -1 3 d 2 3 P 1 3 T 2 3 +2( C ′ 1 + C ′ 2 d + 4 d 2 ) r -2 3 d 4 3 P 2 3 T 1 3 .

## C.6 Parameter-free upper bounds

Corollary 1 presents the optimal choice of parameter B for TEWA-SE when S , ∆ and P are known. When the non-stationarity measures are unknown, the optimal B cannot be directly computed, and we therefore employ the Bandit-over-Bandit (BoB) framework from [27] to adaptively select B from a prespecified set B = { 2 i : i = 0 , 1 , . . . , /floorleft log 2 T /floorright } . BoB has been used in [66] in a similar fashion to obtain parameter-free algorithms. Specifically, BoB divides the time horizon into E = /ceilingleft T/L /ceilingright epochs each with length L , denoted by ( I e ) E e =1 (where the last epoch may be shorter than L ). In the first epoch, it runs TEWA-SE with B = B 1 which is randomly selected from B . For subsequent epochs, it uses the cumulative empirical loss on the current epoch e -1 to select B e ∈ B for the next epoch via EXP3 [59]. That is, BoB computes

<!-- formula-not-decoded -->

where e denotes the base of the exponential function, and then samples i e = i with probability p e,i yielding B e = 2 i e -1 . 5 For i ∈ [ |B| ] , initialized with s 0 ,i = 1 , the quantity s e,i for e ∈ N + is updated by computing

<!-- formula-not-decoded -->

where with M T = 1 + 2 σ √ log( T +1) , the importance-weighted reward r e,i takes the form

Note that conditioned on the event Λ T = { | ξ t | ≤ 2 σ √ log( T +1) , ∀ t ∈ [ T ] } defined above (10), the absolute total reward in each epoch is bounded by Q := max e ∈ [ E ] ∣ ∣ ∑ t ∈ I e (1 -y t ) ∣ ∣ ≤ LM T , which ensures the rescaled reward 1 2 + 1 2 LM T ∑ t ∈ I e (1 -y t ) in (75) remains bounded within [0 , 1] . The pseudo-code for TEWA-SE equipped with BoB is provided in Algorithm 4, with theoretical guarantees detailed in Corollary 3.

<!-- formula-not-decoded -->

Corollary 3 (TEWA-SE with BoB) . Consider any horizon T ∈ N + and assume that, for all t ∈ [ T ] , the loss f t is convex, or strongly-convex with arg min x ∈ R d f t ( x ) ∈ Θ (referred to as the strongly-convex (SC) case).

Then, for the general convex case, Algorithm 4 with epoch size L = ( dT ) 2 3 attains all the regret bounds from Corollary 1 plus an additional term of d 1 3 T 5 6 + d 4 3 T 1 3 . For the SC case, Algorithm 4 with epoch size L = d √ T satisfies all the regret bounds from Corollary 1 plus an additional term of d 1 2 T 3 4 + d √ T . Both results omitted polylogarithmic factors.

5 We adopt clipping (by γ ) following [27, 59], though γ = 0 suffices as discussed in [77] and [9, Section 11.6].

## Algorithm 4 TEWA equipped with Bandit-over-Bandit (BoB)

Input: d, T, L, E = /ceilingleft T/L /ceilingright , ( I e ) E e =1 , B = { 2 i : i = 0 , 1 , . . . , /floorleft log 2 T /floorright } , and γ ∈ (0 , 1) as defined in (73) Initialize: s 0 ,i = 1 ∀ i ∈ [ |B| ] 1: for e = 1 , 2 , . . . , E do 2: Compute p e,i according to (73) ∀ i ∈ [ |B| ] 3: Sample i e = i with probability p e,i , and select B e = 2 i e -1 ∈ B 4: for t ∈ I e do 5: Run TEWA-SE with B = B e to select action z t and observe losses y t = f t ( z t ) + ξ t 6: end for 7: Update s e +1 ,i according to (74) ∀ i ∈ [ |B| ] 8: end for

Proof of Corollary 3. For brevity, we suppress terms that are polylogarithmic in T using /lessorsimilar in this proof. For all B † ∈ B , we have

<!-- formula-not-decoded -->

where z t ( B e ) represents the actual action taken by TEWA-SE in round t of epoch e , and z t ( B † ) denotes the hypothetical action that TEWA-SE would have chosen had its B parameter been set to B † . Term I in (76) can be bounded by applying the classical analysis of EXP3 from [59, Corollary 3.2], combined with (28), as follows

<!-- formula-not-decoded -->

/negationslash

To bound term II, we introduce shorthand F ada ( B , T ) to refer to the upper bound on R ada ( B , T ) in Theorem 1, and F swi ( T, S ) , F dyn ( T, ∆ , S ) and F path ( T, P ) to refer to the upper bounds on R swi ( T, S ) , R dyn ( T, ∆ , S ) and R path ( T, P ) in Corollary 1 for known S, ∆ and P . We also use S e = 1+ ∑ t ∈ I e 1 ( f t = f t -1 ) . By choosing B † = 2 i † in the analysis with i † = ⌊ log 2 T S ⌋ ∧/floorleft log 2 L /floorright , term II can be bounded in terms of the number of switches S by

<!-- formula-not-decoded -->

Combining (77) and (78), we obtain

<!-- formula-not-decoded -->

where we used L = ( dT ) 2 3 for the general convex case, and L = d √ T for the strongly-convex case. Following similar steps, by choosing B † = 2 i † in the analysis with i † = ( ⌊ log 2 T S ⌋ ∨ /floorleft log 2 ( B ∆ ) /floorright ) ∧

/floorleft log 2 L /floorright where B ∆ = ( √ dT ∆ ) 4 5 for the general convex case or B ∆ = ( dT ∆ ) 2 3 for the strongly-convex case, we obtain

<!-- formula-not-decoded -->

The bound on R path ( T, P ) can be established analogously.

## D Proofs of lower bounds

We call π = { z t } ∞ t =1 a randomized procedure if z t = Φ t ( { z k } t -1 k =1 , { y k } t -1 k =1 ) where Φ t are Borel functions, and z 1 ∈ R d is deterministic. We emphasize that, throughout this section, we assume the noise variables { ξ t } T t =1 are independent with cumulative distribution function F satisfying the condition

<!-- formula-not-decoded -->

for some 0 &lt; I 0 &lt; ∞ , 0 &lt; v 0 ≤ ∞ . This condition holds, for instance, if F has a sufficiently smooth density with finite Fisher information. In the special case where F is Gaussian, the inequality (81) holds with v 0 = ∞ . Note that Gaussian noise also satisfies our sub-Gaussian noise assumption in Section 1, which is used in the proof of the upper bounds.

We first restate and prove Theorem 2, which establishes a lower bound on R dyn ( T, ∆ , S ) , and then present and prove Theorem 4, which establishes a lower bound on R path ( T, P ) .

Theorem 2. Let Θ = B d . For α &gt; 0 denote by F α the class of α -strongly convex and smooth functions. Let π = { z t } T t =1 be any randomized algorithm (see Appendix D for a definition). Then there exists T 0 &gt; 0 such that for all T ≥ T 0 it holds that

<!-- formula-not-decoded -->

where c 1 &gt; 0 is a constant independent of d, T , S and ∆ .

Proof of Theorem 2. Let η 0 : R → R be an infinitely many times differentiable function that satisfies

Denote by Ω = { -1 , 1 } d the set of binary sequences of length d , and let η ( x ) = ∫ x -∞ η 0 ( u ) d u . Consider the set of functions f ω : R d → R with ω = ( ω 1 , . . . , ω d ) ∈ { -1 , 1 } d such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where h = min ( d -1 2 , ( T S ) -1 4 , ( dT ∆ ) -1 6 ) , and ι &gt; 0 is to be assigned later. Let L ′ = max x ∈ R | η ′′ ( x ) | . By [78, Lemma 10] we have that if ι ≤ min(1 / 2 η (1) , α /L ′ ) then f ω ∈ F α . Moreover, if ι ≤ α / 2 , the equation ∇ f ω ( x ) = 0 has the solution

<!-- formula-not-decoded -->

This is the unique minimizer of f ω and belongs to x ∗ ( ω ) ∈ Θ = B d because

<!-- formula-not-decoded -->

We consider the following adversarial protocol. At the beginning of the game, the adversary selects N c = min( S, ( T ∆ 2 /d 2 ) 1 3 ) points from Ω , sampled uniformly at random with replacement. Here

without loss of generality we assumed that ( T ∆ 2 ) 1 3 is a positive integer. Denote these points by { ω k } N c k =1 , and then for each k = 1 , 2 , . . . , N c , let

<!-- formula-not-decoded -->

/negationslash

For any ω , ω ′ ∈ Ω let ρ ( ω , ω ′ ) = ∑ d i =1 1 ( ω i = ω ′ i ) be the Hamming distance between ω and ω ′ , with ω = ( ω 1 , . . . , ω d ) and ω ′ = ( ω ′ 1 , . . . , ω ′ d ) . By construction, N c ≤ S and

<!-- formula-not-decoded -->

For any fixed ω 1 , . . . , ω N c ∈ Ω , and 1 ≤ t ≤ T , denote Γ = [ ω 1 | . . . | ω N c ] as the matrix whose columns are the ω k 's. Denote by P Γ ,t the probability measure corresponding to the joint distribution of { z k , y k } t k =1 where y k = f k ( z k ) + ξ k with independent identically distributed ξ k 's such that (81) holds and z k 's are chosen by the algorithm π . We have

<!-- formula-not-decoded -->

where k τ = /floorleft ( τ -1) N c /T /floorright +1 . (We omit explicit mention of the dependence of P Γ ,t and Φ τ on z 2 , . . . , z τ -1 , since z τ for τ ≥ 2 is a Borel function of z 1 , y 1 , . . . , z τ -1 , y τ -1 .) Let E Γ ,t denote the expectation w.r.t. P Γ ,t .

Note that by α -strong convexity of f ω and the fact that x ∗ ( ω ) ∈ arg min x ∈ R d f ω ( x ) from (83), we have

<!-- formula-not-decoded -->

Define the nearest-neighbour estimator

<!-- formula-not-decoded -->

Using this combined with the triangle inequality, we have ‖ x ∗ (ˆ ω t ) -x ∗ ( ω k t ) ‖ ≤ ‖ z t -x ∗ (ˆ ω t ) ‖ + ‖ z t -x ∗ ( ω k t ) ‖ ≤ 2 ‖ z t -x ∗ ( ω k t ) ‖ . Together with (83) this implies that

<!-- formula-not-decoded -->

Summing over 1 , . . . , T , then taking the maximum over Γ = [ ω 1 | . . . | ω N c ] and then the minimum over all estimators ˆ ω 1 , . . . , ˆ ω T with values in Ω , we get

/negationslash

<!-- formula-not-decoded -->

For term I, lower bounding the maximum with the average we can write

<!-- formula-not-decoded -->

/negationslash

/negationslash

/negationslash

Next, for each i = 1 , . . . , d , define Γ k t i = { [ ω 1 | . . . | ω N c ] : ω 1 , . . . , ω N c ∈ Ω , ω k t ,i = 1 } . Given any Γ ∈ Γ k t i , let ¯ Γ = [¯ ω 1 | . . . | ¯ ω N c ] such that ¯ ω k,j = ω k,j for any k = k t , and let ¯ ω k t ,i = -1 and ¯ ω k t ,j = ω k t ,j for j = i . Hence,

/negationslash

/negationslash

<!-- formula-not-decoded -->

Thus, we can write

<!-- formula-not-decoded -->

Since h ≤ min( ( S T ) 1 4 , ( ∆ dT ) 1 6 ) , and by choosing ι ≤ √ log(2) / (4 I 0 η 2 (1)) , we have KL ( P Γ ,t || P ¯ Γ ,t ) ≤ log(2) . Hence, Theorem 2.12 of [79] gives

<!-- formula-not-decoded -->

Substituting this into (86) and our overall bound (85) yields

<!-- formula-not-decoded -->

Finally, substituting the definition of h and noting that ι is independent of d, T, S and ∆ completes the proof.

Theorem 4. Let Θ = B d . For α &gt; 0 denote by F α the class of α -strongly convex and smooth functions. Let π = { z t } T t =1 be any randomized algorithm. Then there exists T 0 &gt; 0 such that for all T ≥ T 0 it holds that

<!-- formula-not-decoded -->

where c 2 &gt; 0 is a constant indepedent of d, T and P .

Proof of Theorem 4. The proof uses the same notation and follows the same steps as in the proof of Theorem 2, but with different choices for the parameters h and N c . Define the set of functions f ω : R d → R with ω ∈ { -1 , 1 } d as they are defined in (82), and choose h = min( d -1 2 , P N c √ d ) and N c = /floorleft P 4 5 T 1 5 d -2 5 /floorright . Then we have that

<!-- formula-not-decoded -->

/negationslash

/negationslash

/negationslash

for any ι ≤ α 2 . Following similar steps as in the proof of Theorem 2 for large enough T (when h = P N c √ d ) we get

<!-- formula-not-decoded -->

where c 2 &gt; is independent of d, T and P

<!-- formula-not-decoded -->

## E Proofs for clipped Exploration by Optimization

We restate and prove Theorem 3 which establishes an adaptive regret guarantee for cExO. In this section, we use 〈 p , f t 〉 = E z ∼ p [ f t ( z )] where p belongs to a probability simplex.

<!-- formula-not-decoded -->

Theorem 3. For T ∈ N + and B ∈ [ T ] , Algorithm 3 calibrated with ε = 1 T , γ = 1 T |C| , η = √ log( γ -1 ) / ( d 4 log( dT ) B ) and log |C| = O ( d log( dT 2 )) satisfies

Proof of Theorem 3. Consider an arbitrary interval [ a, b ] of length b -a +1 ≤ B , and notice that for any q /star ∈ ˜ ∆ ,

<!-- formula-not-decoded -->

In what follows, we choose a suitable q /star and bound term I and term II separately.

Recall that the covering set C is assumed in Section 3 to have a discretization error of ε , implying that there exists a u C ∈ C such that ∑ b t = a f t ( u C ) -min u ∈ Θ ∑ b t = a f t ( u ) ≤ ε B . Define q /star ∈ ˜ ∆ to be the distribution with probability mass given by

<!-- formula-not-decoded -->

This construction ensures that

<!-- formula-not-decoded -->

To bound term I, we first apply Lemma 8 to the sequence of Online Mirror Descent (OMD) updates q t ∈ ˜ ∆ and the sequence of loss estimates s t to obtain where by the definition of q /star ( · ) in (89), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then applying (91) and (92), we have

<!-- formula-not-decoded -->

where (i) follows from the update rule (18) and the precision level assumed for solving the minimization problem (18) (see line 3 of Algorithm 3), and (ii) uses [1, Theorems 8.19 and 8.21] which establish that there exists a universal constant κ such that

<!-- formula-not-decoded -->

Finally, combining (90) and (93) we obtain

<!-- formula-not-decoded -->

where (i) applies ε = 1 T , γ = 1 T |C| and η = √ log( γ -1 ) / ( d 4 log( dT ) B ) , and (ii) is by selecting the covering set C such that log |C| ≤ d log(1 + 16 dT 2 ) (existence given by [1, Definition 8.12 and Exercise 8.13]).

The proof of Theorem 3 above relied on Lemma 8, which we present and prove below.

Lemma 8. Consider Online Mirror Descent (OMD) with KL divergence regularization and fixed learning rate η &gt; 0 applied to a sequence of loss estimates s t ∈ R n for t ∈ N + . When run over a convex and complete domain ˜ ∆ ⊆ ∆ n -1 , the algorithm produces a sequence of updates q t ∈ ˜ ∆ for t ∈ N + . For any comparator in q /star ∈ ˜ ∆ and time interval { t ∈ N + : a ≤ t ≤ b } , it holds that

<!-- formula-not-decoded -->

where S q ( η s ) = max q ′ ∈ ∆ ( C ) 〈 q -q ′ , η s 〉 -KL ( q ′ || q ) .

Proof of Lemma 8. The proof is standard and included for completeness. Let F denote the negentropy F ( q ) = ∑ n i =1 q i log( q i ) for q ∈ ∆ n -1 , and note that

<!-- formula-not-decoded -->

Consider the update rule of the OMD defined in the lemma:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies by the first order optimality condition [9, Proposition 26.14] that, for any q /star ∈ ˜ ∆ and time t ,

Rearranging (95) and applying (94) we obtain

<!-- formula-not-decoded -->

Rearranging (96) and summing over t ∈ [ a, b ] yields

<!-- formula-not-decoded -->

which combined with non-negativity of the KL divergence completes the proof.

Finally, we apply Theorem 3 to prove the bounds on R swi ( T, S ) , R dyn ( T, ∆ , S ) and R path ( T, P ) in Corollary 2, as well as the parameter-free guarantees in Corollary 4.

Corollary 2. For any horizon T ∈ N + , Algorithm 3 calibrated as in Theorem 3 and tuned with interval size B (which determines η ) satisfies the following regret guarantees:

<!-- formula-not-decoded -->

Proof of Corollary 2. We prove these results by applying the adaptive regret guarantee from Theorem 3 and the conversions results from Proposition 1, similarly to the proof of Corollary 1.

Corollary 4 (cExO with BoB) . Let T ∈ N + . By partitioning the time horizon [ T ] into epochs of length L = d 5 2 √ T , and employing Bandit-over-Bandit to select cExO's parameter B for each epoch from the set B = { 2 i : i = 0 , 1 , . . . , /floorleft log 2 T /floorright } , this algorithm achieves all regret bounds in Corollary 2 with an additional term of d 5 4 T 3 4 (up to polylogarithmic factors).

Proof of Corollary 4. The proof is similar to that of Corollary 3 and is therefore omitted.