## Convergence Theorems for Entropy-Regularized and Distributional Reinforcement Learning

Yash Jhaveri ∗ Rutgers University-Newark

Harley Wiltzer ∗ Mila-Québec AI Institute McGill University

Marc G. Bellemare † Mila-Québec AI Institute McGill University

Patrick Shafto

Rutgers University-Newark

David Meger Mila-Québec AI Institute McGill University

## Abstract

In the pursuit of finding an optimal policy, reinforcement learning (RL) methods generally ignore the properties of learned policies apart from their expected return. Thus, even when successful, it is difficult to characterize which policies will be learned and what they will do. In this work, we present a theoretical framework for policy optimization that guarantees convergence to a particular optimal policy, via vanishing entropy regularization and a temperature decoupling gambit . Our approach realizes an interpretable, diversity-preserving optimal policy as the regularization temperature vanishes and ensures the convergence of policy derived objects-value functions and return distributions. In a particular instance of our method, for example, the realized policy samples all optimal actions uniformly. Leveraging our temperature decoupling gambit, we present an algorithm that estimates, to arbitrary accuracy, the return distribution associated to its interpretable, diversity-preserving optimal policy.

## 1 Introduction

In generic Markov Decision Processes (MDPs), many optimal policies exist. Thus, while certain policy optimization approaches can ensure convergent approximation to an optimal policy, they do not have control over which states these policies will visit, which actions they will play, or which long-term returns they can achieve. Indeed, the non-uniqueness of optimal policies renders any discussion of the properties of an optimal policy ambiguous, beyond its expected value.

A partial remedy to this problem is to regularize the RL objective in order to induce uniqueness. One popular approach to regularization is to penalize the value of a policy according to its KL divergence to a reference policy π ref . This branch of RL is known as entropy-regularized RL (ERL). In ERL, for any positive regularization weight τ (also known as temperature ), one and only one policy is optimal. Moreover, in a tabular MDP, τ -optimal policies and their derived objects (value functions, occupancy measures, and return distributions) converge to classically optimal policies and their derived objects. However, beyond tabular MDPs, the evolution of τ -optimal quantities, as a function of the temperature, is not well understood. Thus, as we decay the temperature to zero, we are, in some sense, back to where we started: living in ambiguity.

In this work, we introduce a temperature decoupling gambit , through which we can guarantee the convergence of resulting policies and their derived objects in the vanishing temperature limit.

∗ Equal contribution. Correspondence to yash.jhaveri@rutgers.edu , wiltzerh@mila.quebec † CIFAR AI Chair.

.

Much like how a gambit in chess sacrifices an immediate and shallow proxy of the objective (e.g., material count) for a long term positional advantage, the temperature decoupling gambit plays notably suboptimal policies for the τ -ERL objective to ensure convergence to RL optimality as τ → 0 . This scheme entails estimating action-values under a target regularization temperature while playing policies with an amplified temperature. Furthermore, we characterize this limiting policy as a modification of the reference policy which 'filters out' suboptimal actions. Even when τ -optimal policies converge in the vanishing temperature limit (such as in tabular MDPs), the limiting policy produced by the temperature decoupling gambit is distinct from the limiting policy found otherwise. The limiting policy found via our gambit preserves, quantifiably, more state-wise action diversity. Moreover, we show that this limiting policy achieves a notion of reference-optimality for RL, characterized by a new Bellman-like equation, whose unique fixed point upper bounds the (RL) performance of τ -optimal policies in general.

Our analysis additionally sheds light on the convergence of return distributions-the central objects of study in distributional RL (DRL) [6]. While optimal policies achieve the same return in expectation, they may vary drastically in other statistics, such as variance. In safety-critical applications, for example, understanding the distribution over returns is crucial. DRL provides techniques for estimating return distributions, primarily based on distributional dynamic programming methods which generalize dynamic programming approaches for estimating expected returns. However, it is well-known that existing distributional methods do not produce convergent iterates in the control setting [5]. Leveraging our convergence results for policies in ERL, we define the first algorithm for accurately estimating a reference-optimal return distribution, the return distribution associated to the interpretable, diverse policy realized by the temperature decoupling gambit.

## 2 Preliminaries

Given a Borel set S ⊂ R n , for some n ∈ N , we let M ( S ) and M b ( S ) denote the space of Borel measurable and bounded Borel measurable functions on S respectively. We let P ( S ) denoted the space of Borel probability measures on S . From now on, measurability will always be with respect to Borel sets. Moreover, for any ρ ∈ P ( Y ) with Y ⊂ R m and any measurable function f : Y → S , the push-forward of ρ by f is f # ρ := ρ ◦ f -1 ∈ P ( S ) . Here f -1 is the preimage of f .

We single out two particular functions. The function proj Y k : Y 1 × · · · × Y n → Y k defined by proj Y k ( y 1 , . . . , y k , . . . , y n ) := y k is the projection function of Y n onto Y k . We note that the pushforward of the projection map is marginalization: ν µ := proj Y # µ is the Y -marginal of µ ∈ P ( Y × Z ) . The bootstrap function b a,b : R → R is defined by b a,b ( z ) := a + bz from [6].

Our analysis works with conditional distributions, which we formalize as probability kernels, as well as a tensor-product notation constructing product measures and for disintegrating product measures. For any Y ⊂ R m and Z ⊂ R n , the space of (Borel) probability kernels from Y to Z , denoted K ( Y , P ( Z )) , is the set of all indexed measures λ for which y ↦→ λ y ( S ) is measurable for each S ∈ B ( Z ) , the Borel subsets of Z . Given λ ∈ K ( Y , P ( Z )) and ρ ∈ P ( Y ) , the generalized product measure λ ⊗ ρ ∈ P ( Y × Z ) is defined as follows:

<!-- formula-not-decoded -->

Additionally, we can disintegrate any µ ∈ P ( Y × Z ) as a generalized product between either of its marginals and the induced conditional probabilities:

<!-- formula-not-decoded -->

An important subset of K ( Y , P ( Z )) consists of those kernels with bounded p th moments,

<!-- formula-not-decoded -->

which can be metrized as complete metric spaces. In this work, we consider their metrization via the following metrics based on the Wasserstein metrics [40] d p ,

<!-- formula-not-decoded -->

where p, q ∈ [1 , ∞ ) and ω ∈ P ( Y ) . These metrize topologies on K p ( Y , P ( Z )) akin to the weak topology on probability measures with finite p th moments.

## 2.1 Markov Decision Processes and Reinforcement Learning

A discounted MDP is a five-tuple ( X , A , P, r, γ ) . Here X ⊂ R m is the state space , A ⊂ R n is the action space , r ∈ M b ( X × A ) is the reward function , and γ ∈ (0 , 1) is the discount factor . 1

Central to RL are policies. A policy is a probability kernel π ∈ K ( X , P ( A )) . Policies induce state transition kernels ˆ P π as well as a state-action transition kernels ˇ P π , given by

<!-- formula-not-decoded -->

respectively. Therefore, policies yield sequences of states as well as state-action pairs, labeled ( S π t ) t ≥ 0 and ( X π t , A π t ) t ≥ 0 respectively, whose sequences of laws ( ν π t ) t ≥ 0 and ( µ π t ) t ≥ 0 are given by

<!-- formula-not-decoded -->

for some ν 0 ∈ P ( X ) . Given ν 0 ∈ P ( X ) , the long-term behavior of any policy π can be encoded via its (discounted, state-action) occupancy measure µ π , the set of which we denote by O ( ν 0 ) ,

<!-- formula-not-decoded -->

Policies also induce return distribution functions ζ π ∈ K ( X × A , P ( R )) and η π ∈ K ( X , P ( R )) ,

<!-- formula-not-decoded -->

whose means, the action-value function q π ∈ M b ( X × A ) and the value function v π ∈ M b ( X ) ,

<!-- formula-not-decoded -->

lead to the RL objective: find a π ⋆ ∈ K ( X , P ( A )) such that q π ⋆ ≥ q π for all π . Such a policy is called optimal . Generally, many policies are optimal. However, their associated action-value functions are identical (see [32]). We denote this optimal action-value function by q ⋆ .

## 2.2 Entropy-Regularized Reinforcement Learning

In ERL, the value of a policy is penalized by how far it diverges from a fixed reference policy π ref ∈ K ( X , P ( A )) . In particular, the τ -ERL problem with temperature τ &gt; 0 is

<!-- formula-not-decoded -->

When τ = 0 , we recover the linear programming formulation of the (expected-value) RL objective. In ERL, the regularizer R is strictly convex. Thus, J τ ia strictly concave and its maximizer unique. 2

<!-- formula-not-decoded -->

Given Lemma 2.1, one might hope that the well-posedness of τ -ERL could be realized through simple, yet power methods like the direct method in the calculus of variations. However, outside the tabular case, this is unclear, for many reasons, the first of which is that M b ( X × A ) is not separable.

The well-posedness of τ -ERL, however, can be established through other means. In particular, in τ -ERL, only one optimal policy exists, and it is characterized as a Boltzmann-Gibbs (BG) policy.

Definition 2.2. Let q ∈ M ( X × A ) and τ &gt; 0 . We denote the Boltzmann-Gibbs policy associated to q and τ by G τ q , and it is characterized by

<!-- formula-not-decoded -->

We note that ( G τ q ) x is well-defined if and only if ( V τ q )( x ) ∈ R .

1 We expect many of our results can be extended to Polish spaces.

2 The only work we are aware of that establishes a comparable result is [28]. However, their result is on tabular MDPs and establishes convexity on O ( ν 0 ) , not on all of P ( X × A ) .

More specifically, it is well-known that the optimal policy of τ -ERL is the BG policy associated to the unique fixed point q ⋆ τ of the soft Bellman optimality operator B ⋆ τ : M ( X × A ) → M ( X × A ) ,

<!-- formula-not-decoded -->

(See Lemma A.7.) The following theorem summarizes the well-posedness of τ -ERL.

Theorem 2.3. Let τ &gt; 0 . The policy π τ,⋆ := G τ q ⋆ τ is optimal, and uniquely so. More precisely, for all ν 0 , ν ′ 0 ∈ P ( X ) , we have that arg max O ( ν 0 ) J τ = π τ,⋆ = arg max O ( ν ′ 0 ) J τ . [Proof]

In Appendix A, we prove Theorem 2.3 as well as a collection of supporting and related results that generalize well-known results in tabular MDPs. We include them for completeness.

In the remainder of this work, we study the evolution of τ -optimal objects as τ vanishes. In the tabular regime, where M b ( X × A ) is separable, one can establish the existence and uniqueness of a τ -optimal occupancy measure: µ ⋆ τ . Furthermore, under a compatibility assumption, one can prove that the limit of the sequence ( µ ⋆ τ ) τ&gt; 0 as τ vanishes exists and is unique as well.

Assumption 2.4. The intersection of { arg sup O ( ν 0 ) J 0 } and { R &lt; ∞} is nonempty.

Assumption 2.4 asks that our regularizer isn't identically + ∞ on the set of optimal policies. Without such an assumption, τ -ERL and RL have no meaningful relationship, as we shall see in Section 3.

Theorem 2.5. Suppose that r ∈ M b ( X × A ) and that X × A is finite. For every τ &gt; 0 , let µ ⋆ τ be the maximizer of J τ over O ( ν 0 ) . If Assumption 2.4 holds, the sequence ( µ ⋆ τ ) τ&gt; 0 has a unique setwise limit as τ tends to zero. This limit µ ⋆ 0 is the minimizer of R over arg sup O ( ν 0 ) J 0 . [Proof]

Consequently, in the tabular setting, the sequence ( π τ,⋆ ) τ&gt; 0 has a unique limit.

Remark 2.6. Even if Theorem 2.5 could be extended to hold true in continuous MDPs, occupancy measure convergence does not guarantee policy convergence, outside of the tabular setting. Theorem 2.5 is a statement about a sequence of joint distributions . A policy convergence statement would be one about a sequence of conditional distributions (i.e., probability kernels). In general, the convergence of a sequence of joint distributions does not imply the convergence of the associated sequence of conditional distributions with respect to a fixed marginal (see, e.g., [7, Example 10.4.24]). While it is possible that the structure O ( ν 0 ) permits a type of policy convergence, we are unaware of any such result for continuous MDPs.

## 3 Convergence to Optimality: The Temperature Decoupling Gambit

While ERL has a unique solution, this identifiability comes at a cost with respect to RL: the resulting policy is suboptimal for RL. In this section, we analyze vanishing-temperature limits in τ -ERL. Our main results for this section-Theorems 3.9 and 3.10-show that policies and their return distributions converge under the scheme of Definition 3.7 to interpretable, optimal limits as τ → 0 .

To understand the ways in which τ -ERL converges to RL, we define a (new) π ref -sensitive variant of the Bellman optimality operator, the Bellman reference-optimality operator . We call its unique fixed point the reference-optimal action-value function .

Lemma 3.1. Let r ∈ M b ( X × A ) , γ &lt; 1 , and B ⋆ ref : M ( X × A ) → M ( X × A ) be defined by

<!-- formula-not-decoded -->

Then B ⋆ ref is a contraction on M b ( X × A ) . Thus, it has a unique fixed point q ⋆ ref . [Proof]

Generally, q ⋆ ref is distinct from q ⋆ . Yet, ERL recovers q ⋆ ref in the vanishing temperature limit.

<!-- formula-not-decoded -->

τ → ref →

Theorem 3.2 implies that optimal policies, in general, cannot be recovered by taking vanishing temperature limits in ERL. We formalize a notion of reference-optimality to highlight this distinction.

Definition 3.3. A policy π ∈ K ( X , P ( A )) is said to be reference-optimal (against π ref ) if q π ≥ q ⋆ ref . Moreover, π is said to be ϵ -reference optimal if q π ≥ q ⋆ ref -ϵ .

Generally, q ⋆ ref &lt; q ⋆ . For instance, consider an MDP with one state ⊥ (a bandit), A = [0 , 1] , and π ref ⊥ = U ( A ) . If r ( ⊥ , · ) = δ 1 / 2 , then sup A q ⋆ ( ⊥ , · ) = 1 , while ess sup π ref ⊥ q ⋆ ( ⊥ , · ) = 0 . However, in many interesting cases, reference-optimal policies are optimal in the classic sense. When A is discrete and π ref x is supported on all of A -a ubiquitous assumption in ERL-then indeed q ⋆ = q ⋆ ref . Likewise, when A is continuous and ( P, r ) satisfy certain regularity conditions, then q ⋆ is continuous [20]. In these case, a reference-optimal policy is optimal.

̸

When q ⋆ ref = q ⋆ , even state-of-the-art continuous-control methods, entropy-regularized or otherwise, can at best hope to achieve q ⋆ ref , and not q ⋆ . This is because, when q ⋆ ref = q ⋆ , optimal actions form a measure 0 set. And so, even rich policy classes, such as neural-network-parameterized Gaussian policies [19] or diffusion policies [9] will not sample these actions, with probability 1 . Thus, moving forward, we establish q ⋆ ref as a 'skyline' for optimal performance. In other words, we strive to achieve convergence to reference-optimal policies.

̸

Under the next assumption, we can derive convergent policy optimization schemes as τ tends to zero.

Assumption 3.4. A constant p ref &gt; 0 exists for which

<!-- formula-not-decoded -->

Remark 3.5. If A is discrete and π ref x is uniformly lower bounded, Assumption 3.4 holds. This is a standard assumption. When A is continuous, this assumption is more difficult to guarantee. Intuitively, it asks that there is enough mass surrounding the optima of the entropy-regularized optimal value functions q ⋆ τ for KL(( G τ q ⋆ τ ) x ∥ π ref x ) to remain bounded in the limit.

A result key to the remainder of our work is the following bound on the total variation distance between pairs of BG policies in terms of their temperature and the distance between their potentials.

Theorem 3.6. Let q, q ′ ∈ M ( X × A ) . For any τ &gt; 0 and any x ∈ X ,

<!-- formula-not-decoded -->

In particular,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

While q ⋆ τ and V τ q ⋆ τ converge in the zero-temperature limit, whether or not τ -regularized optimal policies π τ,⋆ converge is still unclear. Indeed, under Assumption 3.4, ∥ q ⋆ ref -q ⋆ τ ∥ ∞ ≲ τ (see Lemma B.10). However, the log-probabilities of an action a under π τ,⋆ are amplified by τ -1 . Hence, the total variation difference between the BG policy at temperature τ and potential q ⋆ ref and π τ,⋆ may not vanish as τ vanishes. Based on this insight, we introduce the temperature decoupling gambit .

Definition 3.7. Given τ &gt; 0 , the temperature decoupling gambit specifies an alternate temperature σ = σ ( τ ) and constructs π τ,σ := G τ q ⋆ σ . In particular, it requires that σ/τ → 0 as τ → 0 .

At any τ &gt; 0 , decoupled-temperature policies π τ,σ are necessarily not optimal for the τ -regularized problem. Nevertheless, unlike π τ,⋆ , the policies π τ,σ produced by the temperature decoupling gambit realize long-term advantages: they have convergence guarantees in the vanishing temperature limit, and they recover an interpretable reference-optimal policy.

Definition 3.8. Let q ⋆ denote the optimal action-value function in a given MDP, and let π ref ∈ K ( X , P ( A )) . The optimality-filtered reference policy π ref ,⋆ is defined by

<!-- formula-not-decoded -->

Here χ Y is the characteristic or indicator function for the measurable set Y .

Heuristically, the optimality-filtered reference π ref ,⋆ x is the restriction of π ref x onto the set of expectedvalue-optimal actions in the state x . 3 When π ref is the uniform random policy, that is, π ref x = U ( A )

3 This is exact when q ⋆ = q ⋆ ref .

for all x ∈ X , we see that π ref ,⋆ x = U ( N ⋆ ref ( x )) -the uniform policy on optimal actions . In a sense, π ref ,⋆ is the most diverse (reference-)optimal policy; it does not discriminate between optimal actions.

In general, even when π τ,⋆ does converge as τ converges to zero, its limit is different from π ref ,⋆ . We demonstrate this explicitly in Section 3.1. On the other hand, our next result proves that the temperature decoupling gambit enables convergence to π ref ,⋆ . 4

Theorem 3.9. Under Assumption 3.4, if σ = σ ( τ ) is such that lim τ → 0 σ/τ = 0 , then π τ,σ x → π ref ,⋆ x as τ → 0 , for all x ∈ X , in TV if A is discrete and weakly if A is continuous. [Proof]

At the heart of the proof of Theorem 3.9 is the following inequality (a direct consequence of Theorem 3.6 and Lemma B.10), which relates the BG policies at temperature τ and potentials q ⋆ σ and q ⋆ ref :

<!-- formula-not-decoded -->

This inequality reduces questions of convergence of G τ q ⋆ σ to those of G τ q ⋆ ref (the vanishing temperature limit of a BG policy with a fixed potential is well-studied). Note that the smaller the fraction σ/τ is, the closer these two policies are. For instance, taking σ ( τ ) = τ 3 ensures that G τ q ⋆ σ is more like G τ q ⋆ ref than taking σ ( τ ) = τ 2 . In particular, it is from this inequality that the temperature decoupling gambit's requirement that σ/τ → 0 as τ → 0 arises.

Beyond enabling policy convergence in the vanishing temperature limit, the temperature decoupling gambit also ensures return distribution function convergence.

Theorem 3.10. Suppose A is discrete and Assumption 3.4 holds. If σ = σ ( τ ) is such that σ/τ → 0 as τ → 0 , then, for any p, p ′ ∈ [1 , ∞ ) and ω ∈ P ( X × A ) , as τ → 0 , the return distribution functions ζ τ,σ of the temperature-decoupled policies π τ,σ satisfy d p ; p ′ ,ω ( ζ τ,σ , ζ π ref ,⋆ ) → 0 . [Proof]

While Theorem 3.10 does not yet provide an algorithm for approximating ζ ⋆ , this result serves as inspiration for such developments in Section 4.

## 3.1 Numerical Demonstration

In this section, we demonstrate that the policies learned via the temperature decoupling gambit differ from those learned in ERL, even in the presence of stochastic updates.

Figure 3.1 shows a given tristate MDP with two actions (blue: a 1 ; green: a 2 ), as well as learned policies ˆ π τ,⋆ and ˆ π τ,σ estimated with soft Q-learning [18]. Here π ref x = U ( A ) for all x ∈ X and γ = 0 . 9 . As this MDP is tabular, Theorem 2.5 implies that the policies π τ,⋆ converge as τ → 0 . Thus, the temperature decoupling gambit is not necessary to guarantee convergence. Yet we see different limiting behavior. As predicted by Theorem 3.9, the estimates ˆ π τ,σ converge to π ref ,⋆ , as τ → 0 . With uniform π ref , this is the policy that samples all optimal actions, given a state, with equal probability. As τ → 0 , the estimates ˆ π τ,⋆ x 0 do converge to a different optimal policy. This difference is in x 0 , where ˆ π τ,⋆ x 0 collapse to δ a 1 . We take σ = τ 2 , in line with Definition 3.7. The two optimal policies found emphasize different notions of diversity. The limit of π τ,⋆ filters out optimal actions in order to play actions more uniformly on average with respect to state occupancy in the long term, while the limit of π τ,σ looks to maximize state-wise action diversity.

Figure 3.1: Differences between ˆ π τ,⋆ and ˆ π τ,σ , approximated with soft Q-learning. Left : Graphical model of the MDP; arrow colors encode actions. Center : Depiction of the estimated policies ˆ π τ,⋆ at each state, as τ → 0 . Right : Depiction of the estimated policies ˆ π τ,σ at each state, as τ → 0 . Summary : Learned policies differ in x 0 , but are otherwise the same.

<!-- image -->

4 We discuss the benefits of this optimal policy in Appendix D.

## 4 Convergent Approximation of Optimal Return Distributions

In this section, we formalize a new branch of DRL and introduce distributional ERL (DERL). 5 Our main results in this section, Theorems 4.5, 4.6, and 4.7, establish convergent iterative schemes for approximate (reference-)optimal return distribution estimation. In Section 4.1, we introduce novel soft distributional Bellman operators, for evaluation and for control, and establish the convergence of their iterates. The behavior of the resulting return distribution approximations in the vanishing temperature limit is treated in Section 4.2. To conclude, a simulation is presented in Section 4.3 to illustrate the resulting optimal return distribution approximations.

## 4.1 Entropy-Regularized Distributional Reinforcement Learning

We begin by defining a soft distributional Bellman operator , as an analogue to the distributional Bellman operator [5, 35]. It, under certain conditions, computes

<!-- formula-not-decoded -->

Notationally, for any π ∈ K ( X , P ( A )) , we define kl [ π ] : X → R via kl [ π ]( x ) = KL( π x ∥ π ref x ) . Definition 4.1. For any τ &gt; 0 , γ &lt; 1 , and π ∈ K ( X , P ( A )) , the soft distributional Bellman operator T π τ is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 4.2. If r M ( X A ) , γ &lt; 1 , and π K ( X , ( A )) is such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

the soft distributional Bellman operator T π τ is a γ -contraction in d p for every τ ≥ 0 . Thus, it has a unique solution to the fixed point equation ¯ ζ = T π τ ¯ ζ , which we denote by ¯ ζ π,τ . [Proof]

Next, we move to policy improvement . In ERL, improving the action-value function q involves policy evaluation with the policy G τ q . We leverage this insight to enable control.

Definition 4.3. For any τ &gt; 0 , the soft distributional optimality operator T ⋆ τ is given by

<!-- formula-not-decoded -->

We proceed by establishing a simple, but useful algebraic property.

<!-- formula-not-decoded -->

τ τ

Now we prove that iterates of T ⋆ τ converge, unlike iterates of T ⋆ [5].

Theorem 4.5. For any ¯ ζ ∈ K p ( X × A , P ( R )) and temperature τ &gt; 0 define the iterates ( ¯ ζ n ) n ∈ N given by ¯ ζ n +1 = T ⋆ τ ¯ ζ n for ¯ ζ 0 = T ⋆ τ ¯ ζ . Then, for ¯ ζ τ,⋆ := ¯ ζ τ,π τ,⋆ ,

<!-- formula-not-decoded -->

where C, C p,τ,γ &lt; ∞ are constants depending on ∥ r ∥ sup , ( p, τ, γ, ∥ r ∥ sup ) respectively. [Proof]

Theorem 4.5 leads to stability in entropy-regularized optimal return distribution estimation. In Figure 4.1, we demonstrate the stability of T ⋆ τ and the instability of T ⋆ . The iterates defined in Theorem 4.5 converge to soft return distributions , which are influenced by stepwise regularization penalties and correspond to policies that are optimal in ERL. To estimate optimal return distributions, we must consider vanishing temperature limits.

5 Independently and concurrently, similar results were established by [26] in the fixed-temperature regime, but only with discrete action spaces and π ref being the uniform policy.

Figure 4.1: Evolution of the soft optimality iterates ( T ⋆ τ ) k ζ ( x, a ) (bottom row) and the iterates of the distributional optimality operator ( T ⋆ ) k ζ ( x, a ) (top row). Video of entire iterate sequence is available at https://harwiltz.github.io/assets/stable-return-distributions/ .

<!-- image -->

## 4.2 Convergent Optimal Return Distribution Estimation in the Vanishing Temperature Limit

In this section, we instantiate the first methods for computing iterates that approximate referenceoptimal return distribution functions in a stable manner.

Theorem 4.6. Suppose Assumption 3.4 holds. Let p, p ′ ∈ [1 , ∞ ) and ω ∈ P ( X × A ) . For any ϵ, δ &gt; 0 , there exists a τ &gt; 0 for which d p ; p ′ ,ω ( ¯ ζ τ,π τ,⋆ , ζ π τ,⋆ ) ≤ δ/ 2 and q π τ,⋆ is ϵ/ 2 -referenceoptimal. In turn, an n ϵ,δ = n ϵ,δ ( τ ) ∈ N exists for which

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 4.6 is the first example of a convergent iterative scheme for approximating the return distribution of a (reference-)optimal policy. While it ensures convergence to a ϵ -reference-optimal return distribution, it is still not possible a priori to characterize which return distribution will be learned. As ϵ → 0 , there may be no stable trend in the return distribution that will be estimated because π τ,⋆ may not converge. To achieve (characterizable) convergence to a reference-optimal return distribution, we turn back to the temperature decoupling gambit.

Theorem 4.7. Suppose Assumption 3.4 holds and A is discrete. Let p, p ′ ∈ [1 , ∞ ) and ω ∈ P ( X × A ) . For any ϵ, δ &gt; 0 and ¯ ζ 0 ∈ K p ( X × A , P ( R )) , there exists τ &gt; 0 , a decoupled σ τ &gt; 0 and n opt , n eval ∈ N such that

<!-- formula-not-decoded -->

Theorem 4.7 outlines an algorithm for estimating ζ π ref ,⋆ . First, approximate ¯ ζ σ,⋆ via n opt applications of T ⋆ σ (control). Second, extract the mean: ˆ q ⋆ σ ≈ q ⋆ σ . Finally, apply T π τ n eval times, with π = G τ ˆ q ⋆ σ (evaluation). If τ ≪ 1 , σ = τ 2 , for example, and n opt , n eval ≫ 1 , then the resulting return distribution is as desired. This ensures convergence, (reference)-optimality, and interpretability of the final iterate.

## 4.3 Numerical Demonstration

Here we validate that ¯ ζ τ,σ approximates ζ π ref ,⋆ . We consider the MDP given in Figure 4.2. Arrow colors correspond to different actions. Dashed lines represent transitions that occur with probability 1 / 2 . In this MDP, different optimal policies have distinct return distributions. From x 1 , the blue action yields return of 2 γ (1 -γ ) -1 , while the green action achieves return 4 γ (1 -γ ) -1 Bernoulli (1 / 2) . In Figures 4.3 and 4.4, we compute estimates ˆ ζ τ,⋆ ≈ ¯ ζ τ,⋆ and ˆ ζ τ,σ ≈ ¯ ζ τ,σ by (soft) distributional dynamic programming using 64-bit precision and

Figure 4.2: An illustrative MDP.

<!-- image -->

32-bit precision respectively. 32-bit precision is the default in many scientific computing libraries, such as Jax [8]. Here γ = 1 / 2 , π ref x = U ( A ) for all x ∈ X , and σ = τ 2 . We consider τ ∈ { 10 -(2 m +1) : m = 0 , 1 , 2 , 3 , 4 } . Our simulation is a practical implementation of Theorem 4.7. First, we approximate n opt = 1000 iterative applications of our soft Bellman optimality operator at τ

Figure 4.3: Estimates of return distributions via soft distributional dynamic programmingˆ η τ,σ using the temperature-decoupling gambit and ˆ η τ,⋆ without-as τ → 0 . As the temperature vanishes, η τ,σ recovers the return distribution of π ref ,⋆ , shown on the right.

<!-- image -->

(control). Then, we extract ˆ q ⋆ τ , an approximation of q ⋆ τ , and construct two policies: the BG policy at τ and the BG policy at τ 1 / 2 , both with potential ˆ q ⋆ τ . Next we approximate n eval = 1000 iterative applications of our soft Bellman operator (policy evaluation) at temperature τ with the first policy and at temperature τ 1 / 2 with the second policy. These yield approximations of ¯ ζ τ,⋆ and ¯ ζ τ,σ , respectively. Figures 4.3 and 4.4 depict the policy-averaged return distributions ˆ η τ,⋆ x 0 and ˆ η τ,σ x 0 compared to the

Figure 4.4: Return distribution estimation with vanishing temperature using soft distributional dynamic programming, with 32-bit floating point precision.

<!-- image -->

baseline η ⋆ x 0 := proj R # ( ζ ⋆ x 0 , ⊗ π ref ,⋆ x 0 ) . The iterates are approximated via categorical representations [5, 34] supported on 121 uniformly-spaced atoms on [ -2 , 8] , and MMD projections [43] with the energy distance kernel E 3 / 2 . In both figures, we see that the sequence of temperature-decoupled return distribution estimates approximate the return distribution associated to π ref ,⋆ (right). Return distributions estimates of ¯ ζ τ,⋆ also converge to those of optimal policies, as predicted by Theorem 4.6, but we find reach different return distributions in each case. While the temperature-decoupling gambit is not impervious to precision issues, it stabilizes BG policy estimation.

## 5 Related Work

Entropy regularization in RL was introduced by [48] for inverse RL, where it is necessary to disambiguate optimal policies and identify the most likely reward function to explain demonstrated behavior. ERL with π ref as the uniform policy-termed maximum entropy or MaxEnt RL, has been highly influential in deep reinforcement learning. Heuristically, MaxEnt RL encourages policies to be more uniform, thereby enhancing exploration, sample-efficiency, behavioral diversity [29, 18, 17], as well as robustness [16, 2, 11, 12]. Heuristic approaches to adaptive temperature schemes in deep

MaxEnt RL have been effective in practice [19, 47]. Policy optimization in MaxEnt RL has been shown to be equivalent to a form of inference, conditional on a notion of behavioral optimality, in a certain graphical model [24, 14], and further characterizations of MaxEnt RL have lead to principled algorithms for efficient exploration [30, 39]. Alternative forms of regularized RL objectives and optimizers have been proposed and analyzed [25, 31, 36, 4, 37, 15].

Policy optimization algorithms for entropy-regularization in general are presented and analyzed by [28]-these methods apply to tabular MDPs and fixed nonzero temperature. [27] provide improved convergence rates for entropy-regularized policy optimization. They also derive convergence results in the vanishing temperature limit, but only in the bandit setting. Exceptionally, [23], based on the work of [1], studies global convergence of policy gradient methods in continuous entropy-regularized MDPs, for fixed and vanishing temperature, with neural network policies via mean-field analysis. However, their analysis requires an extra regularization term to a distribution over neurons, precluding convergence to an optimum of RL. To the best of our knowledge, our work is the first to introduce a convergent policy optimization scheme for general MDPs in the vanishing temperature limit.

Entropy regularization in DRL is largely unexplored. [22] experimented with an adaptation of Rainbow [21] to MaxEntRL, but without analysis or formalism. The concurrent work of [26] also introduced soft distributional Bellman operators, but did not study vanishing temperature limits, and did not establish convergence rates for iterates of T ⋆ τ even for fixed τ . Moreover, the work of [26] established convergence only in the case of discrete A , and only for a uniform reference policy. Works have investigated the challenges of estimating optimal return distributions [5, 42], and more generally, the influence of particular tractable distribution representations on learning dynamics and fixed point accuracy [45, 46, 43, 3]. In [6], the authors show that distributional analogues of B ⋆ produce iterates that converge when there is a unique (deterministic) optimal policy. The interplay between policy optimization stability and return distributions was studied in [33]. Their empirical study found that distributions of returns following stochastic policy gradient updates tend to have long left tails, and called for methods to guide policies into smoother regions ('quiet' neighborhoods) of the return landscape , the manifold of policy returns across parameters. This study focused primarily on deterministic policy gradient methods.

## 6 Discussion

In this work, we have investigated policy and return distribution convergence as the temperature vanishes in ERL. Our findings motivate iterative schemes for achieving convergence results beyond expected returns. However, they come with several limitations. In particular, while we have established policy convergence via the temperature-decoupling gambit, this convergence qualitative. As a consequence, our ability to derive approximation algorithms for ζ π with π = π ref ,⋆ is limited; it is a priori unclear which temperatures are required for ζ τ,σ to be an ϵ -approximation of ζ π with π = π ref ,⋆ in d p ; p,ω and, therefore, to deploy for iterative applications of T ⋆ τ or T π τ with π = G τ q ⋆ σ . At the moment, however, our results ensure that by progressively annealing τ , the scheme discussed in Theorem 4.7 will approach ζ π with π = π ref ,⋆ . Nevertheless, quantifying Theorem 3.10 is an exciting direction for future work. Another exciting direction for future work is to try to incorporate the temperature-decoupling gambit into the many algorithms in ERL/RL.

## Acknowledgments and Disclosure of Funding

The authors wish to thank Wesley Chung, Mark Rowland, Jesse Farebrother, Arnav Kumar Jain, Siddarth Venkatraman, Athanasios Vasileiadis, Aditya Mahajan, and Doina Precup for helpful comments and discussions. HW was supported by the National Sciences and Engineering Research Council of Canada (NSERC) and the Fonds de Recherche du Québec. MGB was supported by the Canada CIFAR AI Chair program and NSERC. This work was supported in part by DARPA HR0011-23-9-0050.

## References

- [1] A. Agazzi and J. Lu. Global optimality of softmax policy gradient with single hidden layer neural networks in the mean-field regime. arXiv preprint arXiv:2010.11858 , 2020.

- [2] Z. Ahmed, N. Le Roux, M. Norouzi, and D. Schuurmans. Understanding the impact of entropy on policy optimization. In Interational Conference on Machine Learning (ICML) , 2019.
- [3] J. Alhosh, H. Wiltzer, and D. Meger. Tractable representations for convergent approximation of distributional HJB equations. Multidisciplinary Conference on Reinforcement Learning and Decision Making (RLDM) , 2025.
- [4] K. Asadi and M. L. Littman. An alternative softmax operator for reinforcement learning. In Interational Conference on Machine Learning (ICML) , 2017.
- [5] M. G. Bellemare, W. Dabney, and R. Munos. A distributional perspective on reinforcement learning. In Interational Conference on Machine Learning (ICML) , 2017.
- [6] M. G. Bellemare, W. Dabney, and M. Rowland. Distributional reinforcement learning . MIT Press, 2023.
- [7] V. I. Bogachev and M. A. S. Ruas. Measure theory , volume 1. Springer, 2007.
- [8] J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, and Q. Zhang. JAX: composable transformations of Python+NumPy programs, 2018.
- [9] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research , 2023.
- [10] R. Dadashi, A. A. Taiga, N. Le Roux, D. Schuurmans, and M. G. Bellemare. The value function polytope in reinforcement learning. In Interational Conference on Machine Learning (ICML) , 2019.
- [11] B. Eysenbach and S. Levine. If MaxEnt RL is the answer, what is the question? arXiv preprint arXiv:1910.01913 , 2019.
- [12] B. Eysenbach and S. Levine. Maximum entropy RL (provably) solves some robust RL problems. In International Conference on Learning Representations (ICLR) , 2021.
- [13] E. A. Feinberg and A. Shwartz. Handbook of Markov decision processes: methods and applications , volume 40. Springer Science &amp; Business Media, 2012.
- [14] M. Fellows, A. Mahajan, T. G. J. Rudner, and S. Whiteson. VIREL: A variational inference framework for reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS) , 2019.
- [15] R. Fox. Toward provably unbiased temporal-difference value estimation. In Optimization Foundations for Reinforcement Learning workshop (OPTRL @ NeurIPS) , 2019.
- [16] R. Fox, A. Pakman, and N. Tishby. Taming the noise in reinforcement learning via soft updates. In Conference on Uncertainty in Artificial Intelligence (UAI) , 2015.
- [17] D. Garg, J. Hejna, M. Geist, and S. Ermon. Extreme Q-learning: MaxEnt RL without entropy. In International Conference on Learning Representations (ICLR) , 2023.
- [18] T. Haarnoja, H. Tang, P. Abbeel, and S. Levine. Reinforcement learning with deep energy-based policies. In Interational Conference on Machine Learning (ICML) , 2017.
- [19] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In Interational Conference on Machine Learning (ICML) , 2018.
- [20] O. Hernández-Lerma and J. B. Lasserre. Discrete-time Markov control processes: basic optimality criteria , volume 30. Springer Science &amp; Business Media, 2012.
- [21] M. Hessel, J. Modayil, H. Van Hasselt, T. Schaul, G. Ostrovski, W. Dabney, D. Horgan, B. Piot, M. Azar, and D. Silver. Rainbow: Combining improvements in deep reinforcement learning. In AAAI Conference on Artificial Intelligence , 2018.
- [22] D. Hu, P. Abbeel, and R. Fox. Count-based temperature scheduling for maximum entropy reinforcement learning. In Deep RL Workshop @ (NeurIPS) , 2021.
- [23] J.-M. Leahy, B. Kerimkulov, D. Siska, and L. Szpruch. Convergence of policy gradient for entropy regularized mdps with neural network approximation in the mean-field regime. In Interational Conference on Machine Learning (ICML) , 2022.

- [24] S. Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review. arXiv preprint arXiv:1805.00909 , 2018.
- [25] M. L. Littman and C. Szepesvári. A generalized reinforcement-learning model: Convergence and applications. In Interational Conference on Machine Learning (ICML) , 1996.
- [26] X. Ma, J. Chen, L. Xia, J. Yang, Q. Zhao, and Z. Zhou. DSAC: Distributional soft actor-critic for risk-sensitive reinforcement learning. Journal of Artificial Intelligence Research , 83, 2025.
- [27] J. Mei, C. Xiao, C. Szepesvari, and D. Schuurmans. On the global convergence rates of softmax policy gradient methods. In Interational Conference on Machine Learning (ICML) , 2020.
- [28] G. Neu, A. Jonsson, and V. Gómez. A unified view of entropy-regularized markov decision processes. arXiv preprint arXiv:1705.07798 , 2017.
- [29] B. O'Donoghue, R. Munos, K. Kavukcuoglu, and V. Mnih. Combining policy gradient and Q-learning. In International Conference on Learning Representations (ICLR) , 2016.
- [30] B. O'Donoghue, I. Osband, and C. Ionescu. Making sense of reinforcement learning and probabilistic inference. In International Conference on Learning Representations (ICLR) , 2020.
- [31] J. Peters, K. Mulling, and Y. Altun. Relative entropy policy search. In AAAI Conference on Artificial Intelligence , 2010.
- [32] M. L. Puterman. Markov decision processes: discrete stochastic dynamic programming . John Wiley &amp; Sons, 2014.
- [33] N. Rahn, P. D'Oro, H. Wiltzer, P.-L. Bacon, and M. Bellemare. Policy optimization in a noisy neighborhood: On return landscapes in continuous control. In Advances in Neural Information Processing Systems (NeurIPS) , 2023.
- [34] M. Rowland, M. Bellemare, W. Dabney, R. Munos, and Y. W. Teh. An analysis of categorical distributional reinforcement learning. In Interational Conference on Machine Learning (ICML) , 2018.
- [35] M. Rowland, R. Dadashi, S. Kumar, R. Munos, M. G. Bellemare, and W. Dabney. Statistics and samples in distributional reinforcement learning. In Interational Conference on Machine Learning (ICML) , 2019.
- [36] J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz. Trust region policy optimization. In Interational Conference on Machine Learning (ICML) , 2015.
- [37] Z. Song, R. E. Parr, and L. Carin. Revisiting the softmax bellman operator: New benefits and new perspective. In Interational Conference on Machine Learning (ICML) , 2019.
- [38] U. Syed, M. Bowling, and R. E. Schapire. Apprenticeship learning using linear programming. In Interational Conference on Machine Learning (ICML) , 2008.
- [39] J. Tarbouriech, T. Lattimore, and B. O'Donoghue. Probabilistic inference in reinforcement learning done right. In Advances in Neural Information Processing Systems (NeurIPS) , 2023.
- [40] C. Villani. Optimal transport: old and new , volume 338. Springer, 2008.
- [41] H. S. Wilf. generatingfunctionology . CRC press, 2005.
- [42] H. Wiltzer, M. G. Bellemare, D. Meger, P. Shafto, and Y. Jhaveri. Action gaps and advantages in continuous-time distributional reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS) , 2024.
- [43] H. Wiltzer, J. Farebrother, A. Gretton, and M. Rowland. Foundations of multivariate distributional reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS) , 2024.
- [44] H. Wiltzer, J. Farebrother, A. Gretton, Y. Tang, A. Barreto, W. Dabney, M. G. Bellemare, and M. Rowland. A distributional analogue to the successor representation. In Interational Conference on Machine Learning (ICML) , 2024.
- [45] H. E. Wiltzer, D. Meger, and M. G. Bellemare. Distributional Hamilton-Jacobi-Bellman equations for continuous-time reinforcement learning. In Interational Conference on Machine Learning (ICML) , 2022.
- [46] R. Wu, M. Uehara, and W. Sun. Distributional offline policy evaluation with predictive error guarantees. In Interational Conference on Machine Learning (ICML) , 2023.

- [47] Y. Xu, D. Hu, L. Liang, S. M. McAleer, P. Abbeel, and R. Fox. Target entropy annealing for discrete soft actor-critic. In Deep RL Workshop @ (NeurIPS) , 2021.
- [48] B. D. Ziebart, A. L. Maas, J. A. Bagnell, and A. K. Dey. Maximum entropy inverse reinforcement learning. In AAAI Conference on Artificial Intelligence , 2008.

## A Entropy-Regularized RL in Continuous MDPs

Here we prove Theorem 2.3 as well as a collection of supporting and related results that generalize well-known results in tabular MDPs.

We start with a characterization the geometry of the space of occupancy measures. The following result extends the well-known counterpart in tabular MDPs [13, 38, 10] to continuous MDPs. While certain parts of this result are proved by [20], not all connections are made, which we state here for the first time.

Theorem A.1. Let O ( ν 0 ) = { µ π : π ∈ K ( X , P ( A )) } the space of all occupancy measures under the initial state distribution ν 0 ∈ P ( X ) . Then O ( ν 0 ) is equivalent to the space of all µ ∈ P ( X × A ) that satisfy

<!-- formula-not-decoded -->

The space O ( ν 0 ) is convex, it is closed under setwise convergence.

Before proceeding with the proof of Theorem A.1, we recall the state occupancy measures ν π , given by

<!-- formula-not-decoded -->

where ( ν π t ) t ≥ 1 is the sequence of laws generated by ˆ P π starting at ν 0 .

Proposition A.2. Let ν π t and µ π t , for t ≥ 1 denote the laws generated by ˆ P π and ˇ P π starting at ν 0 and µ π 0 = π ⊗ ν 0 . Then µ π t = π ⊗ ν π t , for all t ≥ 1 . Hence, given π and ν 0 , the state marginal of the associated occupancy measure µ π is the associated state occupancy measure ν π .

The proof of this proposition will use the following lemma.

Lemma A.3. Under the hypotheses of Proposition A.2, for every t ≥ 1 , the conditional probabilities of µ π t with respect to its state marginal are π .

Proof. It suffices to prove that the conditional probabilities of µ π 1 are π . Let ν 1 denote the state marginal of µ π 1 . By definition,

<!-- formula-not-decoded -->

Thus, for any φ ∈ M b ( X × A ) , with ψ ( x ′ ) := ∫ φ ( x ′ , a ′ ) d π x ′ ( a ′ ) , observe that

<!-- formula-not-decoded -->

So the conditional probabilities of µ with respect to ν are π , as desired.

π 1 1 x

Proof of Proposition A.2. By Lemma A.3, it suffices to show that the state marginal of µ π 1 is ν π 1 . This holds:

<!-- formula-not-decoded -->

By this computation and Lemma A.3 applied successively to each pair ( µ π t +1 , µ π t ) for every t ≥ 1 , we deduce that µ π t = π ⊗ ν π t , for all t ≥ 1 . Finally, by the linearity of the integral, we conclude. Indeed,

<!-- formula-not-decoded -->

Proof of Theorem A.1. We prove this theorem in three steps.

Step 1: O ( ν 0 ) = F ( ν 0 ) . First, recall that proj X # µ π = ν π for any policy π , by Proposition A.2. Thus, we have that for any π and any Borel E ⊂ X ,

<!-- formula-not-decoded -->

This shows that O ( ν 0 ) ⊂ F ( ν 0 ) . It remains to show that F ( ν 0 ) ⊂ O ( ν 0 ) . Let µ ∈ F ( ν 0 ) , and let π µ denote its conditional action probabilities with respect to its state marginal ν µ -that is, µ = π µ ⊗ ν µ . Moreover, let ϕ 0 be any bounded measurable function. By the definition of P , we note that (A.1) can be written as

<!-- formula-not-decoded -->

Defining ϕ 1 ( x ) = ∫ X ϕ 0 ( x ′ ) d ˆ P π µ x ( x ′ ) , the rightmost term ∫ X ϕ 1 ( x 0 ) d ν µ ( x 0 ) can be again expanded via (A.1),

<!-- formula-not-decoded -->

Continuing, we define ϕ n +1 ( x ) = ∫ X ϕ n ( x ′ ) d ˆ P π µ x ( x ′ ) , which is bounded and measurable for each n ∈ N , yielding

<!-- formula-not-decoded -->

By the definition of ϕ n , we have that

<!-- formula-not-decoded -->

Moreover, by the boundedness of ϕ n , we deduce that II n → 0 . Substituting, we have

<!-- formula-not-decoded -->

Since ϕ 0 was an arbitrary bounded and measurable function, it follows that ν µ = ν π µ . Thus, µ = π ⊗ ν µ = µ π µ -the occupancy measure for the policy π µ . Consequently, any µ ∈ F ( ν 0 ) is a member of O ( ν 0 ) .

Step 2: O ( ν 0 ) is convex. The convexity of O ( ν 0 ) follows immediately from the structure of F ( ν 0 ) . Consider any µ 0 , µ 1 ∈ O ( ν 0 ) any α ∈ [0 , 1] , and define µ α = αµ 0 +(1 -α ) µ 1 . For any Borel E ⊂ X , we have that

<!-- formula-not-decoded -->

Since µ 0 , µ 1 ∈ F ( ν 0 ) , they solve (A.1), so we expand the RHS,

<!-- formula-not-decoded -->

So µ α ∈ F ( ν 0 ) = O ( ν 0 ) , as desired.

Step 3: O ( ν 0 ) is closed under setwise convergence. Let ( µ k ) k ∈ N ⊂ F ( ν 0 ) be a sequence that converges setwise to µ . Since ( x, a ) ↦→ P x,a ( E ) is bounded and measurable for any Borel E ⊂ X ,

Likewise, as µ k → µ setwise. Consequently, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first equality follows from (A.3), the second follows as µ k ∈ F ( ν 0 ) , and the final equality follows from (A.2). Thus, we see that µ ∈ F ( ν 0 ) = O ( ν 0 ) .

Now we prove Lemma 2.1.

Lemma 2.1. The functional R : P ( X × A ) → R

Proof. Observe that

We prove this in two steps. First, for every Borel f : X × A → [0 , ∞ ) , we have that is strictly convex. [Source]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, µ = π µ ⊗ ν µ ≪ π ref ⊗ ν µ if π µ x ≪ π ref x for ν µ -almost every x , and

Second, µ = π µ ⊗ ν µ ≪ π ref ⊗ ν µ implies that π µ x ≪ π ref x for ν µ -almost every x . Indeed, suppose that a set S ⊂ X exists such that ν µ ( S ) &gt; 0 and for each x ∈ S , we have that

Let

Then,

<!-- formula-not-decoded -->

This is a contradiction. And so, as desired.

<!-- formula-not-decoded -->

Now recall that

<!-- formula-not-decoded -->

Moreover, note that

In turn,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, R is convex. In particular, R is strictly convex as KL is strictly convex in its first argument.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

With Theorem A.1 and Lemma 2.1 in hand, we use the direct method from the Calculus of Variations to prove the well-posedness of τ -ERL, in the tabular setting.

Remark A.4. The space M b ( X × A ) endowed with the supnorm is a Banach space. Note that M b ( X × A ) ∗ ∼ = ba ( X × A ) , where ba ( X × A ) denotes the set of finitely additive set functions on B ( X × A ) equipped with the total variation norm. Note that the set of probability measures on X × A is a subset of the closed unit ball in ba ( X × A ) , which is weak* compact, by Banach-Alaoglu. The duality pairing for any µ ∈ P ( X × A ) and for any φ ∈ M b ( X × A ) is given by integration: ⟨ µ, φ ⟩ := ∫ φ d µ . In other words, weak* convergence is setwise convergence when P ( X × A ) is considered as a subset of the dual of ba ( X × A ) .

Theorem A.5. Suppose that r ∈ M b ( X × A ) , X × A is finite, and let ν 0 ∈ P ( X ) . A µ ⋆ τ ∈ O ( ν 0 ) that achieves the supremum in (2.2) exists. Moreover, no other occupancy measure does so.

Proof. Let the supremum in (2.2) be denoted by ϑ ⋆ τ and ( µ k ) k ∈ N ⊂ O ( ν 0 ) be such that

<!-- formula-not-decoded -->

In other words, let ( µ k ) k ∈ N ⊂ O ( ν 0 ) be a maximizing sequence. By Remark A.4, owing to the fact that M b ( X × A ) is separable (since X × A is finite), let ( µ k ℓ ) ℓ ∈ N be a weakly* convergent subsequence, with weak* limit µ ∞ . In particular, µ k ℓ → µ ∞ setwise. As O ( ν 0 ) is closed under setwise convergence, by Theorem A.1, we have that µ ∞ ∈ O ( ν 0 ) . Furthermore, π ref ⊗ ν µ k ℓ → π ref ⊗ ν µ ∞ setwise as well. As setwise convergence implies weak convergence and as the KL( µ ∥ µ ′ ) is lower-semicontinuous in the pair ( µ, µ ′ ) in the weak topology, we find that

<!-- formula-not-decoded -->

The penultimate equality uses that r is bounded. Thus, J τ ( µ ∞ ) = ϑ ⋆ τ . The previous argument applies to any sub-sequential weak* limit of our maximizing sequence. But as R is strictly convex, by Lemma 2.1, and O ( ν 0 ) is convex, by Theorem A.1, only one such limit exists.

We now move to prove Theorem 2.3. To do so, we state and prove some helpful results. We begin with policy evaluation.

For any π ∈ K ( X , P ( A )) , define q π τ : X × A → R ∪ {-∞} by

<!-- formula-not-decoded -->

By the tower property of condition expectation, we have that

<!-- formula-not-decoded -->

It is convenient to be able to evaluate a policy π (find q π τ ) in an iterative fashion. This can be done via the soft Bellman operator B π τ : M ( X × A ) → M ( X × A ) defined by

<!-- formula-not-decoded -->

but only on a restricted collection of policies.

Lemma A.6. If r ∈ M b ( X × A ) , γ &lt; 1 , and π is such that (4.1) holds with p = 1 , then the B π τ is contractive on M b ( X × A ) endowed with the supnorm. Its unique fixed point is q π τ .

Proof. Observe that

<!-- formula-not-decoded -->

by (4.1), and

<!-- formula-not-decoded -->

Next, we proceed with policy improvement.

Lemma A.7. If r ∈ M b ( X × A ) and γ &lt; 1 , then the soft Bellman optimality operator is a contraction on M b ( X × A ) endowed with the supremum norm. Thus, it has a unique fixed point q ⋆ τ .

Proof. Observe that and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma A.8. The following equality holds true: q G τ q ⋆ τ τ = q ⋆ τ .

Proof. Observe that

<!-- formula-not-decoded -->

In words, q ⋆ τ is a fixed point of the soft Bellman (policy evaluation) operator with π = G τ q ⋆ τ . As G τ q ⋆ τ is a Boltzmann-Gibbs policy with a bounded potential, by Lemma A.6 and the preceding note, this operator is a contraction with a unique fixed point. Hence,

<!-- formula-not-decoded -->

the unique fixed point of B π τ with π = G τ q ⋆ τ , as desired.

Lemma A.9. For every π ∈ K ( X , P ( A )) , we have that

Proof. First, we prove that

By definition and the Donsker-Varadhan variational principle,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now we conclude. Let q π τ, 0 := max { q π τ , 0 } . By (A.4) and since B ⋆ τ is a monotone operator,

<!-- formula-not-decoded -->

where the final equality holds by Lemma A.7, noting that ∥ q π τ, 0 ∥ sup &lt; ∞ .

Finally, we prove Theorem 2.3.

Theorem 2.3. Let τ &gt; 0 . The policy π τ,⋆ := G τ q ⋆ τ is optimal, and uniquely so. More precisely, for all ν 0 , ν ′ 0 ∈ P ( X ) , we have that arg max O ( ν 0 ) J τ = π τ,⋆ = arg max O ( ν ′ 0 ) J τ . [Source]

Proof. For any π ∈ K ( X , P ( A )) , let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

if µ π ∈ O ( ν 0 ) . Hence, it suffices to show that

<!-- formula-not-decoded -->

Note that

Observe, by Lemma A.8,

<!-- formula-not-decoded -->

Thus, by the Donsker-Varadhan variational principle,

<!-- formula-not-decoded -->

Finally, by Lemma A.9, we have that

<!-- formula-not-decoded -->

for all π , as desired.

To conclude this section, we prove Theorem 2.5.

Theorem 2.5. Suppose that r ∈ M b ( X × A ) and that X × A is finite. For every τ &gt; 0 , let µ ⋆ τ be the maximizer of J τ over O ( ν 0 ) . If Assumption 2.4 holds, the sequence ( µ ⋆ τ ) τ&gt; 0 has a unique setwise limit as τ tends to zero. This limit µ ⋆ 0 is the minimizer of R over arg sup O ( ν 0 ) J 0 . [Source]

Proof. Let µ ⋆ ∈ { arg sup O ( ν 0 ) J 0 } ∩ { R &lt; ∞} . Then,

<!-- formula-not-decoded -->

In turn, for all τ &gt; 0 , we deduce that R ( µ ⋆ τ ) ≤ R ( µ ⋆ ) .

Now let µ 0 be any limit of any setwise convergent subsequence of ( µ ⋆ τ ) τ&gt; 0 (cf. the proof of Theorem A.5 and Remark A.4). As R is weakly lower semi-continuous we find that

<!-- formula-not-decoded -->

Moreover, since R ( µ ⋆ ) &lt; ∞ , by Lemma B.2, and as r ∈ M b ( X × A ) , we deduce that

<!-- formula-not-decoded -->

Therefore, µ 0 ∈ arg sup O ( ν 0 ) J 0 and minimizes R over arg sup O ( ν 0 ) J 0 .

Since R is strictly convex, by Lemma 2.1, and the set arg sup O ( ν 0 ) J 0 is convex, R has at most one minimizer among this set. In turn, only one such limit µ 0 exists, call it µ ⋆ 0 . Hence, µ ⋆ τ → µ ⋆ 0 setwise, as desired.

## B Proofs for Section 3

Before proving the results from Section 3, we introduce some helpful notation. For any q : X × A → R , we define

<!-- formula-not-decoded -->

Additionally, we will define M τ : L ∞ ( X × A ) → R ∪ {∞} according to

<!-- formula-not-decoded -->

We start by proving that B ⋆ ref is contractive on M b ( X × A ) .

<!-- formula-not-decoded -->

Lemma 3.1. Let r ∈ M b ( X × A ) , γ &lt; 1 , and B ⋆ ref : M ( X × A ) → M ( X × A ) be defined by

Then B ⋆ ref is a contraction on M b ( X × A ) . Thus, it has a unique fixed point q ⋆ ref . [Source]

Proof. First, observe that

Second,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The lemma follows by the Banach fixed point theorem.

Next we prove value function convergence.

Theorem 3.2. We have that q q monotonically as τ 0 . [Source]

⋆ τ → ⋆ ref →

Proof. Since q ⋆ τ is bounded (as the fixed point of a contractive operator on M b ( X × A ), there exists q 0 : X × A → R such that q ⋆ τ → q 0 monotonically and pointwise as τ → 0 , as a direct consequence of Lemma B.1. Therefore, by the monotone convergence theorem,

<!-- formula-not-decoded -->

The second step holds since for any f ∈ L ∞ , ∥ f ∥ p converges up to ∥ f ∥ ∞ as p → ∞ . So, since the sequence ( V τ q ⋆ σ ( x )) τ,σ ≥ 0 is monotone and bounded, its limit exists, and coincides with that computed above:

<!-- formula-not-decoded -->

Since q ⋆ τ is the unique fixed point of B ⋆ τ , by the monotone convergence theorem, we have

<!-- formula-not-decoded -->

so that q 0 is a fixed point of B ⋆ ref . Since the B ⋆ ref has a fixed point q ⋆ ref , it follows that q 0 = q ⋆ ref .

Now we prove our core estimate.

Theorem 3.6. Let q, q ′ ∈ M ( X × A ) . For any τ &gt; 0 and any x ∈ X ,

<!-- formula-not-decoded -->

In particular,

<!-- formula-not-decoded -->

Proof. Let π := G τ q, π ′ := G τ q ′ . By Lemma B.6, we have

<!-- formula-not-decoded -->

Moreover, by Lemma B.9,

<!-- formula-not-decoded -->

This concludes the proof of the first claim. Next, we recall that

<!-- formula-not-decoded -->

which is convergent for any y ∈ C . Therefore, for y ∈ (0 , 1) , we have

<!-- formula-not-decoded -->

So, when ∥ q ( x, · ) -q ′ ( x, · ) ∥ L ∞ ( π ref ) &lt; τ/ 2 , it follows that

<!-- formula-not-decoded -->

Finally, we prove policy and return distribution convergence.

Theorem 3.9. Under Assumption 3.4, if σ = σ ( τ ) is such that lim τ → 0 σ/τ = 0 , then π τ,σ x → π ref ,⋆ x as τ → 0 , for all x ∈ X , in TV if A is discrete and weakly if A is continuous. [Source]

Proof. Recall π τ,σ := G τ q ⋆ σ . By Theorem 3.6 and Lemma B.10,

<!-- formula-not-decoded -->

Consequently, π τ,σ x → π ref ,⋆ x if and only if ( G τ q ⋆ ref ) x → π ref ,⋆ x , and in whatever sense the later convergence occurs. In particular, if A is continuous, this is in the weak sense. While, if A is discrete, this in total variation.

Theorem 3.10. Suppose A is discrete and Assumption 3.4 holds. If σ = σ ( τ ) is such that σ/τ → 0 as τ → 0 , then, for any p, p ′ ∈ [1 , ∞ ) and ω ∈ P ( X × A ) , as τ → 0 , the return distribution functions ζ τ,σ of the temperature-decoupled policies π τ,σ satisfy d p ; p ′ ,ω ( ζ τ,σ , ζ π ref ,⋆ ) → 0 . [Source]

Proof. By the distributional Bellman equation [5], we have that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

We now derive a bound on I τ,σ . Starting with a triangle inequality,

<!-- formula-not-decoded -->

where ( a ) follows by the convexity of the Wasserstein metrics [40, 6], ( b ) applies [40, Theorem 6.15], and ( c ) leverages that the support ζ π x ′ ,a ′ lives in a ball of radius ∥ r ∥ sup / (1 -γ ) , for any π and ( x ′ , a ′ ) ∈ X × A .

So, thus far, we have shown that

<!-- formula-not-decoded -->

By Theorem 3.9, the total variation term tends to zero as τ tends to zero. Thus, defining ι ( x ′ , a ′ ) := lim sup τ → 0 d p ( ζ ⋆ x ′ ,a ′ , ζ τ,σ x ′ ,a ′ ) , this implies that

<!-- formula-not-decoded -->

In turn, sup ι ≤ γ sup ι , implying that ι ≡ 0 . Therefore, d p ( ζ ⋆ x,a , ζ τ,σ x,a ) → 0 pointwise over X × A , so by the dominated convergence theorem, d p ; p ′ ,ω ( ζ ⋆ , ζ τ,σ ) → 0 for any p ′ ∈ [1 , ∞ ) and ω ∈ P ( X × A ) .

## B.1 Supplemental Lemmas for Section 3

The following lemma translates immediately from the corresponding result in tabular MDPs; we prove it here for completeness.

Lemma B.1. If τ ≤ σ , then q ⋆ σ ≤ q ⋆ τ .

Proof. By the monotonicity of B ⋆ τ ,

<!-- formula-not-decoded -->

(cf. the proof of Lemma A.9).

Lemma B.2. Let σ = σ ( τ ) and suppose σ → 0 as τ → 0 . Then

Proof. Expanding the KL, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the final inequality holds by Lemma B.1. Since σ = σ ( τ ) ≤ τ , we have where the inequality again is due to Lemma B.1. Consequently, we have

∥ · ∥ ≥ ∥ · ∥

<!-- formula-not-decoded -->

where the penultimate step is due to the fact that V τ q ⋆ τ → v ⋆ ref monotonically, as shown in the proof of Theorem 3.2.

Lemma B.3. For every q ∈ M b ( X × A ) , with the notation above, M τ ( q ) → 0 as τ → 0 . If Assumption 3.4 is satisfied, then for any σ &gt; 0 ,

<!-- formula-not-decoded -->

Proof. First, we observe that lim τ → 0 V τ q ( x ) = lim τ → 0 log ∥ exp( q ( x, · )) ∥ L 1 /τ ( π ref ) = log ∥ exp( q ( x, · )) ∥ L ∞ ( π ref ) = ess sup π ref x q ( x, · ) . This is a monotone limit in τ , as it is known that for any f ∈ L ∞ , ∥ f ∥ p converges up to ∥ f ∥ L ∞ as p →∞ . Thus, we see that

M τ ( q ) = sup x ( ess sup π ref x q ( x, · ) -V τ q ( x ) ) → sup x (ess sup π ref x q ( x, · ) -ess sup π ref x q ( x, · )) = 0

as claimed. Now, under Assumption 3.4, we have

<!-- formula-not-decoded -->

Let B x = { a ∈ A : q ⋆ σ ( x, a ) = ess sup π ref x q ⋆ σ ( x, · ) } . Then, where the final inequality invokes Assumption 3.4.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma B.4. For all τ &gt; 0 and any q L ∞ ( X A )

where B ⋆ ref denotes the Bellman optimality operator (cf. Lemma 3.1).

Proof. A direct calculation gives

<!-- formula-not-decoded -->

Therefore, it immediate follows that

<!-- formula-not-decoded -->

The follow proof is essentially the performance difference bound in [37]. Lemma B.5. For all n ≥ 1 and any τ &gt; 0 ,

<!-- formula-not-decoded -->

If, additionally, Assumption 3.4 is satisfied, then

<!-- formula-not-decoded -->

Proof. We begin with the first statement. Recall that q ⋆ τ is the fixed point of B ⋆ τ , so that q ⋆ τ = B ⋆ τ q ⋆ τ . We will proceed by induction on n . For n = 1 , we observe that

<!-- formula-not-decoded -->

recalling the notation established above. This proves the base case. Now, assume the statement holds for all m ≤ n . We have

<!-- formula-not-decoded -->

∈ × , B ⋆ ref q ≥ B ⋆ τ q,

where the first inequality invokes the induction hypothesis, and the second inequality is due to the base case. Thus, we have shown that the claimed statement holds for any n ∈ N .

When Assumption 3.4 is satisfied, by Lemma B.3, we have M τ ( q ⋆ τ ) ≤ -τ log p ref , and the second statement follows.

Lemma B.6. Let q, q ′ ∈ L ∞ ( X × A ) . Then for any τ &gt; 0 and any x ∈ X ,

<!-- formula-not-decoded -->

Proof. Let π = G τ q and let π ′ = G τ q ′ . By Pinsker's inequality, we have

<!-- formula-not-decoded -->

Since q, q ′ ∈ L ∞ ( X × A ) , π x , π ′ x are mutually absolutely continuous. Expanding the KL divergence, we have

<!-- formula-not-decoded -->

where the last inequality holds since V τ is 1-Lipschitz, as shown in the proof of Lemma B.4. Substituting back into Pinsker's inequality, we have

<!-- formula-not-decoded -->

as claimed.

Lemma B.7. Let π, π ′ ∈ P ( Y ) for some measurable space Y be mutually absolutely continuous. Then

<!-- formula-not-decoded -->

Proof. Define h := d π d π ′ , and write M := ess sup π ′ h, m := ess inf π ′ h . Note that

<!-- formula-not-decoded -->

for any measurable E ⊂ Y . Consequently, we have

<!-- formula-not-decoded -->

Now, we derive the following upper bounds,

<!-- formula-not-decoded -->

Multiplying these inequalities by π ′ ( Y \ E ) and π ′ ( E ) , respectively, and adding the results, we have

<!-- formula-not-decoded -->

In fact, the same bound can be achieved for π ′ ( E ) -π ( E ) ; to see this, note that

<!-- formula-not-decoded -->

so by the same procedure as above, π ′ ( E ) -π ( E ) ≤ ( M -m ) π ′ ( E ) π ′ ( Y \ E ) . Therefore, we have shown that

<!-- formula-not-decoded -->

for any measurable E ⊂ Y . Since π ′ ( E ) π ′ ( Y \ E ) is maximized at π ′ ( E ) = π ′ ( Y \ E ) = 1 / 2 , we have

<!-- formula-not-decoded -->

as claimed.

Lemma B.8. Let u, w ∈ L ∞ ( Y ) for some measurable space Y , and let λ be a measure on Y . Define π u , π w ∈ P ( Y ) absolutely continuous with respect to λ such that d π · d λ ∝ e -· for · ∈ { u, w } . Then

<!-- formula-not-decoded -->

Proof. Firstly, since u, w ∈ L ∞ ( Y ) , it follows that π u , π w are mutually absolutely continuous. Now, define h := d π u d π w , with M := ess sup λ h and m := ess inf λ h . Note that

<!-- formula-not-decoded -->

where Z u , Z w ∈ R are normalizing constants. Defining f := u -w , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consequently, it holds that ess inf λ h ≥ e ess inf λ f -ess sup λ f and ess sup λ h ≤ e ess sup λ f -ess inf λ f . So, by the definition of f , we have m ≥ e -2 ∥ u -w ∥ L ∞ ( λ ) and M ≤ e 2 ∥ u -w ∥ L ∞ ( λ ) . Then, invoking Lemma B.7, we have

<!-- formula-not-decoded -->

Lemma B.9. Let q, q ′ ∈ L ∞ ( X × A ) . Then for any τ &gt; 0 and any x ∈ X ,

<!-- formula-not-decoded -->

Proof. Note that, for any q ∈ M b ( X × A ) , we have

<!-- formula-not-decoded -->

So, invoking Lemma B.8 with u = -τ -1 q ( x, · ) , v = -τ -1 q ′ ( x, · ) , and λ = π ref x , we have

<!-- formula-not-decoded -->

Additionally, we have

Lemma B.10. For every τ &gt; 0 , recalling the notation above,

<!-- formula-not-decoded -->

If Assumption 3.4 is satisfied, then and q ⋆ τ converges uniformly up to q ⋆ ref .

<!-- formula-not-decoded -->

Proof. By Lemma B.4, we have that for any q ∈ L ∞ ( X × A ) ,

<!-- formula-not-decoded -->

Then, by Lemma B.5, we have

<!-- formula-not-decoded -->

proving the first claim. When Assumption 3.4 is satisfied, we have M τ ( q ⋆ τ ) ≤ -τ log p ref by Lemma B.3, so that M τ ( q ⋆ τ ) converges down to 0 , and consequently q ⋆ τ converges up to q ⋆ .

## C Proofs from Section 4

<!-- formula-not-decoded -->

the soft distributional Bellman operator T π τ is a γ -contraction in d p for every τ ≥ 0 . Thus, it has a unique solution to the fixed point equation ¯ ζ = T π τ ¯ ζ , which we denote by ¯ ζ π,τ . [Source]

Proof. To begin, let us show that T π τ maps elements of K p ( X × A , P ( R )) to K p ( X × A , P ( R )) . For any ζ ∈ K p ( X × A , P ( R )) , observe that

<!-- formula-not-decoded -->

by assumption, as desired.

Next, by the convexity of the Wasserstein metric [6, 40], we have

<!-- formula-not-decoded -->

where the second inequality holds since the common transformation b r ( x,a ) -γτ KL( π x ′ ∥ π ref x ′ ) ,γ is affine. As a consequence, we have that

<!-- formula-not-decoded -->

which validates that T π τ is a γ -contraction in d p . Consequently, since ( K p ( X × A , P ( R )) , d p ) is complete and separable [6], it follows that T π τ has a unique fixed point. That ¯ ζ π,τ coincides with this fixed point follows precisely by [6, Proposition 4.9].

Lemma 4.4. For any τ &gt; 0 , QT = B Q

<!-- formula-not-decoded -->

Proof. For any ¯ ζ ∈ K p ( X × A , P ( R )) , we have

<!-- formula-not-decoded -->

Defining q := Q ¯ ζ , this is equivalent to

<!-- formula-not-decoded -->

Moreover, note that

<!-- formula-not-decoded -->

Substituting, we have shown that

<!-- formula-not-decoded -->

Theorem 4.5. For any ¯ ζ ∈ K p ( X × A , P ( R )) and temperature τ &gt; 0 define the iterates ( ¯ ζ n ) n ∈ N given by ¯ ζ n +1 = T ⋆ τ ¯ ζ n for ¯ ζ 0 = T ⋆ τ ¯ ζ . Then, for ¯ ζ τ,⋆ := ¯ ζ τ,π τ,⋆ ,

<!-- formula-not-decoded -->

where C, C p,τ,γ &lt; ∞ are constants depending on ∥ r ∥ sup , ( p, τ, γ, ∥ r ∥ sup ) respectively. [Source]

Proof. We begin by defining some helper notation. For any ¯ ζ ∈ K p ( X × A , P ( R )) , we define ξ ¯ ζ ∈ K p ( X , P ( R × A )) where

<!-- formula-not-decoded -->

In turn,

<!-- formula-not-decoded -->

Next, we define the following helpers,

<!-- formula-not-decoded -->

By [40, Theorem 4.8], we have that for any ( x, a ) ∈ X × A ,

<!-- formula-not-decoded -->

Invoking the triangle inequality together with the expansion of the ξ terms by definition, we have that for any x ∈ X ,

<!-- formula-not-decoded -->

Since the measures being compared in I n are both translated by the same pushforward map, another application of [40, Theorem 4.8] yields the following inequality:

<!-- formula-not-decoded -->

Next, we bound II n . Let C ( ρ 1 , ρ 2 ) be the set of couplings between measures ρ 1 , ρ 2 . Then

<!-- formula-not-decoded -->

for some constant C depending on τ, p, γ , ∥ r ∥ sup where ( a ) applies Minkowski's inequality, noting that the KL terms are independent of κ , and ( b ) invokes Lemma C.5 and Lemma C.6. Indeed, for n large enough, Lemmas C.5 and C.6 assert that C ≲ τ -1 for fixed p , and more generally that C ≲ τ -1 / 2 for any n (and fixed p ). Substituting back into (C.3), we see that

<!-- formula-not-decoded -->

Let a n := d p ( ¯ ζ n , ¯ ζ τ,⋆ ) . We have shown that a n +1 ≤ γa n + C ′ γ 1+ n/p , where C ′ = C ∥ Q ¯ ζ 0 -q ⋆ τ ∥ 1 /p sup is a constant depending on p and τ . We will apply techniques of generatingfunctionology [41] to bound this sequence. We define A : R → R as the formal power series given by

<!-- formula-not-decoded -->

and we will pick off the coefficients a n from the power series representation of A . Our recurrence above, upon multiplying through by y n and summing over n yields

<!-- formula-not-decoded -->

where a 0 = d p ( ¯ ζ 0 , ¯ ζ τ,⋆ ) . Now, the formal power series expansion gives

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where C ′′ = C ′ when p = 1 , and C ′′ = C ′ / ( γ 1 /p -γ ) otherwise-in any case, C ′′ is a constant depending only on p, τ, γ , and the proof is complete.

Theorem 4.6. Suppose Assumption 3.4 holds. Let p, p ′ ∈ [1 , ∞ ) and ω ∈ P ( X × A ) . For any ϵ, δ &gt; 0 , there exists a τ &gt; 0 for which d p ; p ′ ,ω ( ¯ ζ τ,π τ,⋆ , ζ π τ,⋆ ) ≤ δ/ 2 and q π τ,⋆ is ϵ/ 2 -referenceoptimal. In turn, an n ϵ,δ = n ϵ,δ ( τ ) ∈ N exists for which

<!-- formula-not-decoded -->

Proof. By Lemma B.10 and under Assumption 3.4,

<!-- formula-not-decoded -->

choosing τ ≤ τ ϵ := ϵ (1 -γ ) 2 γ log p -1 ref . Hence, by Lemmas 4.4 and A.7

<!-- formula-not-decoded -->

which holds when n ϵ ≥ (log γ ) -1 log ϵ 2 ∥ Q ¯ ζ 0 -q ⋆ ref ∥ sup .

Next, we will show that the soft return distribution estimates will approximate ζ π τ,⋆ . For notational simplicity, define X t := X π τ,⋆ t and A t := A π τ,⋆ t for t ∈ N . Recall that

<!-- formula-not-decoded -->

Moreover, we define ˜ ζ τ,π ⋆,τ x,a := ( -τ KL( π τ,⋆ x ∥ π ref x ) + id ) # ¯ ζ τ,π τ,⋆ x,a , so that

<!-- formula-not-decoded -->

Now, by the triangle inequality, we have

<!-- formula-not-decoded -->

Combining, we have

We proceed by analying I τ . By coupling states and actions, we immediately have

<!-- formula-not-decoded -->

and so, since π τ,⋆ = G τ q ⋆ τ , by virtue of Lemma B.2 we have

<!-- formula-not-decoded -->

Next, we bound II τ . Denote by r π,τ : X × A → R the reward function defined by

The work of [44] shows that, for any policy π , there is a unique ℸ π ∈ K ( X × A , P ( P ( X × A ))) for which ( µ ↦→ (1 -γ ) -1 ( µr )( x, a )) # ℸ π x,a = ζ π,r x,a , where ζ π,r denotes the return distribution function associated to the policy π for the reward function r . Noting that ˜ ζ τ,π τ,⋆ = ζ π τ,⋆ ,r π,τ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the penultimate step is simply a coupling argument (coupling the samples of ℸ π τ,⋆ ). Once again, since lim sup τ → 0 τ KL( π τ,⋆ x ∥ π ref x ) = 0 , and KL( π τ,⋆ x ∥ π ref x ) is bounded by Lemma B.2, the dominated convergence theorem asserts that lim τ → 0 II τ ( x, a ) = 0 pointwise.

Altogether, we have shown that lim τ → 0 (I τ ( x, a ) + II τ ( x, a )) = 0 pointwise, and is bounded as a consequence of Lemma B.2. Thus, by another application of the dominated convergence theorem together with (C.4), we have that

<!-- formula-not-decoded -->

It follows that there exists some τ δ &gt; 0 for which d p ; p ′ ,ω ( ¯ ζ τ,π τ,⋆ , ζ π τ,⋆ ) ≤ δ/ 2 whenever τ ≤ τ δ . For any such τ , by Theorem 4.5, there exists n δ ∈ N for which

<!-- formula-not-decoded -->

For this choice of τ and n δ , by the triangle inequality,

<!-- formula-not-decoded -->

Altogether, taking τ = min { τ ϵ , τ δ } and n = max { n ϵ , n δ } , we have that as well as

To complete the proof, we note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so that G τ Q ¯ ζ n is ϵ -reference-optimal.

Theorem 4.7. Suppose Assumption 3.4 holds and A is discrete. Let p, p ′ ∈ [1 , ∞ ) and ω ∈ P ( X × A ) . For any ϵ, δ &gt; 0 and ¯ ζ 0 ∈ K p ( X × A , P ( R )) , there exists τ &gt; 0 , a decoupled σ τ &gt; 0 and n opt , n eval ∈ N such that where ¯ ζ n +1 = T ⋆ σ ¯ ζ n , ˆ π τ,σ = G τ ¯ ζ n opt , and ˆ ζ n +1 = T ˆ π τ,σ τ ˆ ζ n , for ˆ ζ 0 = ¯ ζ n opt . [Source]

<!-- formula-not-decoded -->

Proof. Appealing to Theorem 3.10, for any δ &gt; 0 , any temperature decoupling gambit yields a τ δ &gt; 0 and an associated decoupled temperature σ δ = σ ( τ δ ) &gt; 0 such that

<!-- formula-not-decoded -->

whenever τ ≤ τ δ . Moreover, as shown in the proof of Theorem 4.6, for small enough τ ′ δ ,

<!-- formula-not-decoded -->

whenever τ ≤ τ ′ δ -here, we recall that ¯ ζ τ,σ is the entropy-regularized return distribution function for the decoupled policy π τ,σ .

Now, define ˆ ζ τ,σ = ( T ˆ π τ,σ τ ) n eval ˆ ζ σ,⋆ , and ˆ ζ σ,⋆ = ( T ⋆ τ σ ) n opt ¯ ζ 0 . By the triangle inequality, we have

<!-- formula-not-decoded -->

Here, ( a ) leverages the fact that ¯ ζ τ,σ is the fixed point of T π τ,σ τ by definition, ( b ) invokes the contractivity of T π τ,σ τ shown in Theorem 4.2 appealing to the fact that π τ,σ is a BG policy for reference π ref , and ( c ) follows by Lemma C.1. As a consequence, again since | γ | &lt; 1 , for sufficiently large n opt , n eval ∈ N , we have

<!-- formula-not-decoded -->

Altogether, by the triangle inequality once again, for the choices of n opt , n eval , τ, σ above,

<!-- formula-not-decoded -->

This completes the proof of the first claim. It remains to show now that G τ Q ˆ ζ n eval is ϵ -referenceoptimal. Towards this end, we note that by Theorem 4.6 and 4 . 7 that there exists τ ϵ &gt; 0 , n ϵ ∈ N such that

<!-- formula-not-decoded -->

whenever max { n eval , n opt } ≥ n ϵ and τ ≤ τ ϵ . To proceed, we note that for any ( x, a ) ∈ X × A ,

<!-- formula-not-decoded -->

where the last inequality holds for small enough τ by Theorem 3.2. Continuing, we have

<!-- formula-not-decoded -->

where the final inequality holds since V τ , as a log-sum-exp, is 1-Lipschitz. Now, again by Theorem 3.2, for small enough τ (inducing small enough σ ), we have

<!-- formula-not-decoded -->

Altogether, we have that

<!-- formula-not-decoded -->

Next, since d 1 ( ρ 1 , ρ 2 ) ≥ E ( Z 1 ,Z 2 ) ∼ ρ 1 ⊗ ρ 2 [ | Z 1 -Z 2 | ] , we have that by (C.5). Now, by yet another triangle inequality,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consequently, we have

<!-- formula-not-decoded -->

Thus, we have shown that G τ Q ˆ ζ n eval is ϵ -reference-optimal, completing the proof.

<!-- formula-not-decoded -->

Proof. By Lemma C.2, we have

<!-- formula-not-decoded -->

It remains to bound ∥ Q ˆ ζ σ,⋆ -q ⋆ σ ∥ sup . However, by Lemma 4.4 and the contractivity of B ⋆ σ , we have that

Since | γ | &lt; 1 , it follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma C.2. Let ζ ∈ K p ( X × A , P ( R )) , τ, σ &gt; 0 , and n eval ∈ N be given. Then d p (( T ˆ π τ ) n eval ζ, ( T π τ ) n eval ζ ) ≲ ( τ -1 Q ζ q ⋆ σ sup ) 1 / 2 p + √ τ -1 Q ζ q ⋆ σ sup + Q ζ q ⋆ σ sup .

<!-- formula-not-decoded -->

Proof. For simplicity, we define ˆ π = G Q ζ and π = G q . We want to bound

By Lemma C.3, we have

<!-- formula-not-decoded -->

τ τ ⋆ σ d p (( T ˆ π τ ) n eval ζ, ( T π τ ) n eval ζ ) .

where

<!-- formula-not-decoded -->

Now (( T G τ q ⋆ σ τ ) n ˆ ζ σ,⋆ ) n ∈ N is the sequence of return distributions generated by iterative applications of a contractive operator on K p ( X × A , P ( R )) . Thus,

<!-- formula-not-decoded -->

where c 3 is a constant depending only on p, γ, σ, τ, ∥ r ∥ sup . It remains to bound c 1 and c 2 . By Theorem 3.6, we have

<!-- formula-not-decoded -->

Thus, since ∥ q ⋆ σ ∥ sup is uniformly bounded for any σ &gt; 0 , we have shown that

<!-- formula-not-decoded -->

Lemma C.3. Let τ &gt; 0 , p ∈ [1 , ∞ ) , q, q ′ ∈ M b ( X × A ) , and ζ, ζ ′ ∈ K p ( X × A , P ( R )) . Then,

<!-- formula-not-decoded -->

where c q,q ′ := min {∥ q ∥ sup , ∥ q ′ ∥ sup } .

Proof. Observe

<!-- formula-not-decoded -->

So by Lemma C.4, we conclude.

Lemma C.4. Let ζ ∈ K ( X × A , P ( R )) and q, q ′ ∈ M b ( X × A ) . For any τ &gt; 0 , defining π · = G τ · for · ∈ { q, q ′ } , denoting c q,q ′ = min {∥ q ′ ∥ sup , ∥ q ∥ sup } , we have

<!-- formula-not-decoded -->

Proof. For notational simplicity, we define

<!-- formula-not-decoded -->

Then, by the definition of T π τ , we have

<!-- formula-not-decoded -->

Following, by [40, Theorem 4.8], we have

<!-- formula-not-decoded -->

We will now estimate the integrand above. By the definition of ξ ζ,q , for any x ∈ X , denoting π q := G τ q , we have

<!-- formula-not-decoded -->

The inequality is due to a technique employed in the proof of Theorem 4.5. Next, by [40, Theorem 6.15], we bound I via

<!-- formula-not-decoded -->

Now, for II , we have

<!-- formula-not-decoded -->

as shown in the proof of Lemma C.5. Therefore, we have shown that

<!-- formula-not-decoded -->

## C.1 Supplemental Lemmas for Section 4

Lemma C.5. Let ¯ ζ ∈ K 1 ( X × A , P ( R )) , and for any n ∈ N , define ¯ ζ n +1 = T ⋆ τ ¯ ζ n , with ¯ ζ 0 = T ⋆ τ ζ . Also, define π n := G τ Q ¯ ζ n . Then for any x ∈ X , denoting C x := ∥ Q ¯ ζ ( x, · ) -q ⋆ τ ( x, · ) ∥ L ∞ ( π ref x ) ,

<!-- formula-not-decoded -->

where C 1 &lt; ∞ is a constant. If τ ≥ 2 γ n C x , then for a constant C 2 &lt; ∞ , we have

<!-- formula-not-decoded -->

Proof. First, observe that

<!-- formula-not-decoded -->

By Lemma 4.4 and the γ -contractivity of B ⋆ τ , we note that

<!-- formula-not-decoded -->

Then, by Theorem 3.6, we have

<!-- formula-not-decoded -->

Note that ∥ q ⋆ τ ∥ sup ≤ ∥ r ∥ sup / (1 -γ ) . Indeed, the upper bound is free; the lower bound comes from comparing q ⋆ τ with q π τ for π = π ref . Altogether, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If τ ≥ 2 γ n C x , then we have the stronger bound

Lemma C.6. Let ¯ ζ ∈ K p ( X × A , P ( R )) . For any n ∈ N , define ¯ ζ n +1 = T ⋆ τ ¯ ζ n , with ¯ ζ 0 = T ⋆ τ ¯ ζ . Denoting by C ( ρ 1 , ρ 2 ) the space of all couplings between the measures ρ 1 , ρ 2 , for all x ∈ X we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where π n := G τ Q ¯ ζ n and C p &lt; ∞ is a constant depending only on p and ∥ r ∥ sup . Moreover, when n &gt; log γ -1 (log 2 ∥ Q ¯ ζ ( x, · ) -q ⋆ τ ( x, · ) ∥ L ∞ ( π ref x ) -log τ ) , we have

<!-- formula-not-decoded -->

Proof. For notational convenience, define ϖ · x := ¯ ζ τ,⋆ x, ⊗ π · x , for · ∈ { n, ( τ, ⋆ ) } . Then,

<!-- formula-not-decoded -->

where ( a ) applies [40, Theorem 6.15], and ( b ) uses that the support of ¯ ζ τ,⋆ is contained in a ball of radius 3 ∥ r ∥ sup / (1 -γ ) . By Lemma B.6, it follows that

<!-- formula-not-decoded -->

where the last inequality holds by Lemma 4.4 and the γ -contractivity of B ⋆ τ with C p := 3 2 p -1 ∥ r ∥ p sup .

Moreover, if n &gt; log γ -1 (log 2 ∥ Q ¯ ζ ( x, · ) -q ⋆ τ ( x, · ) ∥ L ∞ ( π ref x ) -log τ ) , then

<!-- formula-not-decoded -->

for each x ∈ X , so by Theorem 3.6,

<!-- formula-not-decoded -->

## D Comparison between vanishing temperature limits of ERL with and without temperature decoupling

In this section, we compare and contrast the properties of vanishing temperature limits of standard ERL (assuming they exist) with those achieved by the temperature decoupling gambit. As we showed in Theorem 2.3 and Theorem 3.9, both schemes achieve reference-optimality in the limit; yet, their

limits may be notably distinct according to criteria beyond the RL objective, as we saw in Sections 3.1 and 4.3.

In the remainder of this section, we will define ζ ref ,⋆ := ζ π ref ,⋆ as the return distribution function corresponding to the limiting temperature-decoupled policy, and ζ ERL ,⋆ := ζ π ERL ,⋆ as the return distribution function corresponding to the limiting ERL policy π ERL ,⋆ , assuming such a limit exists.

A very nice property of π ref ,⋆ is that it is easy to characterize as the optimality-filtered reference , as per Definition 3.8. In particular, π ref ,⋆ is characterized entirely in terms of the optimal action-value function q ⋆ and the reference policy π ref . On the other hand, as we see explicitly in Section 3.1, π ERL ,⋆ does not have such a simple characterization: it is influenced also by the transition dynamics of the MDP (as well as the q ⋆ and π ref ).

A notable consequence of this fact is that one can reason about π ref ,⋆ generically across MDPs, which is not the case for π ERL ,⋆ . For instance, in any MDP, if π ref is the uniform policy, π ref ,⋆ is the uniform policy on optimal actions. Thus, one can say definitively that all actions leading to optimal behavior are played equally under π ref ,⋆ . But this is not true of π ERL ,⋆ ; in general, it is difficult to characterize exactly how π ERL ,⋆ behaves: among a set of MDPs with equal q ⋆ , the corresponding π ERL ,⋆ can vary significantly.

Similarly, this property of π ref ,⋆ enables one to easily influence the optimal policy that is achieved via temperature decoupling by intervening on π ref . Again, this is possible due to the simple characterization of π ref ,⋆ as the optimality-filtered reference. Suppose, for example, there exists a particular action a scary that you want to avoid whenever possible (e.g., certain controversial phrases in language generation). It may be undesirable to filter this action out completely (say, by choosing π ref to never play a scary ), because perhaps from some states this action is necessary to achieve optimal return. Instead, with temperature-decoupling, you can choose π ref to play this action with very low probability (e.g., π ref x ( a scary ) = p ref for each x ). By Theorem 3.9, a scary will only ever be played when it achieves optimal returns, and moreover, as long as other actions exist that achieve optimal returns, a scary will be played with much lower probability.

The same logic does not hold, in general, for π ERL ,⋆ . As we saw in Section 3.1, π ERL ,⋆ may continue to play a scary with high probability even if π ref plays it with low probability. Suppose, for instance, that after playing a scary in state x , it is optimal to play π ref subsequently for the rest of the episode. Then π ERL ,⋆ may strongly prefer to play a scary from state x , even if other actions can achieve the same expected return. In fact, depending on the transition kernel, the scale of the rewards, and the discount factor, π ERL ,⋆ may play a scary from state x with arbitrarily high probability.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We believe our abstract and introduction outline and faithfully summarize the content of our work.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We have included discussions of the limitations and scope of our results throughout our work, rather than within a specific 'Limitations' section. See, for example, Section 6.

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

Justification: For each theoretical result, we provide rigorous proofs in the appendix, and the assumptions are stated very explicitly.

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

Justification: We present all details about the experiments we ran in order for them to be reproduced.

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

Justification: The experimental results are merely visualizations, all of which can be verified mathematically.

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

Justification: Our experiments are on toy problems where such details are vacuous; though, all experimental details are fully specified in the paper.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: Statistical analysis does not apply for our empirical results, which have deterministic outcomes.

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

Justification: Experiments can be trivially run on any modern computer.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Yes, we have read the NeurIPS code of ethics, and believe our work conforms to it in every respect.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: We believe our work is foundational, without a direct path to negative societal impact.

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

Justification: We believe our work poses no risk of misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: We did not use existing assets.

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

Justification: Our work does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our work did not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our research did not involve crowdsourxcing nor research with human subjects.

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
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.