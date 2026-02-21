## Policy Gradient Methods Converge Globally in Imperfect-Information Extensive-Form Games

Fivos Kalogiannis UCSD CSE La Jolla, CA 92093 fkalogiannis@ucsd.edu

Gabriele Farina

MIT EECS

Cambridge, MA 02139 gfarina@mit.edu

## Abstract

Multi-agent reinforcement learning (MARL) has long been seen as inseparable from Markov games (Littman, 1994). Yet, the most remarkable achievements of practical MARL have arguably been in extensive-form games (EFGs)-spanning games like Poker, Stratego, and Hanabi. At the same time, little is known about provable equilibrium convergence for MARL algorithms applied to EFGs as they stumble upon the inherent nonconvexity of the optimization landscape and the failure of the value-iteration subroutine in EFGs. To this goal, we utilize contemporary advances in nonconvex optimization theory to prove that regularized alternating policy gradient with (i) direct policy parametrization , (ii) softmax policy parametrization , and (iii) softmax policy parametrization with natural policy gradient updates converge to an approximate Nash equilibrium (NE) in the last-iterate in imperfectinformation perfect-recall zero-sum EFGs. Namely, we observe that since the individual utilities are concave with respect to the sequence-form strategy, they satisfy gradient dominance with respect to the behavioral strategy-or, policy , in reinforcement learning terms. We exploit this structure to further prove that the regularized utility satisfies the much stronger proximal Polyak-Łojasiewicz condition. In turn, we show that the different flavors of alternating policy gradient methods converge to an ϵ -approximate NE with a number of iterations and trajectory samples that are polynomial in 1 /ϵ and the natural parameters of the game. Our work is a preliminary-yet principled-attempt in bridging the conceptual gap between the theory of Markov and imperfect-information EFGs while it aspires to stimulate a deeper dialogue between them.

## 1 Introduction

Reinforcement learning (RL) dominates contemporary applied and theoretical research. The flagship of RL, policy optimization methods , appears to lend reasoning capabilities to language models (Shao et al., 2024), defeats human Go world champions (Silver et al., 2016), and navigates real-world roads safely (Lu et al., 2023; Cusumano-Towner et al., 2025). As is evident from even more examples (Vinyals et al., 2019; Schrittwieser et al., 2020), machine gameplay has transformed by incorporating RL techniques into its algorithmic arsenal. Although theoretical literature (Littman, 1994) posits that the canonical model of MARL are Markov games (MGs), MARL has handled imperfect-information extensive-form games (EFGs) with commendable success (Brown and Sandholm, 2019b; Bard et al., 2020; Perolat et al., 2022).

At first, the theory and practice of imperfect-information EFGs can seem saturated. Exhaustive research in the properties of EFGs has exposed its convex structure using sequence-form strategies (Romanovskii, 1962; Koller et al., 1996; V on Stengel, 1996) and yielded the different counterfactual-regret minimization algorithms (CFR) (Zinkevich et al., 2007; Tammelin, 2014; Brown and Sandholm,

2019a). These algorithms can solve games using tabular policies with unmatched computational efficiency. Notwithstanding, these techniques seem to hit a wall when faced with large-scale games whose size makes the use of tabular policies infeasible and calls for a neural network parametrized policy (or, more generally, policy function approximation). The picture is even more grave when CFR needs to be combined with model-free counterfactual value estimation. Its call for importance sampling yields a feedback of prohibitively high variance. Further, CFR's average-iterate convergence makes the task of extracting a single policy network highly nontrivial. Since practitioners have extensively studied policy optimization for imperfect-information games (Lanctot et al., 2017; Srinivasan et al., 2018; Lockhart et al., 2019; Hennes et al., 2020; Rudolph et al., 2025) without offering guarantees of polynomial time convergence, we are naturally lead to the question:

Do policy gradient methods provably converge to an equilibrium in imperfect-information EFGs using a polynomial number of iterations and samples?

<!-- image -->

To answer, we need to face the two obstacles that imperfect-information games raise against optimization, the failure of value iterationwhich we sidestep by solely using policy gradient updates -and a highly nonconvex policy optimization landscapewhich we prove to be benign .

Failure of value iteration In MARL for MGs, the overwhelming majority (Shapley, 1953; Wei et al., 2021; Zhao et al., 2022; Alacaoglu et al., 2022b; Zhang et al., 2019) of existing algorithmic solutions for equilibrium learning or computation makes use of a value iteration subroutine or a value critic -which is in essence a backwards induction of the estimated value of the game. Instead, solving imperfect-information games requires leveraging the opponent's uncertainty about the underlying state. In other words, one needs to trade off exploiting private information and the benefit of keeping it secret. This precludes solving subtree-by-subtree conditioned on private information and leads to the emergence of behaviors such as bluffing at optimality.

Gradient Domination in Nonconvex Problems. Contemporary machine learning is arguably propelled by large-scale optimization of systems of astounding size to perform increasingly elaborate tasks. The corresponding objective functions are by no means convex in terms of parameters, which precludes theoretical guarantees of even reaching a local optimum in a reasonable number of iterations (Murty and Kabadi, 1985). Yet, practice indicates a different reality and theory is gradually catching up. It has painstakingly been demonstrated that the nonconvexity of various ML optimization problems is seriously benign-significantly often, stationarity implies global optimality . Cases in point, gradient domination is exhibited for the loss functions of overparametrized neural networks (Liu et al., 2022a; Scaman et al., 2022), the linear quadratic regulator (Fazel et al., 2018), value functions of Markov decision processes (MDPs) (Agarwal et al., 2021; Bhandari and Russo, 2024), matrix completion (Ge et al., 2016), dictionary learning (Sun et al., 2015), and more. For a thorough discussion of gradient domination and other regularity conditions we refer the reader to (Karimi et al., 2016; Li and Pong, 2018; Drusvyatskiy and Paquette, 2019; Drusvyatskiy and Lewis, 2018; Liao et al., 2024; Rebjock and Boumal, 2024; Oikonomidis et al., 2025) and references therein. With the latter in mind, one could make the case that when game theory researchers seek equilibrium computation in general nonconvex games (Cai et al., 2024a; Angelopoulos et al., 2025) they set the bar too high. Still, the study of benign nonconvexity seems of great importance and rather underexplored (Yang et al., 2020; Mulvaney-Kemp et al., 2023; Vlatakis-Gkaragkounis et al., 2021; Sakos et al., 2023).

## 1.1 Contributions

We answer ( ♥ ) in the affirmative by developing three policy gradient methods (Theorems 3.1 to 3.3). All three algorithmic approaches lead to last-iterate convergence to a regularized NE of the EFG. We contribute,

- a novel decentralized exploration scheme that yields sufficient visitation of all information sets;
- a proof that the nonconvex utilities of the (un-)regularized game satisfy gradient domination;
- guarantee of last-iterate convergence of three different alternating policy gradient (PG) methods: (1) PG with direct parametrization and ℓ 2 -norm regularization (2) PG softmax parametrization and entropy regularization (3) natural policy gradient (NPG) with softmax parametrization and entropy regularization .

On a sidenote, we offer a sharper dependence of the PŁ modulus to the hidden convexity modulus than the one suggested by (Karimi et al., 2016, Appendix G) for constrained optimization.

## 1.2 Overview of Techniques

The theoretical guarantees for our three algorithmic solutions are pinpointed by a simple unifying conceptual principle. That is, the nonconvex optimization problem of computing an equilibrium by directly optimizing the behavioral strategies (or, policies) is a constrained two-sided PŁ optimization problem where alternating gradient descent ascent is known to converge. Namely, we show that the optimization landscape viewed in terms of policies is nonconvex in a rather benign way; the utility is hidden concave . In particular, after appropriate regularization, each utility function satisfies a strong gradient domination property, i.e., the proximal Polyak-Łojasiewicz condition.

Hidden concavity. Going into more detail, utilities in EFGs are concave in terms of sequenceform strategies. We select an appropriate regularizer that enhances concavity to strong concavity. Moreover, enforcing a positive lower bound on the probability of reaching every information set yields a uniform Lipschitz constant for the bijection that maps sequence-form strategies to behavioral policies. Taken together, these two observations imply a strong gradient-domination condition for each player's policy.

PŁ condition. For the sake of offering an intuitive exposition, we forego the nuances of constrained optimization to explain how the PŁ condition is proven to hold. We say that an optimization problem min x f ( x ) exhibits hidden strong convexity when there exists an invertible mapping u = c ( x ) and a function H ( u ) that is µ -strongly convex in u and f ( x ) = H ( c ( x )) . Strong convexity implies that f ( x ) -f ⋆ ≡ H ( u ) -H ⋆ ≤ 1 2 µ ∥∇ u H ( u ) ∥ 2 . Now, a bounded Lipshcitz modulus L c -1 &gt; 0 of the inverse transform, c -1 ( u ) = x , leads to the PŁ inequality f ( x ) -f ⋆ ≤ L 2 c -1 2 µ ∥∇ f ( x ) ∥ 2 by merely applying the chain rule of differentiation. Similar arguments work for the proximal-PŁ condition.

Convergence. Then, alternating gradient descent ascent on min x ∈X max y ∈Y f ( x, y ) ,

<!-- formula-not-decoded -->

is proven to converge to a saddle-point point using a typical Lyapunov function argument. We tune the stepsizes η x , η y in such a way that one player learns faster than the other. Since the function is PŁ, this means that after each update the optimizer is significantly approximated. Intuitively, after enough iterations, the update scheme can be viewed as optimizing for Φ( x ) := max y ∈Y f ( x, y ) as x t +1 ≈ Proj X ( x t -η x ∇ x Φ( x t )) . Crucially, our convergence analysis sets aside the usual regret minimization arguments that are used to either prove average-iterate or best-iterate convergence ( e.g. , Anagnostides et al. (2022); Liu et al. (2024)).

## 1.3 Comparison to Related Work

We point out two particular results (Sokota et al., 2022; Liu et al., 2024) directly related to our endeavor of policy gradient/optimization methods for imperfect-information EFGs. Although the magnetic mirror descent method proposed in (Sokota et al., 2022) does not come with guarantees in EFGs, it exhibits impressive empirical performance. (Liu et al., 2024) lays the foundation of our approach as it introduces the bidilated regularizer although it does not offer a convergence guarantee that is polynomial in the parameters of the game and 1 /ϵ .

Our work follows arguments utilized in the context of policy gradient methods for Markov decision processes (MDPs) and MGs. Namely, we use techniques from (Kalogiannis et al., 2025) that analyzed alternating gradient descent in the constrained parameter case and arguments from (Mei et al., 2020; Cen et al., 2022a) as the entropic bidilated regularizer is almost identical to discounted entropy. Further, we use arguments from (Zhang et al., 2021) to show that the mapping from sequence-form strategies to policies is Lipschitz continuous.

Table 1: Comparison of policy gradient/optimization methods.

|                                          | Altern./Simult. Updates   | Provable Convergence               | Regularization           | Feedback   |
|------------------------------------------|---------------------------|------------------------------------|--------------------------|------------|
| (Liu et al., 2024) (Sokota et al., 2022) | simultaneous simultaneous | yes, best-iterate * no             | bidilated policy entropy | CFR ,Q,Q Q |
| Ours                                     | alternating               | yes, last-iterate, polynomial time | bidilated                | ∇ θ V, Q   |

CFR , Q, Q, ∇ θ V stand for counterfactual value, action-value, traject. action-value, and policy gradient. * Guarantees are pseudo-polynomial in the game-size.

## 2 Preliminaries

In this section we introduce the key ingredients required for our analysis. For IIEFGs, we highlight how the utility is expressed as a concave function of the sequence-form strategies. We also review the-Euclidean or entropic-bidilated regularizer whose strong convexity underpins our gradient-domination arguments. With regards to RL theory, we recall the definition of the value and and action-value functions and show that trajectory samples, or roll-outs , give unbiased Monte-Carlo estimates of both the utility and the bidilated regularizer via the ( REINFORCE ) estimator (Williams, 1992; Sutton et al., 1999). Finally, we review the optimization notions of hidden concavity and gradient dominance, used to prove convergence in of our algorithmic solutions.

## 2.1 Imperfect-Information Extensive-Form Games

We briefly go over the definition of an IIEFG and move on to the sequence-form strategies and the corresponding regularizers.

Definition 1 (IIEFG) . A two player zero-sum extensive-form game, Γ , is defined by the tuple ( T , H , S , A , B , r ) . A special chance player , c , models uncontrollable randomness while,

- T is a rooted game tree of height D ( T ) ,
- H := H 1 ∪H 2 ∪H c is the set of T 's nodes, referred to as histories . Each history, h , belongs to exactly one of the sets H 1 , H 2 , H c depending on the player responsible for taking action at h .
- S := S 1 ∪S 2 is a finite set of information sets ( infosets ). The infosets partition histories, H i , of the acting player i into sets of nodes that are indistinguishable. We will note S := max {|S 1 | , |S 2 |} .
- A := {A s } s ∈S 1 , B := {B s } s ∈S 2 are the action sets of player 1 and 2, respectively. Each infoset s ∈ S has a corresponding set of actions A s , and respectively B s . Further, we will denote A s := |A s | , A := max s A s and B s := |B s | , B max := max s B s .
- r : H → [0 , 1] is a payoff function mapping leaves of T to a payoff for player 1; player 2 gets the opposite payoff.

A perfect recall assumption is made, ensuring that players remember their past observations and actions. This implies that nodes in the same infoset have the same past observation sequence. We will use σ 1 ( s ) , σ 2 ( s ) to denote the last parent infoset-action pair ( s ′ , a ′ ) , s ∈ S 1 and ( s ′ , b ′ ) , s ∈ S 2 encountered when descending from the game tree's root to history h . σ 1 ( · ) , σ 2 ( · ) are either unique for non-root nodes or the null set for the root. We will overload notation σ 1 ( h ) to mean σ 1 ( s ) for the infoset s where h belongs (resp. for σ 2 ( h ) ).

Sequence-Form Strategies A player's behavioral strategy is a probability distribution over actions at each of their infosets. With Σ 1 we denote player 1 's subsequences of play starting at the root. In sequence-form , the strategy of player 1 , µ π 1 1 ∈ R | Σ 1 | , with | Σ 1 | := 1 + ∑ s A s is defined as:

<!-- formula-not-decoded -->

The sequence-form strategy and Σ 2 of player 2 is defined in a symmetric fashion. Introduced in (Romanovskii, 1962; Von Stengel, 1996; Koller et al., 1996), sequence-form strategies are generalizations of simplices and express the sequential structure of an IIEFG. The set of sequence-form strategies,

M 1 , M 2 are convex polytopes as they are is defined only by linear equalities and non-negativity constraints. The chance player's contribution to the probability of reaching history h is given by µ c ( h ) and it is assumed to be strictly positive for reachable nodes. For player 1 , the expected utility is given by the bilinear form:

<!-- formula-not-decoded -->

where R is the matrix representation of payoff function r . Forward, we will refer to behavioral strategies as policies which will be denoted as π 1 , π 2 . The solution concept we are after is an ϵ -approximate Nash equilibrium.

Definition 2 ( ϵ -NE) . A policy profile π ⋆ 1 , π ⋆ 2 is an ϵ -approximate Nash equilibrium of an IIEFG Γ , if, for any policies π 1 and π 2 it holds true that,

<!-- formula-not-decoded -->

The bidilated regularizer. Introduced in (Liu et al., 2024), the unweighted bidilated regularizer is defined using a strongly-convex regularizer ψ ( · ) multiplied by the total probability of reaching the corresponding infoset. Since it depends on both players' policies we write R ( π 1 , π 2 ) , R ( π 2 , π 2 ) , s.t.:

<!-- formula-not-decoded -->

## 2.2 RL Fundamentals

Moving on, we define the value, action-value, and advantage functions in the context of IIEFGs. Inspired by the occupancy measure of MGs, we define the history occupancy measure d π for a given policy profile π := ( π 1 , π 2 ) which simply is the reach probability of each history and comes in handy as a shorthand notation in the description of the algorithms and their analysis. Moreover, we recall the definitions of direct and softmax policy parametrization. Last but not least, we demonstrate how the ( REINFORCE ) gradient estimator computes policy gradients for IIEFGs for both the unregularized and regularized utility.

Value, action-value, and advantage functions. Without loss of generality, we assume that players get a payoff only on a terminal history h . This way we can define the value function of an infoset s , as the expected utility if the game were to start at a history h 0 belonging to s ,

<!-- formula-not-decoded -->

In a similar vein, we define the action-value function, or Q , as the expected utility if the game started at at a history h 0 belonging in s and after the player had taken action a 0 , (or, resp. b 0 ),

<!-- formula-not-decoded -->

Finally, the advantage function is defined for each player as the difference between an action-value and the infoset's value A π 1 ( s, a ) := -V π ( s ) -Q π 1 ( s, a ) and A π 2 ( s, b ) := V π ( s ) -Q π 1 ( s, b ) . Similar to the state occupancy measure of an MG, we can define the history occupancy measure d π : H → [0 , 1] which is defined as, d π ( h ) := E h ′ ∼ π [ ✶ { h ′ = h } ] . Overloading notation, for an infoset s ∈ S d π ( s ) := ∑ h ∈ s d π ( h ) .

Policies. Policies are precisely parametrized behavioral strategies. We will consider two parametrizations of policies, (i) direct parametrization , and (ii) softmax parametrization . For directly parametrized policies, we denote the parameters as x, y which are x ∈ × s ∈S 1 ∆( A s ) , y ∈ × s ∈S 2 ∆( B s ) . The parameters of softmax policies will be denoted χ, θ with χ ∈ R A , A = ∑ s A s and θ ∈ R B , B = ∑ s B s .

Gradient estimation with REINFORCE . The ability to estimate a gradient of the value function using trajectory samples, or roll-outs , has endowed the theory and practice of RL with the rich toolbox of gradient-based optimization. In fact, the ( REINFORCE ) gradient estimator (Williams, 1992; Sutton et al., 1999) is also an unbiased estimator of the policy gradient in the IIEFG setting, and thus provides a sound foundation for our analysis.

Definition 3 ( REINFORCE ) . Let ξ denote a trajectory of infoset and actions sampled by implementing policies π 1 , π 2 , ξ := ( s (1) , a ( k ) , . . . ) . We define REINFORCE , ( ̂ ∇ x , ̂ ∇ y ) , to be the stochastic gradient estimators:

<!-- formula-not-decoded -->

The addition of regularization, leads to the definition a regularized value function, V τ ,

<!-- formula-not-decoded -->

The regularized Q -value and advantage functions, Q π τ , A π τ , are defined accordingly (see Appendix B.2). Furthermore, ( REINFORCE ) can be minimally modified to estimate the policy gradient of the regularized value function without importance sampling (discussed in detail in Appendix F.1).

Assumption 1. For an ε &gt; 0 , both players' policies, for every infoset and action, satisfy

<!-- formula-not-decoded -->

Guaranteeing that ( ε -trunc.) holds is straightforward for directly parametrized policies. The players need to pick policies x, y , from the cartesian product of appropriately truncated simplices, to be denoted X ε , Y ε respectively. As for softmax parametrized policies, ( ε -trunc.) is achieved when both players' parameters are restricted to the polytopes X R , Θ R . To demonstrate, X R is defined in the following manner, X R := { χ ∈ R A , A = ∑ s A s : χ ⊤ s 1 = 0 , ∀ s ∈ S 1 , | χ s,i -χ s,j | ≤ 2 R, ∀ i, j ∈ [ A s ] } , and the definition of Θ R follows suit. We highlight that the images of X R , Ψ R under the softmax map are convex sets (Lemma D.5) and we will denote the resulting truncated policy sets as Π R 1 , Π R 2 .

## 2.3 Hidden Concavity and Gradient Domination

In this subsection, we define the two key backbone concepts of hidden concavity and gradient domination. Gradient domination of a weak or strong form has been extensively investigated in the theory of RL and MARL (Bhandari and Russo, 2024; Agarwal et al., 2021; Mei et al., 2020; Zhang et al., 2019; Daskalakis et al., 2020). Simply put, the nonconvex value function satisfies a gradient-domination property and any stationary point is globally optimal. Thus, any guarantee of convergence to a stationary point is elevated to a guarantee of convergence to global optimality.

Definition 4 (Hidden convexity) . A nonconvex function f : X → R defined over the set X is said to be hidden (strongly) convex if there exists (i) a bijective mapping c : X → U for some convex set U ; (ii) a function H : U → R that is strongly convex with modulus α H ≥ 0 ; such that f ( x ) = H ( c ( x )) , ∀ x ∈ X .

When the Lipschitz continuity modulus of the inverse transform, c -1 , is uniformly bounded it implies the gradient domination condition as shown in (Fatkhullin et al., 2023, Prop. 2) coupled with (Karimi et al., 2016, App. G).

Definition 5 (pPŁ condition (Karimi et al., 2016)) . Assume F : R d → R defined as F ( x ) := f ( x ) + g ( x ) . Let f : R d → R be an ℓ -smooth function and g : R d → R be convex. Define

<!-- formula-not-decoded -->

for a choice of Bregman divergence B ( ·∥· ) . We say that F satisfies the pPŁ condition with modulus α &gt; 0 if, for every x ,

<!-- formula-not-decoded -->

where F ⋆ = min x F ( x ) . When g is the indicator function of a set X we write D X ( x, ℓ ) .

## 3 Main Results

With the latter in hand, we are ready to state our main contributions, (i) the independent exploration strategy , (ii) the gradient domination condition for utilities of EFGs (iii) and the global convergence of three variants of policy gradient methods to an approximate Nash equilibrium.

## 3.1 Efficient Exploration Scheme

We propose a novel approach to exploration. Each player is expected to reach every subsequence with probability at least γ |H| . The rule is simple:

Assumption 2 (Efficient Exploration) . Both players follow the following exploration strategy:

- At the start of each game, the player flips a biased coin that shows 'heads' with probability γ .
- If the coin shows 'heads', the player selects a sequence uniformly at random and then executes it.
- After this sequence, or if the coin shows 'tails', the player resumes play according to their policy.

Remark 1. It is noteworthy that using this exploration strategy, one can exercise direct control over the modulus of gradient domination. Whereas, policy gradient literature (Agarwal et al., 2021; Daskalakis et al., 2020; Mei et al., 2020; Zeng et al., 2022) needs to make an assumption on the boundedness of the distribution mismatch coefficient.

## 3.2 Gradient Domination Property of the Utilities

In this subsection, we establish that the utility of an imperfect-information EFG under different policy parametrizations is pPŁ with regards to the policy. This observation is central in proving convergence of policy gradient methods to a Nash equilibrium. First, we state the weak gradient domination property for the unregularized utilities of the game.

Lemma 3.1 (Utility Weak Gradient Domination) . Let Γ be an imperfect-information EFG, following Assumption 2, then it holds true that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, by picking an appropriate regularization term to each player's utility we can enhance the weak gradient domination property to the much stronger pPŁ condition which ultimately guarantees last-iterate convergence to an equilibrium of the regularized game.

Lemma 3.2 (Utility pPŁ; restated from Lemmata E.1 to E.3) . Let an imperfect-information EFG, Γ , perturbed by a pair of weighted bidilated regularizers ( R 1 , R 2 ) with a coefficient τ &gt; 0 . Also, assume that each player follows Assumption 1 and Assumption 2. Then, each player's utility satisfies the pPŁ condition with a modulus α -1 = 1 τ × poly ( 1 ε , 1 γ , 1 min h µ c ( h ) , |H| , S, A, B, 2 D ( T ) ) .

A key observation in both conditions is that the modulus is a polynomial of the exploration parameter 1 /γ . This stresses the importance of efficient exploration and our corresponding contribution of the scheme in Assumption 2. Also,

## 3.3 Convergence of Alternating Regularized Policy Gradient

Having established the required background and notation, we are ready to present our main results. In Theorem 3.1 we show the convergence of simple alternating regularized policy gradient to an approximate NE in the last iterate. Moving to Theorem 3.2, we prove a similar result for softmaxparametrized policies. Finally, we analyze alternating regularized natural policy gradient through a mirror-descent lens, demonstrate its relationship to multiplicative weight updates of the policies, and prove its convergence to an approximate NE in the last iterate (Theorem 3.3).

Throughout, η x , η y denote the stepsizes and ˆ ∇ τ · denotes the ( REINFORCE ) gradient estimate of the utility w.r.t. to a player's parameters accounting only for their own regularization term.

## 3.3.1 Direct Policy Parametrization

The first result we present is the a simple policy gradient scheme with alternating updates and a Euclidean regularizer. The parameter updates of alternating regularized policy gradient takes the

following form,

<!-- formula-not-decoded -->

where Proj X ε , Proj Y ε denote the Euclidean projection of the parameters to the truncated simplices dictated by ( ε -trunc.). We state our first convergence theorem which settles question ( ♥ ) and defer its formal statement to the Appendix H.1.

Theorem 3.1 (Informal; restated from Thm. H.1) . With direct policy parametrization and the Euclidean bidilated regularizer, alternating policy-gradient algorithm attains a last-iterate ϵ -Nash equilibrium in

<!-- formula-not-decoded -->

using batches of poly ( 1 ϵ , 1 ε , 1 γ , |H| , |S 1 | , |S 2 | , A, B, 2 D ( T ) ) trajectory samples at each step.

Remark 2. We note that the exponential dependence on D ( T ) is still polynomial in the game size as the height has itself logarithmic dependence in size of the game.

## 3.3.2 Softmax Policy Parametrization

We move on to convergence under softmax parametrization and entropic regularization. This choice of parametrization is an important step towards getting provable guarantees for policy gradient methods in imperfect-information EFGs using function approximation ( e.g. neural networks). The projection to X R , Θ R guarantees that ( ε -trunc.) is satisfied,

<!-- formula-not-decoded -->

Theorem 3.2 (Informal; restated from Thm. H.2) . Alternating policy-gradient algorithm with softmax policy parametrization and the entropic bidilated regularizer, converges in expectation in the lastiterate to an ϵ -Nash equilibrium after a number of iterations T , that is

<!-- formula-not-decoded -->

using batches of poly ( 1 ϵ , 1 ε , 1 γ , |H| , |S 1 | , |S 2 | , A, B, 2 D ( T ) ) trajectory samples at each step.

## 3.3.3 Natural Policy Gradient

Finally, we consider the natural policy gradient algorithm (Kakade, 2001) which is an adaptation of natural gradient (Amari, 1998). This algorithm is of particular interest due to its intimate connection to the TRPO, PPO (Schulman et al., 2015, 2017) policy optimization algorithms. Natural policy gradient uses a Fisher information matrix induced by the policy as a preconditioner for policy gradient updates:

<!-- formula-not-decoded -->

We cast natural policy gradient steps as mirror descent steps with a Mahalanobis norm induced by the Fisher information matrix (for a more nuanced discussion on this connection see (Raskutti and Mukherjee, 2015)).The update scheme can be equivalently written as:

<!-- formula-not-decoded -->

More importantly, we note that in policy space, the update scheme of natural policy gradient takes a very simple form which, as expected, reads, for player 1 ( ⊙ is element-wise multiplication):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To see why the second approximate equality holds, we note that the Mahalanobis distance over the parameters induced by the Fisher information matrix of the softmax policy, is a second-order approximation of policy KL divergence. The derivation and an extensive discussion are deferred to Appendices H.3 and I.

Theorem 3.3 (Informal; restated from Thm. H.3) . For an appropriate tuning of η x , η y &gt; 0 , the last-iterate of alternating regularized natural policy gradient ( Alt-RegNPG ) converges in expectation to an ϵ -approximate Nash equilibrium in a number of iterations T that is:

<!-- formula-not-decoded -->

## 4 Empirical Validation

To corroborate our theoretical results, we tested Alt-RegNPG on four different imperfect information EFGs (Kuhn Poker, Leduc Poker, 2 × 2 Abrupt Dark Hex and Liar's Dice). Inspired by MMD (Sokota et al., 2022), we implement two variants of Alt-RegNPG where the (i) the regularization strength diminishes across time along the stepsizes and (ii) the regularizer is the discounted KL divergence from a moving reference policy. We observe that the exploitability ( i.e. max π ′ 1 V π ′ 1 ,π 2 -min π ′ 2 V π 1 ,π ′ 2 ) diminishes across time for our method, and it compares well with CFR and MMD.

Figure 1: Three variants of Alt-RegNPG compared against CFR and MMD.

<!-- image -->

## 5 Discussion

We conclude our main text with a further comparison between MGs and imperfect information IIEFGs to further promote the connection between the two areas. Finally, we state our conclusions and suggestions for future work.

## 5.1 Further comparison of Markov and Imperfect Information Extensive-Form Games

Imperfect-information IIEFGs and MGs both model multi-stage strategic interaction. They differ sharply in what each player can observe while they maintain marked similarities in the way strategies are represented ( behavioral strategies and policies ), the hidden concave representation of utilities (concavity w.r.t. sequence-form strategies and occupancy measures ), and regularization choices for optimization. The table and discussion below summarize this comparison along the axes of observability, strategy space, utility convex reformulation, regularization and optimization landscape. Clearly, an infoset (information set) in an imperfect-information EFG is to a behavioral strategy what a state is to a policy in an MG. However, imperfect information (or partial observability ) leads to a discrepancy between the expected return of an infoset in an EFG and the expected return state

Table 2: Imperfect-information extensive-form games (IIEFG) vs. Markov games (MG).

|                                                        | Game State                                      | Observable State                                                           | Control Variables                                                         | Utility Concave In     |
|--------------------------------------------------------|-------------------------------------------------|----------------------------------------------------------------------------|---------------------------------------------------------------------------|------------------------|
| History h ∈ T each a node of game tree graph           | Infoset s ∈ S T each a disjoint set of multiple | Behavioral Strategy π ( ·&#124; s ) distribution over actions at infoset s | Sequence-form Strategy µ π independent of opponents' strategies           | IIEFG histories h      |
| State s fully observable potentially recurring horizon | by all in the finite of the game                | Markovian Policy π ( ·&#124; s ) distribution over actions at              | State-action Occupancy measure λ π state s depends on opponents' policies | MG players or infinite |

in an MG as highlighted in (Nayyar et al., 2013; Sokota et al., 2023). Interestingly, the concave reparametrization of EFG utilities exhibits a structure more favorable than the corresponding one in MGs. In particular, the utility is concave in sequence-form strategies of IIEFGs and the latter depend solely on a player's own behavioral strategy. This comes in stark contrast to the state-action occupancy measure of MGs which are conditioned on opponents' strategies.

Finally, similarities of the regularization techniques in IIEFGs and MGs are cornerstone to our work. The EFG entropic bidilated regularizer (Liu et al., 2024), R , and the very commonly used MDP discounted entropy (Williams and Peng, 1991; Haarnoja et al., 2018; Mei et al., 2020; Cen et al., 2022a,b), E , are virtually identical. We note that, in IIEFGs a regularizer is mostly used in context of directly optimizing in the sequence-form space. They induce a distance generating function of mirror descent instantiations. Some more recent works have used it to make the game strongly-monotone and guarantee convergence of gradient descent methods (Liu et al., 2022b). Liu et al. (2024), in the context of policy optimization, define the bidilated regularizer whose policy gradients can be estimated without importance sampling. Illustratively, the two regualaizers read side-by-side ( γ is a discount factor of MDPs):

<!-- formula-not-decoded -->

## 5.2 Conclusion

We studied three different policy gradient methods for imperfect-information perfect-recall zerosum IIEFGs under a unifying optimization principle. We managed to provide the first global last-iterate convergence guarantees of policy gradient methods to an ϵ -approximate Nash equilibrium. Furthermore, our analysis requires a number of iterations and samples that is polynomial in 1 /ϵ and the parameters of the game. To do so, we demonstrated that utilities as functions of behavioral strategies (policies) exhibit gradient domination properties even though they are nonconvex; and provided a practical decentralized exploration scheme that implicitly controls the moduli of gradient domination. We departed from the usual route of regret analysis in IIEFGs and opted for more conventional convergence analysis arguments using a Lyapunov function. We hope to motivate further exchange between theoretical MARL research and the theory of IIEFGs as we strongly believe in the potential this communication fosters.

Future directions. Our main objective was proving polynomial time convergence of policy gradient in IIEFGs, our analysis is at places loose. We firmly believe that the convergence rates and constant dependencies can be improved, e.g. , by using the machinery of treeplex norms (Fan et al., 2024), relatively-smooth optimization (Lu et al., 2018; Fatkhullin and He, 2024), and other policy optimization arguments (Zhan et al., 2023; Cen et al., 2022b). To be particular, we would like to see guarantees that do not call for mini-batching and possibly use variance reduction techniques. Moreover, fundamental questions about the limit points of policy gradient methods in IIEFGs (similar to those of (Giannou et al., 2022) for MGs) are open. More broadly, do forms of benign nonconvexity (like hidden convexity) refine the results of (Cai et al., 2024b; Angelopoulos et al., 2025)?

## Acknowledgments

This work was supported in part by the NSF AI Institute for Learning-Enabled Optimization at Scale (TILOS, CCF-2112665), NSF Award CCF-244306, and the Office of Naval Research (ONR grants N000142412631 and N00014-25-1-2296). GF is supported in part by an AI2050 Early Career Fellowship.

## References

- Alekh Agarwal, Sham M Kakade, Jason D Lee, and Gaurav Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. Journal of Machine Learning Research , 22(98):1-76, 2021.
- Ahmet Alacaoglu, Luca Viano, Niao He, and Volkan Cevher. A natural actor-critic framework for zero-sum markov games. In International Conference on Machine Learning , pages 307-366. PMLR, 2022a.
- Ahmet Alacaoglu, Luca Viano, Niao He, and Volkan Cevher. A natural actor-critic framework for zero-sum Markov games. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning , volume 162 of Proceedings of Machine Learning Research , pages 307-366. PMLR, 17-23 Jul 2022b. URL https://proceedings.mlr.press/v162/alacaoglu22a.html .
- Shun-Ichi Amari. Natural gradient works efficiently in learning. Neural computation , 10(2):251-276, 1998.
- Ioannis Anagnostides, Ioannis Panageas, Gabriele Farina, and Tuomas Sandholm. On last-iterate convergence beyond zero-sum games. In International Conference on Machine Learning , pages 536-581. PMLR, 2022.
- Anastasios N Angelopoulos, Michael I Jordan, and Ryan J Tibshirani. Gradient equilibrium in online learning: Theory and applications. arXiv preprint arXiv:2501.08330 , 2025.
- Nolan Bard, Jakob N Foerster, Sarath Chandar, Neil Burch, Marc Lanctot, H Francis Song, Emilio Parisotto, Vincent Dumoulin, Subhodeep Moitra, Edward Hughes, et al. The hanabi challenge: A new frontier for ai research. Artificial Intelligence , 280:103216, 2020.
- Jalaj Bhandari and Daniel Russo. Global optimality guarantees for policy gradient methods. Operations Research , 72(5):1906-1927, 2024.
- Ronen I Brafman and Moshe Tennenholtz. R-max-a general polynomial time algorithm for nearoptimal reinforcement learning. Journal of Machine Learning Research , 3(Oct):213-231, 2002.
- Noam Brown and Tuomas Sandholm. Solving imperfect-information games via discounted regret minimization. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 33, pages 1829-1836, 2019a.
- Noam Brown and Tuomas Sandholm. Superhuman ai for multiplayer poker. Science , 365(6456): 885-890, 2019b.
- Yang Cai, Constantinos Daskalakis, Haipeng Luo, Chen-Yu Wei, and Weiqiang Zheng. On tractable phi-equilibria in non-concave games. arXiv preprint arXiv:2403.08171 , 2024a.
- Yang Cai, Gabriele Farina, Julien Grand-Cl´ ement, Christian Kroer, Chung-Wei Lee, Haipeng Luo, and Weiqiang Zheng. Fast last-iterate convergence of learning in games requires forgetful algorithms. arXiv preprint arXiv:2406.10631 , 2024b.
- Shicong Cen, Chen Cheng, Yuxin Chen, Yuting Wei, and Yuejie Chi. Fast global convergence of natural policy gradient methods with entropy regularization. Operations Research , 70(4): 2563-2578, 2022a.
- Shicong Cen, Yuejie Chi, Simon S Du, and Lin Xiao. Faster last-iterate convergence of policy optimization in zero-sum markov games. arXiv preprint arXiv:2210.01050 , 2022b.

- Marco Cusumano-Towner, David Hafner, Alex Hertzberg, Brody Huval, Aleksei Petrenko, Eugene Vinitsky, Erik Wijmans, Taylor Killian, Stuart Bowers, Ozan Sener, et al. Robust autonomy emerges from self-play. arXiv preprint arXiv:2502.03349 , 2025.
- Constantinos Daskalakis, Dylan J Foster, and Noah Golowich. Independent policy gradient methods for competitive reinforcement learning. Advances in neural information processing systems , 33: 5527-5540, 2020.
- Damek Davis and Dmitriy Drusvyatskiy. Stochastic subgradient method converges at the rate o (k-1/4) on weakly convex functions. arXiv preprint arXiv:1802.02988 , 2018.
- Dmitriy Drusvyatskiy and Adrian S Lewis. Error bounds, quadratic growth, and linear convergence of proximal methods. Mathematics of Operations Research , 43(3):919-948, 2018.
- Dmitriy Drusvyatskiy and Courtney Paquette. Efficiency of minimizing compositions of convex functions and smooth maps. Mathematical Programming , 178:503-558, 2019.
- Zhiyuan Fan, Christian Kroer, and Gabriele Farina. On the optimality of dilated entropy and lower bounds for online learning in extensive-form games. arXiv preprint arXiv:2410.23398 , 2024.
- Gabriele Farina, Christian Kroer, and Tuomas Sandholm. Optimistic regret minimization for extensiveform games via dilated distance-generating functions. Advances in neural information processing systems , 32, 2019.
- Ilyas Fatkhullin and Niao He. Taming nonconvex stochastic mirror descent with general bregman divergence. In International Conference on Artificial Intelligence and Statistics , pages 3493-3501. PMLR, 2024.
- Ilyas Fatkhullin, Niao He, and Yifan Hu. Stochastic optimization under hidden convexity. arXiv preprint arXiv:2401.00108 , 2023.
- Maryam Fazel, Rong Ge, Sham M. Kakade, and Mehran Mesbahi. Global convergence of policy gradient methods for the linear quadratic regulator. In International Conference on Machine Learning , 2018. URL https://api.semanticscholar.org/CorpusID:51881649 .
- Bolin Gao and Lacra Pavel. On the properties of the softmax function with application in game theory and reinforcement learning. arXiv preprint arXiv:1704.00805 , 2017.
- Rong Ge, Jason D Lee, and Tengyu Ma. Matrix completion has no spurious local minimum. Advances in neural information processing systems , 29, 2016.
- Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. A theory of regularized markov decision processes. In International conference on machine learning , pages 2160-2169. PMLR, 2019.
- Angeliki Giannou, Kyriakos Lotidis, Panayotis Mertikopoulos, and Emmanouil-Vasileios VlatakisGkaragkounis. On the convergence of policy gradient methods to nash equilibria in general stochastic games. Advances in Neural Information Processing Systems , 35:7128-7141, 2022.
- Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning , pages 1861-1870. Pmlr, 2018.
- Elad Hazan et al. Introduction to online convex optimization. Foundations and Trends® in Optimization , 2(3-4):157-325, 2016.
- Daniel Hennes, Dustin Morrill, Shayegan Omidshafiei, R´ emi Munos, Julien Perolat, Marc Lanctot, Audrunas Gruslys, Jean-Baptiste Lespiau, Paavo Parmas, Edgar Du` e˜ nez Guzm´ an, and Karl Tuyls. Neural replicator dynamics: Multiagent learning via hedging policy gradients. In Proceedings of the 19th International Conference on Autonomous Agents and MultiAgent Systems , AAMAS '20, page 492-501, Richland, SC, 2020. International Foundation for Autonomous Agents and Multiagent Systems. ISBN 9781450375184.
- Samid Hoda, Andrew Gilpin, Javier Pena, and Tuomas Sandholm. Smoothing techniques for computing nash equilibria of sequential games. Mathematics of Operations Research , 35(2): 494-512, 2010.

- Sashank J Reddi, Suvrit Sra, Barnabas Poczos, and Alexander J Smola. Proximal stochastic methods for nonsmooth nonconvex finite-sum optimization. Advances in neural information processing systems , 29, 2016.
- Sham M Kakade. A natural policy gradient. Advances in neural information processing systems , 14, 2001.
- Fivos Kalogiannis, Emmanouil-Vasileios Vlatakis-Gkaragkounis, Ian Gemp, and Georgios Piliouras. Solving zero-sum convex markov games. In Forty-second International Conference on Machine Learning , 2025.
- Hamed Karimi, Julie Nutini, and Mark Schmidt. Linear convergence of gradient and proximalgradient methods under the polyak-łojasiewicz condition. In Joint European conference on machine learning and knowledge discovery in databases , pages 795-811. Springer, 2016.
- Daphne Koller, Nimrod Megiddo, and Bernhard Von Stengel. Efficient computation of equilibria for extensive two-person games. Games and economic behavior , 14(2):247-259, 1996.
- Christian Kroer, Kevin Waugh, Fatma Kılınc ¸-Karzan, and Tuomas Sandholm. Faster algorithms for extensive-form game solving via improved smoothing functions. Mathematical Programming , 179 (1):385-417, 2020.
- Marc Lanctot, Vinicius Zambaldi, Audrunas Gruslys, Angeliki Lazaridou, Karl Tuyls, Julien P´ erolat, David Silver, and Thore Graepel. A unified game-theoretic approach to multiagent reinforcement learning. Advances in neural information processing systems , 30, 2017.
- Tor Lattimore and Csaba Szepesv´ ari. Bandit algorithms . Cambridge University Press, 2020.
- Guoyin Li and Ting Kei Pong. Calculus of the exponent of kurdyka-łojasiewicz inequality and its applications to linear convergence of first-order methods. Foundations of computational mathematics , 18(5):1199-1232, 2018.
- Feng-Yi Liao, Lijun Ding, and Yang Zheng. Error bounds, pl condition, and quadratic growth for weakly convex functions, and linear convergences of proximal point methods. In 6th Annual Learning for Dynamics &amp; Control Conference , pages 993-1005. PMLR, 2024.
- Michael L Littman. Markov games as a framework for multi-agent reinforcement learning. In Machine learning proceedings 1994 , pages 157-163. Elsevier, 1994.
- Chaoyue Liu, Libin Zhu, and Mikhail Belkin. Loss landscapes and optimization in over-parameterized non-linear systems and neural networks. Applied and Computational Harmonic Analysis , 59: 85-116, 2022a.
- Mingyang Liu, Asuman Ozdaglar, Tiancheng Yu, and Kaiqing Zhang. The power of regularization in solving extensive-form games. arXiv preprint arXiv:2206.09495 , 2022b.
- Mingyang Liu, Gabriele Farina, and Asuman Ozdaglar. A policy-gradient approach to solving imperfect-information games with iterate convergence. arXiv preprint arXiv:2408.00751 , 2024.
- Weiming Liu, Huacong Jiang, Bin Li, and Houqiang Li. Equivalence analysis between counterfactual regret minimization and online mirror descent. In International Conference on Machine Learning , pages 13717-13745. PMLR, 2022c.
- Edward Lockhart, Marc Lanctot, Julien P´ erolat, Jean-Baptiste Lespiau, Dustin Morrill, Finbarr Timbers, and Karl Tuyls. Computing approximate equilibria in sequential adversarial games by exploitability descent. arXiv preprint arXiv:1903.05614 , 2019.
- Haihao Lu, Robert M Freund, and Yurii Nesterov. Relatively smooth convex optimization by first-order methods, and applications. SIAM Journal on Optimization , 28(1):333-354, 2018.
- Yiren Lu, Justin Fu, George Tucker, Xinlei Pan, Eli Bronstein, Rebecca Roelofs, Benjamin Sapp, Brandyn White, Aleksandra Faust, Shimon Whiteson, et al. Imitation is not enough: Robustifying imitation with reinforcement learning for challenging driving scenarios. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , pages 7553-7560. IEEE, 2023.

- Jincheng Mei, Chenjun Xiao, Csaba Szepesvari, and Dale Schuurmans. On the global convergence rates of softmax policy gradient methods. In International conference on machine learning , pages 6820-6829. PMLR, 2020.
- Julie Mulvaney-Kemp, SangWoo Park, Ming Jin, and Javad Lavaei. Dynamic regret bounds for constrained online nonconvex optimization based on polyak-lojasiewicz regions. IEEE Transactions on Control of Network Systems , 10(2):599-611, 2023. doi: 10.1109/TCNS.2022.3203798.
- Remi Munos, Julien Perolat, Jean-Baptiste Lespiau, Mark Rowland, Bart De Vylder, Marc Lanctot, Finbarr Timbers, Daniel Hennes, Shayegan Omidshafiei, Audrunas Gruslys, et al. Fast computation of nash equilibria in imperfect information games. In International Conference on Machine Learning , pages 7119-7129. PMLR, 2020.
- Katta G Murty and Santosh N Kabadi. Some np-complete problems in quadratic and nonlinear programming. Technical report, 1985.
- Ofir Nachum, Mohammad Norouzi, Kelvin Xu, and Dale Schuurmans. Bridging the gap between value and policy based reinforcement learning. Advances in neural information processing systems , 30, 2017.
- Ashutosh Nayyar, Aditya Mahajan, and Demosthenis Teneketzis. Decentralized stochastic control with partial history sharing: A common information approach. IEEE Transactions on Automatic Control , 58(7):1644-1658, 2013.
- Gergely Neu, Anders Jonsson, and Vicenc ¸ G´ omez. A unified view of entropy-regularized markov decision processes. arXiv preprint arXiv:1705.07798 , 2017.
- Konstantinos Oikonomidis, Emanuel Laude, and Panagiotis Patrinos. Forward-backward splitting under the light of generalized convexity. arXiv preprint arXiv:2503.18098 , 2025.
- Julien Perolat, Bruno Scherrer, Bilal Piot, and Olivier Pietquin. Approximate dynamic programming for two-player zero-sum markov games. In International Conference on Machine Learning , pages 1321-1329. PMLR, 2015.
- Julien Perolat, Remi Munos, Jean-Baptiste Lespiau, Shayegan Omidshafiei, Mark Rowland, Pedro Ortega, Neil Burch, Thomas Anthony, David Balduzzi, Bart De Vylder, et al. From poincar´ e recurrence to convergence in imperfect information games: Finding equilibrium via regularization. In International Conference on Machine Learning , pages 8525-8535. PMLR, 2021.
- Julien Perolat, Bart De Vylder, Daniel Hennes, Eugene Tarassov, Florian Strub, Vincent de Boer, Paul Muller, Jerome T Connor, Neil Burch, Thomas Anthony, et al. Mastering the game of stratego with model-free multiagent reinforcement learning. Science , 378(6623):990-996, 2022.
- Garvesh Raskutti and Sayan Mukherjee. The information geometry of mirror descent. IEEE Transactions on Information Theory , 61(3):1451-1457, 2015.
- Quentin Rebjock and Nicolas Boumal. Fast convergence to non-isolated minima: four equivalent conditions for c 2 functions. Mathematical Programming , pages 1-49, 2024.
- I Romanovskii. Reduction of a game with complete memory to a matrix game. Soviet Mathematics , 3:678-681, 1962.
- Max Rudolph, Nathan Lichtle, Sobhan Mohammadpour, Alexandre Bayen, J Zico Kolter, Amy Zhang, Gabriele Farina, Eugene Vinitsky, and Samuel Sokota. Reevaluating policy gradient methods for imperfect-information games. arXiv preprint arXiv:2502.08938 , 2025.
- Iosif Sakos, Emmanouil-Vasileios Vlatakis-Gkaragkounis, Panayotis Mertikopoulos, and Georgios Piliouras. Exploiting hidden structures in non-convex games for convergence to nash equilibrium. Advances in Neural Information Processing Systems , 36:66979-67006, 2023.
- Kevin Scaman, Cedric Malherbe, and Ludovic Dos Santos. Convergence rates of non-convex stochastic gradient descent under a generic lojasiewicz condition and local smoothness. In International conference on machine learning , pages 19310-19327. PMLR, 2022.

- Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering atari, go, chess and shogi by planning with a learned model. Nature , 588(7839):604-609, 2020.
- John Schulman, Sergey Levine, P. Abbeel, Michael I. Jordan, and Philipp Moritz. Trust region policy optimization. ArXiv , abs/1502.05477, 2015. URL https://api.semanticscholar. org/CorpusID:16046818 .
- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. ArXiv , abs/1707.06347, 2017. URL https://api.semanticscholar. org/CorpusID:28695052 .
- Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 , 2024.
- Lloyd S Shapley. Stochastic games. Proceedings of the national academy of sciences , 39(10): 1095-1100, 1953.
- David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. nature , 529(7587):484-489, 2016.
- Samuel Sokota, Ryan D'Orazio, J Zico Kolter, Nicolas Loizou, Marc Lanctot, Ioannis Mitliagkas, Noam Brown, and Christian Kroer. A unified approach to reinforcement learning, quantal response equilibria, and two-player zero-sum games. arXiv preprint arXiv:2206.05825 , 2022.
- Samuel Sokota, Ryan D'Orazio, Chun Kai Ling, David J Wu, J Zico Kolter, and Noam Brown. Abstracting imperfect information away from two-player zero-sum games. In International Conference on Machine Learning , pages 32169-32193. PMLR, 2023.
- Sriram Srinivasan, Marc Lanctot, Vinicius Zambaldi, Julien P´ erolat, Karl Tuyls, R´ emi Munos, and Michael Bowling. Actor-critic policy optimization in partially observable multiagent environments. Advances in neural information processing systems , 31, 2018.
- Ju Sun, Qing Qu, and John Wright. Complete dictionary recovery over the sphere. In 2015 International Conference on Sampling Theory and Applications (SampTA) , pages 407-410. IEEE, 2015.
- Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems , 12, 1999.
- Oskari Tammelin. Solving large imperfect information games using cfr+. arXiv preprint arXiv:1407.5042 , 2014.
- Oriol Vinyals, Igor Babuschkin, Junyoung Chung, Michael Mathieu, Max Jaderberg, Wojciech M Czarnecki, Andrew Dudzik, Aja Huang, Petko Georgiev, Richard Powell, et al. Alphastar: Mastering the real-time strategy game starcraft ii. DeepMind blog , 2:20, 2019.
- Emmanouil-Vasileios Vlatakis-Gkaragkounis, Lampros Flokas, and Georgios Piliouras. Solving min-max optimization with hidden structure via gradient descent ascent. Advances in Neural Information Processing Systems , 34:2373-2386, 2021.
- Bernhard Von Stengel. Efficient computation of behavior strategies. Games and Economic Behavior , 14(2):220-246, 1996.
- Chen-Yu Wei, Chung-Wei Lee, Mengxiao Zhang, and Haipeng Luo. Last-iterate convergence of decentralized optimistic gradient descent/ascent in infinite-horizon competitive markov games. In Conference on learning theory , pages 4259-4299. PMLR, 2021.
- Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning , 8:229-256, 1992.

- Ronald J Williams and Jing Peng. Function optimization using connectionist reinforcement learning algorithms. Connection Science , 3(3):241-268, 1991.
- Junchi Yang, Negar Kiyavash, and Niao He. Global convergence and variance reduction for a class of nonconvex-nonconcave minimax problems. Advances in Neural Information Processing Systems , 33:1153-1165, 2020.
- Sihan Zeng, Thinh Doan, and Justin Romberg. Regularized gradient descent ascent for two-player zero-sum markov games. Advances in Neural Information Processing Systems , 35:34546-34558, 2022.
- Wenhao Zhan, Shicong Cen, Baihe Huang, Yuxin Chen, Jason D Lee, and Yuejie Chi. Policy mirror descent for regularized reinforcement learning: A generalized framework with linear convergence. SIAM Journal on Optimization , 33(2):1061-1091, 2023.
- Junyu Zhang, Alec Koppel, Amrit Singh Bedi, Csaba Szepesvari, and Mengdi Wang. Variational policy gradient method for reinforcement learning with general utilities. Advances in Neural Information Processing Systems , 33:4572-4583, 2020.
- Junyu Zhang, Chengzhuo Ni, Csaba Szepesvari, Mengdi Wang, et al. On the convergence and sample efficiency of variance-reduced policy gradient method. Advances in Neural Information Processing Systems , 34:2228-2240, 2021.
- Kaiqing Zhang, Zhuoran Yang, and Tamer Basar. Policy optimization provably converges to nash equilibria in zero-sum linear quadratic games. Advances in Neural Information Processing Systems , 32, 2019.
- Runyu Zhang, Qinghua Liu, Huan Wang, Caiming Xiong, Na Li, and Yu Bai. Policy optimization for markov games: Unified framework and faster convergence. Advances in Neural Information Processing Systems , 35:21886-21899, 2022.
- Yulai Zhao, Yuandong Tian, Jason Lee, and Simon Du. Provably efficient policy optimization for two-player zero-sum markov games. In International Conference on Artificial Intelligence and Statistics , pages 2736-2761. PMLR, 2022.
- Martin Zinkevich, Michael Johanson, Michael Bowling, and Carmelo Piccione. Regret minimization in games with incomplete information. Advances in neural information processing systems , 20, 2007.

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Proofs of all claims are provided in the appendix

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discussed them in the conclusion section

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate 'Limitations' section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: Yes found in the appendix.

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

Justification: experiments are small scale. code will be uploaded

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

Justification: code and proofs are in the supplemental material

## Guidelines:

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

Justification: code is shared

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: small scale experiments, confidence intervals included

## Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer 'Yes' if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
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

Justification: description of laptop

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: we follow the NeurIPS code of ethics

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: work is theoretical. probably unlikely that it will have direct societal impacts Guidelines:

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

Justification: we cite previous work

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

Justification: no new assets released

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: no crowdsourcing

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: theoretical research

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification:

Guidelines: only editing grammar

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Appendix

| A   | Further Related Work                 | Further Related Work                                                |   25 |
|-----|--------------------------------------|---------------------------------------------------------------------|------|
| B   | Further Preliminaries on IIEFGs      | Further Preliminaries on IIEFGs                                     |   26 |
|     | B.1                                  | The Behavioral and Sequence-Form Strategies . . . . . . . . .       |   26 |
|     | B.2                                  | Value, Action-Value, and Advantage Functions . . . . . . . .        |   28 |
|     | B.3                                  | Continuity of the Utility . . . . . . . . . . . . . . . . . . . . . |   29 |
|     | B.4                                  | Properties of the Bidilated Regularizer . . . . . . . . . . . . .   |   30 |
| C   | Efficient Exploration                | Efficient Exploration                                               |   34 |
| D   | Regarding the Policy Parametrization | Regarding the Policy Parametrization                                |   35 |
|     | D.1                                  | Definitions . . . . . . . . . . . . . . . . . . . . . . . . . . . . |   35 |
|     | D.2                                  | Properties under Parameter Constraints . . . . . . . . . . . . .    |   35 |
| E   | Gradient Domination                  | Gradient Domination                                                 |   38 |
|     | E.1                                  | Direct Policy Parametrization pPŁ . . . . . . . . . . . . . . .     |   38 |
|     | E.2                                  | Softmax Policy Parametrization pPŁ . . . . . . . . . . . . . .      |   38 |
|     | E.3                                  | Mahalanobis-pPŁ . . . . . . . . . . . . . . . . . . . . . . . .     |   39 |
|     | E.4                                  | Weak Gradient Domination . . . . . . . . . . . . . . . . . . .      |   40 |
| F   | Gradient Estimators                  | Gradient Estimators                                                 |   40 |
|     | F.1                                  | A Policy Gradient Theorem . . . . . . . . . . . . . . . . . . .     |   40 |
| G   | Optimization Lemmata                 | Optimization Lemmata                                                |   45 |
|     | G.1                                  | A Variation of the Descent Lemma . . . . . . . . . . . . . . .      |   47 |
|     | G.2                                  | Min-Max Optimization . . . . . . . . . . . . . . . . . . . . .      |   48 |
|     | G.3                                  | Regarding the Mahalanobis Distance . . . . . . . . . . . . . .      |   50 |
|     | G.4                                  | Alternating Mirror Descent using a Changing Mahalanobis DGF         |   51 |
| H   | Convergence Analysis                 | Convergence Analysis                                                |   55 |
|     | H.1                                  | Direct Policy Parametrization . . . . . . . . . . . . . . . . . .   |   55 |
|     | H.2                                  | Softmax Policy Parametrization . . . . . . . . . . . . . . . .      |   57 |
|     | H.3                                  | Natural Policy Gradient . . . . . . . . . . . . . . . . . . . . .   |   59 |
| I   | Proximity of Projections             | Proximity of Projections                                            |   61 |

## A Further Related Work

In this section we attempt discussing related work. Arguably, since our work lies in the intersection of several already broad themes, we encourage the reader to follow references in the cited works.

Relevant MARL for MG works In MDP and MG literature, policy optimization seems to come in two flavors-an online learning (Hazan et al., 2016; Lattimore and Szepesv´ ari, 2020) approach and a stochastic optimization one. In the current work, we opt for the second approach.

The approach of (Zeng et al., 2022) which considers zero-sum Markov games is particularly similar to ours. Yet, we highlight that they make a rather strong assumption; they assume that the probability of playing each action in the support of the regularized Nash equilibrium is lower-bounded by a constant independent of the regularization coefficient τ . In turn, we contribute the two-sided pPŁ condition for IIEFGs and, importantly, circumvent such an assumption by exercising direct control over the minimum probability of playing any action by projecting the parameters of the softmax parameters onto a convex polytope.

Theory of Policy Gradient Methods The policy gradient method was introduced for Markov decision processes in (Williams, 1992; Sutton et al., 1999). Ever since provable guarantees have been yielded by a number of works for different variations of the algorithm:

- (Agarwal et al., 2021) prove the convergence of directly parametrized policy gradient. They use the convergence result of gradient descent for smooth nonconvex function along a gradient domination lemma to demonstrate a O (1 /ϵ 2 ) convergence rate to optimality. Later, (Zhang et al., 2020, 2021) use the hidden concave structure of the problem to improve the convergence rate to O (1 /ϵ ) .
- (Mei et al., 2020) provide the first non-asymptotic convergence rate result for the policy gradient method using discounted entropy regularization (the analogue of bidilated entropy regularization). The proof of convergence uses a novel nonuniform PŁ condition.
- (Cen et al., 2022a) analyze natural policy gradient (NPG) with discounted entropy regularization. Natural policy gradient can be seen as a form of preconditioned gradient descent. Natural policy gradient effectively boils down to policy multiplicative weight updates using the Q -functions as feedback. The analysis of convergence uses a linear dynamical system.

Regularized Markov Decision Processes Regularization in RL seems to have a very broad development. It was theoretically analyzed by (Haarnoja et al., 2018; Nachum et al., 2017; Geist et al., 2019). Regularization helps with both the optimization landscape (Mei et al., 2020) as well as learning policies from offline data (Neu et al., 2017).

RL &amp; Regularization in IIEFGs Applying RL in IIEFGs, in the sense of using policy gradients and action-value functions is not a new endeavor. It has been extensively studied from both theoretical and practical viewpoints (Munos et al., 2020; Sokota et al., 2022; Rudolph et al., 2025). Yet, a provable convergence guarantee for policy gradient methods like ours was missing. Furthermore, using regularization has also been investigated in (Perolat et al., 2021; Liu et al., 2022b, 2024) to get favorable convergence guarantees to equilibria, to guarantee uniqueness of equilibria and continuity of best-response maps Sokota et al. (2023).

Markov Games MGs have been extensively studied through the lens of policy gradient and policy optimization methods. For the zero-sum setting there have been numerous algorithmic approaches using multiple techniques (Brafman and Tennenholtz, 2002; Perolat et al., 2015; Alacaoglu et al., 2022a; Wei et al., 2021; Zhang et al., 2022).

## B Further Preliminaries on IIEFGs

## B.1 The Behavioral and Sequence-Form Strategies

In this subsection, we investigate the continuity of the sequence-form map and that of its inverse.

Lemma B.1. Under Assumption 2, the transforms c -1 1 : M 1 →X γ , c -1 2 : M 2 →Y γ are Lipschitz continuous. I.e., for any µ 1 , µ ′ 1 , it holds true that,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We will first observe the difference in c -1 1 in the ( s, a ) -th entry of the the vector-valued mapping:

<!-- formula-not-decoded -->

As a reminder, for all s ∈ S 1 it holds that µ 1 ( s ) ≥ γ |H| by Assumption 2. Proceeding towards the desired inequality,

<!-- formula-not-decoded -->

We need to upper bound the second term by some quantity proportional to ∥ µ 1 -µ ′ 1 ∥ . We first note that by the triangular inequality,

<!-- formula-not-decoded -->

and for any µ 2 , µ ′ 2 ,

where the last inequality is due to the fact that ∥ x ∥ 1 ≤ √ d ∥ x ∥ , ∀ x ∈ R d . As such, we can note that,

<!-- formula-not-decoded -->

Plugging this inequality into (1) yields the desired bound.

Lemma B.2. The sequence-form strategy µ 1 = c 1 ( π 1 ) is a ( √ | Σ 1 | D ( T )) -Lipschitz and ( √ | Σ 1 | D ( T )) -smooth function of the behavioral strategy π 1 . That is,

<!-- formula-not-decoded -->

for any π 1 , π ′ 1 , where J c 1 ( · ) denotes the Jacobian of the sequence-form map.

Proof. For the continuity of µ 1 we observe that each entry of the Jacobian, J c 1 ( π 1 ) , is in [0 , 1] as a product of variables in [0 , 1] . Further, the number of non-zero elements of each row of J c 1 is bounded by the height of the tree, D ( T ) . We can then write,

<!-- formula-not-decoded -->

Now, for the continuity of the Jacobian, J c 1 ( · ) , we make some observations on the Hessian tensor. In particular, for the matrix corresponding to a single entry of µ 1 , with index i , it is the case that all entries are in [0 , 1] and are at most D ( T ) 2 in number. Then, we consider ∥∇ 2 c ( π 1 ) ∥ op :=

√ ∑

( ∑

∑

)

2

sup

∥

u

∥

2

=

∥

v

∥

2

=1

i

j

k

[

∇

c

(

π

1

)]

ijk

u

j

v

k

where j, k

index entries of

π

1

. In this case, by bounding each ( ∑ j ∑ k [ ∇ 2 c ( π 1 )] ijk u j v k ) by an upper bound on its Frobenius norm, we conclude that,

<!-- formula-not-decoded -->

Lemma B.3. The sequence-form strategy µ 1 = c 1 ( π χ ) is a ( √ | Σ 1 | D ( T )) -Lipschitz and ( √ | Σ 1 | D ( T )) -smooth function of the parameters of softmax policy π χ , χ . That is,

<!-- formula-not-decoded -->

for any χ, χ ′ .

Proof. We know that the softmax map is 1 2 -Lipschitz continuous and it has a 8 -Lipschitz Jacobian Lemma D.2. Treating c 1 ( π χ ) as a composition of the sequence-form map and the softmax map, we can conclude that,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

2

## B.2 Value, Action-Value, and Advantage Functions

On notation. In this subsection, we will use the following shorthand notations,

- σ 1 ( h ) , σ 2 ( h ) returns the last history before h where player 1 (player 2, resp.) took an action,
- h ∈ s signifies that history h belongs in the infoset s ,
- h ′ ⪰ T h, h ′ ⪰ T ( h, a ) signifies that h ′ is a successor/child node of h , ( h, a ) ;
- h ∈ ξ, ( h, a ) ∈ ξ signifies that h , h, a belongs in the game trajectory ξ from the root to a terminal node.

Occupancy measure For a policy pair π := ( π 1 , π 2 ) , we define d π : S → [0 , 1] to be a finite measure over all the infosets-summing over all infosets s ∈ S yields the depth of the game tree D ( T ) -where for any infoset s ∈ S ,

<!-- formula-not-decoded -->

The value function of each infoset is defined as,

<!-- formula-not-decoded -->

Also, the action-value function reads:

<!-- formula-not-decoded -->

We define the advantage function to be:

<!-- formula-not-decoded -->

Finally, let a policy pair π 1 , π 2 and π := ( π 1 , π 2 ) . Let π 1 be parametrized by some vector θ . We compute the policy gradient for θ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Where we have used the following fact,

<!-- formula-not-decoded -->

Further, for direct policy parametrization, we get,

<!-- formula-not-decoded -->

For the softmax policy parametrization, (2) yields,

<!-- formula-not-decoded -->

## B.3 Continuity of the Utility

We briefly consider the Lipschitz continuity of the utility w.r.t. direct and softmax policy parametrizations.

Lemma B.4. The utility of an IIEFG function as a function of direct-parametrized policies is (max i ∈{ 1 , 2 } √ | Σ i | D ( T )) -smooth.

Proof. Let u := R µ π 2 2 . It is a vector in R | Σ 1 | with entries in [ -1 , 1] . As such,

<!-- formula-not-decoded -->

from which we write,

<!-- formula-not-decoded -->

Where, we used Lemma B.2 in the first inequality.

Lemma B.5. The utility function as a function of softmax-parametrized policies is 16(max i ∈{ 1 , 2 } √ | Σ i | D ( T )) -smooth.

Proof. We treat the utility function as a composition of the utility as a function of the policy and the softmax map ( i.e. , Lemma B.4 along with Lemma D.2).

## B.4 Properties of the Bidilated Regularizer

Introduced in (Liu et al., 2024), the bidilated regularizer offers an alternative to the commonly used dilated regularizer (Hoda et al., 2010). It can be seamlessly used along Q feedback by dropping the need of importance sampling which would be necessary for the dilated regularizer when the gradient is estimated through trajectory roll-outs. The purpose of this refined regularizer was introducing a distance generating function in the sequence-form space that would not necessitate importance sampling.

## B.4.1 Strong Convexity Modulus

Lemma B.6. For a choice of strongly convex function ψ , and a weighting scheme { w 1 ,s } s ∈S 1 , { w 2 ,s } s ∈S 2 and let α dil &gt; 0 be the modulus of the weighted dilated regularizer. Then, the corresponding bidiliated regularizer is strongly convex,

<!-- formula-not-decoded -->

Proof. These calculations were used in the proof of (Liu et al., 2024, Lemma D.1); we repeat them for completeness. For an appropriate choice of weights { w 1 ,s } s ∈S 1 , { w 2 ,s } s ∈S 2 , the weighted bidilated regularizer is defined as,

<!-- formula-not-decoded -->

We can slightly refine (Liu et al., 2024, Lemma C.1) in order to compute an explicit lower bound on the convexity modulus of different weighted bidilated regularizer depending on the choice of ψ . From the fact that R 1 ( µ π 1 1 , µ π 2 2 ) is linear in µ π 2 2 and the definition of the Bregman divergence, we conclude that,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By (Liu et al., 2022c, Lemma D.2) we know that,

<!-- formula-not-decoded -->

As such, for the strong convexity modulus of the weighted R ψ 1 relative to the choice of norm appropriate for ψ , we write,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By (Farina et al., 2019, Corollary 1), we know that there exists a weighting scheme, such that the Euclidean dilated regularizer is 1 -strongly convex w.r.t. the ℓ 2 -norm. The procedure assigns weights to nodes in a bottom-up fashion.

- At each leaf node s , the weights are set to

<!-- formula-not-decoded -->

- For an internal node s , let s a , s a ′ , . . . denote its child nodes under actions a, a ′ , . . . . For each action a , compute

<!-- formula-not-decoded -->

- The node's weights are then set to

<!-- formula-not-decoded -->

Corollary B.1 (Euclidean Regularizer) . There exists a choice of weights, with max s w 1 ,s , max s w 2 ,s = Θ(2 D ( T ) ) , and under the assumption that min s µ 2 ( s ) ≥ γ , the bidilated Euclidean regularizer has a strong convexity modulus w.r.t. the ℓ 2 -norm, α bi ,

<!-- formula-not-decoded -->

(Kroer et al., 2020, Theorem 2) states that a recursion defines weights with max s w 1 ,s , max s w 2 ,s = Θ(2 D ( T ) ) such that the entropic dilated regularizer is strongly convex w.r.t. the ℓ 2 -norm.

Corollary B.2 (Entropic Regularizer) . There exists a choice of weights, and under the assumption that min s µ 2 ( s ) ≥ γ , the bidilated entropic regularizer has a strong convexity modulus w.r.t. the ℓ 2 -norm, α bi ,

<!-- formula-not-decoded -->

## B.4.2 Lipschitz Moduli

Here, we establish the Lipschitz continuity of the regularizers and that of their gradients.

## Euclidean regularizer

Lemma B.7. The weighted Euclidean bidilated regularizer is ℓ -smooth with

<!-- formula-not-decoded -->

Proof. We write the bidilated regularizer as

<!-- formula-not-decoded -->

For a fixed π 2 , we have

<!-- formula-not-decoded -->

where, f ( π 1 , π 2 ) , g ( π 1 ) ∈ R |H| with f ( π 1 , π 2 ) = ∑ h ∈ s µ c ( h ) µ π 2 2 ( σ 2 ( h )) µ π 1 1 ( σ 1 ( h )) and g s ( π 1 ) = w 1 ,s ∥ π 1 ( ·| s ) ∥ 2 . We write:

<!-- formula-not-decoded -->

- For g , we see that L g := √ S max s w 1 ,s and ℓ g := 2 √ S max s w 1 ,s by the properties of the weighted ℓ 2 -norm and the fact that π 1 ( ·| s ) lies in the simplex, i.e. , ∥ π 1 ( ·| s ) ∥ 2 ≤ 1 . Also, the weight w 1 ,s only scales the local quadratic term.
- For f , similar to Lemma B.2 and Lemma B.4, L f ≤ max i ∈{ 1 , 2 } | Σ i | √ D ( T ) S and ℓ f ≤ max i ∈{ 1 , 2 } | Σ i | D ( T ) √ S . Also, it holds that max π 1 ,π 2 ∥ f ( π 1 ) ∥ ≤ √ S .

Concluding,

<!-- formula-not-decoded -->

Symmetrically,

<!-- formula-not-decoded -->

Now, we need to bound the Lipschitz modulus of ∇ π 1 R eucl 2 ( π 1 , π 2 ) . Similarly, we write,

<!-- formula-not-decoded -->

We see that the the vector f ( π 1 , π 2 ) (occupancy measure of player 2 ) has entries that are products of entries of µ 1 , µ 2 , µ c . Hence, L f = max i ∈{ 1 , 2 } | Σ i | √ D ( T ) S and ℓ f = max i ∈{ 1 , 2 } | Σ i | D ( T ) √ S.

<!-- formula-not-decoded -->

## Entropic regularizer

Lemma B.8. The weighted entropic bidilated regularizer is ℓ -smooth with

<!-- formula-not-decoded -->

Proof. We write R 2 as the inner product of f ( π χ ) := d π χ ,π θ and g := [ π θ ( b | s ) log π θ ( b | s )] s,b . For notational convenience, we suppress dependence of f, g on π θ .

<!-- formula-not-decoded -->

We now bound the Lipschitz modulus of the gradient using the chain rule:

<!-- formula-not-decoded -->

For the Lipschitz modulus of ∇ χ R 1 ( π χ , π θ ) , we re-purpose the lengthy calculations found in the proof of (Mei et al., 2020, Lemma 14), we consider χ = χ 0 + αu for some u, χ ∈ R A , α ∈ R ,

<!-- formula-not-decoded -->

hence, (since ∥ x ∥ 2 ≤ √ S 1 ∥ x ∥ ∞ ),

<!-- formula-not-decoded -->

or, L g = max s w 1 ,s log A √ S . Similarly,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

or, ℓ g = 3max s w 1 ,s (1 + log A ) S . Hence, ∇ χ R 1 is ℓ -smooth with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C Efficient Exploration

Throughout our proofs, we have kept our complexity results parametric w.r.t. 1 /γ . Anaive exploration rule that would dictate that the player merely picks behavioral strategies over the ε -truncated simplex will give a γ = O ( ε D ( T ) ) . We propose a different approach to exploration. In particular, every player is expected to reach every prefix subsequence with a probability γ | Σ i | where | Σ i | := 1 + ∑ s ∈S 1 |A s | denotes the set of all possible 'prefix' sequences of player i . The rule is simple,

- at the beginning of each game, the player throws a biased coin which lands on 'heads' with probability γ . If so happens, the player executes a sequence of actions with probability 1 | Σ i | . Afterwards, the player continues to play according to their own behavioral strategy.
- In the case that the coin lands on 'tails', the player simply plays according to their behavioral strategy.

We observe that in sequence-form, this means that µ 1 ( σ ( s )) ≥ γ | Σ 1 | + γ | Σ 1 | ∑ s ′ ∈ Σ 1 ✶ { s ′ ⪰ T s } (in words, the amount of 'probability flow' reaching the corresponding sequence σ ( s ) for s , is at least as much as γ | Σ 1 | plus the flow that passes through σ ( s ) to visit its children). In other words, the sequence-form strategies are truncated by a set of linear constraints and as long as γ ≤ 1 | Σ 1 | , there set of feasible sequence-form strategies is non-empty. We now observe that the mapping, from µ to the part component of the behavioral policy the agent can in fact control, is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

✶ The 'probability flow' passing through the edge ( s, a ) breaks down to a controllable part due to the policy π ( a | s ) and an uncontrollable one due to the exploration scheme. In particular, the uncontrollable 'probability flow' is precisely γ | Σ 1 | × ∑ s ′ ∈ Σ 1 ✶ { s ′ ⪰ T ( s, a ) } -i.e. , proportional to the number of nodes of the subtree rooted at the next node after ( s, a ) where player 1 acts. As such, the Lipschitz continuity of mapping µ ↦→ π , is Lipschitz continuous with a modulus, by following the same line of arguments as the ones in Lemma B.1.

In short, we are only adding an additional linear constraint on the feasibility set of µ π 1 1 (and µ π 2 2 , respectively). Granted that that this new feasibility set is always non-empty, this γ -truncated treeplex remains a convex polytope. Finally we note that for any player i , | Σ i | ≤ |H| .

Proposition 1. Let Γ be an n -player imperfect-information EFG Γ with perfect recall. Also, assume that players follow the exploration scheme of Assumption 2. Then, an ϵ -NE on the explorationinduced γ -truncated treeplices, is an ( ϵ +2[1 -(1 -γ ) n ] ) -NE of the original game.

Proof. Let π ⋆ be a joint policy profile, V π ⋆ i will be the utility of player under no exploration under joint policy π ⋆ and V π ⋆ γ,i the utility of player i under the exploration scheme. When the exploration scheme is followed, there is still a probability (1 -γ ) n that no player follows it for a particular episode. Hence, for any π ⋆ ,

<!-- formula-not-decoded -->

where, r i, max , r i, min signify the maximum and minimum value of payoff r i for player i . With the same line of reasoning, ∣ ∣ ∣ max π ′ i V π ′ i ,π ⋆ -i i -max π ′ i V π ′ i ,π ⋆ -i γ,i ∣ ∣ ∣ ≤ 1 -(1 -γ ) n . Now, assume { π ⋆ i } i ∈ [ n ] to be an ϵ -NE. Fixing a player i , we want to compute the difference in the optimality gap on the γ -truncated treeplex versus the entire treeplex. Now, by definition of the ϵ -NE,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## D Regarding the Policy Parametrization

## D.1 Definitions

Direct policy parametrization. Both players parameterize their policies (or behavioral strategies), π 1 : S 1 →A and π 2 : S 2 →B , using a concatenation of |S 1 | and |S 2 | probability vectors over the (potentially truncated) probability simplex ∆( A s ) , ∆( B s ) for all s in S 1 and S 2 respectively. The parameter space of player 1 is denoted by X := ∏ s ∈S 1 ∆( A s ) , while the parameter space of player 2 by Y := ∏ s ∈S 2 ∆( B s ) .

Softmax policy parametrization. Softmax parametrized policies have a well-known definition. The parameters of the corresponding policies are denoted χ, θ with χ ∈ R A , A = ∑ s A s and θ ∈ R B , B = ∑ s B s . For each infoset s , the policy is

<!-- formula-not-decoded -->

Now, since we want to have control over the minimum eigenvalue of the Jacobian of softmax( · ) , we restrict the parameter space to the following convex polytopes,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## D.2 Properties under Parameter Constraints

Lemma D.1. Let J := J ( θ ) ∈ R d × d

<!-- formula-not-decoded -->

Further, the vector 1 is an eigenvector of J with a corresponding eigenvalue of 0 . The rest of the eigenvalues are

<!-- formula-not-decoded -->

Proof. For brevity, define σ := softmax( θ ) , and let diag( v ) be the d × d diagonal matrix 'whose diagonal entries are given by v ∈ R d ,

<!-- formula-not-decoded -->

First, we observe that the all-ones vector 1 ∈ R d is an eigenvector of J with a corresponding eigenvalue of 0 ,

<!-- formula-not-decoded -->

By Weyl's inequality for two Hermitian matrices, A,B , we know that their eigenvalues indexed in a descending order λ 1 ( A ) ≥ · · · ≥ λ d ( A ) satisfy,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- λ + min ( J ) ≥ min i ∈ [ d ] σ i ( θ ) -by taking i = d and j = d -1 ;
- σ ↓ 2 ≤ λ max ( J ) ≤ max i ∈ [ d ] σ i ( θ ) -by taking i = 2 , j = 1 for the LHS and i = 1 , j = 1 for the RHS.

<!-- formula-not-decoded -->

Lemma D.2 ((Zhang et al., 2021, Lemma 5.3)) . The softmax map is 8 -smooth.

Lemma D.3. The softmax map, softmax : R d → R d , has an 3 √ 2 d 3 / 2 -smooth gradient.

Proof. Again we use σ := softmax( θ ) for brevity. We compute the second order derivatives:

<!-- formula-not-decoded -->

Every term is a function of θ and it is true in general that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As such, we can write,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma D.4. Assume θ ∈ R d with θ ∈ Θ R := { θ ∈ R d : θ ⊤ 1 = 0 and | θ i -θ j | ≤ 2 R, ∀ i, j ∈ [ d ] } . Then, the following bounds hold true,

- min i ∈ [ d ] softmax i ( θ ) ≥ 1 1+( d -1) e 2 R ;
- max i ∈ [ d ] softmax i ( θ ) ≥ 1 1+( d -1) e -2 R .

Proof.

Minimum probability lower bound. W.l.o.g. we minimize the first coordinate. We write,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By observing that,

We can lower bound the value as,

<!-- formula-not-decoded -->

̸

It suffices to maximize the quantity max j =1 ,θ ∈ Θ R { θ j -θ 1 } as the RHS quantity is non-increasing in max j =1 ,θ ∈ Θ R { θ j -θ 1 } . I.e., the largest difference between two coordinates of a vector in the sphere is 2 R . The minimum is achieved when θ j -θ 1 = 2 R and θ j = θ k , ∀ j, k ≥ 2 .

̸

Maximum probability lower bound. Similarly, w.l.o.g, it suffices to maximize softmax 1 ( θ ) for θ ∈ Θ R .

̸

<!-- formula-not-decoded -->

where the inequality follows from the convexity of e x . For any θ ∈ Θ R the point ( θ ) = ( θ 1 , . . . θ i d -1 , . . . ) is also in Θ R due to the convexity of the set (it is a linear polytope). We can simply optimize the objective,

<!-- formula-not-decoded -->

Due to the objective function's monotonicity in b -a , the program can be simplified even more into,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, it is clear that the last objective is minimized for a -b = -2 R . Letting ε ≤ ( d -1) -2

.

In this vein, if we want to bound the minimum probability of the softmax parametrized policy by ε &gt; 0 for some R &gt; 0 , we need to set R ≤ 1 / 2 log ( 1 -ε ε ( d -1) ) . Then, it is also the case that max θ ∈ Θ R ,i softmax i ( θ ) ≥ 1 -ε 1 -ε + ε ( d -1) 2 ≥ 1 -ε -ε ( d -1) 2 .

Proposition 2. Let p be a probability vector in ∆ d -1 and define θ ( p ) to be the set of θ such that softmax( θ ) = p . For any two θ, θ ′ ∈ θ ( p ) , there exists a c ∈ R such that θ = θ ′ + c 1 .

Proof. By assumption, softmax( θ ) = softmax( θ ′ ) = p . For every entry i ,

<!-- formula-not-decoded -->

Letting Z := ∑ d i e θ i , Z ′ := ∑ d i , we observe,

<!-- formula-not-decoded -->

Hence, any two θ, θ ′ that map to the same probability vector are translations of each other in the direction of 1 .

Proposition 3. Let p ∈ ∆ d -1 be a probability vector and the set, θ ( p ) , of vectors θ ∈ R d such that softmax( θ ) = p . For the vector θ ⋆ := arg min θ ∈ Θ( p ) ∥ θ ∥ 2 it holds true that,

<!-- formula-not-decoded -->

Proof. The set θ ( p ) takes the form θ ( p ) := { ( θ i = log p i + c ) | c ∈ R } = { θ 0 + c 1 | c ∈ R } for an appropriate choice of θ 0 . Picking an arbitrary θ 0 ∈ θ ( p ) to use as a reference, we can write the problem of minimizing ∥ θ ∥ 2 as,

<!-- formula-not-decoded -->

By the first-order optimality conditions, c = -1 d θ ⊤ 0 1 . Plugging back this for θ ⋆ , we see θ ⋆ = θ 0 -1 d 1 ( θ ⊤ 0 1 ) . We see that, 1 ⊤ θ ⋆ = 1 ⊤ θ 0 -d d θ ⊤ 0 1 = 0 .

Lemma D.5. Assume a fixed 0 &lt; R &lt; ∞ and define the set Θ R to be Θ R := { θ ∈ R d : θ ⊤ 1 = 0 and | θ i -θ j | ≤ 2 R, ∀ i, j ∈ [ d ] } . Then, softmax(Θ R ) is a convex set.

Proof. For any p ∈ ∆ d -1 for which e -2 R ≤ p i p j ≤ e 2 R , ∀ i, j ∈ [ d ] , there exists θ ∈ Θ R such that softmax( θ ) = p . To see this, we apply the logarithm on the inequalities,

<!-- formula-not-decoded -->

A vector χ with entries χ i := log p i clearly implements p . By (3) we see that subtracting κ = max j log p j +min k log p k 2 from all entries yields a softmax-equivalent vector χ ′ i := log p i -κ with -R ≤ χ ′ i ≤ R . Conversely, for any θ ∈ Θ R , e -2 R ≤ softmax i ( θ ) softmax j ( θ ) ≤ e 2 R .

Now, the set defined by the inequalities p ∈ ∆ d -1 , e -2 R ≤ p i p j ≤ e 2 R , is clearly a linear polytope and as such, convex.

## E Gradient Domination

In this section we prove the gradient domination properties of the utilities of the game with different policy parametrizations. Further, for clarity, in place of V x,y τ we will use V τ ( x, y ) ; and in place of V π χ ,π θ τ we will use V τ ( χ, θ ) .

## E.1 Direct Policy Parametrization pPŁ

Lemma E.1. The utility of the game regularized with the weighted bidilated Euclidean regularizer with a weighting scheme defined in Appendix B.4.1, satisfies the pPŁ condition for directly parametrized policies,

<!-- formula-not-decoded -->

Proof. We write the utility function of the regularized game,

<!-- formula-not-decoded -->

For player 1 , we know that the function H eucl τ is strongly convex with an appropriate weighting scheme { w 1 ,s } , (correspondingly { w 2 ,s } for player 2 ),

<!-- formula-not-decoded -->

Strong convexity implies the KŁ condition for µ 1 . In turn, using the bound on the Lipschitz continuity modulus of the map µ 1 ↦→ x ,

<!-- formula-not-decoded -->

Now, we know that α eucl bi = γ min h µ c ( h ) |H| (Corollary B.1). The conclusion follows from Lemma G.2.

## E.2 Softmax Policy Parametrization pPŁ

Lemma E.2. The utility of the game with softmax-parametrized policies satisfies the two-sided pPŁ condition,

<!-- formula-not-decoded -->

where ℓ is the smoothness constant of the softmax-parametrized utility function.

Proof. The main challenge in proving this lemma is the fact that the softmax mapping is not a bijection; this is manifested with a rank-deficient Jacobian of the mapping.

Concretely, from (4), we know that the KŁ-condition holds for the policies. What remains to show is that the KŁ-condition also holds for the parameters χ (and θ ).

For some R &gt; 0 , let X R := softmax( X R ) be the convex set of softmax-parametrized policies where X R := { θ ∈ R A , A = ∑ s A s : χ ⊤ s 1 = 0 , ∀ s ∈ S 1 , | χ s,i -χ s,j | ≤ 2 R, ∀ i, j ∈ [ A s ] } . By overloading notation, let V ( π χ , π θ ) be the loss function of the minimizing player as a function of policies π χ , π θ and V ( χ, θ ) the utility as a function of parameters χ, θ .

Now, we note that the subgradient s ∈ ∂ π χ ( V ( π χ , π θ ) + I X R ( π χ )) that minimizes ∥ s ∥ is such that s ⊤ 1 = 0 . So when picking a norm-minimizing s , it suffices to look at the set of subgradients that are perpendicular to 1 . Further, the chain rule applied on V ( π χ , π θ ) + I X R ( π χ ) yields,

<!-- formula-not-decoded -->

Moreover, we note that by the symmetry of J ( χ ) ,

<!-- formula-not-decoded -->

From inclusion (5) we infer that:

<!-- formula-not-decoded -->

Lemma D.4 provides the bound λ + min ( J ( χ )) ≥ 1 1+( B -1) e 2 R and the conclusion is proven.

## E.3 Mahalanobis-pPŁ

Lemma E.3. The utility of the game with softmax-parametrized policies satisfies the two-sided Mahalanobis pPŁ condition,

<!-- formula-not-decoded -->

Proof. We invoke (6) and the fact that ∥ w ∥ 2 M -1 ≥ λ + min ( M -1 ) ∥ w ∥ 2 for any ⟨ w,v ⟩ = 0 , ∀ v ∈ ker ( M -1 ) . Also, we use Equation ( ε -trunc.) and Assumption 2 to bound λ + min ( M -1 ) . In detail, we know that,

<!-- formula-not-decoded -->

When M := F ( χ, θ ) , it is true that γ 2 min h µ c ( h ) |H| 2 ε ≤ λ max ( F ( χ, θ )) ≤ 1 .

The spectrum of the Fisher Information Matrix With the same arguments used in Lemma D.1, we can conclude that,

- λ min ( F ( χ, θ )) = 0 ;
- λ + min ( F ( χ, θ ) s ) ≥ d ( s ) min a π χ ( a | s ) ;
- d χ,θ ( s ) min s,a π χ ( a | s ) ≤ λ max ( F ( χ, θ ) s ) ≤ d χ,θ ( s ) max a π χ ( a | s ) + 1 .

Hence,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, d χ,θ ( s ) ≥ γ 2 min h µ c ( h ) |H| 2 by Assumption 2.

## E.4 Weak Gradient Domination

We now conclude this section with a proof of the weak gradient domination condition.

Lemma E.4 (Utility Weak Gradient Domination) . Let Γ be an IIEFG satisfying satisfying Assumption 2. Then, it holds true that,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We use (Fatkhullin et al., 2023, Prop. 2) by using the fact that the diameter of the treeplex is at most √ 2 |H|| A | and the fact that the Lipschitz of µ π 1 1 → π 1 is |H| √ A γ . Then, we use the fact that max ∥ y -x ∥≤ 1 ,y ∈X ⟨∇ f ( x ) , x -y ⟩ = min v ∈ ∂ x ( f + I X ( x )) ∥ v ∥ .

## F Gradient Estimators

In this section, we demonstrate that the well-known stochastic gradient estimator, REINFORCE , can be used yield an unbiased estimate of bounded variance of the gradients of the non-regularized and regularized imperfect-information game.

## F.1 A Policy Gradient Theorem

We define a trajectory ξ to be a sequence of consecutive history-action pairs, ξ = ( ( h (1) , a (1) i (1) ) , ( h (2) , a (2) i (2) ) , . . . ) . The length of trajectory ξ is noted as K ξ and it is bounded by the game-tree's height, D ( T ) . We define K to be the set of all trajectories and note that it is finite. After a policy profile, ( π 1 , π 2 ) , is fixed, the probability of each trajectory ξ ∈ K taking place is the product of the probability of each consecutive action,

<!-- formula-not-decoded -->

where i ( k ) denotes the player that takes an action at timestep k .

Lemma F.1. Under the assumption of ( ε -trunc.) , it holds true that the gradient estimator ( REINFORCE ) is unbiased,

<!-- formula-not-decoded -->

and also, its variance is bounded:

<!-- formula-not-decoded -->

where A,B denote the maximum available number of action in any infoset for player 1 and 2 respectively.

Proof. We first show that the gradient estimator is unbiased. Indeed,

<!-- formula-not-decoded -->

The proof for ̂ ∇ y uses an identical argument. We will now proceed to show that the variance of the ( REINFORCE ) gradient estimator is bounded:

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

With the latter expression, proving the desired properties is easier.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma F.2. The variance of ( REINFORCE ) for softmax-parametrized policies is bounded as σ 2 θ , σ 2 χ ≤ 2 D ( T ) 2 .

Proof. We see that ∇ θ log π θ ( a | s ) = e s,a -π θ ( ·| s ) . From then on, ∥∇ θ log π θ ( a | s ) ∥ ≤ √ 2 with probability 1 . Then, the proof follows arguments similar to the previous one.

Policy gradient of the bidilated regularizer Wedefine the policy gradient estimator of the bidilated regularizer, ̂ ∇ x R 1 , as:

<!-- formula-not-decoded -->

We will demonstrate that this gradient estimator is, in fact, both unbiased and enjoys a variance that is bounded. We start with a preliminary proposition about an alternative expression of the regularizer. Proposition 4. For a policy profile π 1 , π 2 , the bidilated regularizer, R 1 can be alternatively defined as:

<!-- formula-not-decoded -->

= ∑ ξ ∈K ( ∇ x P π 1 ,π 2 ( ξ ))   K ξ ∑ k ψ ( π 1 ( s ( k ) ))   + ∑ ξ ∈K P π 1 ,π 2 ( ξ )   ∇ x K ξ ∑ k ψ ( π 1 ( s ( k ) )) = ∑ ξ ∈K ( P π 1 ,π 2 ( ξ ) ∇ x log P π 1 ,π 2 ( ξ ))   K ξ ∑ k ψ ( π 1 ( s ( k ) ))   ︸ ︷︷ ︸ ϖ 1 + ∑ ξ ∈K P π 1 ,π 2 ( ξ )   K ξ ∑ k ∇ x ψ ( π 1 ( s ( k ) ))   ︸ ︷︷ ︸ ϖ 2 For ϖ 1 , let us denote r ξ = ∑ K ξ k ψ ( π 1 ( s ( k ) )) , ϖ 1 = r ξ ∑ ξ ∈K P π 1 ,π 2 ( ξ ) ∇ x log P π 1 ,π 2 ( ξ ) = ∑ ξ ∈K r ξ P ξ ∇ x log P ξ = ∑ ξ ∈K r ξ P ξ K ξ ∑ k =1 ( ∇ x log π i ( k ) ( a ( k ) i ( k ) | h ( k ) ) ) = E ξ ∼ π 1 ,π 2   r ξ K ξ ∑ k =1 ∇ x log π i ( k ) ( a ( k ) i ( k ) | h ( k ) )   = E ξ ∼ π 1 ,π 2   r ξ K ξ ∑ k =1 ∇ x log π 1 ( a ( k ) | s ( k ) )   . For ϖ 2 , we write, ))

<!-- formula-not-decoded -->

We will use similar arguments for the variance in the case of the ( REINFORCE ) gradient estimator.

<!-- formula-not-decoded -->

For ϑ 1 , similar to Lemma F.1, we see that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Whereas, for ϑ 2 ,

<!-- image -->

Finally, we note that when Assumption 2 is followed, then ( REINFORCE ) is also an unbiased estimator of bounded variance (same bounds as previously) of the perturbed version of the game. The reasoning is the same (when a player is exploring the gradient of the probability of an action is zero) and as such we omit it.

## G Optimization Lemmata

Definition 6 (Stationarity Proxies) . Assume a function F : f + I X ( · ) such that f : X → R is ℓ -smooth relative to ∥·∥ M and I X ( · ) is the indicator function of the set X . We define the following stationarity proxies ,

- gradient of the Mahalanobis proximal mapping (MPM) ,

<!-- formula-not-decoded -->

with prox F/ρ ( · ) := arg min x ′ { F ( x ′ ) + ρ 2 ∥ · -x ′ ∥ 2 M } .

- Mahalanobis gradient mapping (MGM) ,

<!-- formula-not-decoded -->

where x + := arg min x ∈X ∥ ∥ x -ρ M -1 ∇ f ( x ) ∥ ∥ 2 M ,

- Mahalanobis forward-backward mapping (MFBM) ,

<!-- formula-not-decoded -->

Lemma G.1. The following properties hold true for the proximal point and the Mahalanobis Moreau envelope,

- ∇ F ρ ( x ) = 1 ρ ( x -ˆ x )
- dist(0 , ∂F (ˆ x )) ≤ ∥∇ F ρ ( x ) ∥ M -1
- F (ˆ x ) ≤ F ρ (ˆ x ) ≤ F ( x )

Proof. The first and last items follow easily from the definition and standard arguments (Davis and Drusvyatskiy, 2018). The middle one uses the optimality condition of ˆ x := prox ρF ( x ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, we conclude that min s ˆ x ∈ ∂F (ˆ x ) ∥ s ˆ x ∥ 2 M -1 ≤ 1 ρ 2 ∥ x -ˆ x ∥ 2 M .

Definition 7 (pPŁ, KŁ) . Let f : X → R be an L -Lipschitz continuous function with ℓ -Lipschitz continuous gradient. Then,

- Proximal Polyak-Łojasiewicz (pPŁ): f is said to satisfy the proximal Polyak-Łojasiewicz condition if ∃ α &gt; 0 s.t.

<!-- formula-not-decoded -->

- Kurdyka-Łojasiewicz (KŁ): f is said to satisfy if ∃ α s.t.

<!-- formula-not-decoded -->

The definitions for the Mahalanobis analogues of pPŁ and KŁ follow straightforward extension.

Lemma G.2. Let f be an ℓ -smooth function relative to ∥·∥ 2 M defined over the convex set X . If f satisfies the (Mahalanobis) KŁ condition with modulus α kl , it also satisfies the (Mahalanobis) pPŁ condition with a modulus of α ppl = α kl 202 .

from which we conclude,

Proof. First, we define F ( x ) := f ( x ) + I X ( x ) , with I X ( · ) being the indicator function. We highlight that since I X ( · ) is convex and f is ℓ -smooth (relative to ∥·∥ 2 M ), then F is ℓ -weakly convex (relative to ∥·∥ 2 M ). This means that the proximal point of the function F/ρ is well defined for any ρ &gt; ℓ .

Now, assume a point x ∈ X and ˆ x := prox F/ρ ( x ) . By assumption, for any ˆ x ∈ X , it holds true that,

<!-- formula-not-decoded -->

where s ˆ x ∈ ∂F (ˆ x ) . The latter implies that for the gradient of the Mahalanobis-Moreau envelope of F , it holds that,

<!-- formula-not-decoded -->

where (7) follows from the fact that F is an ℓ -weakly convex function, and for every v ∈ ∂F ( x ) . To see this, we write that due to weak convexity (relative to ∥·∥ 2 M ) ,

<!-- formula-not-decoded -->

Collecting the terms,

<!-- formula-not-decoded -->

A direct generalization of (Karimi et al., 2016, Lemma 1), implies that for the MFBM and a choice of ρ 1 , ρ 2 &gt; 0 such that ρ 1 &gt; ρ 2 , then D ( x, ρ 1 ) ≥ D ( x, ρ 1 ) . As such, we write,

<!-- formula-not-decoded -->

We can pick ρ = 4 ℓ which then yields,

<!-- formula-not-decoded -->

Observing that α ≤ ℓ in general, we re-write:

<!-- formula-not-decoded -->

Now, from (Fatkhullin and He, 2024, Lemmata 4.1 &amp; 4.2), we know that,

<!-- formula-not-decoded -->

which we plugin in the former inequality to finally conclude that,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark 3. The latter lemma provides a bound that is significantly tighter than the one implied by the analysis found (Karimi et al., 2016, Appendix G) which connects the moduli of the KŁand pPŁconditions.

## G.1 A Variation of the Descent Lemma

The following lemma is a consequence of the three-point identity of the Mahalanobis norm and the smoothness of f .

Lemma G.3 ((J Reddi et al., 2016, Lemma 1)) . Let f : X → R be an ℓ -smooth function relative to ∥·∥ M t and a point x ∈ X ⊆ R d . Also, define the vector v ∈ R d and y ∈ X to be

<!-- formula-not-decoded -->

Then, the following inequality holds true:

<!-- formula-not-decoded -->

Lemma G.4. Let X ⊆ R d be a closed convex set, and let f : X → R be an ℓ -smooth function relative to ∥·∥ M t for some ℓ &gt; 0 . Suppose η &gt; 0 with η ≤ 1 5 ℓ . For any x ∈ X and any vector v ∈ R d , define x + = Proj X , M t ( x -ηv ) . Then the following inequality holds:

<!-- formula-not-decoded -->

Proof. First, we define x + := Proj X , M t ( x -1 ρ M -1 t ∇ f ( x ) ) .

- Invoking ℓ -smoothness relative to ∥·∥ M t of f for x, x + and assuming ρ &gt; 0 with ρ ≥ ℓ ,

<!-- formula-not-decoded -->

- Invoking Lemma G.3 with x = x , y = x + , z = x , v = ∇ f ( x )

<!-- formula-not-decoded -->

- Again, invoking Lemma G.3 but with x = x , y = x + , z = x + , v ,

<!-- formula-not-decoded -->

Combining the previous inequalities as 1 / 3 × (8) and 2 / 3 × (9), and letting 1 /ρ = η ≤ 1 ℓ yields,

<!-- formula-not-decoded -->

Adding (10),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (11) follows from the application of Young's inequality on

<!-- formula-not-decoded -->

- (12) follows by dropping the non-positive terms; non-positivity follows from the choice of the step-size, η ≤ 1 5 ℓ .

## G.2 Min-Max Optimization

Lemma G.5. Let f : X × Y be an ℓ -smooth function, ρ &gt; 0 , two points y, y ′ ∈ Y , and a point x ∈ X . Then, the following inequality holds:

<!-- formula-not-decoded -->

Proof. We define x, x ′ ∈ X to be:

<!-- formula-not-decoded -->

By the definition of D X ( x, ρ ; y ′ ) we write:

<!-- formula-not-decoded -->

Considering the difference D X ( x, ρ ; y ) -D X ( x, ρ ; y ′ ) we see that:

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

where the last line follows from a slight sharpening of the proof of Lemma G.5 (for the function h ( y, x ) = -f ( x, y ) and M = I ). Finally, piecing inequalities (13), (14), and (15) together,

<!-- formula-not-decoded -->

What is left to do is to observe the following, due to Danskin's theorem and ℓ -smoothness of f ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We note that:

- The first inequality follows from the triangle inequality.
- In the second inequality, we applied the reverse triangle inequality.
- The third uses the Cauchy-Schwarz inequality.
- Finally, the second to last uses Lemma G.9 while, the last one, invokes the ℓ -Lipschitz continuity of the gradient.

Lemma G.6. Let f : X × Y be an ℓ -smooth function such that for any x ∈ X , f ( x, · ) satisfies the proximal-PŁ condition with modulus α &gt; 0 . Then, the function Φ( x ) := arg max y ∈Y f ( x, y ) is ℓ ⋆ -smooth, with

<!-- formula-not-decoded -->

Proof. We effectively need to show Lipschitz continuity of the maximizers y ⋆ ( · ) := arg max x and the proof will follow from Danskin's lemma and f 's own ℓ -smoothness. So, we write by the quadratic growth condition,

<!-- formula-not-decoded -->

We denote D Y ( · , ρ ; x ) := -2 ρ arg min z ∈Y {⟨-∇ f ( x, y ) , z -y ⟩ + ρ 2 ∥ y -z ∥ 2 } and by the proximalPŁ condition, we write,

<!-- formula-not-decoded -->

Now, we aim to bound D Y ( y, ℓ ; x ) by ∥ y ⋆ ( x ) -y ⋆ ( x ′ ) ∥ 2 . We observe that,

<!-- formula-not-decoded -->

.

Lemma G.8 ((Kalogiannis et al., 2025, Lemma D.4)) . Let f : X × Y be an ℓ -smooth function. Additionally, assume that f ( · , y ) is α x -pPŁ for all y ∈ Y and f ( x, · ) is α y -pPŁ for all x ∈ X . Then, the function Φ( x ) := max y ∈Y f ( x, y ) is α x -pPŁ .

## G.3 Regarding the Mahalanobis Distance

Throughout, we will refer to a positive-semidefinite matrix M ∈ R d × d and its Moore-Penrose pseudoinverse M † ∈ R d × d . Although in general a PSD matrix cannot define a distance, restricting x, y ∈ R d such that ( x -y ) ∈ ker( M ) ⊥ , then ∥ x -y ∥ 2 M := ( x -y ) ⊤ M ( x -y ) satisfies all properties of a metric. As we shall see, this seemingly arbitrary assumption is satisfied for every pair of consecutive updates of natural policy gradient steps. The matrix rank-deficient matrix we are interested in is policy gradient Fisher information matrix, and for softmax policy parametrization, it is rank deficient in the direction 1 ∈ R d . Further, the gradient ∇ f ( x ) as

Proposition 5. Assume that θ 0 = 0 . Also, let v ⊤ t 1 = 0 , ∀ t ∈ { 1 , 2 , 3 , . . . } . Then, setting θ t +1 = θ t -η M † v t guarantees that,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proposition 6. Let Θ ⊆ R d be a convex compact set. Assume that θ 0 = 0 . Also, let v ⊤ t 1 = 0 , ∀ t ∈ { 1 , 2 , 3 , . . . } . Then, the following minimization problem has a unique solution,

<!-- formula-not-decoded -->

Further, it is equivalent to the minimization problem,

<!-- formula-not-decoded -->

Proof. It is clear that, for θ, χ ∈ Θ , θ ⊤ 1 = χ ⊤ 1 = 0 the function ∥ θ ∥ 2 M , ∥ θ -χ ∥ 2 M is strongly convex in θ . Hence, both problems attain a unique minimum.

For the first problem, the first-order optimality conditions for the write,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Noting that, ( θ + -( θ t -η M † v t ) ) ⊤ 1 = 0 and ( θ -θ ) ⊤ 1 = 0 ,

<!-- formula-not-decoded -->

But, since the matrix M is PSD and the last inequality is a condition on the sign of the inner-product, it can be written equivalently as,

<!-- formula-not-decoded -->

The final inequality, is exactly the first-order optimality condition for the second minimization problem.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The latter inequality follows from (16) and completes the proof.

Lemma G.7 ((Kalogiannis et al., 2025, Lemma D.3)) . Let f : X × Y be an ℓ -smooth function. Additionally, assume that f ( · , y ) is α x -pPŁ for all y ∈ Y and f ( x, · ) is α y -pPŁ for all x ∈ X . Then, it holds true that:

<!-- formula-not-decoded -->

## G.4 Alternating Mirror Descent using a Changing Mahalanobis DGF

## G.4.1 Supporting Lemmata

Lemma G.9. Let v 1 , v 2 be vectors in R d and X ⊆ R d be a compact convex set and a scalar η &gt; 0 . Also, let points x + 1 , x + 2 ∈ X such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, it holds true that:

.

## Smoothness Relative to the Mahalanobis Distance

Proposition 7. Let f be a function ℓ -smooth relative to the ℓ 2 -distance. Then, it is ℓ λ min ( M t ) -smooth relative to the Mahalanobis distance induced by a positive definite matrix M t .

Proof. We will merely demonstrate that if f is ℓ -smooth (relative to ℓ 2 -distance) it is also the case that:

<!-- formula-not-decoded -->

For one direction we use vector norm equivalence to write:

<!-- formula-not-decoded -->

Correspondingly for the opposite direction:

<!-- formula-not-decoded -->

## G.4.2 Convergence of Alternating Descent-Ascent

Through, we consider this section, we consider the iteration following scheme,

<!-- formula-not-decoded -->

We make a standard assumption on the gradient estimators and their second moments.

Assumption 3 (Unbiased Gradient Estimators and Bounded Second Moments) . For all iterations t , the gradient estimators ˆ g x ( x t , y t ) and ˆ g y ( x t , y t ) satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Theorem G.1. Let f : X × Y → R an ℓ -smooth function and bounded in the interval ∆ f . Further, assume X , Y to be two convex sets with Euclidean diameters, diam( X ) , diam( Y ) . Moreover, assume that f satisfies a two-sided pPŁ condition with moduli α x for all y ∈ Y and α y for any x ∈ X . Additionally, let (ˆ g x , ˆ g y ) be an inexact stochastic gradient oracle satisfying Assumption 3.

- When M · t = I , after T iterations of (Alt-GDA) with a choice of stepsizes η x = α 2 y 960 ℓ 3 and η y = 1 5 ℓ , it holds true that:

<!-- formula-not-decoded -->

where, ∆ f := max x ∈X ,y ∈Y f ( x, y ) -min x ∈X ,y ∈Y f ( x, y ) and c 1 , c 2 ∈ O (1) .

- For a general positive definite choice of M · t (Mahalanobis metric), after T iterations of (Alt-GDA) with a choice of stepsizes η x = α 2 y 960 ℓ 3 λ 2 max and η y = 1 5 ℓλ max , it holds true that: E Φ( x T ) -Φ ⋆ + 1 10 ( E Φ( x T ) -E f ( x T , y T ))

<!-- formula-not-decoded -->

where, ∆ f := max x ∈X ,y ∈Y f ( x, y ) -min x ∈X ,y ∈Y f ( x, y ) , λ max := max t λ max ( M -1 · ,t ) and c 1 , c 2 ∈ O (1) .

Proof. To prove convergence we will use the Lyapunov function L ( x, y ) := U ( x, y ) + cW ( x, y ) with U ( x, y ) := E [Φ( x ) -Φ ⋆ ] , W ( x, y ) := E [Φ( x ) -f ( x, y )] and c &gt; 0 . Intuitively, U ( x, y ) measures x 's success in achieving the unique minmax value Φ ⋆ , while W ( x, y ) measures y 's success in achieving to be a best-response to its corresponding x . We begin with some preliminary work to ultimately setup a recursion on L .

Descent on Φ In order to guarantee descent, by Lemma G.6, Proposition 7, and Lemma G.4, it suffices to pick η x ≤ 1 5 ℓλ max ( M x,t ) . Then, we can write,

<!-- formula-not-decoded -->

Equivalently, subtracting Φ ⋆ from both sides yields,

<!-- formula-not-decoded -->

Further, a simple re-arrangement reads,

<!-- formula-not-decoded -->

Ascent on f ( x, · ) Requiring that η y ≤ 1 5 ℓλ max ( M y,t ) , (Proposition 7 and Lemma G.4), we write:

<!-- formula-not-decoded -->

Invoking Lemma G.8, multiplying by -1 , and adding Φ( x t +1 ) will yield,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As a reminder, Φ is a pPL function relative to the Mahalanobis distance induced by M t by Lemma G.8.

Upper bound on the descent of f ( · , y ) From the smoothnes of f :

<!-- formula-not-decoded -->

Re-arranging to isolate f ( x t , y t ) -f ( x t +1 , y t ) ,

<!-- formula-not-decoded -->

Putting the pieces together for Φ( x t ) -f ( x t , y t ) , we get:

<!-- formula-not-decoded -->

Decrease in the Lyapunov function We consider the Lyapunov function L ( x, y ) := U ( x, y ) + cW ( x, y ) with U ( x, y ) := E [Φ( x ) -Φ ⋆ ] , W ( x, y ) := E [Φ( x ) -f ( x, y )] and shorthand notation U t = U ( x t , y t ) , W t = W ( x t , y t ) . Here U t measures primal suboptimality via the PL condition on Φ , while W t captures the dual gap Φ( x t ) -f ( x t , y t ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (17) uses the fact that a ≤ | a -b | + b for a = D X ( x t , 1 /η x ; y t ) , b = D Φ X ( x t , 1 /η x ) . This decomposition isolates the term |D X -D Φ X | , which can then be controlled using the Mahalanobis continuity lemma in y .
- (18) uses Lemma G.5 and Danskin's theorem; this yields a bound |D X - D Φ X | ≤ 3 λ max ( M -1 x,t ) ℓ 2 ∥ y t -y ⋆ ( x t ) ∥ 2 .

<!-- formula-not-decoded -->

We then collect the coefficients in front of U t and W t in the previous inequality into ϖ 1 and ϖ 2 , respectively, so that the Lyapunov recursion can be written compactly as U t +1 + cW t +1 ≤ ϖ 1 U t + cϖ 2 W t + noise. I.e. ,

<!-- formula-not-decoded -->

For ϖ 1 , letting c = 1 / 10

<!-- formula-not-decoded -->

For ϖ 2 , we distinguish two cases relevant to our algorithms, M t = I and a general choice of M t .

- For M t = I , it holds that λ max ( M -1 · ,t ) = 1 , and α qg = α y . So we write

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let α 2 y η y η x ℓ 2 = 192 . Then, choosing η y = 1 5 ℓ yields η x = α 2 y 960 ℓ 3 .

- For a general choice of M t , let λ max := max { λ max ( M -1 x,t ) , λ max ( M -1 y,t ) } and α y ← min { α qg , α y } ,

<!-- formula-not-decoded -->

Similarly, we need to set

<!-- formula-not-decoded -->

This in turn yields η y = 1 5 λ max ℓ and η x = α y 2 960 ℓ 3 λ 2 max .

Remark 4. In fact, M t is allowed to be positive semidefinite as long as the gradient throuhgout the iterations is in the kernel of M t .

## H Convergence Analysis

## H.1 Direct Policy Parametrization

Theorem H.1. With direct policy parametrization and the Euclidean bidilated regularizer, alternating policy-gradient algorithm attains a last-iterate ϵ -Nash equilibrium in

<!-- formula-not-decoded -->

using batches of poly ( 1 ϵ , 1 γ , |H| , A, B, 2 D ( T ) , 1 min h µ c ( h ) , |S 1 | , |S 2 | ) trajectory samples at each step.

Proof. The proof follows as an application of Theorem G.1. In a central role lies Lemma E.1, which provides a two-sided pPŁ condition for the regularized game under direct policy parametrization, while in a supportive one the smoothness lemmata of the value function and the Euclidean bidilated regularizer when the policy is directly parametrized.

First, we relate equilibria of the regularized, truncated, exploration-perturbed game to equilibria of the original game. An ϵ -NE of the regularized game is an ϵ ′ -NE of the unregularized game where

<!-- formula-not-decoded -->

The term contains the optimization error ϵ , the regularization error (controlled by τ ), the truncation error (controlled by ε through the minimum action probability), and the exploration-induced error (controlled by γ ). To make each contribution O ( ϵ ) we choose

- γ = Θ( ϵ ) ,
- τ = Θ ( ϵ max i ∈{ 1 , 2 } |S i | 2 D ( T ) ) ,

<!-- formula-not-decoded -->

We now instantiate Theorem G.1. By Lemma E.1 the utility of the regularized game satisfies the two-sided pPŁ condition with moduli

<!-- formula-not-decoded -->

Combining the smoothness of the value function with that of the Euclidean bidilated regularizer (Lemmata B.4 and B.7 ) yields an overall smoothness constant

<!-- formula-not-decoded -->

The stochastic gradients used by Alt-RegPG are given by the REINFORCE estimator together with the gradient estimators for the bidilated regularizer; by Lemma F.1 and the analysis of Appendix F.1 they are unbiased and have bounded per-trajectory variance

<!-- formula-not-decoded -->

If each update averages a mini-batch of M i.i.d. trajectories, ̂ ∇ x = 1 M ∑ M m =1 ̂ ∇ ( m ) x and ̂ ∇ y = 1 M ∑ M m =1 ̂ ∇ ( m ) y , then the averaged estimators have variances

<!-- formula-not-decoded -->

with per-trajectory bounds σ 2 x ≤ A 2 D ( T ) 2 /ε and σ 2 y ≤ B 2 D ( T ) 2 /ε . Substituting these into Theorem G.1, the stochastic error terms are controlled (up to absolute constants) by σ 2 x / ( Mα x ) and ℓ σ 2 y / ( Mα x α 2 y ) . Requiring each to be at most ϵ leads to the condition

<!-- formula-not-decoded -->

Using the explicit forms of α x , α y from Lemma E.1 and the per-trajectory variance bounds from Lemma F.1, this can be summarized as choosing

<!-- formula-not-decoded -->

Writing S := max {|S 1 | , |S 2 |} and using the tunings γ = Θ( ϵ ) , τ = Θ ( ϵ/ ( S 2 D ( T ) ) ) , and ε = Θ ( ϵ/ ( SA ) ) from above, together with

<!-- formula-not-decoded -->

a direct substitution yields the explicit bounds

<!-- formula-not-decoded -->

For small ϵ the second constraint dominates, so it is sufficient to choose

<!-- formula-not-decoded -->

which spells out the precise dependence of the mini-batch size on ϵ , A , B , D ( T ) , |S 1 | , |S 2 | , |H| , and min h µ c ( h ) .

Under these conditions, Theorem G.1 prescribes the concrete stepsizes

<!-- formula-not-decoded -->

owing to the symmetric pPŁ moduli α x = α y from Lemma E.1. The resulting duality-gap decay is exp ( -α x α 2 y 960 ℓ 3 T ) , so driving the deterministic term below ϵ requires

<!-- formula-not-decoded -->

where ∆ f is the payoff range appearing in Theorem G.1. Substituting the smoothness estimate from Corollary B.1 and Lemma B.7,

<!-- formula-not-decoded -->

yields the following dependencies on the game parameters:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, substituting the choices of γ, τ, ε from above into the expression for T yields

<!-- formula-not-decoded -->

as claimed in the statement of the theorem.

## H.2 Softmax Policy Parametrization

Theorem H.2. Alternating policy-gradient algorithm with softmax policy parametrization and the entropic bidilated regularizer converges in expectation in the last-iterate to an ϵ -Nash equilibrium after a number of iterations T given by

<!-- formula-not-decoded -->

using batches of poly ( 1 ϵ , 1 γ , |H| , A, B, 2 D ( T ) , 1 min h µ c ( h ) , |S 1 | , |S 2 | ) trajectory samples at each step.

Proof. The theorem follows as a corollary of Theorem G.1. By Lemma E.2, the regularized game under softmax parametrization satisfies the two-sided pPŁ condition with moduli

<!-- formula-not-decoded -->

up to absolute constants. An ϵ -NE for the regularized game is also an ϵ ′ -NE for the unregularized game where

<!-- formula-not-decoded -->

Then, we need to tune:

- γ = Θ( ϵ ) ;

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We recall the smoothness parameter of the softmax-parametrized regularized utility function is

<!-- formula-not-decoded -->

by combining the Lipschitz bounds on the utility and the weighted entropic bidilated regularizer (Lemma B.8). Then, from Theorem G.1 we tune,

<!-- formula-not-decoded -->

where we set ℓ := ℓ softmax , and α x , α y are the softmax pPŁ moduli of the two players. Invoking Lemma E.2 for player 2 yields

<!-- formula-not-decoded -->

and therefore, prior to relating R to the truncation level ε ,

<!-- formula-not-decoded -->

Finally, using the explicit relationship between R and the minimum action probability (so that ( 1 + ( B -1) e 2 R ) 4 can be expressed as a polynomial in 1 /ε ) and simplifying constants leads to the following convenient. And, subsequently,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, plugging the explicit expressions for α x , α y from above into the generic bound T = Θ ( |H| 9 ℓ 3 ( 1+( A -1) e 2 R ) 2 ( 1+( B -1) e 2 R ) 4 τ 3 (min h ∈H µ c ( h )) 3 γ 9 log 1 ε ) yields the precise parameter dependence

<!-- formula-not-decoded -->

Using the relationship between R and the minimum action probability to upper-bound ( 1 + ( A -1) e 2 R ) and ( 1 + ( B -1) e 2 R ) by polynomials in 1 /ε and then substituting the tunings of γ, τ, ε we obtain an explicit dependence on the game parameters. Writing S := max {|S 1 | , |S 2 |} and using the smoothness estimate ℓ softmax together with the truncation relation ε , a straightforward calculation yields

<!-- formula-not-decoded -->

As in the direct-parametrization case, we now quantify the effect of stochastic gradients. For softmax-parametrized policies, Lemma F.2 shows that the REINFORCE estimator (combined with

the estimator for the entropic bidilated regularizer) is unbiased and has bounded variance per-trajector with σ 2 χ , σ 2 θ ≤ Θ ( D ( T ) 2 + τ 2 D ( T ) ) = O ( D ( T ) 2 ) . We will control the stochastic error using mini-batches.

Substituting these into Theorem G.1 with the softmax pPŁ moduli from Lemma E.2,

<!-- formula-not-decoded -->

the stochastic error terms are controlled by

<!-- formula-not-decoded -->

Requiring each to be at most ϵ gives the condition

<!-- formula-not-decoded -->

The second term dominates for small ϵ , so it suffices to enforce

<!-- formula-not-decoded -->

To relate the dependence on R to the truncation level, we use Lemma D.4, which implies that if the minimum action probability under the softmax parametrization is at least ε , then 1+( A -1) e 2 R ≤ 1 ε , and 1 + ( B -1) e 2 R ≤ 1 ε , so

<!-- formula-not-decoded -->

Combining this with ℓ -smoothness from above yields the bound

<!-- formula-not-decoded -->

Finally, we denote S := max {|S 1 | , |S 2 |} and substitute the terms γ, τ, ε , together with the definition of ℓ softmax , a direct calculation shows that it is sufficient to choose

<!-- formula-not-decoded -->

## H.3 Natural Policy Gradient

## H.3.1 The Fisher Information Matrix

<!-- formula-not-decoded -->

The matrix F ( χ ) is a blog diagonal matrix with its ( s, s ) -block being the matrix:

<!-- formula-not-decoded -->

Its pseudo-inverse, F † , is again a block-diagonal matrix, with an ( s, s ) -block,

<!-- formula-not-decoded -->

Interestingly, the matrix Z := F † J softmax ( χ ) is a block-diagonal matrix with entries 1 d χ,θ ( s ) I |A s |×|A s | on diagonal ( s, s ) -block.

The spectrum of the Fisher Information Matrix With the same arguments used in Lemma D.1, we can conclude that,

- λ min ( F ( χ, θ )) = 0 ;
- λ + min ( F s ( χ, θ )) ≥ d χ,θ ( s ) min a π χ ( a | s ) ;

<!-- formula-not-decoded -->

Hence,

- λ + min ( F ( χ, θ )) ≥ min s,a d χ,θ ( s ) π χ ( a | s ) ;

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem H.3. Alternating natural policy-gradient algorithm with softmax policy parametrization and the entropic bidilated regularizer converges in expectation in the last-iterate to an ϵ -Nash equilibrium after a number of iterations T , that is

<!-- formula-not-decoded -->

.

Proof. This theorem is again an application of Theorem G.1, now in its Mahalanobis form. For natural policy gradient, the updates are mirror-descent steps with a Mahalanobis metric induced by the Fisher information matrices, so we run Alt-GDA with M x,t = F χ ( χ t , θ t ) and M y,t = F θ ( χ t , θ t ) .

By Lemma E.3, for a general positive-semidefinite metric matrix M the game satisfies a two-sided Mahalanobis pPŁ condition with moduli

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

When we specialize M to the Fisher information matrices, the spectrum bounds in the previous subsection together with Assumption 2 and the truncation assumption imply

<!-- formula-not-decoded -->

and hence, over the image of the Fisher matrices,

<!-- formula-not-decoded -->

Substituting these bounds for λ max ( M -1 ) into the expressions above yields Mahalanobis pPŁ moduli

<!-- formula-not-decoded -->

The Mahalanobis version of Theorem G.1 prescribes stepsizes (up to constants)

<!-- formula-not-decoded -->

where ℓ is the Euclidean smoothness constant of the objective and λ max := max t λ max ( M -1 · ,t ) . We use the Euclidean smoothness constant ℓ := ℓ softmax as in the softmax-parametrized policy-gradient case; writing Σ := max i ∈{ 1 , 2 } | Σ i | and S := max {|S 1 | , |S 2 |} ,

<!-- formula-not-decoded -->

where the final inequality uses the tuning of τ and ϵ &lt; 1 . By the Smoothness Relative to the Mahalanobis Distance (as used in the proof of Theorem G.1), we have

<!-- formula-not-decoded -->

and hence the stepsizes can be expressed as

<!-- formula-not-decoded -->

As in the softmax-parametrized policy-gradient case, we relate equilibria of the truncated, regularized, exploration-perturbed game to equilibria of the original game. An ϵ -NE of the perturbed game is an ϵ ′ -NE of the unregularized game with

<!-- formula-not-decoded -->

so, as before, we choose

- γ = Θ( ϵ ) ;

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining these tunings with the expressions for ˜ α x , ˜ α y , the smoothness ℓ softmax , the bound on λ max , and the generic iteration bound T = Θ ( ℓ 3 softmax λ 2 max / (˜ α x ˜ α 2 y ) log(1 /ε ) ) and using Lemma D.4 to relate R to the truncation level ε yields

<!-- formula-not-decoded -->

## I Proximity of Projections

In this section, we consider that the update rules:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and demonstrate that (19) and (20) are sufficiently close. For brevity we consider only the maximizer's updates and drop the minimizer's variables from the notation. I.e., our goal is to bound ∥ θ kl -θ F ∥ . We begin by defining the two objective functions that each projection optimizes,

<!-- formula-not-decoded -->

where lse( θ ) := log ∑ e θ i . Then, we write,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Further, for the gradient of L kl we write,

<!-- formula-not-decoded -->

Now, from stationarity of the optimal, for a v in the normal cone of Θ R at θ kl ,

<!-- formula-not-decoded -->

Therefore, we can bound the stationarity of θ kl for the objective of L F ( · ) :

<!-- formula-not-decoded -->

where we use:

- L F is the Lipschitz continuity modulus of the operator norm of F ( · ) ,
- Proposition 8,
- α = min θ ∈ Θ R ′ λ + min ( F ( θ )) , and
- the fact that when τ is tuned as dictated in Theorem H.3:

<!-- formula-not-decoded -->

Proposition 8. Consider the update rules (19) and (20). It is the case that:

<!-- formula-not-decoded -->

where α = min θ ∈ Θ R ′ λ + min ( F ( θ )) and R ′ = R + η √ SB .

Proof. We begin by stating a useful fact.

Fact 1. Let lse be the function lse( θ ) := log ∑ i e θ i . Then, softmax( θ ) = ∇ lse( θ ) . Further, L kl ( · ) is strictly convex on Θ ⊥ := { θ ∈ R d | θ ⊤ 1 = 0 } and its Bregman divergence is:

<!-- formula-not-decoded -->

Further, it is the case that:

<!-- formula-not-decoded -->

The arguments for this fact can be found in (Gao and Pavel, 2017). By standard calculations we can see that:

<!-- formula-not-decoded -->

with α := λ + min ( F ( θ )) . From Fact 1 we can conclude that,

<!-- formula-not-decoded -->

where we let Θ R ′ for R ′ = R + η √ SB . From 1 2 -smoothness of L kl ( · ) we write,

<!-- formula-not-decoded -->

Since θ kl = arg min θ ∈ Θ R L kl ( θ ) , it follows that

<!-- formula-not-decoded -->

which concludes the claim.