## Certifying Concavity and Monotonicity in Games via Sum-of-Squares Hierarchies

Vincent Leon 1 Iosif Sakos 2 Ryann Sim 2 Antonios Varvitsiotis 2 , 3 , 4

1 UIUC 2 SUTD 3 NUS CQT 4 Archimedes/Athena RC

leon18@illinois.edu

{iosif\_sakos, ryann\_sim, antonios}@sutd.edu.sg

## Abstract

Concavity and its refinements underpin tractability in multiplayer games, where players independently choose actions to maximize their own payoffs which depend on other players' actions. In concave games, where players' strategy sets are compact and convex, and their payoffs are concave in their own actions, strong guarantees follow: Nash equilibria always exist and decentralized algorithms converge to equilibria. If the game is furthermore monotone , an even stronger guarantee holds: Nash equilibria are unique under strictness assumptions. Unfortunately, we show that certifying concavity or monotonicity is NP-hard, already for games where utilities are multivariate polynomials and compact, convex basic semialgebraic strategy sets-an expressive class that captures extensive-form games with imperfect recall. On the positive side, we develop two hierarchies of sum-of-squares programs that certify concavity and monotonicity of a given game, and each level of the hierarchies can be solved in polynomial time. We show that almost all concave/monotone games are certified at some finite level of the hierarchies. Subsequently, we introduce the classes of SOS-concave/monotone games, which globally approximate concave/monotone games, and show that for any given game we can compute the closest SOS-concave/monotone game in polynomial time. Finally, we apply our techniques to canonical examples of extensiveform games with imperfect recall.

## 1 Introduction

Game theory models settings where multiple decision-makers independently maximize personal objectives that depend on the actions of others. Formally, a game with n players is modeled by assigning to each player i a strategy set X i ⊂ R m i and a utility function u i ( x i , x -i ) , where x i ∈ X i is player i 's action and x -i denotes the actions of all other players. The interdependence of players' utilities makes analyzing the collective behavior of such systems both rich and challenging.

The canonical solution concept in game theory is the Nash equilibrium [42], a product distribution over strategies in which no player can unilaterally deviate to improve their utility, given the strategies of the other players. While Nash equilibria are guaranteed to exist in finite normal-form games, several key questions must be addressed in games with continuous, infinite action spaces: Do Nash equilibria always exist? If one exists, is it unique (thereby avoiding the equilibrium selection problem)? And crucially, can it be computed efficiently using distributed algorithms?

Extensive research has identified concavity , and its refinements, as key enablers in addressing these fundamental questions. In the setting of games, this entails assuming that each strategy set X i is compact and convex. Furthermore, concavity of players' utilities can manifest in at least two

distinct forms. First, we have the class of concave games , where for each player i , the function x i ↦→ u i ( x i , x -i ) is continuous and concave for every fixed x -i . Second, we have the more restrictive class of monotone games , where utility functions are smooth and the (negative) concatenated gradient map

<!-- formula-not-decoded -->

is a monotone operator . Every monotone game is concave, but the converse does not necessarily hold. Concave and monotone games were first studied in the seminal work of Rosen [49], who established that Nash equilibria always exist in these class of games, significantly extending the guarantees of classical results such as von Neumann's minimax theorem [62] for two-player zerosum (normal-form) games, Nash's aforementioned result for finite normal-form games, and Sion's minimax theorem for two-player convex-concave games [53]. Moreover, Rosen showed that strictly monotone games have a unique Nash equilibrium. At the same time, concave/monotone games have been extensively studied due to their inherent expressibility - they have been used to model various fundamental settings in economics and optimization, including but not limited to resource allocation [50], Cournot competition [17] and robust power management [67].

Finally, concave and monotone games have also received considerable attention in the context of equilibrium computation. A substantial body of work has analyzed decentralized dynamics that achieve strong performance guarantees in concave games [17, 56, 28, 57]. Most recently, [20] established a O (polylog T ) regret bound for uncoupled learning dynamics in general convex games, extending classical results beyond structured settings such as normal-form and extensive-form games. In monotone games, decentralized dynamics have also been shown to converge in the last-iterate sense to Nash equilibria [38, 8, 9, 25, 18]. Moreover, in strictly monotone games, one can guarantee last-iterate convergence to the unique Nash equilibrium [67, 7, 51, 54], further underscoring the computational tractability of this class.

However, despite their favorable properties, it is not clear how to efficiently verify concavity and monotonicity. For instance, establishing that u i is concave over X i requires checking that the Hessian ∇ 2 x i u i ( x i , x -i ) is negative semidefinite for every x i ∈ X i and every x -i ∈ X -i , an infinite family of conditions. In view of this, a fundamental computational challenge arises:

## Is it possible to efficiently verify that a game is concave or monotone?

Our Techniques and Contributions. Our starting point is to demonstrate that deciding whether a game is concave or monotone is computationally hard, cf. Theorem 3.1. We establish this hardness result for the class of polynomial games [14, 30, 45] in which each player's utility is a multivariate polynomial and players' strategy sets are compact convex basic semialgebraic sets - that is, sets defined by polynomial equality and inequality constraints. This class is highly expressive, capturing for instance extensive-form games with imperfect recall [46]. Our hardness result builds on recent advances in polynomial optimization [4, 2], which show that unless P = NP , there is no polynomial-time (or even pseudo-polynomial-time) algorithm that can decide whether a multivariate polynomial of degree four (or any higher even degree) is globally convex. This result presents a challenge for game theorists. On one hand, concave/monotone games are expressive classes of games that capture many applications and have desirable equilibriation properties. However, verifying their concavity/monotonicity is hard for the class of polynomial games over convex compact basic semialgebraic sets.

Motivated by this, we next seek to identify tractable sufficient conditions for concavity and monotonicity, as well as special classes of games for which these properties can be efficiently certified. Our approach is based on the observation that, since polynomial games are smooth, these properties can be verified via the positive semidefiniteness of the Hessian and the symmetrized Jacobian, respectively. As a concrete example, a polynomial game is concave if, for each player i , the (negative) Hessian of the utility function is positive semidefinite for all x i ∈ X i and x -i ∈ X -i . By the variational characterization of positive semidefiniteness, this is equivalent to requiring that p i ( x, y ) := -y ⊤ ∇ 2 x i u i ( x i , x -i ) y ≥ 0 , for all x ∈ × n i =1 X i and y ∈ B , where B ⊂ R m i is the unit ball. Since u i is a polynomial and X is a closed basic semialgebraic set, the function p i ( x, y ) is a polynomial over a semialgebraic domain.

Although testing nonnegativity of polynomials is, in general, computationally hard [41], a powerful approach from polynomial optimization, pioneered in [45, 35], is to seek a sum-of-squares (SOS) decomposition that certifies nonnegativity. This idea has also been recently used to develop certifi-

cates for the global convexity of polynomials [5]. Searching for an SOS decomposition of bounded degree can be done in polynomial time via semidefinite programming.

In our setting, the application of the SOS framework leads to a hierarchy of increasingly stronger sufficient conditions for certifying concavity or monotonicity, each of which can be checked in polynomial time via semidefinite programming. At the ℓ -th level of the hierarchy, we check whether p i ( x, y ) admits a degreeℓ SOS decomposition over X × B . While the SOS framework does not eliminate the inherent hardness of the problem, it offers a practical trade-off: by relaxing the problem into a sequence of SDPs, one obtains a hierarchy of increasingly tight sufficient conditions with provable convergence in the limit. The main limitation is that the size of the resulting SDPs grows with the level of the hierarchy.

Leveraging these ideas, our main contributions are summarized below:

- We construct a hierarchy of optimization problems that provide increasingly strong certificates of monotonicity/concavity for polynomial games over compact, convex basic semialgebraic sets (cf. Theorem 3.2). Furthermore, each level of the hierarchy can be solved in polynomial time via semidefinite programming.
- We show that for every strictly monotone/strictly concave game, a certificate is always found at a finite level of the hierarchy (cf. Statement 4 in Theorem 3.2). More importantly, we show that for almost all monotone/concave games, such a certificate can be obtained at some finite level of the hierarchy (cf. Theorem 3.3).
- We define subclasses of monotone/concave polynomial games over compact, convex basic semialgebraic sets, called ℓ -SOS-monotone (resp. ℓ -SOS-concave) games, for which monotonicity (resp. concavity) can be certified by the ℓ -th level of the hierarchy (cf. Definition 4.1). We show that this class of games globally approximates the class of monotone (resp. concave) games, and importantly, given any polynomial game, the closest ℓ -SOSmonotone (resp. ℓ -SOS-concave) game can be computed by solving a single SDP (cf. Theorem 4.3).
- We apply our proposed methods to several canonical and new examples of extensive-form games with imperfect recall (cf. Section 6). We show examples of how our hierarchies can be used to verify monotonicity/concavity in these games, as well as to find the closest ℓ -SOS-monotone (resp. ℓ -SOS-concave) game with respect to an appropriate norm.

## 2 Preliminaries

## 2.1 Polynomial Games over Semialgebraic Sets

We consider an n -player continuous game denoted by G = G ( J n K , X , u ) . For each player i ∈ J n K , we denote their set of actions by X i ⊆ R m i and their payoff function by u i : X → R , where the set of joint actions X def = X 1 × · · · × X n is a compact, convex set . Each player i selects an action x i ∈ X i . We denote by x def = ( x 1 , . . . , x n ) the joint action profile of all players, and by X def = X 1 ×··· × X n ⊆ R m their joint action space, where m def = m 1 + · · · + m n . We also denote by u def = ( u 1 , . . . , u n ) the ensemble of the players' payoff functions.

In this work, we focus on games G where u 1 , . . . , u n are polynomial functions, and X is a basic semialgebraic set. In particular, we assume that

<!-- formula-not-decoded -->

where g 1 , . . . , g m g , h 1 , . . . , h m h ∈ R [ x ] .

We refer to d = max { deg( u 1 ) , . . . , deg( u n ) , deg( g 1 ) , . . . , deg( g m g ) , deg( h 1 ) , . . . , deg( h m h ) } as the degree of the game. For each n, d ∈ N , we use G ( n,d ) to denote the set of n -player, d -degree polynomial games over X .

G ( n,d ) is isomorphic to R M , where M def = n · ( m + d d ) . In particular, we define the isomorphism

<!-- formula-not-decoded -->

where vec( u i ) is the coefficient vector of u i for each i ∈ J n K . Throughout the paper, we also consider the topology on G ( n,d ) induced by the norm

<!-- formula-not-decoded -->

When necessary, we use the convention x = ( x i , x -i ) to distinguish the action x i of player i in a joint action x ∈ X from the actions of the rest of the players. In a similar vein, we use X -i to denote the joint action space of all players except player i .

A fundamental equilibrium concept in game theory is the Nash equilibrium (NE) [42], which are strategy profiles from which players have no incentive to unilaterally deviate. Concretely, a joint action profile x ∗ ∈ X is a NE of a game G if

<!-- formula-not-decoded -->

## 2.2 Sum-of-Squares Optimization

Given a closed basic semialgebraic set X as in (1), the quadratic module Q ( X ) of X is a set of functions defined as

<!-- formula-not-decoded -->

where Σ[ x ] ⊂ R [ x ] is the set of sum-of-squares (SOS) polynomials on variables x , i.e., the set of all polynomials of the form

<!-- formula-not-decoded -->

Furthermore, for all d ≥ 0 , we define Q d ( X ) as the restriction of Q ( X ) to Putinar-type decompositions of degree at most 2 d given by deg( σ 0 ) , . . . , deg( σ m g ) , deg( p 1 ) , . . . , deg( p m h ) ≤ 2 d .

As part of the analysis in Section 3.1, we require that the quadratic module Q ( X ) is Archimedean, a property formally given for completeness in the following definition.

Definition 2.1. A quadratic module Q ( X ) is called Archimedean if there exists N ∈ N such that

<!-- formula-not-decoded -->

## 2.3 Concave &amp; Monotone Games

In this section, we introduce two important subclasses of continuous games, concave games and monotone games, both defined in [49]. These classes are particularly significant due to their implications for the existence and uniqueness of Nash equilibria.

Definition 2.2 (Concave Games) . A game G is concave if, for all players i ∈ J n K , the function x i ↦→ u i ( x i , x -i ) is concave for every fixed x -i ∈ X -i . Furthermore, if G is polynomial then it is concave if and only if the Hessian matrices of the payoff functions u 1 , . . . , u n with respect to x 1 , . . . , x n , respectively, are negative semidefinite, i.e.,

<!-- formula-not-decoded -->

Rosen [49] proved that a Nash equilibrium exists in every concave game, thereby extending Nash's equilibrium existence result to a broad class of continuous games. He also identified an important subclass of concave games with additional structural properties, which are now typically referred to as monotone games [38].

Definition 2.3 (Monotone Games) . A game G is monotone if the negative of its concatenated gradient mapping, referred to by Rosen as the pseudogradient,

<!-- formula-not-decoded -->

is a monotone operator on X , i.e.,

<!-- formula-not-decoded -->

Furthermore, if G is polynomial, it is well-known that (cf. [48, Proposition 12.3]) it is monotone if and only if the symmetrized Jacobian matrix with respect to v ( x ) is negative semidefinite, i.e.,

<!-- formula-not-decoded -->

where, for all x ∈ X , J ( x ) is the Jacobian matrix of v ( x ) (see Appendix B for a definition of J ( x ) ).

It is easy to verify that if a game G is monotone, then it is also concave; however, the converse does not hold. We now turn our attention to the strict versions of these definitions.

Definition 2.4 (Strictly Concave/Monotone Games) . Consider a polynomial game G over a basic semialgebraic set X . Then, G is strictly concave over X if

<!-- formula-not-decoded -->

Furthermore, G is strictly monotone over X if SJ is negative definite on X , i.e.,

<!-- formula-not-decoded -->

Finally, Rosen [49] also studied the class of diagonally strictly concave games, defined as those for which equality in (10) holds if and only if x = x ′ .

Several important connections and inclusions between the aforementioned game classes we study are summarized in Figure 1. The proofs for these inclusions follow directly from the definitions of the games and standard results from [48]. Of particular interest to us is the fact that, if G is strictly monotone (i.e., it satisfies (13)), then G is both diagonally strictly concave and strictly concave. Moreover, [49] also proved that diagonally strictly concave games admit a unique Nash equilibrium.

Figure 1: Connections and inclusions among the game classes we study.

<!-- image -->

## 3 Certifying Concavity and Monotonicity in Polynomial Games

As discussed in the introduction, concave and monotone games are highly expressive and have strong theoretical properties, including the existence of Nash equilibria, uniqueness under strictness conditions, and convergence of distributed dynamics to equilibrium. Given these favorable features, a natural question arises: can concavity/monotonicity be efficiently certified? In this section, we investigate this question in the setting of polynomial games with semialgebraic strategy sets.

To investigate hardness of deciding concavity/monotonicity, we leverage recent breakthroughs in polynomial optimization, particularly recent works on the complexity of certifying convexity of polynomials. Specifically, it has been shown in [4] that deciding whether a quartic (multivariate) polynomial is globally convex is NP-hard. Subsequently, [2] demonstrated that determining whether a cubic polynomial is convex over a box is also NP-hard. Building on these results, the starting point of this work is the observation that verifying whether a polynomial game belongs to the class of concave or monotone games is also NP-hard. This result is given below, and proven in Appendix C.1.

Theorem 3.1. Let G ( J n K , X , u ) be a polynomial game over a compact convex basic semialgebraic set. If for some player i , u i is a polynomial of degree at least 3 with respect to x i ∈ X i , verifying whether G is concave or monotone is strongly NP-hard.

Motivated by the hardness result, it is crucial to identify tractable sufficient conditions for concavity and monotonicity, which gives rise to non-trivial subclasses of concave and monotone games. This can be achieved by using the technique of sum-of-squares optimization, together with the positive semidefiniteness of the Hessian or the symmetrized Jacobian matrix of the game. Throughout the remainder of the paper, for brevity we focus only on the class of monotone games . Analogous results hold for concave games with minor modifications, and we describe them in Section 5.

## 3.1 Sum-of-Squares Certificates for Concavity &amp; Monotonicity

We introduce a hierarchy of increasingly strong sufficient conditions for certifying concavity and monotonicity, based on SOS certificates for the associated quadratic forms defined by the Hessian and the symmetrized Jacobian matrices of G . The starting point for this observation is that, for any fixed x ∈ X , and considering the symmetrized Jacobian, we have

<!-- formula-not-decoded -->

Consequently, using the Rayleigh-Ritz theorem , it follows that G is monotone if and only if

<!-- formula-not-decoded -->

where B def = { y ∈ R m ∣ ∣ y T y = 1 } . The crucial observation here is that the function ( x, y ) ↦→ y T SJ ( x ) y is a polynomial in x, y , since the Jacobian matrix SJ ( x ) is polynomial in x . Moreover, X and B are compact basic semialgebraic sets. Therefore, max x ∈X λ max ( SJ ( x ) ) can be written as the solution to the following polynomial maximization problem:

<!-- formula-not-decoded -->

Finally, although polynomial optimization is in general NP-hard, the solution to a polynomial optimization problem, i.e., max x ∈X λ max ( SJ ( x ) ) can be approximated via the SOS framework. This is formally stated in the main theorem of this section, the proof of which is given in Appendix C.2:

Theorem 3.2. Let G ( J n K , X , u ) be a polynomial game over a compact, convex basic semialgebraic set X . Assume the quadratic module Q ( X ) is Archimedean. For any ℓ ∈ N consider the hierarchy of SOS optimization problems:

<!-- formula-not-decoded -->

where Q ℓ ( X × B ) denotes the restriction of Q ( X × B ) to polynomials of degree at most ℓ . Then, the following statements are true:

- 1) For all ℓ , we have that SOS ℓ ( G ) ≥ max x ∈X λ max ( SJ ( x ) ) .
- 2) The sequence ( SOS ℓ ( G ) ) ℓ ≥ 0 is nonincreasing.
- 3) lim ℓ →∞ SOS ℓ ( G ) = max x ∈X λ max ( SJ ( x ) ) .
- 4) G is strictly monotone if, and only if, there exists some finite level ℓ such that SOS ℓ ( G ) &lt; 0 .
- 5) For any level ℓ , the program in (17) can be formulated as an semidefinite program (SDP) and solved in polynomial time.

Theorem 3.2 shows how a sequence of SDPs, which can be solved efficiently (Statement 5), can be used to approximate max x ∈X λ max ( SJ ( x ) ) , and therefore certify whether G is monotone. In particular, Statements 1 to 3 guarantee that SOS ℓ ( SJ ) , for ℓ ≥ 0 , gives progressively tighter upper bounds for max x ∈X λ max ( SJ ( x ) ) . If for any finite ℓ we obtain SOS ℓ ( SJ ) ≤ 0 , it follows that max x ∈X λ max ( SJ ( x ) ) ≤ 0 , and therefore G is monotone. Additionally, if at some ℓ we get SOS ℓ ( SJ ) &lt; 0 , it follows that max x ∈X λ max ( SJ ( x ) ) &lt; 0 , and therefore G is strictly monotone.

Importantly, Statement 3 guarantees that whenever G is monotone, even if no finite ℓ exists such that SOS ℓ ( SJ ) ≤ 0 , the sequence ( SOS ℓ ( SJ ) ) ℓ ≥ 0 nonetheless converges (asymptotically) to a non-positive value. Moreover, whenever G is not only monotone but also strictly monotone, by Statement 4 we are guaranteed the existence of a finite ℓ . In fact, it turns out that generic monotone polynomial games over compact, convex semialgebraic sets are almost always strictly monotone. In particular, in the following theorem we show that for all G of degree at least 2 , the set of polynomial monotone games that are not strictly monotone form a set with zero Lebesgue measure.

Theorem 3.3. For almost all monotone games, monotonicity can be certified at a finite level ℓ of the SOS hierarchy (17) , i.e., SOS ℓ ( G ) ≤ 0 . Concretely, for all d ≥ 2 , the set of monotone polynomial games of degree d over a compact basic semialgebraic set X that are not strictly monotone has zero Lebesgue measure.

The proof of this result is given in Appendix C.3. At this point, we have shown that the monotonicity of almost all polynomial monotone games G over a compact, convex semialgebraic set can be certified by a solution SOS ℓ G ( SJ ) at some finite level ℓ G of the SOS hierarchy in (17). However, for an arbitrary game G , the required level ℓ G may be large . Thus, in practice, certifying the monotonicity of G via the SOS hierarchy in (17) may be computationally infeasible. To reflect this limitation, in the following section, we introduce and study a subclass of monotone games called ℓ -SOS-monotone games, for which monotonicity can be certified in polynomial time via semidefinite programming.

## 4 SOS-Concave &amp; SOS-Monotone Games

Motivated by the convergence guarantees of the SOS hierarchy established in Theorem 3.2, in this section, we define and analyze a subclass of polynomial monotone games over a compact, convex basic semialgebraic set for which monotonicity can be certified at some fixed level ℓ of the SOS hierarchy. These are games whose monotonicity can be verified in polynomial time with respect to the level ℓ . We refer to such games as ℓ -SOS-monotone.

Definition 4.1 ( ℓ -SOS-Monotone Game) . Consider a polynomial game G ∈ G ( n,d ) over a compact, convex basic semialgebraic set X . For all ℓ ≥ 0 , we say that G is ℓ -SOS-monotone if

<!-- formula-not-decoded -->

We denote the set of ℓ -SOS-monotone games by G sosm( n,d,ℓ ) . Furthermore, we say that G is SOSmonotone if there exists ℓ ∈ N such that G is ℓ -SOS-monotone.

The following theorem is an immediate consequence of Statement 4 in Theorem 3.2 and the measure-theoretic result in Theorem 3.3:

Theorem 4.2. For all d ≥ 2 , the set of monotone polynomial games of degree d over a compact, convex basic semialgebraic set X that are not SOS-monotone has zero Lebesgue measure.

Next, we show that for every ℓ ≥ 0 , the set of ℓ -SOS-monotone games is a global approximator to the set of monotone games, i.e., SOS-monotone games are dense in monotone games. In particular, given some polynomial game G ∗ over a convex, compact basic semialgebraic set, we can compute the closest ℓ -SOS-monotone game G in polynomial time. Moreover, since SOS-monotone games are dense in monotone games, as ℓ → ∞ , the projections G of G ∗ in the set of ℓ -SOS-monotone games converge to the closest monotone game to G ∗ ; not just the closest SOS-monotone game. The proof of the following theorem can be found in Appendix C.4.

Theorem 4.3. For all d ≥ 2 , the set of SOS-monotone games of degree d over a compact basic semialgebraic set X is dense in the set of monotone games of degree d over X . Furthermore, given any polynomial game G ∗ ∈ G ( n,d ) over X , and any fixed ℓ ≥ 0 , we can compute the closest ℓ -SOSmonotone game to G ∗ by the program

<!-- formula-not-decoded -->

which can be formulated as an SDP.

In Theorem 4.3 and throughout our experiments, distance between games is measured via the norm ∥·∥ in Eq. (3). Beyond the aforementioned norm, the optimization framework in Theorem 4.3 extends naturally to any function ∥·∥ that measures deviations on G ( n,d ) , whose epigraph is semidefinite representable and for which ∥ G k -G ∥ → 0 as k → ∞ , for all monotone games G and some sequence of SOS-monotone games.

For example, we give another example of a valid deviation operator. Let G quad ( J n K , X , u quad ) be the SOS-monotone game with the payoff functions u quad i ( x ) = -∥ x i ∥ 2 2 , for all i ∈ J n K . The gauge is given by

<!-- formula-not-decoded -->

where for all ε ≥ 0 , G + ε · G quad denotes the polynomial game G ′ ( J n K , X , u ′ ) with the payoff functions u ′ i ( x ) = u i ( x ) + ε · u quad i ( x ) . Furthermore, the corresponding SDP is given by the ℓ -th level of the SOS hierarchy in Theorem 3.2.

## 5 Modifications for the Certification of Concavity

The results in Sections 3.1 and 4 can be equivalently stated in relation to concave polynomial games over a compact, convex basic semialgebraic set, subject to minor modifications. By definition, a polynomial game G over a compact, convex basic semialgebraic set X is concave if and only if the Hessian matrices H u i ( x ) are negative semidefinite, for all x ∈ X and i ∈ J n K . Furthermore, G is strictly concave if and only if H u i ( x ) are all negative definite. For each i , consider the SOS hierarchy ( SOS i,ℓ ( G ) ) ℓ ≥ 0 given in Eq. (17), where we substitute SJ ( x ) with H u i ( x ) . Then, the SOS-based hierarchy

<!-- formula-not-decoded -->

provides analogous guarantees as in the case of monotone games. In particular, Theorem 3.2 and Theorem 4.3, as well as Definition 4.1 can be written analogously with respect to the SOS-based hierarchy in Eq. (21). Meanwhile, Theorems 3.3 and 4.2 can be written for concave games directly without further modifications. For completeness, we provide the definition of ℓ -SOS-Concave games here:

Definition 5.1 ( ℓ -SOS-Concave Game) . Consider a polynomial game G ∈ G ( n,d ) over a compact, convex basic semialgebraic set X . For all ℓ ≥ 0 , we say that G is ℓ -SOS-concave if

<!-- formula-not-decoded -->

## 6 Application: Extensive-Form Games with Imperfect Recall

As described concisely in [20], the class of concave games has many modern applications. Similarly, monotone games have been studied extensively due to their desirable equilibrium properties (see e.g. [38, 19, 9] and references therein). In this section, we highlight extensive-form games (EFGs) with imperfect recall, leveraging the fact that they can be viewed as polynomial games over compact, convex basic semialgebraic sets. We also utilize our theoretical results and proposed game classes to study canonical examples of these games. We will defer further discussion on applications to economic markets to Appendix F.

The study of extensive-form or sequential games is arguably as classical as that of normal-form games. The reader is referred to [44, Sections II and III] for a review of standard concepts. Moreover, for the sake of notational brevity and readability, we defer formal definitions of EFGs and related concepts to Appendix D. One of the most important results in extensive-form games is Kuhn's theorem [33], which establishes a connection between mixed strategies and behavioral strategies in EFGs with perfect recall (wherein players effectively never forget the history of information sets visited and actions played). Relaxing the perfect recall assumption results in games where players can forget prior information, which introduces additional computational challenges.

The canonical example of an imperfect recall game is that of the absent-minded taxi driver (Figure 3), introduced in [46]. Furthermore, [46] showed that the expected utility of any player in an EFG with imperfect recall can be written as a polynomial, where each variable is associated with an information set (i.e., a collection of decision nodes which a player cannot distinguish between). In

Figure 2: A Game with No Nash Equilibria

<!-- image -->

C

1

P1

E

4

Figure 3: The Absent-minded Taxi Driver

particular, these utilities define an n -variable polynomial game G ( J n K , X , u ) over the simplex. For clarity, we derive the corresponding polynomial utility function of the game in Figure 3. Since the player has imperfect recall and cannot remember if they are in the first or second decision node, they will select a distribution over { C, E } to be applied to both decision nodes. If the player selects C with probability x 1 and E with probability x 2 , then their expected payoff is given by x 2 1 +4 x 1 x 2 .

In general, though, a Nash equilibrium might not exist in EFGs with imperfect recall. For instance, the game in Figure 2 was introduced by [64] and does not have a Nash equilibrium. Several recent works have further established hardness results for deciding the existence of or computing NE in EFGs with imperfect recall [31, 59, 60, 23]. Theorem C.1 additionally guarantees the hardness of verifying concavity/monotonicity of EFGs with imperfect recall over simplex action sets.

## 6.1 Experimental Methodology &amp; Results

Our results in Sections 3.1 and 4 motivate two lines of investigation: verifying monotonicity/concavity and computing the closest SOS-monotone/concave game.

Examples 1 &amp; 2. First, we use the SDP hierarchy in Eq. (17) to certify SOS-monotonicity of the absent-minded taxi driver game in Figure 3. Next, we use the program in Eq. (19) to find an SOSmonotone game G which is closest to the zero-sum game in Figure 2 (which does not have a NE in behavioral strategies), in the sense of the norm defined in Eq. (3). By further enforcing that the new game has to be zero-sum and the monomial basis of the game is maintained, we are able to obtain the closest SOS-monotone game G ′ given by:

<!-- formula-not-decoded -->

and u ′ 2 = -u ′ 1 . The distance between the two games is ∥ G -G ′ ∥ = 10 , and since the modified game is SOS-monotone, it has a NE in behavioral strategies.

The above examples are applied to canonical EFGs with imperfect recall-going forward, we utilize our framework to study larger EFGs and aim to study the scalability of our approach. For brevity, full experimental details are deferred to Appendix E.

Example 3: A degree-4 strictly monotone general-sum game. [4, Theorem 2.3] introduces a method to construct (strictly) convex polynomials of degree 4. Using this method, we construct a two-player game with degree-4 polynomial utility functions that is strictly concave. P1 and P2 choose their actions ( x 1 , x 2 ) and ( y 1 , y 2 ) from a two-dimensional simplex respectively. By running our hierarchy of SOS optimization problems in Eq. (17) for monotonicity, we obtain an objective value -1 at level 4 , thus certifying that the game is strictly monotone and also SOS-monotone.

Example 4: A degree-5 zero-sum game. We create a two-player zero-sum EFG with imperfect recall as shown in Figure E.4, where the payoffs on each leaf are for P1. In this example, P1 makes four moves before P2 makes a move, and P1 is absent-minded. By letting x denote the probability that P1 chooses L and y denote the probability that P2 chooses l , we obtain the payoffs for P1 and P2 as follows:

<!-- formula-not-decoded -->

P1

C

E

0

and u 2 = -u 1 . We run our program in Eq. (19) to find the closest SOS-monotone game. Two additional constraints are imposed to retain the properties of the original EFG: The modified game has to be zero-sum, and the information structure of the original EFG has to be preserved. To preserve the information structure of the game, we select the monomial basis for the new payoff functions to be precisely the monomial basis that can appear in the original game. The following modified payoff functions are found:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Example 5: A degree-8 general-sum game. Weconstruct a two-player EFG with imperfect recall where P1 makes six moves before P2 makes two moves. There is one information set for P1 and one information set for P2. P1 has three actions to choose from with probability x 1 , x 2 , and 1 -x 1 -x 2 , respectively. P2 also has three actions to choose from with probability y 1 , y 2 , and 1 -y 1 -y 2 , respectively. Hence, the game tree has nine layers, including the root and the leaves, and the payoff functions are degree-8 polynomials with monomial basis

<!-- formula-not-decoded -->

where ⊗ is the tensor product. The size of the monomial basis is 168. We do not restrict the EFG to be zero-sum, but instead randomly generate the payoff functions for P1 and P2 by independently sampling the coefficient of each monomial in the basis from a uniform distribution on [ -1 , 1] .

We run our program in Eq. (19) to find the closest SOS-monotone game with the additional constraint that the information structure of the original EFG has to be preserved, i.e. the new payoff functions have to be polynomials with the same monomial basis. As in Example 3, we defer the full payoff functions of the game to Appendix E.

On Scalability. A natural limitation of our framework is scalability-while SDPs can be solved with arbitrary accuracy in polynomial time using interior point methods, they are among the most expensive convex relaxations to solve. In practice, 'SOS problems involving degree-4 or 6 polynomials are currently limited, roughly speaking, to a handful or a dozen variables' [3]. We compare the compute times of our proposed hierarchies when applied to the larger-scale examples above. Indeed, while the SOS hierarchies in Examples 3 and 4 can be solved in ≈ 0 . 052 and ≈ 0 . 009 seconds respectively, the much larger program for Example 5 took ≈ 37 . 53 seconds to solve using a standard, off-the-shelf solver. This further motivates future work on scaling our approach using existing methods in the literature [3, 37, 66, 39, 27]. Our code 1 is implemented using the SumOfSquares package for Julia [36, 63] and run on a MacBook Air with 16 GB RAM.

## 7 Discussion

In this paper, we have shown that verifying concavity and monotonicity in polynomial games is in general NP-hard. For polynomial games over compact, convex basic semialgebraic sets, we utilize SOS techniques to construct SDP hierarchies that can certify concavity and monotonicity. Moreover, we show that almost all concave/monotone games are strict, and thus can be certified at a finite level of the respective hierarchy. Finally, we introduced ℓ -SOS-concave and ℓ -SOS-monotone games, which are certified at some fixed level ℓ of the respective SOS hierarchy. This leads to an application for EFGs of imperfect recall, where we are able to find the closest (in terms of an appropriate norm) SOS-concave/monotone game to a canonical EFG which has no Nash equilibria. In addition, in light of the experiments in Section 6.1, our work motivates the design of application-specific programs which can find close concave/monotone games while also maintaining structural properties of the original game.

Broader Impact. While our results are primarily theoretical, we acknowledge that there could be potential societal consequences of our work, none of which we feel must be specifically highlighted.

1 Code used to generate the experiments in Section 6 can be found in our github repo.

Acknowledgements This work is supported by the MOE Tier 2 Grant (MOE-T2EP20223-0018), Ministry of Education Singapore (SRG ESD 2024 174), the CQT++ Core Research Funding Grant (SUTD) (RS-NRCQT-00002), the National Research Foundation Singapore and DSO National Laboratories under the AI Singapore Programme (Award Number: AISG2-RP-2020-016), and partially by Project MIS 5154714 of the National Recovery and Resilience Plan, Greece 2.0, funded by the European Union under the NextGenerationEU Program. The authors also thank anonymous reviewers for their insightful feedback during the review process.

## References

- [1] Lukáš Adam, Rostislav Horˇ cík, Tomáš Kasl, and Tomáš Kroupa. Double oracle algorithm for computing equilibria in continuous games. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 35, pages 5070-5077, 2021.
- [2] Amir Ali Ahmadi and Georgina Hall. On the complexity of detecting convexity over a box. Mathematical Programming , 182(1):429-443, 2020.
- [3] Amir Ali Ahmadi and Anirudha Majumdar. DSOS and SDSOS optimization: more tractable alternatives to sum of squares and semidefinite optimization. SIAM Journal on Applied Algebra and Geometry , 3(2):193-230, 2019.
- [4] Amir Ali Ahmadi, Alex Olshevsky, Pablo A Parrilo, and John N Tsitsiklis. NP-hardness of deciding convexity of quartic polynomials and related problems. Mathematical Programming , 137:453-476, 2013.
- [5] Amir Ali Ahmadi and Pablo A Parrilo. On the equivalence of algebraic conditions for convexity and quasiconvexity of polynomials. In 49th IEEE Conference on Decision and Control (CDC) , pages 3343-3348. IEEE, 2010.
- [6] Kenneth J. Arrow and Gerard Debreu. Existence of an equilibrium for a competitive economy. Econometrica , 22(3):265-290, 1954.
- [7] Wenjia Ba, Tianyi Lin, Jiawei Zhang, and Zhengyuan Zhou. Doubly optimal no-regret online learning in strongly monotone games with bandit feedback. Operations Research , 2025.
- [8] Yang Cai, Argyris Oikonomou, and Weiqiang Zheng. Finite-time last-iterate convergence for learning in multi-player games. Advances in Neural Information Processing Systems , 35:33904-33919, 2022.
- [9] Yang Cai and Weiqiang Zheng. Doubly optimal no-regret learning in monotone games. In International Conference on Machine Learning , pages 3507-3524. PMLR, 2023.
- [10] Xi Chen and Shang-Hua Teng. Spending is not easier than trading: On the computational equivalence of Fisher and Arrow-Debreu equilibria. In International Symposium on Algorithms and Computation , pages 647-656. Springer, 2009.
- [11] Mandar Datar. On stability and learning of competitive equilibrium in generalized Fisher market models: A variational inequality approach. arXiv preprint arXiv:2501.07265 , 2025.
- [12] Nikhil R Devanur, Christos H Papadimitriou, Amin Saberi, and Vijay V Vazirani. Market equilibrium via a primal-dual-type algorithm. In The 43rd Annual IEEE Symposium on Foundations of Computer Science, 2002. Proceedings. , pages 389-395. IEEE, 2002.
- [13] Nikhil R Devanur, Christos H Papadimitriou, Amin Saberi, and Vijay V Vazirani. Market equilibrium via a primal-dual algorithm for a convex program. Journal of the ACM (JACM) , 55(5):1-18, 2008.
- [14] Melvin Dresher, Samuel Karlin, and Lloyd S Shapley. Polynomial games. Contributions to the Theory of Games I , 24:161-180, 1950.
- [15] Edmund Eisenberg. Aggregation of utility functions. Management Science , 7(4):337-350, 1961.

- [16] Edmund Eisenberg and David Gale. Consensus of subjective probabilities: The pari-mutuel method. The Annals of Mathematical Statistics , 30(1):165-168, 1959.
- [17] Eyal Even-Dar, Yishay Mansour, and Uri Nadav. On the convergence of regret minimization dynamics in concave games. In Proceedings of the Forty-First Annual ACM Symposium on Theory of Computing , pages 523-532, 2009.
- [18] Francisco Facchinei and Christian Kanzow. Generalized Nash equilibrium problems. Annals of Operations Research , 175(1):177-211, 2010.
- [19] Francisco Facchinei and Jong-Shi Pang. Finite-dimensional variational inequalities and complementarity problems . Springer, 2003.
- [20] Gabriele Farina, Ioannis Anagnostides, Haipeng Luo, Chung-Wei Lee, Christian Kroer, and Tuomas Sandholm. Near-optimal no-regret learning dynamics for general convex games. Advances in Neural Information Processing Systems , 35:39076-39089, 2022.
- [21] Yuan Gao and Christian Kroer. First-order methods for large-scale market equilibrium computation. Advances in Neural Information Processing Systems , 33:21738-21750, 2020.
- [22] Yuan Gao and Christian Kroer. Infinite-dimensional Fisher markets and tractable fair division. Operations Research , 71(2):688-707, 2023.
- [23] Hugo Gimbert, Soumyajit Paul, and B. Srivathsan. A bridge between polynomial optimization and games with imperfect recall. In AAMAS '20 , page 456-464. International Foundation for Autonomous Agents and Multiagent Systems, 2020.
- [24] Denizalp Goktas and Amy Greenwald. Convex-concave min-max Stackelberg games. Advances in Neural Information Processing Systems , 34:2991-3003, 2021.
- [25] Noah Golowich, Sarath Pattathil, and Constantinos Daskalakis. Tight last-iterate convergence rates for no-regret learning in multi-player games. Advances in Neural Information Processing Systems , 33:20766-20778, 2020.
- [26] Baining Guo. On the difficulty of deciding the convexity of polynomials over simplexes. International Journal of Computational Geometry &amp; Applications , 6(02):227-229, 1996.
- [27] Qiushi Han, Chenxi Li, Zhenwei Lin, Caihua Chen, Qi Deng, Dongdong Ge, Huikang Liu, and Yinyu Ye. A low-rank ADMM splitting approach for semidefinite programming. arXiv preprint arXiv:2403.09133 , 2024.
- [28] Yu-Guan Hsieh, Kimon Antonakopoulos, and Panayotis Mertikopoulos. Adaptive learning in continuous games: Optimal regret bounds and convergence to Nash equilibrium. In Conference on Learning Theory , pages 2388-2422. PMLR, 2021.
- [29] Kamal Jain, Vijay V. Vazirani, and Yinyu Ye. Market equilibria for homothetic, quasi-concave utilities and economies of scale in production. In Proceedings of the Sixteenth Annual ACMSIAM Symposium on Discrete Algorithms , SODA '05, page 63-71, USA, 2005. Society for Industrial and Applied Mathematics.
- [30] Samuel Karlin. Mathematical Methods and Theory in Games, Programming and Economics. Vol. 2: The Theory of Infinite Games . Addison-Wesley Publishing Company, 1959.
- [31] Daphne Koller and Nimrod Megiddo. The complexity of two-person zero-sum games in extensive form. Games and Economic Behavior , 4(4):528-552, 1992.
- [32] Tomáš Kroupa and Tomáš Votroubek. Multiple oracle algorithm to solve continuous games. In International Conference on Decision and Game Theory for Security , pages 149-167. Springer, 2022.
- [33] Harold W Kuhn. Extensive games and the problem of information. Contributions to the Theory of Games , 2(28):193-216, 1953.
- [34] Rida Laraki and Jean B Lasserre. Semidefinite programming for min-max problems and games. Mathematical Programming , 131:305-332, 2012.

- [35] Jean Bernard Lasserre. Moments, Positive Polynomials and Their Applications . Imperial College Press, 2009.
- [36] Benoît Legat, Chris Coey, Robin Deits, Joey Huchette, and Amelia Perry. Sum-of-squares optimization in Julia. In The First Annual JuMP-dev Workshop , 2017.
- [37] Anirudha Majumdar, Georgina Hall, and Amir Ali Ahmadi. Recent scalability improvements for semidefinite programming with applications in machine learning, control, and robotics. Annual Review of Control, Robotics, and Autonomous Systems , 3(1):331-360, 2020.
- [38] Panayotis Mertikopoulos and Zhengyuan Zhou. Learning in games with continuous action sets and unknown payoff functions. Mathematical Programming , 173:465-507, 2019.
- [39] Renato DC Monteiro, Arnesh Sujanani, and Diego Cifuentes. A low-rank augmented Lagrangian method for large-scale semidefinite programming based on a hybrid convexnonconvex approach. arXiv preprint arXiv:2401.12490 , 2024.
- [40] Theodore S Motzkin and Ernst G Straus. Maxima for graphs and a new proof of a theorem of Turán. Canadian Journal of Mathematics , 17:533-540, 1965.
- [41] Katta G. Murty and Santosh N. Kabadi. Some NP-complete problems in quadratic and nonlinear programming. Mathematical Programming , 39(2):117-129, June 1987.
- [42] John Nash. Non-cooperative games. The Annals of Mathematics , 54(2):286-295, September 1951.
- [43] Jiawang Nie and Xindong Tang. Nash equilibrium problems of polynomials. Mathematics of Operations Research , 49(2):1065-1090, 2024.
- [44] Martin J Osborne and Ariel Rubinstein. A course in game theory . MIT press, 1994.
- [45] Pablo A Parrilo. Polynomial games and sum of squares optimization. In Proceedings of the 45th IEEE Conference on Decision and Control , pages 2855-2860. IEEE, 2006.
- [46] Michele Piccione and Ariel Rubinstein. On the interpretation of decision problems with imperfect recall. Games and Economic Behavior , 20(1):3-24, 1997.
- [47] R. Tyrrell Rockafellar. Convex Analysis . Princeton University Press, 1970.
- [48] R Tyrrell Rockafellar and Roger J-B Wets. Variational Analysis . Springer Berlin, Heidelberg, 1998.
- [49] J Ben Rosen. Existence and uniqueness of equilibrium points for concave n-person games. Econometrica: Journal of the Econometric Society , pages 520-534, 1965.
- [50] Tim Roughgarden and Florian Schoppmann. Local smoothness and the price of anarchy in splittable congestion games. Journal of Economic Theory , 156:317-342, 2015.
- [51] William H Sandholm. Population games and deterministic evolutionary dynamics. In Handbook of Game Theory with Economic Applications , volume 4, pages 703-778. Elsevier, 2015.
- [52] Abraham Seidenberg. A new decision method for elementary algebra. Annals of Mathematics ,
19. 60(2):365-374, 1954.
- [53] Maurice Sion. On general minimax theorems. Pac. J. Math. , 8:171-176, 1958.
- [54] Sylvain Sorin and Cheng Wan. Finite composite games: Equilibria and dynamics. Journal of Dynamics and Games , 3(1):101-120, 2016.
- [55] Noah D Stein, Asuman Ozdaglar, and Pablo A Parrilo. Separable and low-rank continuous games. International Journal of Game Theory , 37(4):475-504, 2008.
- [56] Noah D Stein, Pablo A Parrilo, and Asuman Ozdaglar. Correlated equilibria in continuous games: Characterization and computation. Games and Economic Behavior , 71(2):436-455, 2011.

- [57] Gilles Stoltz and Gábor Lugosi. Learning correlated equilibria in games with compact sets of strategies. Games and Economic Behavior , 59(1):187-208, 2007.
- [58] Alfred Tarski. A decision method for elementary algebra and geometry. Journal of Symbolic Logic , 17(3), 1952.
- [59] Emanuel Tewolde, Caspar Oesterheld, Vincent Conitzer, and Paul W Goldberg. The computational complexity of single-player imperfect-recall games. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence , pages 2878-2887, 2023.
- [60] Emanuel Tewolde, Brian Hu Zhang, Caspar Oesterheld, Manolis Zampetakis, Tuomas Sandholm, Paul Goldberg, and Vincent Conitzer. Imperfect-recall games: Equilibrium concepts and their complexity. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence , pages 2994-3004, 2024.
- [61] Lou Van den Dries. Tame topology and o-minimal structures , volume 248. Cambridge University Press, 1998.
- [62] John von Neumann. Zur theorie der gesellschaftsspiele. Mathematische Annalen , 100(1):295320, 1928.
- [63] Tillmann Weisser, Benoît Legat, Chris Coey, Lea Kapelevich, and Juan Pablo Vielma. Polynomial and moment optimization in Julia and JuMP. In JuliaCon , 2019.
- [64] Philipp C Wichardt. Existence of Nash equilibria in finite extensive form games with imperfect recall: A counterexample. Games and Economic Behavior , 63(1):366-369, 2008.
- [65] Jiayi Zhao, Denizalp Goktas, and Amy Greenwald. Fisher markets with social influence. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 37, pages 5900-5909, 2023.
- [66] Yang Zheng, Giovanni Fantuzzi, and Antonis Papachristodoulou. Sparse sum-of-squares (SOS) optimization: A bridge between DSOS/SDSOS and SOS optimization for sparse polynomials. In 2019 American Control Conference (ACC) , pages 5513-5518. IEEE, 2019.
- [67] Zhengyuan Zhou, Panayotis Mertikopoulos, Aris L Moustakas, Nicholas Bambos, and Peter Glynn. Robust power management via learning and game design. Operations Research , 69(1):331-345, 2021.

## A Additional Related Work

Polynomial Games and Semidefinite Programming. Initially introduced and studied by [14], polynomial games were viewed as a bridge between finite and continuous games. Although [14] characterized and proved the existence of equilibria in these games, providing computational guarantees for general polynomial games has proven to be a challenging task. [45, 34] used semidefinite programming methods to find the value of two-player zero-sum polynomial games, and similar techniques apply to separable games (where utilities take a sum-of-products form) [55]. Recently, [43] also used semidefinite programming techniques to solve for Nash equilibria in n -player polynomial games, or otherwise detect the nonexistence of equilibria. Beyond polynomial games, oracle-based methods have been used to approximately solve continuous games [1, 32].

## B Additional Preliminaries

Given a game G with utility functions u i ( x ) , the standard Jacobian is defined as follows:

<!-- formula-not-decoded -->

## C Omitted Proofs from Main Text

## C.1 Proof of Theorem 3.1

As mentioned earlier, several works have studied the hardness of verifying convexity in multivariate polynomials [4, 2]. We state the main theorem for hardness of verifying convexity over a box here for completeness:

Theorem C.1 ([2, Theorem 2.3]) . Deciding whether a polynomial of degree at least 3 is convex over a box D def = { x ∈ R n | α i ≤ x i ≤ β i i ∈ J n K } , for α, β ∈ R n , is strongly NP-hard.

Using this result, we are able to prove NP-hardness of verifying concavity and monotonicity in polynomial games.

Theorem 3.1. Let G ( J n K , X , u ) be a polynomial game over a compact convex basic semialgebraic set. If for some player i , u i is a polynomial of degree at least 3 with respect to x i ∈ X i , verifying whether G is concave or monotone is strongly NP-hard.

Proof. To prove the statement for concave games, let p : R m → R be a polynomial of degree 3 or higher, and let D def = { x ∈ R m | α j ≤ x j ≤ β j j ∈ J m K } , for α, β ∈ R n and α j ≤ β j , be a box over R m . Consider a two-player polynomial game G ( J 2 K , D × D , u ) , where u 1 ( x ) = p ( x 1 ) , and u 2 ( x ) = 0 for all x ∈ D × D . Then, since u 2 is concave over D , the game G is concave if and only if p is concave over D . Thus, from Theorem C.1 it follows that verifying whether G is concave is strongly NP-hard.

To prove the statement for monotone games, let p : R m → R be a polynomial of degree 3 or higher, and let D def = { x ∈ R m | α j ≤ x j ≤ β j j ∈ J m K } , for α, β ∈ R n and α j ≤ β j , be a box over R m . Note that the description of D satisfies Eq. (1), and hence D is a convex and compact semialgebraic set. Define X 1 def = [ α 1 , β 1 ] × · · · × [ α m -1 , β m -1 ] , and X 2 def = [ α m , β m ] . Clearly, X 1 × X 2 = D . Consider a two-player polynomial game G ([2] , X 1 ×X 2 , u ) , where u 1 ( x ) = u 2 ( x ) = p ( x ) for all x ∈ X . Then, the game G is monotone if and only if the operator ( -∇ x 1 u 1 ( x ) , -∇ x 2 u 2 ( x ) ) = -∇ p ( x ) is monotone over X 1 ×X 2 , or equivalently p is concave over D . Thus, from Theorem C.1 it follows that verifying whether G is a monotone game is strongly NP-hard.

As a direct consequence of the hardness of verifying the concavity of degree-4 polynomials over the simplex [26, 4], we can obtain the following NP-hardness result for verifying concavity/monotonicity in polynomial games with simplex action sets.

Theorem C.2. Let G ( J n K , X , u ) be a polynomial game where X i ∈ ∆ m i . If for some player i , u i is a polynomial of degree at least 4 with respect to x i ∈ X i , then verifying whether G is concave/monotone is strongly NP-hard.

Proof. We provide a similar construction to the proof of Theorem 3.1. Here we show hardness for verifying concavity - the proof for hardness of verifying monotonicity is similar. Let p : R m → R be a polynomial of degree 4 or higher, and let ∆ def = { x ∈ R m | ∑ i ∈ J m K x i = 1 , x i ≥ 0 for i = 1 , . . . , m } be the m -dimensional simplex. Consider a two-player polynomial game G ( J 2 K , ∆ × ∆ , u ) , where u 1 ( x ) = p ( x 1 ) , and u 2 ( x ) = 0 for all x ∈ ∆ × ∆ . u 2 is concave over ∆ , so the game G is concave if and only if p is concave over ∆ . Thus, from [4, Theorem 2.1] (and as indirectly argued in [26] utilizing [40, Theorem 1]), it follows that verifying whether G is concave is strongly NP-hard.

## C.2 Proof of Theorem 3.2

Theorem 3.2. Let G ( J n K , X , u ) be a polynomial game over a compact, convex basic semialgebraic set X . Assume the quadratic module Q ( X ) is Archimedean. For any ℓ ∈ N consider the hierarchy of SOS optimization problems:

<!-- formula-not-decoded -->

where Q ℓ ( X × B ) denotes the restriction of Q ( X × B ) to polynomials of degree at most ℓ . Then, the following statements are true:

- 1) For all ℓ , we have that SOS ℓ ( G ) ≥ max x ∈X λ max ( SJ ( x ) ) .
- 2) The sequence ( SOS ℓ ( G ) ) ℓ ≥ 0 is nonincreasing.
- 3) lim ℓ →∞ SOS ℓ ( G ) = max x ∈X λ max ( SJ ( x ) ) .
- 4) G is strictly monotone if, and only if, there exists some finite level ℓ such that SOS ℓ ( G ) &lt; 0 .
- 5) For any level ℓ , the program in (17) can be formulated as an SDP and solved in polynomial time.

Proof. To prove Statement 1, we start by considering some arbitrary ℓ ≥ 0 . Observe that, if the program in (17) is infeasible, then SOS ℓ ( SJ ) = ∞ , and therefore SOS ℓ ( SJ ) ≥ λ max ( SJ ( x ) ) for all x ∈ X . On the other hand, if the program in (17) is feasible, SOS ℓ ( SJ ) -y T SJ ( x ) y ∈ Q ℓ ( X ×B ) , i.e., there exist σ ∗ 1 , . . . , σ ∗ m g ∈ Σ[ x ] and p ∗ 0 , . . . , p ∗ m h ∈ R [ x ] such that

<!-- formula-not-decoded -->

In particular, since h 1 ( x ) = · · · = h m h ( x ) = 0 for all x ∈ X ; and 1 -y T y = 0 for all y ∈ B , by the above we also have that

<!-- formula-not-decoded -->

where the last inequality follows because g j ( x ) ≥ 0 for all x ∈ X , and σ 0 , . . . , σ m g ∈ Σ[ x, y ] . Furthermore, since SJ is symmetric, the maximum eigenvalue of SJ ( x ) is given by

<!-- formula-not-decoded -->

Alas, we have established that, SOS ℓ ( SJ ) ≥ λ max ( SJ ( x ) ) for all x ∈ X and ℓ ≥ 0 .

To prove Statement 2, observe that as ℓ increases, the feasible set of the minimization program in (17) is expanding, and therefore ( SOS ℓ ( SJ ) ) ℓ ≥ 0 is nonincreasing.

To prove Statement 3, we, first, prove that lim ℓ →∞ SOS ℓ ( SJ ) exists, i.e., the sequence ( SOS ℓ ( SJ ) ) ℓ ≥ 0 converges. In particular, observe that, since the quadratic module Q ( X ) is Archimedean, the set X is compact, and thus, the maximum max x ∈X λ max ( SJ ( x ) ) exists. Then, by Statement 1, it follows that

<!-- formula-not-decoded -->

Consequently, the sequence ( SOS ℓ ( SJ ) ) ℓ ≥ 0 is nonincreasing (Statement 2) and bounded from below by (Equation (A27)), and therefore it converges.

Now, recall that lim ℓ →∞ SOS ℓ ( SJ ) = max x ∈X λ max ( SJ ( x ) ) if for every ϵ &gt; 0 , there exists ℓ 0 ≥ 0 such that | SOS ℓ ( SJ ) -max x ∈X λ max ( SJ ( x ) ) | ≤ ϵ for all ℓ ≥ ℓ 0 . We are going to use Putinar's Positivstellensatz to show that for every ϵ &gt; 0 such an ℓ 0 exists.

First, observe that X × B is a basic semialgebraic set. In particular, we have that

<!-- formula-not-decoded -->

Furthermore, it is not difficult to show that the quadratic module Q ( X × B ) is Archimedean.

Indeed, since Q ( X ) is Archimedean, there exists N ∈ N such that N -∑ m i =1 x 2 i ∈ Q ( X ) . Therefore, there exist σ 0 , . . . , σ m g ∈ Σ[ x ] , and p 1 , . . . , p m h ∈ R [ x ] such that

<!-- formula-not-decoded -->

Define the polynomial functions σ ′ 0 , . . . , σ ′ m g , p ′ 0 , . . . , p ′ m h : R m × R m → R given by

<!-- formula-not-decoded -->

and observe that, since σ 0 , . . . , σ m g ∈ Σ[ x ] , it follows that σ ′ 0 , . . . , σ ′ m g ∈ Σ[ x, y ] . Moreover, observe that

<!-- formula-not-decoded -->

and therefore, N + 1 -∑ m i =1 x 2 i -∑ m i y 2 i ∈ Q ( X × B ) . Thus, we conclude that Q ( X × B ) is Archimedean.

Next, observe that, by definition, max x ∈X λ max ( SJ ( x ) ) = max x ∈X y ∈B y T SJ ( x ) y , which also implies that

<!-- formula-not-decoded -->

Therefore, the polynomial

<!-- formula-not-decoded -->

is positive over X × B for all ϵ &gt; 0 . Thus, by Putinar's Positivstellensatz, it follows that q ϵ ( x, y ) ∈ Q ( X × B ) , i.e., there exist σ 0 , . . . , σ m g ∈ Σ[ x, y ] , and p 0 , . . . , p m h ∈ R [ x, y ] such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let ℓ 0 ≥ 0 be the smallest number such that 2 ℓ 0 ≥ max { deg( σ 0 ) , . . . , deg( σ m g ) , deg( p 0 ) , . . . , deg( p m h ) } . Then, it follows that ( max x ∈X λ max ( SJ ( x ) ) + ϵ, σ, p ) is a solution to the SOS program in (17), where ℓ = ℓ 0 . Thus, by the optimality of SOS ℓ 0 ( SJ ) , and since the sequence ( SOS ℓ ( SJ ) ) ℓ ≥ 0 is nonincreasing (Statement 1), it follows that

<!-- formula-not-decoded -->

Alas, by (A27), we conclude that

<!-- formula-not-decoded -->

To prove Statement 4, recall that a polynomial game is strictly monotone if, and only if,

<!-- formula-not-decoded -->

First, suppose that G is not strictly monotone. Then, by the above, we have that max x ∈X λ max ( SJ ( x ) ) ≥ 0 . Thus, by Statement 1, it follows that

<!-- formula-not-decoded -->

Next, suppose instead that G is strictly monotone. Then, max x ∈X λ max ( SJ ( x ) ) &lt; 0 , and therefore, it exists ϵ &gt; 0 such that max x ∈X λ max ( SJ ( x ) ) + ϵ &lt; 0 . Moreover, by Statement 3, we also have that

<!-- formula-not-decoded -->

Therefore, by definition, there exists ℓ 0 ≥ 0 such that | SOS ℓ 0 ( SJ ) -max x ∈X λ max ( SJ ( x ) ) | ≤ ϵ , and thus

<!-- formula-not-decoded -->

Statement 5 follows from standard results in semidefinite programming.

## C.3 Proof of Theorem 3.3

Theorem 3.3. For almost all monotone games, monotonicity can be certified at a finite level ℓ of the SOS hierarchy (17) , i.e., SOS ℓ ( G ) ≤ 0 . Concretely, for all d ≥ 2 , the set of monotone polynomial games of degree d over a compact basic semialgebraic set X that are not strictly monotone has zero Lebesgue measure.

Proof. Let G m( n,d ) and G sm( n,d ) denote the sets of n -player, d -degree polynomial monotone and strictly monotone games, respectively. We are going to show that given a compact, convex basic semialgebraic set X of joint actions, the set G m( n,d ) \ G sm( n,d ) has zero Lebesgue measure. In particular, define µ as the canonical dim( G m( n,d ) ) -dimensional Lebesgue measure on aff( G m( n,d ) ) , i.e., the affine hull of G m( n,d ) . First, we show that G m( n,d ) is µ -measurable and therefore the restriction of µ to G m( n,d ) (denoted by µ ↾ G m( n,d ) ) is well-defined. Then, we show that µ ↾ G m( n,d ) ( G m( n,d ) \ G sm( n,d ) ) = 0 .

To begin with, observe that by definition:

<!-- formula-not-decoded -->

where for each G ∈ G m( n,d ) , SJ G is the symmetrized Jacobian matrix of the pseudo-gradient v G of G . Next, observe that the map ( G , x ) ↦→ SJ G ( x ) is polynomial in G and x . Moreover, the determinant A ↦→ det( A ) is also polynomial in A . Therefore, for all G ∈ G m( n,d ) and x ∈ X , the principal minors f I : ( G , x ) ↦→ det ( SJ G , I ( x ) ) , I ∈ 2 J m K , of SJ G ( x ) are polynomial in G and x . Thus, by Sylvester's Criterion :

<!-- formula-not-decoded -->

It follows by the Tarski-Seidenberg theorem [58, 52] that G m( n,d ) is a basic semialgebraic set, and therefore a Borel set. Thus, G m( n,d ) is µ -measurable.

Next, observe that G m( n,d ) is convex. Indeed, for all x ∈ X , define

<!-- formula-not-decoded -->

Observe that the map J x : G ↦→ SJ G ( x ) linear. Since G ( n,d ) is a vector space, S x ≡ J x ( G ( n,d ) ) is a vector subspace, and therefore convex. Thus,

<!-- formula-not-decoded -->

is the (uncountable) intersection of convex sets, and therefore it is convex. Note that since the empty set is convex, the statement in Eq. (A44) remains valid even if the intersection is empty.

Let us now consider the set

<!-- formula-not-decoded -->

Observe that for all d ≥ 0 , G sm( n,d ) is non-empty as the game G ′ with payoff functions u i : x ↦→ 1 2 ∥ x i ∥ 2 , for all i ∈ J n K , is strictly monotone. We show that G sm( n,d ) ⊇ int( G m( n,d ) ) with respect to the relative topology.

Let G 0 ∈ int( G m( n,d ) ) , and suppose G 0 / ∈ G sm( n,d ) . Then, by definition, there exists x 0 ∈ X such that SJ G 0 ( x ) ̸≻ 0 , i.e., it exists a vector u ∈ R m \ { 0 } such that u T SJ G 0 ( x 0 ) u = 0 . Define L : G ↦→ u T SJ G ( x 0 ) u . Since J x 0 is a linear map, it follows that L a linear functional. In particular, we have that L ( G 0 ) = 0 . Moreover, since by definition SJ G ( x ) ⪰ 0 for all G ∈ G m( n,d ) and x ∈ X , we also have that L ( G m( n,d ) ) ≥ 0 . Finally, since G sm( n,d ) is non-empty, by definition we have that L ( G sm( n,d ) ) &gt; 0 , and therefore L is non-trivial, i.e., L ̸≡ 0 . Thus, L describes a non-trivial supporting hyperplane to G m( n,d ) containing { G 0 } . Then, by a version of the Separating Hyperplane theorem [47, Theorem 11.6, p. 100], we may conclude that G 0 / ∈ int( G m( n,d ) ) , which is a contradiction. Thus, G 0 ∈ G sm( n,d ) , and therefore it follows that G sm( n,d ) ⊇ int( G m( n,d ) ) .

Using ∂ to denote the boundary of a set, we conclude that, G m( n,d ) \ G sm( n,d ) ⊂ ∂ ( G m( n,d ) ) . Moreover as established before, G m( n,d ) is a basic semialgebraic set. Thus, by [61, Theorem 1.8, p. 67], it follows that

<!-- formula-not-decoded -->

.

which, since G m( n,d ) is µ -measurable, allows us to conclude that µ ↾ G m( n,d ) ( G m( n,d ) \G sm( n,d ) ) = 0

## C.4 Proof of Theorem 4.3

Theorem 4.3. For all d ≥ 2 , the set of SOS-monotone games of degree d over a compact basic semialgebraic set X is dense in the set of monotone games of degree d over X . Furthermore, given any polynomial game G ∗ ∈ G ( n,d ) over X , and any fixed ℓ ≥ 0 , we can compute the closest ℓ -SOSmonotone game to G ∗ by the program

<!-- formula-not-decoded -->

which can be formulated as an SDP.

Proof. Let G m( n,d ) denote the set of n -player, d -degree polynomial monotone games. First, we show that G sosm( n,d ) is dense in G m( n,d ) , i.e.,

<!-- formula-not-decoded -->

Let G ∈ G m( n,d ) . Furthermore, for each k ∈ N , define G k as the n -player, d -degree polynomial game over X with utility functions u k, 1 , . . . , u k,n : X → R given by

<!-- formula-not-decoded -->

Then, the pseudo-gradient v k of G k is given by

<!-- formula-not-decoded -->

and the symmetrized Jacobian matrix SJ k of v k is given by

<!-- formula-not-decoded -->

Moreover, since G is monotone, it follows

<!-- formula-not-decoded -->

and therefore G k is strictly monotone. Subsequently, it follows that G k is SOS-monotone, i.e., G k ∈ G sosm( n,d ) .

Now, consider the sequence ( G k ) ∞ k =1 of SOS-monotone games. Observe that

<!-- formula-not-decoded -->

In other words, ( G k ) ∞ k =1 converges to G . Thus, by definition, cl G m( n,d ) G sosm( n,d ) ≡ G m( n,d ) , i.e., G sosm( n,d ) is dense in G m( n,d ) .

Next, we show that the program in (19) may be formulated as a SDP and solved in O ( log ( 1 ϵ ) · Poly( ℓ 2 ) ) , where G ∗ ( J n K , X , u ∗ ) ∈ G ( n,d ) , and ℓ ≥ 0 .

First, by Definition 4.1, the program in (19) is equivalent to

<!-- formula-not-decoded -->

where SJ ( x ) is the (symmetrized) Jacobian matrix of the pseudo-gradient of G ( J n K , X , u ) . Define the map f : ( vec( u 1 ) , . . . , vec( u n ) , x, y ) ↦→ -y T SJ ( x ) y . Now, observe that ( vec( u 1 ) , . . . , vec( u n ) ) ↦→ SJ ( x ) is an affine map , and therefore f is affine in ( vec( u 1 ) , . . . , vec( u n ) ) , and polynomial in ( x, y ) .

Next, by the definition of ∥·∥ , we have that

<!-- formula-not-decoded -->

Thus, the program (A53) is equivalent to

<!-- formula-not-decoded -->

Moreover, the condition λ ≥ max i ∈ J n K ∥ vec( u i ) -vec( u ∗ i ) ∥ ∞ is equivalent to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, the program in (A55) is also equivalent to

<!-- formula-not-decoded -->

Finally, observe that, as f is affine in vec( u 1 ) , . . . , vec( u n ) , all the constraints of the program in (A57) are affine in λ , vec( u 1 ) , . . . , vec( u n ) . Then, by definition, the program in (A57) is a SOS minimization program, and therefore it may be reformulated as a SDP.

## D Extensive-Form Games with Imperfect Recall

## D.1 EFG Preliminaries

For completeness, we provide the necessary notation for EFGs with imperfect recall and their connection to polynomial optimization. First, we note that in standard game theory, strategies for EFGs lie in the simplex.

Definition D.1. An n -player extensive form game Γ is a tuple Γ := ⟨H , A , Z , u, I⟩ where:

- The set H denotes the states of the game which are decision points for the players. The states h ∈ H form a tree rooted at an initial state r ∈ H .
- Each state h ∈ H is associated with a set of available actions A ( h ) .
- The set N := { 1 , . . . , n, c } denotes the set of players of the game. Each state h ∈ H admits a label Label( h ) ∈ N which denotes the acting player at state h . The letter c denotes a special player called a chance player . Each state h ∈ H with Label( h ) = c is additionally associated with a function σ h : A ( h ) ↦→ [0 , 1] where σ h ( a ) denotes the probability that the chance player selects action a ∈ A ( h ) at state h , ∑ a ∈A ( h ) σ h ( a ) = 1 .
- Next( a, h ) denotes the state h ′ := Next(a , h) which is reached when player i := Label( h ) takes action a ∈ A ( h ) at state h . H i ⊆ H denotes the states h ∈ H with Label( h ) = i .
- Z denotes the terminal states of the game corresponding to the leafs of the tree. At each z ∈ Z no further action can be chosen, so A ( z ) = ∅ for all z ∈ Z . Each terminal state z ∈ Z is associated with value u ( z ) , where u : Z → R is called the utility function of the game.
- The game states H are further partitioned into information sets ascribed to each player, namely I i ∈ ( I 1 , . . . , I n ) . Each information set I ∈ I i encodes groups of nodes that the acting player i cannot distinguish between, and thus the available actions within each infoset must be the same. Moreover, the player must play the same strategy in all nodes of the infoset. Formally, if h 1 , h 2 ∈ I , then A ( h 1 ) = A ( h 2 ) . With slight abuse of notation, we can consider A ( I ) to be the set of shared available actions for the player in infoset I .
- For notational convenience, we ascribe a singleton information set to each chance node and define I c as the collection of these chance node information sets. For each non-terminal node h ∈ H / ∈ Z , we thus define I h ∈ ( I 1 , . . . , I n ) ∪ I c to be its infoset.

The standard assumption in the literature is that of perfect recall , wherein no player ever forgets their past history (i.e. their past information sets and actions taken in those information sets) or any information acquired. Formally, for any infoset I ∈ I i and for any two nodes h 1 , h 2 ∈ I , the sequence of Player i 's actions from r to h 1 and from r to h 2 must coincide, otherwise they would be able to distinguish between the nodes. Finally, the game is called perfect recall if all players have perfect recall. Otherwise, the game is said to have the imperfect recall property. The notion of perfect recall has been crucial to establishing convergence results to pure Nash equilibria in extensive-form games, primarily via the concept of behavioral strategies :

Definition D.2 (Behavioral Strategy) . Consider the infosets belonging to player i , denoted I ∈ I i . Let ∆( A ( I )) denote the set of probability distributions on the simplex over actions in A ( I ) . The set of behavioral strategies of a player is denoted by σ i : I i →∪ I ∈I i ∆( A ( I )) . In particular, at each of their infosets I , player i selects a probability distribution over their available actions at the infoset, σ i ( ·| I ) ∈ ∆( A ( I )) . Finally, the joint behavioral strategy for all players is denoted σ := ( σ i ) i ∈N .

Similarly, mixed strategies can be defined in the following way:

Definition D.3 (Mixed Strategy in Extensive-Form Games) . Denote by S i the set of all possible actions across all game states H for player i in Γ . Then, for all pure actions in the game s ∈ S i , Player i 's mixed strategy µ i is given by the probability distribution defined by the probabilities µ i ( s ) of playing strategy s .

Intuitively, one can view behavioral strategies as players randomizing between their possible actions between each information set, and mixed strategies as players randomizing over their strategy sequences prior to playing the game (i.e. ex ante). Kuhn's theorem provides a meaningful connection between behavioral strategies and mixed strategies in EFGs with perfect recall:

Theorem D.4 (Kuhn's Theorem [33]) . If player i in an extensive form game has perfect recall, then for any mixed strategy µ of player i there exists an equivalent behavioral strategy σ of player i .

Moreover, computing behavioral strategies in two-player zero-sum games of perfect recall is possible in polynomial time [31]. However, once the assumption of perfect recall is relaxed (i.e. when players have imperfect recall), Kuhn's theorem no longer holds and finding a solution even in the two-player zero-sum case becomes NP-complete [31].

## D.2 Imperfect Recall Games

[64] introduced an example of a game with no Nash equilibria in behavioral strategies (Figure 2). Subsequently, a variation of the original game called the forgetful penalty shoot-out game was introduced in [60] and proceeds as follows: Player 1 decides whether to kick a ball Left or Right before the whistle is blown, then decides again right before kicking the ball. At the second decision node, the player has forgotten their previous decision. If the decisions at the two nodes match, Player 1 manages to aim at the goal, during which Player 2 has to decide to dive Left or Right to stop the ball. Otherwise, the shot goes wide. This game also has no Nash equilibria in behavioral strategies.

When studying imperfect recall games, a key question to ask is whether one should consider mixed strategies or behavioral strategies. In particular, Kuhn's Theorem no longer holds and the convenient sequence form representation is not well-defined. Indeed, mixed strategies require players to select actions according to a distribution over all possible strategy sequences. For instance, a mixed strategy for Player 1 in the forgetful penalty shoot-out game (Figure 2) could look something like: Kick Left twice in a row with probability 0 . 5 , and kick Right twice in a row with probability 0 . 5 . However, this requires the players to have some memory of their previous actions. In contrast, behavioral strategies are more natural in imperfect recall games as they do not necessitate a memory requirement, a point which is argued for in Kuhn's original treatment of perfect recall games [33].

Following the work of [46, 59], we show a construction from imperfect recall EFGs to polynomial utilities via behavioral strategies. First, let P ( h ′ | σ, h ) denote the realization probability of reaching h ′ given that players using strategy σ are at state h . Note that if h / ∈ hist ( h ′ ) (i.e., if h ′ is not reachable from h ) then the probability is 0 . Intuitively, the realization probability given a behavioral strategy is just the product of choice probabilities along the path from h to h ′ . In order to formally define P ( h ′ | σ, h ) , we will need some additional notation. First, any node h ∈ H uniquely corresponds to a history hist( h ) from root r to h .

- Function δ ( h ) : H → N denotes the depth of the game tree starting from node h ∈ H .
- Function ν ( h, d ) : H× N →H identifies the node ancestor at depth d ≤ δ from node h .
- Function α ( h, d ) : H× N →∪ h ∈H A ( h ) identifies the action ancestor at depth d ≤ δ from node h .

Together, the sequence ( ν ( h, 0) , ν ( h, 1) , . . . , ν ( h, δ ( h ))) uniquely identifies the history of nodes from r to h . Likewise, the sequence ( α ( h, 0) , α ( h, 1) , . . . , α ( h, δ ( h ) -1)) uniquely identifies the

history of actions taken from r to h . Then, the realization probability of node h ′ from h if the players use joint strategy profile σ is given by:

Definition D.5 (Realization Probability) .

<!-- formula-not-decoded -->

Definition D.6 (Expected Utility for Player i ) . For player i at node h ∈ H \ Z , if strategy profile σ is played, their expected utility is given by U i ( σ | h ) := ∑ z ∈Z ( P ( z | σ, h ) · u i ( z )) . In its complete form, we can write the expected utility for each player as follows:

<!-- formula-not-decoded -->

With some abuse of notation, we can write P ( h | σ ) := P ( h | σ, r ) where r is the root node, and similarly U i ( σ ) := U i ( σ | r ) . Notice that by definition, the expected utility of each player is a polynomial function. In particular, P ( z | σ, h ) · u i ( z ) is a monomial in σ multiplied by a scalar.

Going forward, we establish several results connecting EFGs with imperfect recall and polynomial games, utilizing some additional notation.

- ℓ i denotes the number of infosets of player i , i.e. ℓ i := |I i | . Moreover, fix an ordering ( I 1 i , . . . , I ℓ i i ) of infosets in I i .
- m j i denotes the number of actions in a given infoset I j i ∈ I i of player i , i.e. m j i := |A ( I j i ) | . Moreover, fix an ordering ( a 1 i , . . . , a ℓ i i ) of actions in A ( I j i ) .
- The strategy set of a player in information set I is defined on the simplex ∆ |A ( I ) -1 | , where ∆ n -1 := { x ∈ R n : x k ≥ 0 ∀ k, ∑ n k =1 x k = 1 } .
- Subsequently, the strategy set of player i over all of their infosets can be written as a Cartesian product of simplices: S i := × ℓ i j =1 ∆ m j i -1 . Moreover, the strategy set over all players is S := × n i =1 × ℓ i j =1 ∆ m j i -1 .
- A joint strategy σ ∈ S for the players can hence be uniquely written as a vector σ = ( σ j ik ) ijk ∈ × n i =1 × ℓ i j =1 ∆ m j i -1 ⊂ × n i =1 × ℓ i j =1 R m j i .

Firstly, note that each infoset belonging to a player of an EFG with imperfect recall induces an additional variable in the expected utility function. Clearly, the resultant polynomial utilities can themselves be viewed as a polynomial game in the sense of [14, 45, 55] (and also falling in our definition of polynomial games G ), with the following definition of Nash equilibrium in behavioral strategies:

Definition D.7 (Nash Equilibrium in Behavioral Strategies) . A joint behavioral strategy σ ∗ ∈ × n i =1 × ℓ i j =1 ∆ m j i -1 is called a Nash equilibrium if for all players i ∈ N :

<!-- formula-not-decoded -->

i.e. no player has incentive to deviate from the behavioral strategy σ ∗ in any of their information sets.

Remark D.8. The definition of Nash equilibria in our setting directly implies that any solution of the corresponding polynomial game defined using the polynomial utility functions is also a solution to the original EFG. In particular, the constructed polynomial utilities can be viewed as a generic polynomial game with utilities u i ( x ) . Here, x denotes the joint action profile of all players (see Section 2). Here, the number of variables in x is equal to the total number of infosets over all players, ∑ i ∈N ℓ i . A joint state x ∗ is called a Nash equilibrium if the following holds for all players i ∈ N : u i ( x ∗ ) ≥ u i ( x i , x ∗ -i ) ∀ x i ∈ S i . At a Nash equilibrium in the polynomial game, no player has incentive to unilaterally deviate in any of the variables they control. This is precisely the definition of Nash equilibrium in behavioral strategies for the original EFG.

## E Additional Experimental Details

Example 1: The absent-minded taxi driver in Figure 3. In the case of the game in Figure 3, we let x denote the probability of choosing C and 1 -x be the probability of choosing E . We use the SDP hierarchy in Eq. (17) to certify SOS-monotonicity of the polynomial u ( x ) = -3 x 2 +4 x . We select ℓ = 2 and obtain SOS 2 ( G ) ≈ -6 &lt; 0 . Then, by Statement 4 of Theorem 3.2, the game is strictly monotone. This additionally guarantees that the solution of the game is unique [49].

Example 2: A game with no Nash equilibria in Figure 2. Next, it follows as a consequence of Theorem 4.3 that we can use the program in Eq. (19) to find an SOS-monotone game G which is closest to the zero-sum game in Figure 2, in the sense of the norm defined in Eq. (3). By letting x 1 denote the probability that P1 selects L at information set { a 1 } , x 2 denote the probability that P1 selects L at information set { a 2 , a 3 } , and y denote the probability that P2 selects r , we obtain the payoff functions for P1 and P2 as follows:

<!-- formula-not-decoded -->

and u 2 = -u 1 , respectively. Recall from [64] that this two-player zero-sum EFG does not have a NE and that the game is neither concave nor monotone. We first run our hierarchy of SOS optimization problems in Eq. (17) at level 2, and we attain an objective value of SOS 2 ( G ) ≈ 10 &gt; 0 . Then, we run our program in Eq. (19) with additional constraints that G has to be zero-sum and that the information structure of the EFG has to be preserved. To preserve the information structure of the game, we select the monomial basis for the new payoff functions to be precisely the monomial basis that can appear in the original game. We obtain the closest SOS-monotone game G ′ given by:

<!-- formula-not-decoded -->

and u ′ 2 = -u ′ 1 . The distance between the two games, ∥ G -G ′ ∥ , which is defined in Eq. (3), is in this case simply ∥ vec( u 1 ) -vec( u ′ 1 ) ∥ ∞ and equals 10 . On the other hand, the payoff function of this game is multilinear and indeed the zero-sum game is monotone if and only if the term x 1 x 2 has coefficient 0 . This is in line with the experimental results. The modified game G ′ has zero symmetrized Jacobian matrix and is, thus, negative semidefinite. Moreover, since the modified game is SOS-monotone, it has a NE in behavioral strategies.

Example 3. We generate a two-player polynomial game with the following payoff functions:

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

P1 and P2 choose their actions ( x 1 , x 2 ) and ( y 1 , y 2 ) from a two-dimensional simplex respectively. This game is certified strictly monotone and SOS-monotone, as we run our hierarchy of SOS optimization problems in Eq. (17) and obtain an objective value -1 at level 4 .

Example 4. The game which was constructed is given in Figure E.4:

Figure E.4: Another Zero-sum Game with No Nash Equilibria

<!-- image -->

Example 5. We construct a two-player EFG with imperfect recall where P1 makes six moves before P2 makes two moves. The utility functions for both players are degree-8 polynomials with 4 variables, and we do not restrict the game to be zero-sum. The size of the monomial basis is 168, and we randomly generate the payoff functions for P1 and P2 by independently sampling the coefficient of each monomial in the basis from a uniform distribution on [ -1 , 1] . After rounding the coefficients to 2 s.f. for brevity, the payoff function for P1 is:

```
u 1 ( x, y ) = -0 . 42 -0 . 46 y 2 -0 . 98 y 1 +0 . 93 x 2 -0 . 11 x 1 -0 . 78 y 2 2 -0 . 85 y 1 y 2 -0 . 21 y 2 1 +0 . 92 x 2 y 2 + 0 . 48 x 2 y 1 +0 . 29 x 2 2 -0 . 37 x 1 y 2 -0 . 97 x 1 y 1 +0 . 65 x 1 x 2 -0 . 44 x 2 1 -0 . 86 x 2 y 2 2 -0 . 35 x 2 y 1 y 2 -0 . 03 x 2 y 2 1 +0 . 02 x 2 2 y 2 -0 . 46 x 2 2 y 1 +0 . 78 x 3 2 -0 . 87 x 1 y 2 2 -0 . 59 x 1 y 1 y 2 +0 . 46 x 1 y 2 1 +0 . 12 x 1 x 2 y 2 + 0 . 37 x 1 x 2 y 1 -0 . 31 x 1 x 2 2 +0 . 89 x 2 1 y 2 +0 . 81 x 2 1 y 1 -0 . 22 x 2 1 x 2 +0 . 92 x 3 1 +0 . 85 x 2 2 y 2 2 +0 . 34 x 2 2 y 1 y 2 -0 . 20 x 2 2 y 2 1 +0 . 53 x 3 2 y 2 -0 . 97 x 3 2 y 1 +0 . 08 x 4 2 +0 . 98 x 1 x 2 y 2 2 +0 . 03 x 1 x 2 y 1 y 2 -0 . 07 x 1 x 2 y 2 1 + 0 . 10 x 1 x 2 2 y 2 -0 . 04 x 1 x 2 2 y 1 -0 . 33 x 1 x 3 2 +0 . 41 x 2 1 y 2 2 +0 . 61 x 2 1 y 1 y 2 -0 . 39 x 2 1 y 2 1 -0 . 71 x 2 1 x 2 y 2 + 0 . 84 x 2 1 x 2 y 1 +0 . 69 x 2 1 x 2 2 +0 . 44 x 3 1 y 2 +0 . 13 x 3 1 y 1 +0 . 05 x 3 1 x 2 +0 . 92 x 4 1 -0 . 10 x 3 2 y 2 2 -0 . 55 x 3 2 y 1 y 2 -0 . 61 x 3 2 y 2 1 +0 . 74 x 4 2 y 2 -0 . 65 x 4 2 y 1 -0 . 74 x 5 2 -0 . 14 x 1 x 2 2 y 2 2 +0 . 77 x 1 x 2 2 y 1 y 2 -0 . 30 x 1 x 2 2 y 2 1 + 0 . 41 x 1 x 3 2 y 2 +0 . 66 x 1 x 3 2 y 1 -0 . 62 x 1 x 4 2 -0 . 39 x 2 1 x 2 y 2 2 +0 . 11 x 2 1 x 2 y 1 y 2 -0 . 71 x 2 1 x 2 y 2 1 -0 . 14 x 2 1 x 2 2 y 2 +0 . 56 x 2 1 x 2 2 y 1 +0 . 60 x 2 1 x 3 2 +0 . 26 x 3 1 y 2 2 +0 . 34 x 3 1 y 1 y 2 -0 . 47 x 3 1 y 2 1 +0 . 87 x 3 1 x 2 y 2 + 0 . 29 x 3 1 x 2 y 1 +0 . 94 x 3 1 x 2 2 -0 . 42 x 4 1 y 2 +0 . 35 x 4 1 y 1 -0 . 12 x 4 1 x 2 -0 . 43 x 5 1 +0 . 04 x 4 2 y 2 2 +0 . 15 x 4 2 y 1 y 2 -0 . 93 x 4 2 y 2 1 +0 . 58 x 5 2 y 2 -0 . 12 x 5 2 y 1 -0 . 71 x 6 2 +0 . 93 x 1 x 3 2 y 2 2 +0 . 81 x 1 x 3 2 y 1 y 2 +0 . 57 x 1 x 3 2 y 2 1 + 0 . 32 x 1 x 4 2 y 2 -0 . 16 x 1 x 4 2 y 1 -0 . 46 x 1 x 5 2 +0 . 85 x 2 1 x 2 2 y 2 2 -0 . 83 x 2 1 x 2 2 y 1 y 2 +0 . 07 x 2 1 x 2 2 y 2 1 -0 . 60 x 2 1 x 3 2 y 2 +0 . 07 x 2 1 x 3 2 y 1 +0 . 44 x 2 1 x 4 2 -0 . 14 x 3 1 x 2 y 2 2 -0 . 91 x 3 1 x 2 y 1 y 2 -0 . 82 x 3 1 x 2 y 2 1 + 0 . 52 x 3 1 x 2 2 y 2 +0 . 79 x 3 1 x 2 2 y 1 +0 . 39 x 3 1 x 3 2 -0 . 04 x 4 1 y 2 2 -0 . 61 x 4 1 y 1 y 2 -0 . 37 x 4 1 y 2 1 +0 . 31 x 4 1 x 2 y 2 -0 . 85 x 4 1 x 2 y 1 +0 . 90 x 4 1 x 2 2 +0 . 49 x 5 1 y 2 -0 . 24 x 5 1 y 1 -0 . 66 x 5 1 x 2 +0 . 58 x 6 1 -0 . 57 x 5 2 y 2 2 +0 . 72 x 5 2 y 1 y 2 -0 . 35 x 5 2 y 2 1 +0 . 50 x 6 2 y 2 +0 . 77 x 6 2 y 1 -0 . 06 x 1 x 4 2 y 2 2 +0 . 89 x 1 x 4 2 y 1 y 2 -0 . 48 x 1 x 4 2 y 2 1 -0 . 69 x 1 x 5 2 y 2 + 0 . 61 x 1 x 5 2 y 1 +0 . 43 x 2 1 x 3 2 y 2 2 +0 . 16 x 2 1 x 3 2 y 1 y 2 -0 . 58 x 2 1 x 3 2 y 2 1 +0 . 40 x 2 1 x 4 2 y 2 -0 . 34 x 2 1 x 4 2 y 1 + 0 . 50 x 3 1 x 2 2 y 2 2 +0 . 20 x 3 1 x 2 2 y 1 y 2 -0 . 77 x 3 1 x 2 2 y 2 1 +0 . 01 x 3 1 x 3 2 y 2 +0 . 05 x 3 1 x 3 2 y 1 -0 . 80 x 4 1 x 2 y 2 2 -0 . 04 x 4 1 x 2 y 1 y 2 +0 . 26 x 4 1 x 2 y 2 1 -0 . 98 x 4 1 x 2 2 y 2 +0 . 91 x 4 1 x 2 2 y 1 -0 . 77 x 5 1 y 2 2 -0 . 89 x 5 1 y 1 y 2 -0 . 72 x 5 1 y 2 1 +0 . 17 x 5 1 x 2 y 2 +0 . 70 x 5 1 x 2 y 1 +0 . 81 x 6 1 y 2 -0 . 57 x 6 1 y 1 +0 . 31 x 6 2 y 2 2 +0 . 82 x 6 2 y 1 y 2 -0 . 59 x 6 2 y 2 1 -0 . 82 x 1 x 5 2 y 2 2 -0 . 72 x 1 x 5 2 y 1 y 2 +0 . 93 x 1 x 5 2 y 2 1 -0 . 54 x 2 1 x 4 2 y 2 2 +0 . 66 x 2 1 x 4 2 y 1 y 2 + 0 . 69 x 2 1 x 4 2 y 2 1 +0 . 97 x 3 1 x 3 2 y 2 2 +0 . 28 x 3 1 x 3 2 y 1 y 2 +0 . 32 x 3 1 x 3 2 y 2 1 +0 . 34 x 4 1 x 2 2 y 2 2 -0 . 82 x 4 1 x 2 2 y 1 y 2 + 0 . 49 x 4 1 x 2 2 y 2 1 +0 . 60 x 5 1 x 2 y 2 2 -0 . 95 x 5 1 x 2 y 1 y 2 +0 . 14 x 5 1 x 2 y 2 1 +0 . 96 x 6 1 y 2 2 -0 . 39 x 6 1 y 1 y 2 -0 . 28 x 6 1 y 2 1 .
```

Similarly, the payoff function for P2 is:

```
u 2 ( x, y ) = -0 . 99+0 . 85 y 2 +0 . 77 y 1 -0 . 62 x 2 -0 . 88 x 1 -0 . 06 y 2 2 -0 . 32 y 1 y 2 -0 . 57 y 2 1 -0 . 94 x 2 y 2 -0 . 76 x 2 y 1 +0 . 66 x 2 2 -0 . 11 x 1 y 2 -0 . 32 x 1 y 1 -0 . 53 x 1 x 2 +0 . 47 x 2 1 +0 . 78 x 2 y 2 2 +0 . 79 x 2 y 1 y 2 -0 . 98 x 2 y 2 1 +0 . 65 x 2 2 y 2 -0 . 58 x 2 2 y 1 +0 . 01 x 3 2 -0 . 65 x 1 y 2 2 -0 . 35 x 1 y 1 y 2 +0 . 95 x 1 y 2 1 -0 . 86 x 1 x 2 y 2 -0 . 57 x 1 x 2 y 1 +0 . 76 x 1 x 2 2 +0 . 64 x 2 1 y 2 +0 . 28 x 2 1 y 1 +0 . 86 x 2 1 x 2 -0 . 74 x 3 1 +0 . 51 x 2 2 y 2 2 -0 . 72 x 2 2 y 1 y 2 -0 . 41 x 2 2 y 2 1 +0 . 39 x 3 2 y 2 -0 . 70 x 3 2 y 1 +0 . 37 x 4 2 +0 . 17 x 1 x 2 y 2 2 -0 . 12 x 1 x 2 y 1 y 2 -0 . 43 x 1 x 2 y 2 1 + 0 . 80 x 1 x 2 2 y 2 +0 . 34 x 1 x 2 2 y 1 +0 . 91 x 1 x 3 2 +0 . 77 x 2 1 y 2 2 +0 . 69 x 2 1 y 1 y 2 +0 . 64 x 2 1 y 2 1 +0 . 84 x 2 1 x 2 y 2 -0 . 41 x 2 1 x 2 y 1 -0 . 01 x 2 1 x 2 2 -0 . 46 x 3 1 y 2 +0 . 94 x 3 1 y 1 -0 . 33 x 3 1 x 2 +0 . 65 x 4 1 +0 . 06 x 3 2 y 2 2 -0 . 53 x 3 2 y 1 y 2 -0 . 81 x 3 2 y 2 1 +0 . 44 x 4 2 y 2 +0 . 32 x 4 2 y 1 +0 . 74 x 5 2 +0 . 63 x 1 x 2 2 y 2 2 +0 . 96 x 1 x 2 2 y 1 y 2 -0 . 21 x 1 x 2 2 y 2 1 + 0 . 84 x 1 x 3 2 y 2 +0 . 13 x 1 x 3 2 y 1 -0 . 13 x 1 x 4 2 +0 . 15 x 2 1 x 2 y 2 2 -0 . 46 x 2 1 x 2 y 1 y 2 -0 . 90 x 2 1 x 2 y 2 1 -0 . 40 x 2 1 x 2 2 y 2 -0 . 07 x 2 1 x 2 2 y 1 +0 . 93 x 2 1 x 3 2 +0 . 07 x 3 1 y 2 2 -0 . 56 x 3 1 y 1 y 2 +0 . 33 x 3 1 y 2 1 +0 . 08 x 3 1 x 2 y 2 + 0 . 97 x 3 1 x 2 y 1 +0 . 86 x 3 1 x 2 2 -0 . 50 x 4 1 y 2 -0 . 44 x 4 1 y 1 +0 . 59 x 4 1 x 2 +0 . 98 x 5 1 +0 . 32 x 4 2 y 2 2 -0 . 27 x 4 2 y 1 y 2 + 0 . 81 x 4 2 y 2 1 -0 . 39 x 5 2 y 2 -0 . 47 x 5 2 y 1 -0 . 23 x 6 2 +0 . 51 x 1 x 3 2 y 2 2 +0 . 64 x 1 x 3 2 y 1 y 2 +0 . 25 x 1 x 3 2 y 2 1 + 0 . 44 x 1 x 4 2 y 2 +0 . 89 x 1 x 4 2 y 1 -0 . 99 x 1 x 5 2 -0 . 11 x 2 1 x 2 2 y 2 2 +0 . 14 x 2 1 x 2 2 y 1 y 2 +0 . 98 x 2 1 x 2 2 y 2 1 -0 . 94 x 2 1 x 3 2 y 2 +0 . 53 x 2 1 x 3 2 y 1 +0 . 82 x 2 1 x 4 2 +0 . 29 x 3 1 x 2 y 2 2 -0 . 68 x 3 1 x 2 y 1 y 2 +0 . 28 x 3 1 x 2 y 2 1 + 0 . 69 x 3 1 x 2 2 y 2 +0 . 08 x 3 1 x 2 2 y 1 -0 . 16 x 3 1 x 3 2 -0 . 20 x 4 1 y 2 2 -0 . 27 x 4 1 y 1 y 2 -0 . 23 x 4 1 y 2 1 +0 . 62 x 4 1 x 2 y 2 -0 . 98 x 4 1 x 2 y 1 -0 . 35 x 4 1 x 2 2 +0 . 39 x 5 1 y 2 +0 . 33 x 5 1 y 1 -0 . 59 x 5 1 x 2 -0 . 23 x 6 1 +0 . 49 x 5 2 y 2 2 -0 . 69 x 5 2 y 1 y 2 + 0 . 93 x 5 2 y 2 1 +0 . 54 x 6 2 y 2 +0 . 38 x 6 2 y 1 -0 . 28 x 1 x 4 2 y 2 2 -0 . 69 x 1 x 4 2 y 1 y 2 -0 . 22 x 1 x 4 2 y 2 1 -0 . 32 x 1 x 5 2 y 2 + 0 . 58 x 1 x 5 2 y 1 +0 . 60 x 2 1 x 3 2 y 2 2 -0 . 99 x 2 1 x 3 2 y 1 y 2 +0 . 64 x 2 1 x 3 2 y 2 1 +0 . 69 x 2 1 x 4 2 y 2 +0 . 79 x 2 1 x 4 2 y 1 + 0 . 45 x 3 1 x 2 2 y 2 2 -0 . 58 x 3 1 x 2 2 y 1 y 2 +0 . 59 x 3 1 x 2 2 y 2 1 +0 . 39 x 3 1 x 3 2 y 2 -0 . 95 x 3 1 x 3 2 y 1 +0 . 68 x 4 1 x 2 y 2 2 -0 . 50 x 4 1 x 2 y 1 y 2 -0 . 02 x 4 1 x 2 y 2 1 +0 . 60 x 4 1 x 2 2 y 2 -0 . 54 x 4 1 x 2 2 y 1 -0 . 80 x 5 1 y 2 2 -0 . 22 x 5 1 y 1 y 2 + 1 . 00 x 5 1 y 2 1 +0 . 99 x 5 1 x 2 y 2 +0 . 82 x 5 1 x 2 y 1 -0 . 17 x 6 1 y 2 +0 . 32 x 6 1 y 1 -0 . 56 x 6 2 y 2 2 +0 . 61 x 6 2 y 1 y 2 + 0 . 98 x 6 2 y 2 1 +0 . 76 x 1 x 5 2 y 2 2 +0 . 38 x 1 x 5 2 y 1 y 2 -0 . 16 x 1 x 5 2 y 2 1 -0 . 16 x 2 1 x 4 2 y 2 2 -0 . 23 x 2 1 x 4 2 y 1 y 2 -0 . 19 x 2 1 x 4 2 y 2 1 +0 . 43 x 3 1 x 3 2 y 2 2 +0 . 88 x 3 1 x 3 2 y 1 y 2 +0 . 33 x 3 1 x 3 2 y 2 1 +0 . 09 x 4 1 x 2 2 y 2 2 -0 . 46 x 4 1 x 2 2 y 1 y 2 + 0 . 60 x 4 1 x 2 2 y 2 1 +0 . 17 x 5 1 x 2 y 2 2 +0 . 81 x 5 1 x 2 y 1 y 2 -0 . 24 x 5 1 x 2 y 2 1 +0 . 05 x 6 1 y 2 2 -0 . 72 x 6 1 y 1 y 2 +0 . 31 x 6 1 y 2 1 .
```

We run our program in Eq. (19) at level 8 of the hierarchy to find the closest SOS-monotone game with an additional constraint that the information structure of the original EFG (i.e., the monomial basis) has to be preserved. This results in a modified game with the following payoff functions:

```
u ′ 1 ( x, y ) = -0 . 92 -0 . 96 y 2 -1 . 49 y 1 +0 . 42 x 2 -0 . 61 x 1 -1 . 28 y 2 2 -1 . 35 y 1 y 2 -0 . 72 y 2 1 +1 . 03 x 2 y 2 + 0 . 76 x 2 y 1 -0 . 21 x 2 2 +0 . 07 x 1 y 2 -0 . 70 x 1 y 1 +0 . 32 x 1 x 2 -0 . 94 x 2 1 -0 . 41 x 2 y 2 2 -0 . 33 x 2 y 1 y 2 + 0 . 38 x 2 y 2 1 -0 . 47 x 2 2 y 2 -0 . 86 x 2 2 y 1 +0 . 29 x 3 2 -0 . 39 x 1 y 2 2 -0 . 29 x 1 y 1 y 2 +0 . 27 x 1 y 2 1 -0 . 34 x 1 x 2 y 2 + 0 . 05 x 1 x 2 y 1 -0 . 81 x 1 x 2 2 +0 . 40 x 2 1 y 2 +0 . 35 x 2 1 y 1 -0 . 60 x 2 1 x 2 +0 . 43 x 3 1 +0 . 35 x 2 2 y 2 2 +0 . 19 x 2 2 y 1 y 2 -0 . 53 x 2 2 y 2 1 +0 . 08 x 3 2 y 2 -0 . 58 x 3 2 y 1 -0 . 39 x 4 2 +0 . 51 x 1 x 2 y 2 2 -0 . 02 x 1 x 2 y 1 y 2 +0 . 21 x 1 x 2 y 2 1 -0 . 12 x 1 x 2 2 y 2 -0 . 44 x 1 x 2 2 y 1 -0 . 14 x 1 x 3 2 -0 . 08 x 2 1 y 2 2 +0 . 24 x 2 1 y 1 y 2 -0 . 78 x 2 1 y 2 1 -0 . 67 x 2 1 x 2 y 2 + 0 . 44 x 2 1 x 2 y 1 +0 . 19 x 2 1 x 2 2 -0 . 02 x 3 1 y 2 -0 . 30 x 3 1 y 1 +0 . 14 x 3 1 x 2 +0 . 42 x 4 1 -0 . 46 x 3 2 y 2 2 -0 . 49 x 3 2 y 1 y 2 -0 . 29 x 3 2 y 2 1 +0 . 30 x 4 2 y 2 -0 . 46 x 4 2 y 1 -0 . 56 x 5 2 -0 . 43 x 1 x 2 2 y 2 2 +0 . 50 x 1 x 2 2 y 1 y 2 -0 . 40 x 1 x 2 2 y 2 1 + 0 . 17 x 1 x 3 2 y 2 +0 . 41 x 1 x 3 2 y 1 -0 . 20 x 1 x 4 2 -0 . 11 x 2 1 x 2 y 2 2 +0 . 26 x 2 1 x 2 y 1 y 2 -0 . 39 x 2 1 x 2 y 2 1 -0 . 49 x 2 1 x 2 2 y 2 +0 . 10 x 2 1 x 2 2 y 1 +0 . 14 x 2 1 x 3 2 -0 . 07 x 3 1 y 2 2 +0 . 30 x 3 1 y 1 y 2 -0 . 18 x 3 1 y 2 1 +0 . 48 x 3 1 x 2 y 2 -0 . 07 x 3 1 x 2 y 1 +0 . 44 x 3 1 x 2 2 -0 . 81 x 4 1 y 2 -0 . 02 x 4 1 y 1 +0 . 33 x 4 1 x 2 -0 . 82 x 5 1 -0 . 23 x 4 2 y 2 2 -0 . 17 x 4 2 y 1 y 2 -0 . 60 x 4 2 y 2 1 +0 . 21 x 5 2 y 2 -0 . 02 x 5 2 y 1 -0 . 35 x 6 2 +0 . 55 x 1 x 3 2 y 2 2 +0 . 47 x 1 x 3 2 y 1 y 2 +0 . 22 x 1 x 3 2 y 2 1 + 0 . 14 x 1 x 4 2 y 2 -0 . 18 x 1 x 4 2 y 1 -0 . 03 x 1 x 5 2 +0 . 41 x 2 1 x 2 2 y 2 2 -0 . 57 x 2 1 x 2 2 y 1 y 2 -0 . 32 x 2 1 x 2 2 y 2 1 -0 . 59 x 2 1 x 3 2 y 2 -0 . 22 x 2 1 x 3 2 y 1 -0 . 01 x 2 1 x 4 2 -0 . 33 x 3 1 x 2 y 2 2 -0 . 53 x 3 1 x 2 y 1 y 2 -0 . 51 x 3 1 x 2 y 2 1 + 0 . 10 x 3 1 x 2 2 y 2 +0 . 33 x 3 1 x 2 2 y 1 -0 . 07 x 3 1 x 3 2 -0 . 43 x 4 1 y 2 2 -0 . 22 x 4 1 y 1 y 2 +0 . 00 x 4 1 y 2 1 +0 . 08 x 4 1 x 2 y 2 -0 . 62 x 4 1 x 2 y 1 +0 . 40 x 4 1 x 2 2 +0 . 02 x 5 1 y 2 +0 . 02 x 5 1 y 1 -0 . 17 x 5 1 x 2 +0 . 11 x 6 1 -0 . 17 x 5 2 y 2 2 +0 . 34 x 5 2 y 1 y 2 -0 . 12 x 5 2 y 2 1 +0 . 09 x 6 2 y 2 +0 . 33 x 6 2 y 1 +0 . 01 x 1 x 4 2 y 2 2 +0 . 50 x 1 x 4 2 y 1 y 2 -0 . 21 x 1 x 4 2 y 2 1 -0 . 25 x 1 x 5 2 y 2 + 0 . 22 x 1 x 5 2 y 1 +0 . 05 x 2 1 x 3 2 y 2 2 +0 . 05 x 2 1 x 3 2 y 1 y 2 -0 . 43 x 2 1 x 3 2 y 2 1 +0 . 05 x 2 1 x 4 2 y 2 -0 . 22 x 2 1 x 4 2 y 1 + 0 . 07 x 3 1 x 2 2 y 2 2 -0 . 02 x 3 1 x 2 2 y 1 y 2 -0 . 61 x 3 1 x 2 2 y 2 1 -0 . 24 x 3 1 x 3 2 y 2 -0 . 07 x 3 1 x 3 2 y 1 -0 . 43 x 4 1 x 2 y 2 2 + 0 . 04 x 4 1 x 2 y 1 y 2 +0 . 06 x 4 1 x 2 y 2 1 -0 . 62 x 4 1 x 2 2 y 2 +0 . 44 x 4 1 x 2 2 y 1 -0 . 47 x 5 1 y 2 2 -0 . 43 x 5 1 y 1 y 2 -0 . 28 x 5 1 y 2 1 -0 . 01 x 5 1 x 2 y 2 +0 . 32 x 5 1 x 2 y 1 +0 . 35 x 6 1 y 2 -0 . 12 x 6 1 y 1 +0 . 04 x 6 2 y 2 2 +0 . 41 x 6 2 y 1 y 2 -0 . 16 x 6 2 y 2 1 -0 . 43 x 1 x 5 2 y 2 2 -0 . 30 x 1 x 5 2 y 1 y 2 +0 . 51 x 1 x 5 2 y 2 1 -0 . 15 x 2 1 x 4 2 y 2 2 +0 . 27 x 2 1 x 4 2 y 1 y 2 + 0 . 28 x 2 1 x 4 2 y 2 1 +0 . 52 x 3 1 x 3 2 y 2 2 -0 . 08 x 3 1 x 3 2 y 1 y 2 +0 . 15 x 3 1 x 3 2 y 2 1 +0 . 06 x 4 1 x 2 2 y 2 2 -0 . 40 x 4 1 x 2 2 y 1 y 2 + 0 . 12 x 4 1 x 2 2 y 2 1 +0 . 26 x 5 1 x 2 y 2 2 -0 . 53 x 5 1 x 2 y 1 y 2 +0 . 09 x 5 1 x 2 y 2 1 +0 . 46 x 6 1 y 2 2 +0 . 02 x 6 1 y 1 y 2 +0 . 10 x 6 1 y 2 1 ,
```

and

```
u ′ 2 ( x, y ) = -1 . 5+0 . 34 y 2 +0 . 26 y 1 -1 . 12 x 2 -1 . 38 x 1 -0 . 55 y 2 2 -0 . 15 y 1 y 2 -1 . 06 y 2 1 -0 . 83 x 2 y 2 -0 . 48 x 2 y 1 +0 . 16 x 2 2 +0 . 32 x 1 y 2 -0 . 06 x 1 y 1 -1 . 03 x 1 x 2 -0 . 04 x 2 1 +0 . 30 x 2 y 2 2 +0 . 65 x 2 y 1 y 2 -0 . 94 x 2 y 2 1 +0 . 37 x 2 2 y 2 -0 . 23 x 2 2 y 1 -0 . 49 x 3 2 -0 . 40 x 1 y 2 2 -0 . 05 x 1 y 1 y 2 +0 . 46 x 1 y 2 1 -0 . 90 x 1 x 2 y 2 -0 . 60 x 1 x 2 y 1 +0 . 25 x 1 x 2 2 +0 . 46 x 2 1 y 2 +0 . 22 x 2 1 y 1 +0 . 35 x 2 1 x 2 -1 . 24 x 3 1 +0 . 03 x 2 2 y 2 2 -0 . 45 x 2 2 y 1 y 2 -0 . 68 x 2 2 y 2 1 +0 . 18 x 3 2 y 2 -0 . 41 x 3 2 y 1 -0 . 13 x 4 2 -0 . 21 x 1 x 2 y 2 2 -0 . 08 x 1 x 2 y 1 y 2 -0 . 14 x 1 x 2 y 2 1 + 0 . 70 x 1 x 2 2 y 2 +0 . 34 x 1 x 2 2 y 1 +0 . 41 x 1 x 3 2 +0 . 32 x 2 1 y 2 2 +0 . 50 x 2 1 y 1 y 2 +0 . 17 x 2 1 y 2 1 +0 . 57 x 2 1 x 2 y 2 -0 . 38 x 2 1 x 2 y 1 -0 . 51 x 2 1 x 2 2 -0 . 30 x 3 1 y 2 +0 . 82 x 3 1 y 1 -0 . 83 x 3 1 x 2 +0 . 15 x 4 1 -0 . 41 x 3 2 y 2 2 -0 . 23 x 3 2 y 1 y 2 -0 . 95 x 3 2 y 2 1 +0 . 31 x 4 2 y 2 +0 . 49 x 4 2 y 1 +0 . 24 x 5 2 +0 . 27 x 1 x 2 2 y 2 2 +0 . 68 x 1 x 2 2 y 1 y 2 -0 . 11 x 1 x 2 2 y 2 1 + 0 . 58 x 1 x 3 2 y 2 -0 . 06 x 1 x 3 2 y 1 -0 . 63 x 1 x 4 2 -0 . 15 x 2 1 x 2 y 2 2 -0 . 28 x 2 1 x 2 y 1 y 2 -0 . 56 x 2 1 x 2 y 2 1 -0 . 34 x 2 1 x 2 2 y 2 -0 . 01 x 2 1 x 2 2 y 1 +0 . 42 x 2 1 x 3 2 -0 . 30 x 3 1 y 2 2 -0 . 40 x 3 1 y 1 y 2 -0 . 13 x 3 1 y 2 1 -0 . 27 x 3 1 x 2 y 2 + 0 . 85 x 3 1 x 2 y 1 +0 . 36 x 3 1 x 2 2 -0 . 18 x 4 1 y 2 -0 . 33 x 4 1 y 1 +0 . 09 x 4 1 x 2 +0 . 48 x 5 1 -0 . 12 x 4 2 y 2 2 -0 . 06 x 4 2 y 1 y 2 + 0 . 43 x 4 2 y 2 1 -0 . 36 x 5 2 y 2 -0 . 22 x 5 2 y 1 -0 . 73 x 6 2 +0 . 11 x 1 x 3 2 y 2 2 +0 . 34 x 1 x 3 2 y 1 y 2 +0 . 09 x 1 x 3 2 y 2 1 + 0 . 26 x 1 x 4 2 y 2 +0 . 73 x 1 x 4 2 y 1 -1 . 50 x 1 x 5 2 -0 . 33 x 2 1 x 2 2 y 2 2 +0 . 18 x 2 1 x 2 2 y 1 y 2 +0 . 88 x 2 1 x 2 2 y 2 1 -0 . 84 x 2 1 x 3 2 y 2 +0 . 43 x 2 1 x 3 2 y 1 +0 . 32 x 2 1 x 4 2 -0 . 01 x 3 1 x 2 y 2 2 -0 . 44 x 3 1 x 2 y 1 y 2 +0 . 52 x 3 1 x 2 y 2 1 + 0 . 46 x 3 1 x 2 2 y 2 -0 . 00 x 3 1 x 2 2 y 1 -0 . 66 x 3 1 x 3 2 +0 . 00 x 4 1 y 2 2 +0 . 01 x 4 1 y 1 y 2 -0 . 69 x 4 1 y 2 1 +0 . 28 x 4 1 x 2 y 2 -0 . 78 x 4 1 x 2 y 1 -0 . 86 x 4 1 x 2 2 +0 . 64 x 5 1 y 2 +0 . 16 x 5 1 y 1 -1 . 09 x 5 1 x 2 -0 . 73 x 6 1 +0 . 05 x 5 2 y 2 2 -0 . 39 x 5 2 y 1 y 2 + 0 . 52 x 5 2 y 2 1 +0 . 22 x 6 2 y 2 +0 . 19 x 6 2 y 1 -0 . 47 x 1 x 4 2 y 2 2 -0 . 85 x 1 x 4 2 y 1 y 2 -0 . 06 x 1 x 4 2 y 2 1 -0 . 34 x 1 x 5 2 y 2 + 0 . 26 x 1 x 5 2 y 1 +0 . 30 x 2 1 x 3 2 y 2 2 -0 . 82 x 2 1 x 3 2 y 1 y 2 +0 . 56 x 2 1 x 3 2 y 2 1 +0 . 59 x 2 1 x 4 2 y 2 +0 . 67 x 2 1 x 4 2 y 1 + 0 . 18 x 3 1 x 2 2 y 2 2 -0 . 51 x 3 1 x 2 2 y 1 y 2 +0 . 51 x 3 1 x 2 2 y 2 1 +0 . 24 x 3 1 x 3 2 y 2 -0 . 77 x 3 1 x 3 2 y 1 +0 . 38 x 4 1 x 2 y 2 2 -0 . 31 x 4 1 x 2 y 1 y 2 +0 . 23 x 4 1 x 2 y 2 1 +0 . 60 x 4 1 x 2 2 y 2 -0 . 47 x 4 1 x 2 2 y 1 -0 . 34 x 5 1 y 2 2 +0 . 12 x 5 1 y 1 y 2 + 0 . 52 x 5 1 y 2 1 +0 . 58 x 5 1 x 2 y 2 +0 . 51 x 5 1 x 2 y 1 +0 . 11 x 6 1 y 2 +0 . 16 x 6 1 y 1 -0 . 34 x 6 2 y 2 2 +0 . 19 x 6 2 y 1 y 2 + 0 . 56 x 6 2 y 2 1 +0 . 41 x 1 x 5 2 y 2 2 +0 . 10 x 1 x 5 2 y 1 y 2 -0 . 38 x 1 x 5 2 y 2 1 -0 . 13 x 2 1 x 4 2 y 2 2 -0 . 21 x 2 1 x 4 2 y 1 y 2 -0 . 04 x 2 1 x 4 2 y 2 1 +0 . 04 x 3 1 x 3 2 y 2 2 +0 . 55 x 3 1 x 3 2 y 1 y 2 +0 . 05 x 3 1 x 3 2 y 2 1 +0 . 08 x 4 1 x 2 2 y 2 2 -0 . 18 x 4 1 x 2 2 y 1 y 2 + 0 . 23 x 4 1 x 2 2 y 2 1 -0 . 15 x 5 1 x 2 y 2 2 +0 . 60 x 5 1 x 2 y 1 y 2 -0 . 03 x 5 1 x 2 y 2 1 -0 . 04 x 6 1 y 2 2 -0 . 29 x 6 1 y 1 y 2 -0 . 14 x 6 1 y 2 1 .
```

## F Application: Economic Markets

Fisher markets are a special case of Arrow-Debreu markets [6] where competitive equilibria can be efficiently computed for specific classes of utility functions. In particular, Fisher markets are markets with n buyers and m divisible goods, and a certain amount of each good j in the market, denoted c j &gt; 0 . Each buyer i comes to the market with a budget w i &gt; 0 , and their objective is to obtain a bundle of goods b i ∈ R m + that maximizes their utility function u i : R m + → R + .

Computing competitive equilibria in Fisher markets is known to be PPAD-complete [10], but the works of Eisenberg and Gale [15, 16] showed that equilibrium computation is efficient if the buyers' utilities are continuous, concave and homogeneous. In recent years, many works have also leveraged techniques from algorithmic game theory to design algorithms that can compute competitive equilibria in Fisher markets in a decentralized fashion [12, 29, 13, 21, 24]. A majority of these prior works focus on Fisher markets where buyers' utilities are linear, quasilinear or Leontief.

In an effort to model more complex utility structures in markets, [22] initiated the first study on linear Fisher markets with a continuum of items. Subsequently, [65] introduced a variant of Fisher markets which captures the impact of social influence on buyers' utilities, showing that these markets can be viewed as pseudo-games , a construction from [6] which led directly to Rosen's definition of concave games. [11] also utilized a variational inequality approach to study monotone variants of these games, and presented decentralized algorithms that converge to equilibria. Indeed, SOSconcave and SOS-monotone games allow the study of Fisher markets with social influence for which concavity/monotonicity is verifiable. In particular, an economist who is constructing a model for a market can use our proposed methods in two ways. First, they can use the hierarchy in Eq. (17) to verify whether a game is concave or monotone. They can also use the hierarchy in Eq. (19) to search within the class of SOS-concave/monotone games in order to ensure that their market model satisfies equilibrium existence and even uniqueness.

## NeurIPS Paper Checklist

## 1) Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Our claims are proven rigorously, and all assumptions are clearly provided.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2) Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We provide a discussion section that outlines the limitations of the proposed techniques, particularly scalability.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3) Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: Each theoretical result has a corresponding proof, either in the main text or in the appendix. Where appropriate, we have also included high-level proof sketches.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4) Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We provide the full code used to obtain our experimental results.

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

## 5) Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: Yes, the code utilizes standard packages and is provided fully.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6) Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: All experimental details are given in the main paper.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7) Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: The experiments are deterministic.

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

## 8) Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The code can be run with minimal compute, and details are given in the main text.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9) Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The work is primarily theoretical and the experiments do not have a human element.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10) Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The work is primarily theoretical, but we mention potential broader impacts of our work in the discussion section.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11) Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The results and code in the paper have almost no potential risk for misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12) Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The SumOfSquares package used for our experiments are appropriately cited.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13) New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: Our codebase is documented and provided alongside the relevant scripts.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14) Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not utilize any crowdsourcing.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15) Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [No]

Justification: The paper is theoretical in nature and thus has no human subjects involved.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16) Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development (i.e. theoretical results/proofs and experiments) of this work did not involve LLMs.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.