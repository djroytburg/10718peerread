## The Complexity of Symmetric Equilibria in Min-Max Optimization and Team Zero-Sum Games ∗

Ioannis Anagnostides 1 , Ioannis Panageas 2 , Tuomas Sandholm 1,3 , and Jingming Yan 2

2

{ianagnos,sandholm}@cs.cmu.edu ,

1 Carnegie Mellon University University of California, Irvine 3 Strategy Robot, Inc. 3 Strategic Machine, Inc. 3 Optimized Markets, Inc. {ipanagea,jingmy1}@uci.edu

## Abstract

We consider the problem of computing stationary points in min-max optimizaadversary-complete, resolving the complexity of Nash equilibria in such settings.

tion, with a focus on the special case of Nash equilibria in (two-)team zero-sum games. We first show that computing ϵ -Nash equilibria in 3 -player adversarial team games-wherein a team of 2 players competes against a single is CLS Our proof proceeds by reducing from symmetric ϵ -Nash equilibria in symmetric , identical-payoff, two-player games, by suitably leveraging the adversarial player so as to enforce symmetry-without disturbing the structure of the game. In particular, the class of instances we construct comprises solely polymatrix games, thereby also settling a question left open by Hollender, Maystre, and Nagarajan (2024). Moreover, we establish that computing symmetric (first-order) equilibria in symmetric min-max optimization is PPAD -complete, even for quadratic functions. Building on this reduction, we show that computing symmetric ϵ -Nash equilibria in symmetric, 6 -player ( 3 vs. 3 ) team zero-sum games is also PPAD -complete, even for ϵ = poly (1 /n ) . As a corollary, this precludes the existence of symmetric dynamics-which includes many of the algorithms considered in the literatureconverging to stationary points. Finally, we prove that computing a non-symmetric poly (1 /n ) -equilibrium in symmetric min-max optimization is FNP -hard.

## 1 Introduction

We consider computing local equilibria in constrained min-max optimization problems of the form

<!-- formula-not-decoded -->

where X ⊆ R d x and Y ⊆ R d y are convex and compact constraint sets, and f : X×Y → R is a smooth objective function. Tracing all the way back to Von Neumann's celebrated minimax theorem [von Neumann, 1928] and the inception of game theory, such problems are attracting renewed interest in recent years propelled by a variety of modern machine learning applications, such as generative modeling [Goodfellow et al., 2014], reinforcement learning [Daskalakis et al., 2020, Bai and Jin, 2020, Wei et al., 2021], and adversarial robustness [Madry et al., 2018, Cohen et al., 2019, Bai et al., 2021, Carlini et al., 2019]. Another prominent class of problems encompassed by (1) concerns computing Nash equilibria in (two-)team zero-sum games [Zhang et al., 2023, 2021, Basilico et al., 2017, von Stengel

∗ The authors are ordered alphabetically.

and Koller, 1997, Carminati et al., 2023, Orzech and Rinard, 2023, Farina et al., 2018, Zhang and An, 2020, Celli and Gatti, 2018, Schulman and Vazirani, 2017], which is a primary focus of this paper.

Perhaps the most natural solution concept-guaranteed to always exist-pertaining to (1), when f is nonconvex-nonconcave, is a pair of strategies ( x ∗ , y ∗ ) such that both players (approximately) satisfy the associated first-order optimality conditions [Tsaknakis and Hong, 2021, Jordan et al., 2023, Ostrovskii et al., 2021, Nouiehed et al., 2019], as formalized in the definition below.

Definition 1.1. A point ( x ∗ , y ∗ ) ∈ X × Y is an ϵ -first-order Nash equilibrium of (1) if

<!-- formula-not-decoded -->

Definition 1.1 can be equivalently recast as a variational inequality (VI) problem: if z := ( x , y ) and F : z ↦→ F ( z ) := ( ∇ x f ( x , y ) , -∇ y f ( x , y )) , we are searching for a point z ∗ ∈ Z := X × Y such that ⟨ z -z ∗ , F ( z ∗ ) ⟩ ≥ -2 ϵ for all z ∈ Z . Yet another equivalent definition is instead based on approximate fixed points of gradient descent/ascent (GDA); namely, Definition 1.1 amounts to bounding the gradient mappings

<!-- formula-not-decoded -->

for some approximation parameter ϵ ′ &gt; 0 that is (polynomially) dependent on ϵ &gt; 0 , where ∥ · ∥ is the (Euclidean) ℓ 2 norm and Π( · ) is the projection operator. Other definitions that differentiate between the order of play between players-based on the notion of a Stackelberg equilibrium-have also been considered in the literature [Jin et al., 2020].

The complexity of min-max optimization is well-understood in certain special cases, such as when f is convex-concave ( e.g. , Korpelevich [1976], Mertikopoulos et al. [2019], Cai et al. [2022], Choudhury et al. [2023], Gorbunov et al. [2022], and references therein), or more broadly, nonconvexconcave [Lin et al., 2020, Xu et al., 2023, Luo et al., 2020]. However, the complexity of general min-max optimization problems, when the objective function f is nonconvex-nonconcave, has remained wide open despite intense efforts in recent years. Daskalakis et al. [2021] made progress by establishing certain hardness results targeting the more challenging setting in which there is a joint (that is, coupled) set of constraints. In fact, it turns out that their lower bounds apply even for linear-nonconcave objective functions ( cf. Bernasconi et al. [2024]), showing that their hardness result is driven by the presence of joint constraints-indeed, under uncoupled constraints, many efficient algorithms attaining Definition 1.1 (for linear-nonconcave problems) have been documented in the literature. In the context of min-max optimization, the most well-studied setting posits that players have independent constraints; this is the primary focus of our paper.

## 1.1 Our results

We establish new complexity lower bounds in min-max optimization for computing equilibria in the sense of Definition 1.1; our main results are gathered in Table 1.

Table 1: The main results of this paper. NE stands for Nash equilibrium and FONE for first-order Nash equilibrium (Definition 1.1). We also abbreviate symmetric to 'sym.' (second column).

| Class of problems      | Eq. concept          | Complexity                                                                 | Even for                                                                                      |
|------------------------|----------------------|----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| Adversarial team games | ϵ -NE sym. ϵ -FONE ϵ | CLS -complete (Theorem 1.3) PPAD -complete (Theorem NP -hard (Theorem 1.6) | 3 -player ( 2 vs 1 ), polymatrix polymatrix, team 0 -sum, ϵ = 1 / n c quadratics, ϵ = 1 / n c |
| Symmetric min-max      | non-sym. -FONE       | 1.7)                                                                       |                                                                                               |

Adversarial team games We first examine an important special case of (1): adversarial team games [von Stengel and Koller, 1997]. Here, a team of n players with identical interests is competing against a single adversarial player. (In such settings, Definition 1.1 captures precisely the Nash equilibria of the game.) The computational complexity of this problem was placed by Anagnostides et al. [2023] in the complexity class CLS -which stands for continuous local search [Daskalakis and Papadimitriou, 2011]. Further, by virtue of a result of Babichenko and Rubinstein [2021], computing Nash equilibria in adversarial team games when n ≫ 1 is CLS -complete. In the context of this prior work, an important question left open by Anagnostides et al. [2023] concerns the case where n is a small constant, a regime not captured by the hardness result of Babichenko and Rubinstein [2021]

pertaining to identical-interest games-in such games, one can simply identify the strategy leading to the highest payoff, which is tractable when n is small.

We show that even when n = 2 , computing an ϵ -Nash equilibrium in adversarial team games is CLS -complete. (The case where n = 1 amounts to two-player zero-sum games, known to be in P .)

Theorem 1.2. Computing an ϵ -Nash equilibrium in 3 -player (that is, 2 vs. 1 ) adversarial team games is CLS -complete.

Coupled with earlier results, Theorem 1.2 completely characterizes the complexity landscape for computing Nash equilibria in adversarial team games.

Our proof is based on a recent hardness result of Ghosh and Hollender [2024] ( cf. Tewolde et al. [2025]), who proved that computing a symmetric ϵ -Nash equilibrium in a symmetric two-player game with identical payoffs is CLS -complete. The key idea in our reduction is that one can leverage the adversarial player so as to enforce symmetry between the team players, without affecting the equilibria of the original game; the basic gadget underpinning this reduction is analyzed in Section 3.1.

Incidentally, our CLS -hardness reduction hinges on a polymatrix adversarial team game, thereby addressing another open question left recently by Hollender et al. [2025].

Theorem 1.3. Theorem 1.2 holds even when one restricts to polymatrix, 3 -player adversarial team games.

We complement the above hardness result by further characterizing the complexity of deciding whether an adversarial team game admits a unique (approximate) Nash equilibrium (Theorem 3.5).

Symmetric min-max optimization As we have seen, symmetry plays a key role in the proof of Theorem 1.2, but that result places no restrictions on whether the equilibrium is symmetric or not-this is indeed the crux of the argument. The next problem we consider concerns computing symmetric equilibria in symmetric min-max optimization problems, in the following natural sense.

Definition 1.4 (Symmetric min-max optimization) . A function f : X × Y → R is called antisymmetric if X = Y and

<!-- formula-not-decoded -->

Furthermore, a point ( x , y ) ∈ X × Y is called symmetric if x = y . The associated min-max optimization problem is called symmetric if the underlying function f is antisymmetric . 2

Symmetric zero-sum games are ubiquitous in the literature and in practical applications alike. Many popular recreational games used for AI benchmarking, such as poker and battleship, are symmetric when roles are assigned at random; the symmetry assumption is particularly natural as it ensures that no player has an a priori advantage before the game begins.

The study of symmetric equilibria has a long history in the development of game theory, propelled by Nash's pathbreaking PhD thesis [Nash, 1950] ( cf. Gale et al. [1951]), and has remained a popular research topic ever since [Tewolde et al., 2025, Emmons et al., 2022, Garg et al., 2018, Ghosh and Hollender, 2024, Mehta, 2014]. Classic examples in game theory, including rock-paper-scissors and matching pennies, are also symmetric; these games were already discussed in the original work of von Neumann [1928].

It is not hard to see that symmetric min-max optimization problems, in the sense of Definition 1.4, always admit symmetric first-order Nash equilibria. What is more, we show that computing such a symmetric equilibrium is in the complexity class PPAD [Papadimitriou, 1994]; this is based on an argument of Etessami and Yannakakis [2010], and complements Daskalakis et al. [2021], who proved that the problem of computing approximate fixed points of gradient descent/ascent-which they refer to as GDAFIXEDPOINT-lies in PPAD . In a celebrated series of work, it was shown that PPAD captures the complexity of computing Nash equilibria in finite games [Daskalakis et al., 2009, Chen et al., 2009]. In this context, we establish that PPAD also characterizes the complexity of computing symmetric first-order Nash equilibria in symmetric min-max optimization problems:

Theorem 1.5. Computing a symmetric 1 / n c -approximate first-order Nash equilibrium in symmetric n -dimensional min-max optimization is PPAD -complete for any constant c &gt; 0 .

2 The nomenclature of this definition is consistent with the usual terminology in the context of (two-player) zero-sum games: a symmetric zero-sum game is one in which the the underlying game matrix A is antisymmetric (that is, skew-symmetric), so that ⟨ x , A y ⟩ = -⟨ y , A x ⟩ for all ( x , y ) .

Barring major complexity breakthroughs, Theorem 1.5 precludes the existence of algorithms with complexity polynomial in the dimension and 1 /ϵ , where ϵ &gt; 0 measures the precision (per Definition 1.1), under the symmetry constraint of Definition 1.4. This stands in contrast to (nonconvex) minimization problems, wherein gradient descent converges to stationary points at a rate of poly (1 /ϵ ) ; even in the regime where ϵ = 1 / exp( n ) , computing a stationary point of a smooth function is in CLS [Daskalakis and Papadimitriou, 2011], which is a subclass of PPAD [Fearnley et al., 2023]. In fact, our reduction also rules out the existence of polynomial-time algorithms even when ϵ = Θ(1) under some well-believed complexity assumptions (Corollary 4.3).

The proof of Theorem 1.5 is elementary, and is based on the PPAD -hardness of computing symmetric Nash equilibria in symmetric two-player games. Importantly, our reduction gives an immediate, and significantly simpler, proof (Theorem 4.4) of the PPAD -hardness result of Daskalakis et al. [2021], while being applicable even with respect to quadratic and anti-symmetric functions defined on a product of simplexes.

Independent and concurrent work Bernasconi et al. [2024] also considerably simplified the proof of Daskalakis et al. [2021]. Our hardness result hinges on the intermediate problem of finding a Nash equilibrium in symmetric two-player games [Chen et al., 2009], whereas Bernasconi et al. [2024] showed their hardness result via the problem of finding a Nash equilibrium in (multi-player) polymatrix two-action games. The main qualitative difference between the two is that ours applies to simplex domains while the result of Bernasconi et al. [2024] to box domains. The basic idea of both reductions then is that one can enforce the symmetry constraint x ≈ y via coupled constraints.

̸

As a byproduct of Theorem 1.5 and the result of Bernasconi et al. [2024], it follows that any symmetric dynamics-whereby both players follow the same online algorithm, as formalized in Definition 4.5-cannot converge to a first-order Nash equilibrium in polynomial time, subject to PPAD = P (Theorem 4.6). This already captures many natural dynamics for which prior papers in the literature ( e.g. , Kalogiannis et al. [2023b]) have painstakingly shown lack of convergence; Theorem 4.6 provides a complexity-theoretic justification for such prior results, while precluding a much broader family of algorithms.

The complexity of non-symmetric equilibria Remaining on symmetric min-max optimization, one natural question arising from Theorem 1.5 concerns the complexity of non-symmetric equilibriadefined as having distance at least δ &gt; 0 . Unlike their symmetric counterparts, non-symmetric first-order Nash equilibria are not guaranteed to exist. In fact, we establish the following result.

Theorem 1.6. For a symmetric min-max optimization problem, constants c 1 , c 2 &gt; 0 , and ϵ = n -c 1 , it is NP -hard to distinguish between the following two cases under the promise that one of them holds:

- any ϵ -first-order Nash equilibrium ( x ∗ , y ∗ ) satisfies ∥ x ∗ -y ∗ ∥ ≤ n -c 2 , and
- there is an ϵ -first-order Nash equilibrium ( x ∗ , y ∗ ) such that ∥ x ∗ -y ∗ ∥ ≥ Ω(1) .

The main technical piece is Theorem 4.7, which concerns symmetric, identical-interest, two-player games. It significantly refines the hardness result of McLennan and Tourky [2010] by accounting even for poly (1 /n ) -Nash equilibria.

Team zero-sum games Finally, building on the reduction of Theorem 1.5 coupled with the gadget behind Theorem 1.2, we establish similar complexity results for team zero-sum games, which generalize adversarial team games by allowing the presence of multiple adversaries. In particular, a symmetric two-team zero-sum game and a symmetric equilibrium thereof are in accordance with Definition 1.4-no symmetry constraints are imposed within the same team, but only across teams. We obtain a result significantly refining Theorem 1.5.

Theorem 1.7. Computing a symmetric 1 / n c -Nash equilibrium in symmetric, 6 -player ( 3 vs. 3 ) team zero-sum polymatrix games is PPAD -complete for some constant c &gt; 0 .

Unlike our reduction in Theorem 1.5 that comprises quadratic terms, the crux in team zero-sum games is that one needs to employ solely multilinear terms. The basic idea is to again use the gadget underpinning Theorem 1.2, which enforces symmetry without affecting the equilibria of the game, thereby (approximately) reproducing the objective function that establishes Theorem 1.5.

It is interesting to note that the class of polymatrix games we construct to prove Theorem 1.7 belongs to a certain family introduced by Cai and Daskalakis [2011]: one can partition the players into 2

groups so that any pairwise interaction between players of the same group is a coordination game, whereas any pairwise interaction across groups is a zero-sum game. Cai and Daskalakis [2011] showed that computing a Nash equilibrium is PPAD -hard in the more general case where there are 3 groups of players. While the complexity of that problem under 2 groups remains wide open, Theorem 1.7 shows PPAD -hardness for computing symmetric Nash equilibria in such games.

Taken together, our results bring us closer to characterizing the complexity of computing equilibria in min-max optimization.

## 1.2 Further related work

Adversarial team games have been the subject of much research tracing back to the influential work of von Stengel and Koller [1997], who introduced the concept of a team maxmin equilibrium (TME) ; a TME can be viewed as the best Nash equilibrium for the team. Notwithstanding its intrinsic appeal, it turns out that computing a TME is FNP -hard [Borgs et al., 2010]. Indeed, unlike two-player zero-sum games, team zero-sum games generally exhibit a duality gap -characterized in the work of Schulman and Vazirani [2017].

This realization has shifted the focus of contemporary research to exploring more permissive solution concepts. One popular such relaxation is TMECor , which enables team players to ex ante correlate their strategies [Zhang et al., 2023, 2021, Basilico et al., 2017, Carminati et al., 2022, Farina et al., 2018, Zhang and An, 2020, Celli and Gatti, 2018]. Yet, in the context of extensive-form games, computing a TMECor remains intractable; Zhang et al. [2023] provided an exact characterization of its complexity. Team zero-sum games can be thought of as two-player zero-sum games but with imperfect recall , and many natural problems immediately become hard without perfect recall ( e.g. , Tewolde et al. [2023]). Parameterized algorithms have been developed for computing a TMECor based on some natural measure of shared information [Zhang et al., 2023, Carminati et al., 2022]. Beyond adversarial team games, Carminati et al. [2023] recently explored hidden-role games, wherein there is uncertainty regarding which players belong in the same team, a feature that often manifests itself in popular recreational games-and used certain cryptographic primitives to solve them.

In contrast, this paper focuses on the usual Nash equilibrium concept, being thereby orthogonal to the above line of work. One drawback of Nash equilibria in adversarial team games is that the (worst-case) value of the team can be significantly lower compared to TME [Basilico et al., 2017]. On the other hand, Anagnostides et al. [2023] showed that ϵ -Nash equilibria in adversarial team games admit an FPTAS , which stands in stark contrast to TME, and indeed, Nash equilibria in general games [Daskalakis et al., 2009, Chen et al., 2009]. This was further strengthened by Kalogiannis et al. [2023a, 2024] for computing ϵ -Nash equilibria in adversarial team Markov games-the natural generalization to Markov (aka. stochastic) games.

Related to Definition 1.1 is the natural notion of a local min-max equilibrium [Daskalakis and Panageas, 2018, Daskalakis et al., 2021]. It is easy to see that any local min-max equilibrium-with respect to a sufficiently large neighborhood of ( x ∗ , y ∗ ) -must satisfy Definition 1.1 [Daskalakis et al., 2021]. Unlike first-order Nash equilibria, local min-max equilibria are not guaranteed to exist.

Finally, Mehta et al. [2015] showed that in two-player symmetric games, deciding whether a nonsymmetric Nash equilibrium exists is NP -hard, which directly relates to our Theorem 1.6.

## 2 Preliminaries

Notation We use boldface lowercase letters, such as x , y , z , to represent vectors, and boldface capital letters, such as A , C , for matrices. We denote by x i the i th coordinate of a vector x ∈ R n . We use the shorthand notation [ n ] := { 1 , 2 , . . . , n } . ∆ n := { x ∈ R n ≥ 0 : ∑ n i =1 x i = 1 } is the probability simplex on R n . For i ∈ [ n ] , e i ∈ ∆ n is the i th unit vector. ⟨· , ·⟩ denotes the inner product. For a vector x ∈ R n , ∥ x ∥ 2 = √ ⟨ x , x ⟩ is its Euclidean norm. For m ≤ n , x [1 ··· m ] ∈ R m is the vector containing the first m coordinates of x . We sometimes use the O ( · ) , Θ( · ) , Ω( · ) notation to suppress absolute constants. A continuously differentiable function f is L -smooth if its gradient is L -Lipschitz continuous with respect to ∥ · ∥ 2 ; that is, ∥∇ f ( x ) -∇ f ( x ′ ) ∥ 2 ≤ L ∥ x -x ′ ∥ 2 for all x , x ′ .

Two-player games In a two-player game, represented in normal-form game, each player has a finite set, let [ n ] , of actions. Under a pair of actions ( i, j ) ∈ [ n ] × [ n ] , the utility of the row player is given by R i,j , where R ∈ Q n × n is the payoff matrix of the row player. Further, we let C ∈ Q n × n be the payoff matrix of the column player. Players are allowed to randomize by selecting mixed strategies-points in ∆ n . Under a pair of mixed strategies ( x , y ) ∈ ∆ n × ∆ n , the expected utility of the players is given by ⟨ x , R y ⟩ and ⟨ x , C y ⟩ , respectively. The canonical solution concept in such games is the Nash equilibrium [Nash, 1951], which is recalled below.

Definition 2.1. A pair of strategies ( x ∗ , y ∗ ) is an ϵ -Nash equilibrium of ( R , C ) if

<!-- formula-not-decoded -->

Symmetric two-player games One of our reductions is based on symmetric two-player games, meaning that R = C ⊤ . Abasic fact is that any symmetric game admits a symmetric Nash equilibrium ( x ∗ , x ∗ ) . Further, computing a Nash equilibrium in a general game can be reduced to computing a symmetric Nash equilibrium in a symmetric game [Nisan et al., 2007, Theorem 2.4]. In conjunction with the hardness result of Chen et al. [2009], we state the following consequence.

Theorem 2.2 (Chen et al., 2009) . Computing a symmetric 1 / n c -Nash equilibrium in a symmetric two-player game is PPAD -hard for any constant c &gt; 0 .

Team zero-sum games A (two-)team zero-sum game is a multi-player game-represented in normal form for the purposes of this paper-in which the players' utilities have a certain structure; namely, we can partition the players into two (disjoint) subsets, such that each player within the same team shares the same utility, whereas players in different teams have opposite utilities-under any possible combination of actions. An adversarial team game is a specific type of team zero-sum game wherein one team consists of a single player. As in Definition 2.1 for two-player games, an ϵ -Nash equilibrium is a tuple of strategies such that no unilateral deviation yields more than an ϵ additive improvement in the utility of the deviator.

## 3 Complexity of adversarial team games

We begin by examining equilibrium computation in adversarial team games.

## 3.1 CLS -completeness for 3-player games

Computing ϵ -Nash equilibria in adversarial team games was placed in CLS by Anagnostides et al. [2023], but whether CLS tightly characterizes the complexity of that problem remained open-that was only known when the number of players is large, so that the hardness result of Babichenko and Rubinstein [2021] can kick in. Our reduction here answers this question in the affirmative.

We rely on a recent hardness result of Ghosh and Hollender [2024] concerning symmetric, two-player games with identical payoffs. We summarize their main result below.

Theorem 3.1 (Ghosh and Hollender, 2024) . Computing an ϵ -Nash equilibrium in a symmetric, identical-payoffs, two-player game is CLS -complete.

Now, let A ∈ Q n × n be the common payoff matrix of a two-player game, which satisfies A = A ⊤ so that the game is symmetric. Without loss of generality, we will assume that A i,j ≤ -1 for all i, j ∈ [ n ] . Wedenote by A min and A max the minimum and maximum entry of A , respectively (which satisfy A max , A min ≤ -1 ). The basic idea of our proof is to suitably use the adversarial player so as to force the other two players to play roughly the same strategy (Lemma 3.2), while (approximately) maintaining the structure of the game (Lemma 3.3). The formal proofs are in Section A.1.

Definition of the adversarial team game Based on A , we construct a 3 -player adversarial team game as follows. The utility function of the adversary reads

<!-- formula-not-decoded -->

The adversary selects a strategy z ∈ ∆ 2 n +1 , while the team players, who endeavor to minimize (2), select strategies x ∈ ∆ n and y ∈ ∆ n , respectively. (While the range of the utilities in (2) grows with 1 / ϵ , normalizing to [ -1 , 1] maintains all of the consequences by suitably adjusting the approximation.)

The first important lemma establishes that, in equilibrium, x ≈ y . The basic argument proceeds as follows. By construction of (2), the adversary would be able to secure a large payoff whenever there is a coordinate i ∈ [ n ] such that | x i -y i |≫ 0 -by virtue of the second term in (2). But that cannot happen in equilibrium, for Player x (or symmetrically Player y ) can simply neutralize that term in the adversary's utility by playing x = y .

Lemma 3.2 (Equilibrium forces symmetry) . Consider an ϵ 2 -Nash equilibrium ( x ∗ , y ∗ , z ∗ ) of the adversarial team game (2) with ϵ 2 ≤ 1 / 2 . Then, ∥ x ∗ -y ∗ ∥ ∞ ≤ 2 ϵ .

Having established that x ≈ y , the next step is to make sure that the adversarial player does not distort the original game by much. In particular, we need to make sure that the effect of the second term in (2) is negligible. We do so by showing that z 2 n +1 ≈ 1 (Lemma 3.3).

The argument here is more subtle; roughly speaking, it goes as follows. Suppose that z i ≫ 0 or z n + i ≫ 0 for some i ∈ [ n ] . Since Player z is approximately best responding, it would then follow that | y ∗ i -x ∗ i |≫ 0 -otherwise Player z would prefer to switch to action 2 n +1 . But, if | y ∗ i -x ∗ i |≫ 0 , Player x could profitably deviate by reallocating probability mass by either removing from or adding to i (depending on whether y ∗ i -x ∗ i &gt; 0 ), which leads to a contradiction.

Lemma 3.3 (Most probability mass in a 2 n +1 ) . Given any ϵ 2 -Nash equilibrium ( x ∗ , y ∗ , z ∗ ) of the adversarial team game (2) with ϵ ≤ 1 / 10 , z j ≤ 9 ϵ for all j ∈ [2 n ] . In particular, z 2 n +1 ≥ 1 -18 nϵ .

By combining Lemmas 3.2 and 3.3, we can complete the reduction from symmetric two-player games with common payoffs to 3 -player adversarial team games, as stated below.

Theorem 3.4. Given any ϵ 2 -Nash equilibrium ( x ∗ , y ∗ , z ∗ ) in the adversarial team game (2) , with ϵ ≤ 1 / 10 , ( y ∗ , y ∗ ) is a symmetric (21 n +1) | A min | ϵ -Nash equilibrium of the symmetric, two-player game ( A , A ) (that is, A = A ⊤ ).

## 3.2 The complexity of determining uniqueness

Another natural question concerns the complexity of determining whether an adversarial team game admits a unique Nash equilibrium. Our next theorem establishes NP -hardness for a version of that problem that accounts for approximate Nash equilibria.

Theorem 3.5. For polymatrix, 3 -player adversarial team games, constants c 1 , c 2 &gt; 0 , and ϵ = n -c 1 , it is NP -hard to distinguish between the following two cases under the promise that one of them holds:

- any two ϵ -Nash equilibria have ℓ 1 -distance at most n -c 2 , and
- there are two ϵ -Nash equilibria that have ℓ 1 -distance Ω(1) .

We will discuss more about the proof of this theorem later in Section 4.2 when we examine the complexity of computing non-symmetric equilibria in symmetric min-max optimization problems. It is also interesting to point out that an adversatial team game can have a unique Nash equilibrium supported on irrational numbers, as we show in Section A.2.

## 4 Complexity of equilibria in symmetric min-max optimization

This section characterizes the complexity of computing symmetric first-order Nash equilibria (Definition 1.1) in symmetric min-max optimization problems in the sense of Definition 1.4; namely, when f ( x , y ) = -f ( y , x ) for all ( x , y ) ∈ X × Y and X = Y .

## 4.1 Problem definitions and hardness results for symmetric equilibria

Given a continuously differentiable function f : D → R , we set F GDA : D → D to be

<!-- formula-not-decoded -->

the norm of which measures the fixed-point gap and corresponds to the update rule of GDA with stepsize equal to one; we recall that Player x is the minimizer, while Player y is the maximizer. The domain D is a compact subset of R d for some d ∈ N . Moreover, the projection operator ∏ is applied

jointly on D . 3 When D can be expressed as a Cartesian product X × Y , the domain set is called uncoupled (and the projection can be done independently), otherwise it is called coupled (or joint ).

We begin by introducing the problem of computing fixed points of gradient descent/ascent (GDA) for domains expressed as the Cartesian product of polytopes, modifying the computational problem GDAFIXEDPOINT introduced by Daskalakis et al. [2021].

## GDAFIXEDPOINT Problem.

INPUT:

- Precision parameter ϵ &gt; 0 and smoothness parameter L ,
- Polynomial-time Turing machine C f evaluating a L -smooth function f : X×Y → R and its gradient ∇ f : X ×Y → R d , where X = { x : A x x ≤ b x } and Y = { y : A y y ≤ b y } are nonempty, bounded polytopes described by input matrices A x ∈ R m x × d x , A y ∈ R m y × d y and vectors b x ∈ R m x , b y ∈ R m y , with d := d x + d y .

<!-- formula-not-decoded -->

Based on GDAFIXEDPOINT, we introduce the problem SYMGDAFIXEDPOINT, which captures the problem of computing symmetric (approximate) fixed points of GDA for symmetric min-max optimization problems. We define our computational problems as promise problems.

## SYMGDAFIXEDPOINT Problem.

INPUT:

- Precision parameter ϵ &gt; 0 and smoothness parameter L ,
- Polynomial-time Turing machine C f evaluating a L -smooth, antisymmetric function f : X × X → R and its gradient ∇ f : X × X → R 2 d , where X = { x : A x ≤ b } is a nonempty, bounded polytope described by an input matrix A ∈ R m × d and vector b ∈ R m .

<!-- formula-not-decoded -->

We start by showing that SYMGDAFIXEDPOINT also lies in PPAD ; the fact that GDAFIXEDPOINT is in PPAD -even under coupled domains-was shown to be the case by Daskalakis et al. [2021]. The detailed proof is included in the appendix.

Lemma 4.1. SYMGDAFIXEDPOINT is a total search problem and lies in PPAD .

Having established that SYMGDAFIXEDPOINT belongs in PPAD , we now state the first main hardness result of this section.

Theorem 4.2 (Complexity for symmetric equilibrium) . SYMGDAFIXEDPOINT is PPAD -complete, even for quadratic functions.

The basic idea of the proof is to consider the objective

<!-- formula-not-decoded -->

where A is symmetric and C is skew-symmetric. Theorem 4.2 then follows from some elementary calculations, as we show in Section A.3.

For symmetric first-order Nash equilibria, our argument establishes PPAD -hardness for any ϵ ≤ 1 / n c , where c &gt; 0 (as claimed in Theorem 1.5). Moreover, leveraging the hardness result of Rubinstein [2016], we can also immediately obtain constant inapproximability under the so-called exponentialtime hypothesis (ETH) for PPAD -which postulates than any algorithm for solving ENDOFALINE, the prototypical PPAD -complete problem, requires 2 ˜ Ω( n ) time.

Corollary 4.3. Computing an Θ(1) -approximate first-order Nash equilibrium in symmetric n -dimensional min-max optimization requires n ˜ Ω(log n ) time, assuming ETH for PPAD .

3 This is the 'safe' version of GDA because it ensures that the mapping always lies in D . One could also project independently on D ( y ) = { x ′ : ( x ′ , y ) ∈ D} and D ( x ) = { y ′ : ( x , y ′ ) ∈ D} ; see Daskalakis et al. [2021] for further details and the polynomial equivalence for finding fixed points for both versions.

The argument of Theorem 4.2 can be slightly modified to imply the main result of Daskalakis et al. [2021]-with simplex instead of box constraints-as stated below.

Theorem 4.4 ( PPAD -hardness for coupled domains) . The problem GDAFIXEDPOINT is PPAD -hard when the domain is a joint polytope, even for quadratic functions.

The main idea is to add coupled constraints in order to force symmetry : -δ ≤ x i -y i ≤ δ for all i ∈ [ n ] , where, if ϵ is the approximation accuracy, δ is of order Θ ( ϵ 1 / 4 ) . Compared to the equilibrium studied in Daskalakis et al. [2021], the symmetric equilibrium considered in our work is stronger in that it accounts for all deviations, not merely ones on the coupled feasibility set. We present the proof of Theorem 4.4 in Section A.3.

Hardness results for symmetric dynamics Another interesting consequence of Theorem 4.2 is that it precludes convergence under a broad class of algorithms in general min-max optimization. Definition 4.5 (Symmetric learning algorithms for min-max) . Let T ∈ N . A deterministic, polynomial-time learning algorithm A proceeds as follows for any time t ∈ [ T ] . It outputs a strategy as a function of the history H ( t ) it has observed so far (where H (1) := ∅ ), and then receives as feedback g ( t ) . It then updates H ( t +1) := ( H ( t ) , g ( t ) ) . A symmetric learning algorithm in min-max optimization consists of Player x employing algorithm A with history H ( t ) x := ( ∇ x f ( x ( t ) , y ( t ) )) T t =1 , and Player y employing the same algorithm with history H ( t ) y := ( -∇ y f ( x ( t ) , y ( t ) )) T t =1 .

Note that a consequence of the above definition is that both players initialize from the same strategy. Many natural and well-studied algorithms in min-max optimization adhere to Definition 4.5. Besides the obvious example of gradient descent/ascent, we mention extragradient descent(/ascent), optimistic gradient descent(/ascent), and optimistic multiplicative weights-all assumed to be executed simultaneously. A simple non-example is alternating gradient descent(/ascent) [Wibisono et al., 2022, Bailey et al., 2020], wherein players do not update their strategies simultaneously.

Theorem 4.6. No symmetric learning algorithm (per Definition 4.5) can converge to ϵ -first-order Nash equilibria in min-max optimization in polynomial time when ϵ = 1 / n c , unless PPAD = P .

This is a consequence of our argument in Theorem 4.2: under Definition 4.5 and the min-max optimization problem (3), it follows inductively that x ( t ) = y ( t ) and H ( t ) x = H ( t ) y for all t ∈ [ T ] . But computing a symmetric first-order Nash equilibrium is PPAD -hard when ϵ = 1 / n c (Theorem 4.2).

̸

Assuming that P = PPAD , Theorem 4.6, and in particular its instantiation in team zero-sum games (Theorem 1.7), significantly generalizes some impossibility results shown by Kalogiannis et al. [2023b] concerning certain algorithms, such as optimistic gradient descent(/ascent)-our hardness result goes much further, precluding any algorithm subject to Definition 4.5, albeit being conditional.

## 4.2 The complexity of non-symmetric fixed points

An immediate question raised by Theorem 4.2 concerns the computational complexity of finding non-symmetric fixed points of GDA for symmetric min-max optimization problems. Since totality is not guaranteed, unlike SYMGDAFIXEDPOINT, we cannot hope to prove membership in PPAD . In fact, we show that finding a non-symmetric fixed point of GDA is FNP -hard. To do so, we first define formally the computational problem of interest.

## NONSYMGDAFIXEDPOINT Problem.

INPUT:

- Parameters ϵ, δ &gt; 0 and Lipschitz constant L and
- Polynomial-time Turing machine C f evaluating a L -smooth antisymmetric function f : X × X → R and its gradient ∇ f : X × X → R 2 d , where X = { x : A x ≤ b } is a nonempty, bounded polytope described by a matrix A ∈ R m × d and vector b ∈ R m .

OUTPUT: A point ( x ∗ , y ∗ ) ∈ X × X such that ∥ x ∗ -y ∗ ∥ 2 ≥ δ and ∥ ( x ∗ , y ∗ ) -F GDA ( x ∗ , y ∗ ) ∥ 2 ≤ ϵ if it exists, otherwise return NO .

We establish that NONSYMGDAFIXEDPOINT is FNP -hard. Our reduction builds on the hardness result of McLennan and Tourky [2010]-in turn based on earlier work by Gilboa and Zemel [1989],

Conitzer and Sandholm [2008]-which we significantly refine in order to account for poly (1 /n ) -Nash equilibria. Our result, which forms the basis for Theorem 1.6 and Theorem 3.5, is summarized below.

Theorem 4.7. For symmetric, identical-interest, two-player games, constants c 1 , c 2 &gt; 0 , and ϵ = n -c 1 , it is NP -hard to distinguish between the following two cases under the promise that one of them holds:

- any two symmetric ϵ -Nash equilibria have ℓ 1 -distance at most n -c 2 , and
- there are two symmetric ϵ -Nash equilibria that have ℓ 1 -distance Ω(1) .

The proof of Theorem 1.6 now follows by considering the antisymmetric function f ( x , y ) := y ⊤ B y -x ⊤ B x for a suitable matrix B (defined per the hard instance from Theorem 4.7 based on k -clique). FNP -hardness follows similarly by considering a search version of maximum clique.

Finally, the proof of Theorem 3.5 that was claimed earlier follows immediately by combining Theorem 4.7 with the reduction of Section 3.1, and in particular, Lemmas 3.2 and 3.3.

## 4.3 Team zero-sum games

Our previous hardness result concerning symmetric min-max optimization problems does not have any immediate implications for (normal-form) team zero-sum games since the class of hard instances we constructed earlier contains a quadratic term. Our next result provides such a hardness result by combining the basic gadget we introduced in Section 3.1 in the context of adversarial team games; the basic pieces of the argument are similar to the ones we described in Section 3.1, and so the proof is deferred to Section A.5. Our goal is to prove the following.

Theorem 1.7. Computing a symmetric 1 / n c -Nash equilibrium in symmetric, 6 -player ( 3 vs. 3 ) team zero-sum polymatrix games is PPAD -complete for some constant c &gt; 0 .

Let us describe the class of 3 vs. 3 team zero-sum games upon which our hardness result is based on. Based on (2), we define the auxiliary function

<!-- formula-not-decoded -->

In what follows, the 3 players of the one team will be identified with ( x , y , z ) , while the 3 players of the other team with (ˆ x , ˆ y , ˆ z ) . We define the utility of the latter team to be

<!-- formula-not-decoded -->

where A is symmetric and C is skew-symmetric. The rest of the argument follows Section 3.1.

## 5 Conclusion and open problems

We have provided a number of new complexity results concerning min-max optimization in general, and team zero-sum games in particular (see Table 1). There are many interesting avenues for future research. The complexity of computing first-order Nash equilibria (equivalently, the GDAFIXEDPOINT problem) remains wide open, but our hardness results suggest a possible approach: as we have seen, in symmetric min-max optimization, computing either symmetric or non-symmetric equilibria is intractable, so it would be enough if one could establish this using the same underlying function-that is, somehow combine our two reductions into one. It would also be interesting to see whether our hardness results can be extended to more structured min-max optimization problems, such as adversarial training and GANs.

## Acknowledgments

I.P. and J.Y. were supported by NSF grant CCF-2454115. I.P. would like to acknowledge ICS research award and a start-up grant from UCI. Part of this work was done while I.P. and J.Y. were visiting Archimedes Research Unit. T.S. is supported by the Vannevar Bush Faculty Fellowship ONR N00014-23-1-2876, National Science Foundation grants RI-2312342 and RI-1901403, ARO award W911NF2210266, and NIH award A240108S001. We are grateful to Alexandros Hollender for many valuable discussions.

## References

- Ioannis Anagnostides, Fivos Kalogiannis, Ioannis Panageas, Emmanouil-Vasileios VlatakisGkaragkounis, and Stephen McAleer. Algorithms and complexity for computing Nash equilibria in adversarial team games. In Conference on Economics and Computation (EC) , 2023.
- Yakov Babichenko and Aviad Rubinstein. Settling the complexity of Nash equilibrium in congestion games. In Symposium on Theory of Computing (STOC) , 2021.
- Tao Bai, Jinqi Luo, Jun Zhao, Bihan Wen, and Qian Wang. Recent advances in adversarial training for adversarial robustness. In International Joint Conference on Artificial Intelligence (IJCAI) , 2021.
- Yu Bai and Chi Jin. Provable self-play algorithms for competitive reinforcement learning. In International Conference on Machine Learning (ICML) , Proceedings of Machine Learning Research, 2020.
- James P. Bailey, Gauthier Gidel, and Georgios Piliouras. Finite regret and cycles with fixed step-size via alternating gradient descent-ascent. In Conference on Learning Theory (COLT) , 2020.
- Nicola Basilico, Andrea Celli, Giuseppe De Nittis, and Nicola Gatti. Team-maxmin equilibrium: Efficiency bounds and algorithms. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI) , 2017.
- Martino Bernasconi, Matteo Castiglioni, Andrea Celli, and Gabriele Farina. On the role of constraints in the complexity of min-max optimization, 2024.
- Marie Louisa Tølbøll Berthelsen and Kristoffer Arnsfelt Hansen. On the computational complexity of decision problems about multi-player Nash equilibria. In Dimitris Fotakis and Evangelos Markakis, editors, Algorithmic Game Theory , pages 153-167, Cham, 2019. Springer International Publishing. ISBN 978-3-030-30473-7.
- Christian Borgs, Jennifer T. Chayes, Nicole Immorlica, Adam Tauman Kalai, Vahab S. Mirrokni, and Christos H. Papadimitriou. The myth of the folk theorem. Games and Economic Behavior , 70(1): 34-43, 2010.
- Yang Cai and Constantinos Daskalakis. On minmax theorems for multiplayer games. In Symposium on Discrete Algorithms (SODA) , 2011.
- Yang Cai, Argyris Oikonomou, and Weiqiang Zheng. Finite-time last-iterate convergence for learning in multi-player games. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS) , 2022.
- Nicholas Carlini, Anish Athalye, Nicolas Papernot, Wieland Brendel, Jonas Rauber, Dimitris Tsipras, Ian J. Goodfellow, Aleksander Madry, and Alexey Kurakin. On evaluating adversarial robustness, 2019.
- Luca Carminati, Federico Cacciamani, Marco Ciccone, and Nicola Gatti. A marriage between adversarial team games and 2-player games: Enabling abstractions, no-regret learning, and subgame solving. In International Conference on Machine Learning (ICML) , 2022.
- Luca Carminati, Brian Hu Zhang, Gabriele Farina, Nicola Gatti, and Tuomas Sandholm. Hiddenrole games: Equilibrium concepts and computation. In Proceedings of the ACM Conference on Economics and Computation (EC) , 2023.
- Andrea Celli and Nicola Gatti. Computational results for extensive-form adversarial team games. In Proceedings of the AAAI Conference on Artificial Intelligence , 2018.
- Xi Chen, Xiaotie Deng, and Shang-Hua Teng. Settling the complexity of computing two-player Nash equilibria. J. ACM , 56(3):14:1-14:57, 2009.
- Sayantan Choudhury, Eduard Gorbunov, and Nicolas Loizou. Single-call stochastic extragradient methods for structured non-monotone variational inequalities: Improved analysis under weaker conditions. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS) , 2023.

- Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized smoothing. In International Conference on Machine Learning (ICML) , pages 1310-1320, 2019.
- Vincent Conitzer and Tuomas Sandholm. New complexity results about Nash equilibria. Games and Economic Behavior , 63(2):621-641, 2008.
- Constantinos Daskalakis and Ioannis Panageas. The limit points of (optimistic) gradient descent in min-max optimization. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS) , 2018.
- Constantinos Daskalakis and Christos H. Papadimitriou. Continuous local search. In Symposium on Discrete Algorithms (SODA) , 2011.
- Constantinos Daskalakis, Paul W. Goldberg, and Christos H. Papadimitriou. The complexity of computing a Nash equilibrium. SIAM J. Comput. , 39(1):195-259, 2009.
- Constantinos Daskalakis, Dylan J Foster, and Noah Golowich. Independent policy gradient methods for competitive reinforcement learning. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS) , 2020.
- Constantinos Daskalakis, Stratis Skoulakis, and Manolis Zampetakis. The complexity of constrained min-max optimization. In Symposium on Theory of Computing (STOC) , 2021.
- Scott Emmons, Caspar Oesterheld, Andrew Critch, Vincent Conitzer, and Stuart Russell. For learning in symmetric teams, local optima are global Nash equilibria. In International Conference on Machine Learning (ICML) , 2022.
- Kousha Etessami and Mihalis Yannakakis. On the complexity of Nash equilibria and other fixed points. SIAM J. Comput. , 39(6):2531-2597, 2010.
- Gabriele Farina, Andrea Celli, Nicola Gatti, and Tuomas Sandholm. Ex ante coordination and collusion in zero-sum multi-player extensive-form games. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS) , 2018.
- John Fearnley, Paul Goldberg, Alexandros Hollender, and Rahul Savani. The complexity of gradient descent: CLS = PPAD ∩ PLS. J. ACM , 70(1):7:1-7:74, 2023.
- D. Gale, H. W. Kuhn, and A. W. Tucker. On Symmetric Games , pages 81-88. Princeton University Press, 1951.
- Jugal Garg, Ruta Mehta, Vijay V. Vazirani, and Sadra Yazdanbod. ∃ r-completeness for decision versions of multi-player (symmetric) Nash equilibria. ACM Trans. Econ. Comput. , 6(1), 2018.
- Saeed Ghadimi and Guanghui Lan. Accelerated gradient methods for nonconvex nonlinear and stochastic programming. Mathematical Programming , 156(1):59-99, 2016.
- Abheek Ghosh and Alexandros Hollender. The complexity of symmetric bimatrix games with common payoffs. In Conference on Web and Internet Economics (WINE) , 2024.
- Itzhak Gilboa and Eitan Zemel. Nash and correlated equilibria: Some complexity considerations. Games and Economic Behavior , 1(1):80-93, 1989.
- Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Proceedings of the Annual Conference on Neural Information Processing Systems (NIPS) , 2014.
- Eduard Gorbunov, Adrien B. Taylor, and Gauthier Gidel. Last-iterate convergence of optimistic gradient method for monotone variational inequalities. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS) , 2022.
- Alexandros Hollender, Gilbert Maystre, and Sai Ganesh Nagarajan. The complexity of two-team polymatrix games with independent adversaries. In International Conference on Learning Representations (ICLR) , 2025.

- Chi Jin, Praneeth Netrapalli, and Michael I. Jordan. What is local optimality in nonconvex-nonconcave minimax optimization? In International Conference on Machine Learning (ICML) , 2020.
- Michael I. Jordan, Tianyi Lin, and Manolis Zampetakis. First-order algorithms for nonlinear generalized Nash equilibrium problems. Journal of Machine Learning Research , 24:38:1-38:46, 2023.
- Fivos Kalogiannis, Ioannis Anagnostides, Ioannis Panageas, Emmanouil-Vasileios VlatakisGkaragkounis, Vaggos Chatziafratis, and Stelios Andrew Stavroulakis. Efficiently computing Nash equilibria in adversarial team markov games. In International Conference on Learning Representations (ICLR) , 2023a.
- Fivos Kalogiannis, Ioannis Panageas, and Emmanouil-Vasileios Vlatakis-Gkaragkounis. Towards convergence to Nash equilibria in two-team zero-sum games. ICLR , 2023b.
- Fivos Kalogiannis, Jingming Yan, and Ioannis Panageas. Learning equilibria in adversarial team markov games: A nonconvex-hidden-concave min-max optimization problem. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS) , 2024.
- Galina M Korpelevich. The extragradient method for finding saddle points and other problems. Matecon , 12:747-756, 1976.
- Tianyi Lin, Chi Jin, and Michael I. Jordan. On gradient descent ascent for nonconvex-concave minimax problems. In International Conferecne on Machine Learning (ICML) , 2020.
- Luo Luo, Haishan Ye, Zhichao Huang, and Tong Zhang. Stochastic recursive gradient descent ascent for stochastic nonconvex-strongly-concave minimax problems. In Advances in Neural Information Processing Systems , 2020.
- Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations (ICLR) , 2018.
- Andrew McLennan and Rabee Tourky. Simple complexity from imitation games. Games and Economic Behavior , 68(2):683-688, 2010.
- Ruta Mehta. Constant rank bimatrix games are ppad-hard. In Symposium on Theory of Computing (STOC) , 2014.
- Ruta Mehta, Vijay V Vazirani, and Sadra Yazdanbod. Settling some open problems on 2-player symmetric Nash equilibria. In International Symposium on Algorithmic Game Theory , pages 272-284. Springer, 2015.
- Panayotis Mertikopoulos, Bruno Lecouat, Houssam Zenati, Chuan-Sheng Foo, Vijay Chandrasekhar, and Georgios Piliouras. Optimistic mirror descent in saddle-point problems: Going the extra (gradient) mile. In International Conference on Learning Representations (ICLR) , 2019.
- John Nash. Non-cooperative games . PhD thesis, Princeton University, 1950.
- John Nash. Non-cooperative games. Annals of mathematics , pages 286-295, 1951.
- Noam Nisan, Tim Roughgarden, Eva Tardos, and Vijay V. Vazirani. Algorithmic Game Theory . Cambridge University Press, USA, 2007.
- Maher Nouiehed, Maziar Sanjabi, Tianjian Huang, Jason D. Lee, and Meisam Razaviyayn. Solving a class of non-convex min-max games using iterative first order methods. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS) , 2019.
- Idan Orzech and Martin C. Rinard. Correlated vs. uncorrelated randomness in adversarial congestion team games, 2023.
- Dmitrii M. Ostrovskii, Andrew Lowy, and Meisam Razaviyayn. Efficient search of first-order nash equilibria in nonconvex-concave smooth min-max problems. SIAM J. Optim. , 31(4):2508-2538, 2021.

- Christos H. Papadimitriou. On the complexity of the parity argument and other inefficient proofs of existence. Journal of Computer and System Sciences , 48(3):498-532, 1994.
- Aviad Rubinstein. Settling the complexity of computing approximate two-player Nash equilibria. In Symposium on Foundations of Computer Science (FOCS) , 2016.
- Leonard Schulman and Umesh V Vazirani. The duality gap for two-team zero-sum games. In Innovations in Theoretical Computer Science Conference (ITCS) , 2017.
- Ke Sun. Some properties of the Nash equilibrium in 2 × 2 zero-sum games, 2022.
- Emanuel Tewolde, Caspar Oesterheld, Vincent Conitzer, and Paul W. Goldberg. The computational complexity of single-player imperfect-recall games. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI) , 2023.
- Emanuel Tewolde, Brian Hu Zhang, Caspar Oesterheld, Tuomas Sandholm, and Vincent Conitzer. Computing game symmetries and equilibria that respect them. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI) , 2025.
- Ioannis C. Tsaknakis and Mingyi Hong. Finding first-order Nash equilibria of zero-sum games with the regularized nikaido-isoda function. In International Conference on Artificial Intelligence and Statistics (AISTATS) , 2021.
- John von Neumann. Zur Theorie der Gesellschaftsspiele. Mathematische Annalen , 100:295-320, 1928.
- Bernhard von Stengel and Daphne Koller. Team-maxmin equilibria. Games and Economic Behavior , 21(1):309-321, 1997.
- Chen-Yu Wei, Chung-Wei Lee, Mengxiao Zhang, and Haipeng Luo. Last-iterate convergence of decentralized optimistic gradient descent/ascent in infinite-horizon competitive markov games. In Conference on Learning Theory (COLT) , 2021.
- Andre Wibisono, Molei Tao, and Georgios Piliouras. Alternating mirror descent for constrained min-max games. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS) , 2022.
- Zi Xu, Huiling Zhang, Yang Xu, and Guanghui Lan. A unified single-loop alternating gradient projection algorithm for nonconvex-concave and convex-nonconcave minimax problems. Mathematical Programming , 201(1):635-706, 2023.
- Brian Hu Zhang, Gabriele Farina, and Tuomas Sandholm. Team belief DAG: generalizing the sequence form to team games for fast computation of correlated team max-min equilibria via regret minimization. In International Conference on Machine Learning (ICML) , 2023.
- Youzhi Zhang and Bo An. Converging to team-maxmin equilibria in zero-sum multiplayer games. In International Conference on Machine Learning (ICML) , 2020.
- Youzhi Zhang, Bo An, and Jakub Cerný. Computing ex ante coordinated team-maxmin equilibria in zero-sum multiplayer extensive-form games. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI) , 2021.

## A Omitted proofs

This section contains the proofs omitted from the main body.

## A.1 Proofs from Section 3.1

We begin by stating and proving a simple auxiliary lemma.

Lemma A.1. Let ( x ∗ i , x ∗ -i ) be an ϵ 2 -Nash equilibrium of a normal-form game, and a j any action of player i . If u i ( a k , x ∗ -i ) ≤ u i ( a j , x ∗ -i ) -c for some c &gt; 0 and k ∈ [ n ] , then x ∗ i ( a k ) ≤ ϵ 2 / c .

Proof. For the sake of contradiction, suppose that x ∗ i ( a k ) &gt; ϵ 2 c for some k ∈ [ n ] such that u i ( a k , x ∗ -i ) ≤ u i ( a j , x ∗ -i ) -c . Consider the strategy ∆ n ∋ x ′ i = x ∗ i + x ∗ i ( a k ) e j -x ∗ i ( a k ) e k . Then, we have

<!-- formula-not-decoded -->

That is, deviating to x ′ i yields a utility benefit strictly larger than ϵ 2 , which contradicts the assumption that ( x ∗ i , x ∗ -i ) is an ϵ 2 -Nash equilibrium.

We move on to the proof of Lemma 3.2.

Proof of Lemma 3.2. For the sake of contradiction, suppose that x ∗ i -y ∗ i &gt; 2 ϵ for some i ∈ [ n ] (the case where y ∗ i -x ∗ i &gt; 2 ϵ is symmetric, and can be treated analogously). Player z could then choose action a i (with probability 1 ), which secures a utility of

<!-- formula-not-decoded -->

since x ∗ i -y ∗ i &gt; 2 ϵ . At the same time, Player z could choose action a 2 n +1 , which secures a utility of u ( x ∗ , y ∗ , a 2 n +1 ) = ⟨ x ∗ , A y ∗ ⟩ + | A min | . So,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Also, using the fact that ( x ∗ , y ∗ , z ∗ ) is an ϵ 2 -Nash equilibrium,

<!-- formula-not-decoded -->

since we have assumed that A min &lt; 0 . Now, consider the deviation of Player x (from strategy x ∗ ) to x ′ := y ∗ . Then, u ( x ′ , y ∗ , z ∗ ) = ⟨ y ∗ , A y ∗ ⟩ + z ∗ 2 n +1 · | A min | . Thus, combining with (6) and (5),

<!-- formula-not-decoded -->

where we used that A max , A min ≤ -1 and ϵ 2 ≤ 1 / 2 . But (7) contradicts the fact that ( x ∗ , y ∗ , z ∗ ) is an ϵ 2 -Nash equilibrium since deviating to x ′ yields a utility improvement (equivalently, decrease in cost) strictly larger than ϵ 2 . This completes the proof.

Applying Lemma A.1,

We proceed with the proof of Lemma 3.3.

Proof of Lemma 3.3. From Lemma 3.2 it holds that ∥ x ∗ -y ∗ ∥ ∞ ≤ 2 ϵ . Let i ∈ [ n ] . We assume that i is such that x ∗ i -y ∗ i ≥ 0 ; the contrary case is symmetric. We consider two cases. First, suppose that | x ∗ i -y ∗ i | ≤ ϵ / 2 . Then, we have

<!-- formula-not-decoded -->

Thus, by Lemma A.1, we conclude that z i ≤ 2 ϵ 2 . Similarly,

<!-- formula-not-decoded -->

since x ∗ i -y ∗ i ≥ 0 . Again, Lemma A.1 implies that z n + i ≤ ϵ 2 .

It thus suffices to treat the case where | x ∗ i -y ∗ i | &gt; ϵ / 2 (assuming that x ∗ i -y ∗ i ≥ 0 ). It follows that there exists j ∈ [ n ] such that x ∗ j -y ∗ j &lt; 0 . In addition, we observe that u ( x ∗ , y ∗ , a 2 n +1 ) = ⟨ x ∗ , A y ∗ ⟩ + | A min | ≥ ⟨ x ∗ , A y ∗ ⟩ +1 , whereas u ( x ∗ , y ∗ , a j ) &lt; ⟨ x ∗ , A y ∗ ⟩ and u ( x ∗ , y ∗ , a n + i ) &lt; ⟨ x ∗ , A y ∗ ⟩ . As a result, Lemma A.1 implies that z ∗ n + i ≤ ϵ 2 and z ∗ j ≤ ϵ 2 .

Now, consider the deviation

<!-- formula-not-decoded -->

that is, x ′ is the strategy that results from x by reallocating ( x ∗ i -y ∗ i ) probability mass from action a i to action a j . The difference u ( x ′ , y ∗ , z ∗ ) -u ( x ∗ , y ∗ , z ∗ ) can be expressed as

<!-- formula-not-decoded -->

where (8) uses the following:

<!-- formula-not-decoded -->

- x ∗ i -y ∗ i ≤ 2 ϵ (Lemma 3.2);
- z ∗ n + i ≤ ϵ 2 and z ∗ j ≤ ϵ 2 ; and
- ⟨ x ′ -x ∗ , A y ∗ ⟩ ≤ ∥ x ′ -x ∗ ∥ 1 ∥ A y ∗ ∥ ∞ ≤ 2 | x ∗ i -y ∗ i || A min | ≤ 4 ϵ | A min | (since A has negative entries);

Moreover, given that ( x ∗ , y ∗ , z ∗ ) is assumed to be an ϵ 2 -Nash equilibrium, we have

<!-- formula-not-decoded -->

(The utility of Player x is given by -u .) Combining (10) and (9),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In summary, when x ∗ i -y ∗ i ≥ 0 , we have shown that z ∗ i ≤ 9 ϵ and z ∗ n + i ≤ ϵ 2 . The case where y ∗ i -x ∗ i ≥ 0 can be treated similarly.

We continue with the proof of Theorem 3.4, which combines Lemmas 3.2 and 3.3 to complete the CLS -hardness reduction of Section 3.1.

Theorem 3.4. Given any ϵ 2 -Nash equilibrium ( x ∗ , y ∗ , z ∗ ) in the adversarial team game (2) , with ϵ ≤ 1 / 10 , ( y ∗ , y ∗ ) is a symmetric (21 n +1) | A min | ϵ -Nash equilibrium of the symmetric, two-player game ( A , A ) (that is, A = A ⊤ ).

Proof. Since ( x ∗ , y ∗ , z ∗ ) is an ϵ 2 -Nash equilibrium, we have that for any for any deviation y ′ ∈ ∆ n of Player y ,

<!-- formula-not-decoded -->

Moreover, by considering a deviation of Player x again to y ′ ,

<!-- formula-not-decoded -->

Adding (13) and (14), and using the fact that A is a symmetric matrix,

<!-- formula-not-decoded -->

where in (15) we use Lemmas 3.2 and 3.3. Also,

<!-- formula-not-decoded -->

Finally, combining (16) and (17), we conclude that for any y ′ ∈ ∆ n ,

<!-- formula-not-decoded -->

This concludes the proof.

We now restate Theorem 1.2, which establishes the main complexity implication of Theorem 3.1.

Theorem 1.2. Computing an ϵ -Nash equilibrium in 3 -player (that is, 2 vs. 1 ) adversarial team games is CLS -complete.

Proof. CLS -hardness follows directly from Theorem 3.4 and Theorem 3.1 (due to Ghosh and Hollender [2024]). The inclusion was shown by Anagnostides et al. [2023].

## A.2 Irrational Nash equilibria in adversarial team games

We next describe an interesting property for adversarial team games. Namely, similar to general-sum 3 -player games [Nash, 1950], there exist adversarial team games that admit a unique irrational Nash equilibrium, as stated below.

Proposition A.2 (Berthelsen and Hansen, 2019) . There exists a 3 -player adversarial team game with a unique Nash equilibrium that is supported on irrationals.

Although the paper of Berthelsen and Hansen [2019] provides such an instance, no proof is given. Here, as a complement to their work, we provide a relatively simple and general way to analyze the irrational Nash equilibrium in 3 -player adversarial team games.

We consider a 3 -player adversarial team game in which the utility function of the adversary u : { 1 , 2 } × { 1 , 2 } × { 1 , 2 } : ( x , y , z ) ↦→ R reads

|   z | (1, 1)   | (1, 2)   | (2, 1)   | (2, 2)   |
|-----|----------|----------|----------|----------|
|   1 | 1        | 3        | 99 100   | - 1 100  |
|   2 | 9 10     | - 1 10   | 1        | 3        |

The proof of this result makes use of a characterization of Nash equilibria in 2 × 2 two-player zero-sum games, stated below; for the proof, we refer to, for example, Sun [2022, Theorem 1.2].

Lemma A.3. Let A ∈ R 2 × 2 such that

<!-- formula-not-decoded -->

Then, the two-player zero-sum game min x ∈ ∆ 2 max z ∈ ∆ 2 ⟨ x , A z ⟩ admits a unique (exact) Nash equilibrium with value

<!-- formula-not-decoded -->

Furthermore, the unique Nash equilibrium ( x ∗ , z ∗ ) satisfies

<!-- formula-not-decoded -->

Proof of Proposition A.2. By construction of the adversarial team game, the mixed extension of the utility can be expressed as

<!-- formula-not-decoded -->

Suppose that we fix y ∈ ∆ 2 . Then, Players x and y are engaged in a (two-player) zero-sum game with payoff matrix

<!-- formula-not-decoded -->

We now invoke Lemma A.3. Indeed, we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

that is, the precondition of Lemma A.3 is satisfied, and so the value of (18) reads

<!-- formula-not-decoded -->

It is easy to verify that v is a strictly convex function in [0 , 1] , and admits a unique minimum corresponding to y ∗ = ( 611 -9 √ 3 600 , 9 √ 3 -11 600 ) , which is irrational. Now, suppose that ( x ∗ , y ∗ , z ∗ ) is a Nash equilibrium of the adversarial team game. We will first argue that ( x ∗ , z ∗ ) is the unique Nash equilibrium of A ( y ∗ ) . Indeed, suppose that there exists x ′ ∈ ∆ 2 such that ⟨ x ′ , A ( y ∗ ) z ∗ ⟩ &lt; ⟨ x ∗ , A ( y ∗ ) z ∗ ⟩ , or equivalently, u ( x ′ , y ∗ , z ∗ ) &lt; u ( x ∗ , y ∗ , z ∗ ) ; this is a contradiction since ( x ∗ , y ∗ , z ∗ ) is assumed to be a Nash equilibrium. Similar reasoning applies with respect to Player z . Thus, ( x ∗ , z ∗ ) is a Nash equilibrium of A ( y ∗ ) , and thereby uniquely determined by y ∗ -by Lemma A.3 coupled with (19) and (20). Furthermore, given the value of y ∗ , we get that x ∗ = ( 3 - √ 3 6 , 3+ √ 3 6 ) and z ∗ = ( 3+ √ 3 6 , 3 - √ 3 6 ) . Now, consider the utility of Player y

when playing the first action a 1 or the second action a 2 ; plugging in the value of x ∗ and z ∗ , we have u ( x ∗ , e 1 , z ∗ ) = 578+9 √ 3 600 and u ( x ∗ , e 2 , z ∗ ) = 578+9 √ 3 600 . Since u ( x ∗ , e 1 , z ∗ ) = u ( x ∗ , e 2 , z ∗ ) , ( x ∗ , y ∗ , z ∗ ) is a Nash equilibrium.

̸

Moreover, suppose there exists another Nash equilibrium ( x ′ , y ′ , z ′ ) that is different from ( x ∗ , y ∗ , z ∗ ) . As shown above, ( x ′ , z ′ ) is the unique equilibrium of the zero-sum game induced by y ′ . Thus, if we have two different Nash equilibria, it implies that y ′ = y ∗ . We consider the following three cases:

- First, let y ′ be a (fully) mixed strategy. Since x ′ and z ′ forms the unique NE in of A ( y ′ ) , we have

<!-- formula-not-decoded -->

Further, for Player y ,

<!-- formula-not-decoded -->

̸

Since y ′ is a mixed strategy, we have u ( x ′ , e 1 , z ′ ) = u ( x ′ , e 2 , z ′ ) ; solving the equality we get y ′ = ( 611 -9 √ 3 600 , 9 √ 3 -11 600 ) , which contradicts the assumption that y ′ = y ∗ .

- If y ′ = (1 , 0) , we have u ( x ′ , e 1 , z ′ ) = 1199 1210 and u ( x ′ , e 2 , z ′ ) = 589 1210 . Thus, it follows that by unilaterally deviating to play (0 , 1) , Player y can decrease the utility of the adversary, contradicting the fact that ( x ′ , y ′ , z ′ ) is a Nash equilibrium.
- Finally, suppose that y ′ = (0 , 1) . Similarly to the second case, we get u ( x ′ , e 1 , z ′ ) &lt; u ( x ′ , e 2 , z ′ ) , which is a contradiction.

Thus, we conclude that ( x ∗ , y ∗ , z ∗ ) is the unique Nash equilibrium of the 3-player adversarial team game defined above, completing the proof.

A natural question arising from Proposition A.2 concerns the complexity of determining whether an adversarial team game admits a unique Nash equilibrium. Theorem 3.5-that was presented earlier in the main body-establishes NP -hardness for a version of that problem that accounts for approximate Nash equilibria.

## A.3 Proofs from Section 4.1

We continue with the proofs from Section 4.1. We first apply Brouwer's fixed point theorem to show that symmetric min-max optimization problems always admit a symmetric equilibrium.

Lemma A.4. Let X be a convex and compact set. Then, any L -smooth, antisymmetric function (Definition 1.4) f : X × X → R admits a symmetric first-order Nash equilibrium ( x ∗ , x ∗ ) .

Proof. We define the function M : X → X to be

<!-- formula-not-decoded -->

Given that f is L -smooth, we conclude that M ( x ′ ) is ( L +1) -Lipschitz, hence continuous. Therefore, from Brouwer's fixed point theorem, there exists an x ∗ so that M ( x ∗ ) = x ∗ . Moreover, the symmetry

<!-- formula-not-decoded -->

Therefore, ( x ∗ , x ∗ ) is a first-order Nash equilibrium of the symmetric min-max problem with function f .

We now present the proof of Lemma 4.1

Proof of Lemma 4.1. We first define the function (as in Lemma A.4) M : X → X as

<!-- formula-not-decoded -->

where we recall that Π is the projection operator on X . Assuming that the input function f is L -smooth, it follows that M ( x ′ ) is ( L +1) -Lipschitz. Furthermore, projecting on the polytope X takes polynomial time, and so M is polynomial-time computable. As a result, we can use Etessami and Yannakakis [2010, Proposition 2, part 2] (see also Fearnley et al. [2023, Proposition D.1]), where it was shown that finding an ϵ -approximate fixed point of a Brouwer function that is efficiently computable and continuous, when the domain is a bounded polytope, lies in PPAD .

## We proceed with the proof of Theorem 4.2

Proof of Theorem 4.2. We P -time reduce the problem of finding approximate symmetric NE in two-player symmetric games to SYMGDAFIXEDPOINT. Given any two-player symmetric game with payoff matrices ( R , R ⊤ ) of size n × n , we set

<!-- formula-not-decoded -->

We define the quadratic , antisymmetric function

<!-- formula-not-decoded -->

with domain ∆ n × ∆ n . Indeed, to see that f is antisymmetric, one can observe that

<!-- formula-not-decoded -->

Assuming that all entries of R lie in [ -1 , 1] , it follows that the singular values of A and C are bounded by n. As a result f and ∇ x f = -A x -C y , ∇ y f = A y + C x are polynomial time computable and continuous, and ∇ x f, ∇ y f are L -Lipschitz for L ≤ 2 n, thus f is 4 n -smooth.

We assume x is the minimizer and y is the maximizer, and let ( x ∗ , x ∗ ) be an ϵ -approximate fixed point of GDA. We shall show that ( x ∗ , x ∗ ) is an 4 nϵ -approximate NE of the symmetric two-player game ( R , R ⊤ ) . Since ( x ∗ , x ∗ ) is an ϵ -approximate fixed point of GDA, we can use Lemma A.5 and obtain the following variational inequalities:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

implying that (since the diameter of ∆ n is √ 2 in ℓ 2 )

Now, we observe that (VI for NE) implies that ( x ∗ , x ∗ ) is a √ 2 ϵ (2 n +1) -approximate symmetric NE in the two-player symmetric game with payoff matrices ( A + C , A -C ) (recall Definition (2.1)). Since √ 2 ϵ (2 n +1) ≤ 4 nϵ for n ≥ 2 , our claim follows.

By Theorem 2.2 and Lemma 4.1, we conclude that SYMGDAFIXEDPOINT is PPAD -complete, even for quadratic functions that are O ( n ) -smooth, O ( n ) -Lipschitz and ϵ ≤ 1 / n 1+ c , for any c &gt; 0 .

We next state a standard lemma that connects first-order optimality with the fixed-point gap of gradient ascent.

Lemma A.5 ([Ghadimi and Lan, 2016], Lemma 3 for c = 1 ) . Let f ( x ) be an L -smooth function in x ∈ ∆ n . Define the gradient mapping

<!-- formula-not-decoded -->

If ∥ G ( x ∗ ) ∥ 2 ≤ ϵ , that is, x ∗ is an ϵ -approximate fixed point of gradient ascent with stepsize equal to one, then

<!-- formula-not-decoded -->

Similarly, the next lemma makes such a connection for min-max optimization problems with coupled constraints; it is mostly extracted from Daskalakis et al. [2021, Section B.2].

Lemma A.6. Let f ( x , y ) be a G -Lipschitz, L -smooth function defined in some polytope domain D ⊆ ∆ n × ∆ n of diameter D . Define the mapping

<!-- formula-not-decoded -->

If ∥ G ( x ∗ , y ∗ ) ∥ 2 ≤ ϵ , that is, ( x ∗ , y ∗ ) is an ϵ -approximate fixed point of (the safe version) of GDA with stepsize equal to one, then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Let ( x ∆ , y ∆ ) = ( x ∗ -∇ x f ( x ∗ , y ∗ ) , y ∗ + ∇ y f ( x ∗ , y ∗ )) . In Daskalakis et al. [2021, Claim B.2], it was shown that for all ( x , y ) ∈ D , we have

<!-- formula-not-decoded -->

Using the above inequality, it was concluded that ( x ∗ , y ∗ ) is an approximate fixed point of the 'unsafe' version of GDA; specifically,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

We now use Lemma A.5 for both inequalities above, together the fact that D = 2 √ 2 , to conclude that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

We proceed to establish Theorem 4.4.

Proof of Theorem 4.4. The proof follows similar steps with Theorem 4.2, namely, we P -time reduce the problem of finding approximate symmetric NE in two-player symmetric games to GDAFIXEDPOINT with coupled domains. Given a two-player symmetric game with payoff matrices ( R , R ⊤ ) of size n × n , we set A := 1 2 ( R + R ⊤ ) , C := 1 2 ( R -R ⊤ ) and define again the quadratic, antisymmetric function

<!-- formula-not-decoded -->

Moreover, given a parameter δ &gt; 0 (to be specified shortly), we define the joint domain of f to be

<!-- formula-not-decoded -->

Let ( x ∗ , y ∗ ) be an ϵ -approximate fixed point of GDA. We will show that ( x ∗ + y ∗ 2 , x ∗ + y ∗ 2 ) is an O ( ϵ 1 / 4 ) -approximate (symmetric) NE of the game ( R , R ⊤ ) for an appropriate choice of δ.

We set D ( x ∗ ) = { y : ( x ∗ , y ) ∈ D} and D ( y ∗ ) = { x : ( x , y ∗ ) ∈ D} . In words, D ( x ∗ ) and D ( y ∗ ) capture the allowed deviations for y and x respectively. It also holds that f is G -Lipschitz continuous with G = 4 n and also 4 n -smooth (using the same reasoning as in Theorem 4.2).

Since ( x ∗ , y ∗ ) is an ϵ -approximate fixed point of GDA, using Lemma A.6, the following VIs must hold for some positive constant K &lt; 10 and n sufficiently large:

<!-- formula-not-decoded -->

Let D = { z ∈ ∆ n : ∥ ∥ ∥ z -x ∗ + y ∗ 2 ∥ ∥ ∥ ∞ ≤ δ 2 } . By triangle inequality, it follows that D ⊆ D ( y ∗ ) ∩ D ( x ∗ ) . We express the VIs of (25) using a single variable z and common deviation domain:

<!-- formula-not-decoded -->

Multiplying the first inequality by -1 / 2 and the second with 1 / 2 and adding them up gives

<!-- formula-not-decoded -->

Since x ∗ , y ∗ ∈ D , it follows that ⟨ x ∗ -y ∗ , A ( x ∗ -y ∗ ) ⟩ ≤ n ∥ x ∗ -y ∗ ∥ 2 2 ≤ n 2 δ 2 . Combining this fact with (26), we conclude that

<!-- formula-not-decoded -->

(VImedian) shows that by deviating from ( x ∗ + y ∗ 2 , x ∗ + y ∗ 2 ) to some z in D , the payoff cannot increase by more than ( n 2 δ 2 + Kn 3 / 2 √ ϵ ) in the two-player symmetric game with matrices ( R , R ⊤ ) .

We consider any pure strategy e j for j ∈ [ n ] . If ∥ ∥ ∥ e j -x ∗ + y ∗ 2 ∥ ∥ ∥ ∞ ≤ δ 2 then e j ∈ D and it is captured by (VImedian). Suppose that ∥ ∥ ∥ e j -x ∗ + y ∗ 2 ∥ ∥ ∥ ∞ &gt; δ 2 and consider the point z ′ ∈ D on the line segment between e j and x ∗ + y ∗ 2 that intersects the boundary of D . It holds that e j -x ∗ + y ∗ 2 = c ( z ′ -x ∗ + y ∗ 2 ) for some positive c ≤ 2 δ (it cannot be larger because otherwise the infinity norm of the difference between e j and x ∗ + y ∗ 2 would exceed one, which is impossible as they both belong to ∆ n ). Therefore,

<!-- formula-not-decoded -->

From (27), we conclude that x ∗ + y ∗ 2 is ( 2 n 2 δ + 2 Kn 3 / 2 √ ϵ δ ) -approximate NE of the symmetric two-player game ( R , R ⊤ ) . We choose δ = ϵ 1 / 4 n -1 / 4 and we get that x ∗ + y ∗ 2 is an O ( n 7 / 4 ϵ 1 / 4 ) -approximate NE for ( R , R ⊤ ) , and thus the hardness result holds for ϵ of order O ( 1 n 7+ c ) for any constant c &gt; 0 . We note that if instead of an ϵ -approximate fixed point of GDA, we were given an ϵ -approximate first-order NE, then the hardness result would hold for any ϵ of order 1 n c with c &gt; 0 .

Figure 1: An example of matrix A = A ( G ) (right) for graph G (left).

<!-- image -->

## A.4 Proofs from Section 4.2

The main result we want to show is restated below.

Theorem 1.6. For a symmetric min-max optimization problem, constants c 1 , c 2 &gt; 0 , and ϵ = n -c 1 , it is NP -hard to distinguish between the following two cases under the promise that one of them holds:

- any ϵ -first-order Nash equilibrium ( x ∗ , y ∗ ) satisfies ∥ x ∗ -y ∗ ∥ ≤ n -c 2 , and
- there is an ϵ -first-order Nash equilibrium ( x ∗ , y ∗ ) such that ∥ x ∗ -y ∗ ∥ ≥ Ω(1) .

Our reduction builds on the hardness result of McLennan and Tourky [2010]-in turn based on earlier work by Gilboa and Zemel [1989], Conitzer and Sandholm [2008]-which we significantly refine in order to account for poly (1 /n ) -Nash equilibria. We begin by describing their basic approach. Let G = ([ n ] , E ) be an n -node, undirected, unweighted graph, and construct

<!-- formula-not-decoded -->

(Figure 1 depicts an illustrative example.) Based on this matrix, McLennan and Tourky [2010] consider the symmetric, identical-payoff, two-player game ( A , A ) -by construction, A = A ⊤ , and so this game is indeed symmetric. They were able to show the following key property.

Lemma A.7 ([McLennan and Tourky, 2010]) . Let C k ⊆ [ n ] be a maximum clique of G with size k and x ∗ = 1 k ∑ i ∈ C k e i . Then, ( x ∗ , x ∗ ) is a Nash equilibrium of ( A , A ) that attains value -1 k . Furthermore, any symmetric Nash equilibrium not in the form described above has value at most -1 k -1 .

The idea now is to construct a new symmetric, identical-payoff game ( B , B ) , for

<!-- formula-not-decoded -->

where V := -1 k and r = 1 2 ( -1 k -1 k -1 ) = -2 k -1 2( k -1) k . Coupled with Lemma A.7, this new game yields the following NP -hardness result.

Theorem A.8 ([McLennan and Tourky, 2010]) . It is NP -hard to determine whether a symmetric, identical-payoff, two-player game has a unique symmetric Nash equilibrium.

Our goal here is to prove a stronger result, Theorem 4.7, that characterizes the set of ϵ -Nash equilibria even for ϵ = 1 / n c ; this will form the basis for our hardness result in min-max optimization and adversarial team games. To do so, we first derive some basic properties of game (29).

̸

Game (29) always admits the trivial (symmetric) Nash equilibrium ( e n +1 , e n +1 ) . Now, consider any symmetric Nash equilibrium ( x ∗ , x ∗ ) with x ∗ n +1 = 1 . If x ∗ n +1 = 0 , it follows that ( x ∗ [ i ··· n ] , x ∗ [ i ··· n ] ) is a Nash equilibrium of ( A , A ) , which in turn implies that G admits a clique of size k ; this follows from Lemma A.7, together with the fact that -1 / ( k -1) &lt; r &lt; -1 / k .

We now analyze the case where x ∗ n +1 ∈ (0 , 1) . It then follows that ( x ∗ [1 ··· n ] / (1 -x ∗ n +1 ) , x ∗ [1 ··· n ] / (1 -x ∗ n +1 ) ) is a (symmetric) Nash equilibrium of ( A , A ) . Furthermore, the utility of playing action a n +1 is

̸

(1 -x ∗ n +1 ) r + x ∗ n +1 V &gt; r . By Lemma A.7, it follows that ( x ∗ [1 ··· n ] / (1 -x ∗ n +1 ) , x ∗ [1 ··· n ] / (1 -x ∗ n +1 ) ) has a value of V and G admits a clique of size k . As a result, the utility of playing any action a i , with i ∈ supp ( x ∗ ) and i = n +1 , is (1 -x ∗ n +1 ) V + x ∗ n +1 r . At the same time, the utility of playing action a n +1 reads (1 -x ∗ n +1 ) r + x ∗ n +1 V . Equating those two quantities, it follows that x ∗ n +1 = 1 / 2 .

In summary, G contains a clique of size k if and only if game (29) admits a unique symmetric Nash equilibrium, which implies Theorem A.8. What is more, we have shown a stronger property. Namely, any symmetric Nash equilibrium of ( B , B ) has to be in one of the following forms:

<!-- formula-not-decoded -->

In particular, the equilibria in Items 2 or 3-which exist iff G contains a clique of size k -are always far from the one in Item 1. However, this characterization only applies to exact Nash equilibria. In any two-player game Γ , when ϵ is sufficiently small with log(1 /ϵ ) ≤ poly ( | Γ | ) , Etessami and Yannakakis [2010] have shown that any ϵ -Nash equilibrium is within ℓ 1 -distance δ from an exact one, and so the above characterization can be applied; unfortunately, this does not apply (for general games) in the regime we are interested, namely ϵ = poly (1 /n ) .

We address this challenge by refining the result of McLennan and Tourky [2010]. Our main result, which forms the basis for Theorem 1.6 and Theorem 3.5, is recalled below.

Theorem A.9. For symmetric, identical-interest, two-player games, constants c 1 , c 2 &gt; 0 , and ϵ = n -c 1 , it is NP -hard to distinguish between the following two cases:

- any two symmetric ϵ -Nash equilibria have ℓ 1 -distance at most n -c 2 , and
- there are two symmetric ϵ -Nash equilibria that have ℓ 1 -distance Ω(1) .

Our reduction proceeds similarly, but defines A to be the adjacency matrix of G with δ ∈ (0 , 1) in each diagonal entry. Using A , we show that we can refine Lemma A.7 of McLennan and Tourky [2010]. Before we state the key property we prove in Lemma A.11, we recall the following definition.

Definition A.10 (Well-supported NE) . A symmetric strategy profile ( x , x ) is an ϵ -well-supported Nash equilibrium of the symmetric, identical-payoff game ( A , A ) if for all i ∈ [ n ] ,

<!-- formula-not-decoded -->

LemmaA.11. Suppose that the maximum clique in G is of size k . For any symmetric ϵ -well-supported NE (ˆ x , ˆ x ) of ( A , A ) not supported on a clique of size k , we have u (ˆ x , ˆ x ) ≤ 1 -1 k + δ k -2 δ n 2 k 4 +2 ϵ .

Equipped with this property, we will see shortly that a similar argument to the one described earlier concerning game (29) establishes Theorem 4.7. We proceed now with the proof of Theorem 1.6.

Proof of Theorem 1.6. It suffices to consider the antisymmetric function f ( x , y ) := y ⊤ B y -x ⊤ B x , where symmetric matrix B is defined as in (29), using our new matrix A instead of A (see (32)). Any ϵ -first-order Nash equilibrium ( x ∗ , y ∗ ) of this (separable) min-max optimization problem induces, two symmetric ϵ -Nash equilibria-namely, ( x ∗ , x ∗ ) and ( y ∗ , y ∗ ) -in the symmetric, identicalinterest, game ( B , B ) . Using Theorem 4.7, the claim follows.

In what follows, our goal is to establish Theorem 4.7, which forms the basis for Theorems 1.6 and 3.5.

We consider the symmetric game ( A , A ⊤ ) , where A ∈ R n × n is a symmetric matrix. In particular, since A = A ⊤ , the two players in the game share the same payoff matrix. The payoff matrix A is constructed based on an underlying graph G = ([ n ] , E ) and a parameter δ ∈ (0 , 1) as follows:

<!-- formula-not-decoded -->

Lemma A.12. Let (ˆ x , ˆ x ) be an ϵ -well-supported NE of the game where ˆ x is supported on a max clique of size k , denoted as C k , then the value u (ˆ x , ˆ x ) is at least 1 -1 k + δ k -( k -δ 1 -δ ) ϵ and ∥ ˆ x -x ∗ ∥ ∞ ≤ k -δ 1 -δ ϵ where x ∗ = 1 k ∑ i ∈ C k e i .

Proof. Since ˆ x is supported on C k , let i be a coordinate that ˆ x ∗ puts the least probability mass on; that is, i ∈ argmin j ∈ C k ˆ x j . Considering the utility of playing action a i , we have

<!-- formula-not-decoded -->

Moreover, let j ∈ C k be a coordinate such that ˆ x j ≥ 1 k . It should hold that

<!-- formula-not-decoded -->

Since (ˆ x , ˆ x ) is an ϵ -well-supported NE, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, for all coordinates i ∈ C k , we have 1 k -ϵ 1 -δ ≤ ˆ x i ≤ 1 k + ϵ 1 -δ · ( k -1) . It then holds that

<!-- formula-not-decoded -->

Assumption A.13. For the rest of this subsection, we set the parameters as follows:

- n ≥ k ≥ 10;
- ϵ &lt; δ (1 -δ ) / 6 n 7 ;
- δ := 1 / 2 .

(Using the symbolic value of δ is more convenient in our derivations below.)

Proof of Lemma A.11. We let x ∗ = 1 k ∑ i ∈ C k e i . The proof considers the following three cases:

- The symmetric ϵ -well-supported NE (ˆ x , ˆ x ) has support size less than k .
- The symmetric ϵ -well-supported NE (ˆ x , ˆ x ) has support size greater than k .
- The symmetric ϵ -well-supported NE (ˆ x , ˆ x ) has support size equal to k but is not supported on a clique.

We proceed to show that for any of the three cases above, we would not be able to have a symmetric ϵ -well-supported NE that achieves value greater than 1 -1 k + δ k -2 δ n 2 k 4 +2 ϵ , which is a contradiction.

- For the first case, since the support has size less than k , we can find a coordinate i ∈ [ n ] such that ˆ x i ≥ 1 k -1 . Therefore, the value of playing that action is

<!-- formula-not-decoded -->

Since (ˆ x , ˆ x ) is a symmetric ϵ -well-supported NE, we have

<!-- formula-not-decoded -->

where in the last step we use Assumption A.13.

- For the second case, suppose | supp (ˆ x ) | = m &gt; k . By Lemma A.16, since the maximum clique size is k , we can find a set S ⊆ supp (ˆ x ) with at least m -k +1 elements such that for each coordinate i ∈ S , we can find a coordinate j ∈ S such that A i,j = A j,i = 0 . Now we consider the utility of playing action a i and a j , we have

<!-- formula-not-decoded -->

Since (ˆ x , ˆ x ) is a symmetric ϵ -well-supported NE, we have

<!-- formula-not-decoded -->

Now, by moving all the probability mass from action j to action i , we form a new strategy x ′ = ˆ x + ˆ x j · e i -ˆ x j · e j such that

<!-- formula-not-decoded -->

Suppose there is a coordinate i ∈ S such that ˆ x i ≥ 1 nk 2 ; then, from (30), we have

<!-- formula-not-decoded -->

If ˆ x i &lt; 1 nk 2 for any i ∈ S , then there exists a coordinate l ̸∈ S such that ˆ x l &gt; 1 -( m -k +1) · 1 nk 2 m -( m -k +1) ≥ 1 k + 1 k 2 . Then, considering the utility when playing action a l ,

<!-- formula-not-decoded -->

where in (31) we used Assumption A.13.

Since the l th action is played with positive probability and (ˆ x , ˆ x ) is an ϵ -well-supported NE, we have u (ˆ x , ˆ x ) ≤ u ( e l , ˆ x ) + ϵ &lt; 1 -1 k + δ k -2 δ n 2 k 4 +2 ϵ.

- For the third case, since the support is not on a clique, the exists at least coordinates i, j such that ˆ x i &gt; 0 , ˆ x j &gt; 0 , and A i,j = A j,i = 0 . Similarly as case two, if ˆ x i ≥ 1 nk 2 or ˆ x j ≥ 1 nk 2 , then we have

<!-- formula-not-decoded -->

If ˆ x i &lt; 1 nk 2 and ˆ x j &lt; 1 nk 2 , then there exists an coordinate l such that ˆ x l ≥ 1 -2 · 1 nk 2 k -2 &gt; 1 k + 1 k 2 . Same as (31), we conclude that u (ˆ x , ˆ x ) is at most 1 -1 k + δ k -2 δ n 2 k 4 +2 ϵ.

The proof is complete.

We now construct a new symmetric identical payoff game ( B , B ) , where B is defined as

<!-- formula-not-decoded -->

Above, V := 1 -1 k + δ k and r := 1 -1 k + δ k -δ n 2 k 4 + 3 ϵ . Similarly to our discussion after Theorem A.8, it follows that the symmetric (exact) Nash equilibria of this game can only be in one of the following forms:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now show the following lemma.

LemmaA.14. For any ϵ -well-supported NE (ˆ x , ˆ x ) in game ( B , B ) , it holds that ∥ ˆ x -x ∗ ∥ ∞ ≤ 2 n 6 ϵ , where ( x ∗ , x ∗ ) is an exact NE in one of the three cases above (1,2,3).

Proof. First, we observe that since V &gt; r , clearly ( e n +1 , e n +1 ) is a ϵ -well-supported NE; in this case, ∥ x ′ -x ∗ ∥ ∞ = 0 . Furthermore, since r = 1 -1 k + δ k -δ n 2 k 4 + 3 ϵ , the game does not attain any ϵ -NE with value less than 1 -1 k + δ k -δ n 2 k 4 + 2 ϵ. Suppose the game admits an symmetric ϵ -well-supported NE (ˆ x , ˆ x ) where ˆ x is supported only on the first n actions. Since u (ˆ x , ˆ x ) ≥ 1 -1 k + δ k -δ n 2 k 4 +2 ϵ &gt; 1 -1 k + δ k -2 δ n 2 k 4 +2 ϵ, taking x ∗ as in (2), we conclude that ∥ ˆ x -x ∗ ∥ ∞ ≤ k -1 1 -δ ϵ &lt; 2 n 6 ϵ from Lemma A.11.

We proceed to the case where there is a mixed symmetric ϵ -well-supported Nash (ˆ x , ˆ x ) between the last action and the rest of actions such that 0 &lt; ˆ x n +1 &lt; 1 . Denote ˆ x n +1 = α and η ( · ) to denote the renormalization operation. Since ( η (ˆ x [1 ··· n ] ) , η (ˆ x [1 ··· n ] ) ) is a symmetric strategy profile, from McLennan and Tourky [2010, Proposition 4], we conclude that there is at least one coordinate i ∈ supp (ˆ x ) -{ n +1 } such that

<!-- formula-not-decoded -->

where u A is the utility when the payoff matrix is A . Since ( η (ˆ x [1 ··· n ] ) , η (ˆ x [1 ··· n ] ) ) is an ϵ -wellsupported NE, we have

<!-- formula-not-decoded -->

where u B is the utility function when the payoff matrix is B . Plugging in the value of V and r and using Assumption A.13, we find that α ≤ 1 2 +2 n 6 ϵ ≤ 2 3 .

Now, observe that for any action in the support other than the last action a i , the utility of playing such action u B ( a i , ˆ x ) = u A ( a i , ˆ x [1 ··· n ] ) + rα . Since (ˆ x , ˆ x ) is an ϵ -well-supported Nash Equilibrium, we have u B ( a i , ˆ x ) ≥ max j u B ( a j , ˆ x ) -ϵ for all pairs ( i, j ) ∈ supp (ˆ x ) . Since α ≤ 2 3 , it follows that u A ( a i , η (ˆ x [1 ··· n ] ) ) ≥ max j u A ( a j , η (ˆ x [1 ··· n ] ) ) -3 ϵ for any pairs ( i, j ) ∈ supp (ˆ x ) -{ n +1 } . Thus, we conclude that ( η (ˆ x [1 ··· n ] ) , η (ˆ x [1 ··· n ] ) ) forms a symmetric 3 ϵ -well-supported Nash Equilibrium in game ( A , A ) . Further, the value of playing the last action is (1 -α ) r + αV &gt; r , and so the only situation where there is a mixed Nash between the last action and the rest actions is when u A (( η (ˆ x [1 ··· n ] ) , η (ˆ x [1 ··· n ] ) )) ≥ r -ϵ . Therefore, by Lemma A.12 and Lemma A.11, we conclude that ˆ x [1 ··· n ] is supported on a clique of size k. There exits at least one coordinate i ∈ [ n ] , with 0 &lt; ˆ x i and ˆ x i ≥ 1 k , such that

<!-- formula-not-decoded -->

Since (ˆ x , ˆ x ) is an ϵ -well-supported Nash, we have u ( e i , ˆ x ) ≤ u ( e n +1 , ˆ x ) + ϵ , and so this gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using Assumption A.13 and combining with (33),

<!-- formula-not-decoded -->

By taking x as in (3), we conclude that ∥ ˆ x -x ∥ ≤ 2 n ϵ .

∗ ∗ ∞ 6

Theorem A.15. For any ϵ -NE ( x , x ) in game ( B , B ) , it holds that ∥ x -x ∗ ∥ ∞ ≤ n 6 √ ϵ , where ( x ∗ , x ∗ ) is an exact NE in one of the forms specified above (1,2,3).

Proof. Chen et al. [2009, Lemma 3.2] showed that from any ϵ 2 / 8 -NE ( x , y ) in any two player bimatrix game, one can construct (in polynomial time) an ϵ -well-supported NE ( x ′ , y ′ ) such that ∥ x -x ′ ∥ ∞ ≤ ϵ 4 and ∥ y -y ′ ∥ ∞ ≤ ϵ 4 . Setting ϵ ′ := ϵ 2 8 for the ϵ defined in Lemma A.14, the proof follows.

The proof of Theorem 4.7 follows directly by observing that having two symmetric ϵ -NE ( x , x ) and ( y , y ) such that ∥ x -y ∥ ∞ &gt; 2 n 6 √ ϵ would imply that the game ( B , B ) has two distinct exact NE, which in turn implies that there is a clique of size k in the graph.

We next state and prove an auxiliary lemma that was used earlier.

Lemma A.16. For any graph G = ( V, E ) with n vertices, if the maximum clique has size k , then we can form a set S ⊆ V of size at least n -k +1 such that for any vertex i ∈ S , there exists a vertex j ∈ S such that i and j are not connected.

Proof. Suppose the largest set S we can form has cardinality |S| &lt; n -k +1 , this implies there is a set S ′ = V -S with at least n -( n -k ) = k vertices such that each vertex in S ′ is connected to all other vertices in G . However if this is the case, there is at least one vertex v ̸∈ S ′ that are connected to all vertices in S ′ . This contradicts the fact that maximum clique has size k .

We proceed now with the proof of Theorem 1.6.

Proof of Theorem 1.6. It suffices to consider the antisymmetric function f ( x , y ) := y ⊤ B y -x ⊤ B x , where symmetric matrix B is defined as in (29), using our new matrix A instead of A (see (32)). Any ϵ -first-order Nash equilibrium ( x ∗ , y ∗ ) of this (separable) min-max optimization problem induces, two symmetric ϵ -Nash equilibria-namely, ( x ∗ , x ∗ ) and ( y ∗ , y ∗ ) -in the symmetric, identicalinterest, game ( B , B ) . Using Theorem 4.7, the claim follows.

## A.5 Proofs from Section 4.3

We conclude with the missing proofs from Section 4.3. We first introduce two lemmas which are useful in later proofs.

The first key lemma, which mirrors Lemma 3.2, shows that, in an approximate Nash equilibrium of (4), x ≈ y and ˆ x ≈ ˆ y . This is crucial as it allows us to construct-up to some small error-quadratic terms in the utility function, as in our hardness result for symmetric min-max optimization.

Lemma A.17. Let ( x ∗ , y ∗ , z ∗ , ˆ x ∗ , ˆ y ∗ , ˆ z ∗ ) be an ϵ 2 -Nash equilibrium of (4) with ϵ 2 ≤ 1 / 2 . Then, ∥ x ∗ -y ∗ ∥ ∞ ≤ 2 ϵ and ∥ ˆ x ∗ -ˆ y ∗ ∥ ∞ ≤ 2 ϵ .

Proof. For the sake of contradiction, suppose that ∥ x ∗ -y ∗ ∥ ∞ &gt; 2 ϵ . Without loss of generality, let us further assume that there is some coordinate i such that x ∗ i -y ∗ i &gt; 2 ϵ . The payoff difference for Player ˆ z when playing the i th action compared to action a 2 n +1 reads

<!-- formula-not-decoded -->

By Lemma A.1, it follows that ˆ z ∗ 2 n +1 &lt; ϵ 2 | A min | ≤ ϵ 2 . Moreover, given that ( x ∗ , y ∗ , z ∗ , ˆ x ∗ , ˆ y ∗ , ˆ z ∗ ) is an ϵ 2 -Nash equilibrium, we have

<!-- formula-not-decoded -->

Now, considering the deviation of Player y to y ′ := x ∗ ,

<!-- formula-not-decoded -->

which contradicts the fact that ( x ∗ , y ∗ , z ∗ , ˆ x ∗ , ˆ y ∗ , ˆ z ∗ ) is an ϵ 2 -Nash equilibrium. We conclude that ∥ x ∗ -y ∗ ∥ ∞ ≤ 2 ϵ ; the proof for the fact that ∥ ˆ x ∗ -ˆ y ∗ ∥ ∞ ≤ 2 ϵ follows similarly.

Next, following the argument of Lemma 3.3, we show that, in equilibrium, Players z and ˆ z place most of their probability mass on action a 2 n +1 , thereby having only a small effect on the game between Players x and y vs. ˆ x and ˆ y .

Lemma A.18. Let ( x ∗ , y ∗ , z ∗ , ˆ x ∗ , ˆ y ∗ , ˆ z ∗ ) be an ϵ 2 -Nash equilibrium of (4) with ϵ ≤ 1 / 10 . Then, z j , ˆ z j ≤ 9 ϵ for all j ∈ [2 n ] .

Proof. We will prove that z j ≤ 9 ϵ for all j ∈ [2 n ] ; the corresponding claim for Player ˆ z follows similarly. Fix i ∈ [ n ] . Lemma A.17 shows that | y ∗ i -x ∗ i | ≤ 2 ϵ . We shall consider two cases.

First, suppose that | y ∗ i -x ∗ i | ≤ ϵ / 2 . Then,

<!-- formula-not-decoded -->

By Lemma A.1, it follows that ˆ z i ≤ 2 ϵ 2 , and similar reasoning yields ˆ z n + i ≤ 2 ϵ 2 . On the other hand, suppose that | y ∗ i -x ∗ i | &gt; ϵ / 2 . Without loss of generality, we can assume that y ∗ i -x ∗ i ≥ 0 ; the contrary case is symmetric. Since x ∗ ∈ ∆ n and y ∗ ∈ ∆ n , there is some coordinate j ∈ [ n ] such that

y ∗ j -x ∗ j &lt; 0 . As before, by Lemma A.1, it follows that ˆ z ∗ i ≤ ϵ 2 and ˆ z ∗ n + j ≤ ϵ 2 . Now, we consider the deviation

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

At the same time, since ( x ∗ , y ∗ , z ∗ , ˆ x ∗ , ˆ y ∗ , ˆ z ∗ ) is an ϵ 2 -Nash equilibrium, we have

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

We conclude that ˆ z ∗ i ≤ ϵ 2 and ˆ z ∗ n + i ≤ 9 ϵ . The case where x ∗ i -y ∗ i ≥ 0 can be treated similarly.

Armed with those two basic lemmas, we are ready to complete the proof of Theorem 1.7.

Proof of Theorem 1.7. Suppose that ( x ∗ , y ∗ , z ∗ , ˆ x ∗ , ˆ y ∗ , ˆ z ∗ ) is an ϵ 2 -Nash equilibrium. We have that for any y ′ ∈ ∆ n ,

<!-- formula-not-decoded -->

Moreover, considering a deviation from x ∗ to y ′ ,

<!-- formula-not-decoded -->

Summing (36) and (37),

<!-- formula-not-decoded -->

Moreover,

<!-- formula-not-decoded -->

Combining (38) and (39), we get that for all y ′ ∈ ∆ n ,

<!-- formula-not-decoded -->

Similarly, we can show that for all ˆ x ′ ∈ ∆ n ,

<!-- formula-not-decoded -->

Taking y ′ = ˆ x ′ in (40) and summing with (41), we get that for all ˆ x ′ ∈ ∆ n ,

<!-- formula-not-decoded -->

where we used the fact that C is skew-symmetric. Now, using the fact that the Nash equilibrium is symmetric, so that x ∗ = ˆ x ∗ , we have

<!-- formula-not-decoded -->

Setting A := -1 2 ( R + R ⊤ ) and C := R ⊤ -R , (42) shows that

<!-- formula-not-decoded -->

for any ˆ x ′ ∈ ∆ n ; ergo , ( x ∗ , x ∗ ) is a symmetric (21 n +1) | A min | ϵ -Nash equilibrium of the symmetric (two-player) game ( R , R ⊤ ) , and the proof follows from Theorem 2.2.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We provide proofs for all the claims made in the abstract and introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We formally state all assumptions needed for our results.

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

Justification: All assumptions are stated in the main body and the proofs are in the appendix.

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

Justification: The paper does not include experiments.

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

Justification: The paper does not include experiments.

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

Justification: The paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper does not include experiments.

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

Justification: The paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in the paper conforms with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: We do not foresee any direct societal impact.

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

Justification: The paper poses no such risks.

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

Justification: This paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

## Answer: [NA]

Justification: The development of the paper does not involve LLMs.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.