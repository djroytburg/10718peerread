## Efficient Kernelized Learning in Polyhedral Games beyond Full Information: From Colonel Blotto to Congestion Games

Andreas Kontogiannis *

NTUA &amp; Archimedes / Athena RC

Vasilis Pollatos *

NKUA &amp; Archimedes / Athena RC

## Panayotis Mertikopoulos

Univ. Grenoble Alpes, CNRS, Inria, Grenoble INP LIG 38000 Grenoble, France &amp; Archimedes / Athena RC

## Abstract

We examine the problem of efficiently learning coarse correlated equilibria (CCE) in polyhedral games, that is, normal-form games with an exponentially large number of actions per player and an underlying combinatorial structure-such as the classic Colonel Blotto game or congestion games. Achieving computational efficiency in this setting requires learning algorithms whose regret and per-iteration complexity scale at most polylogarithmically with the size of the players' action sets. This challenge has recently been addressed in the full-information setting, primarily through the use of kernelization; however, in the more realistic partial information setting, the situation is much more challenging, and existing approaches result in suboptimal and impractical runtime complexity to learn CCE. We address this gap via a novel kernelization-based framework for payoff-based learning in polyhedral games, which we then apply to certain key classes of polyhedral games-namely Colonel Blotto, graphic matroid and network congestion games. In so doing, we obtain a range of computationally efficient payoff-based learning algorithms which significantly improve upon prior work in terms of the runtime for learning CCE.

## 1 Introduction

Learning dynamics for computing equilibria in games have been extensively studied over recent decades. The origins trace back to the work of Brown and Robinson in the 1950s [21; 51], who introduced and analyzed fictitious play. A major conceptual breakthrough came with Blackwell's approachability theorem [16], which laid the foundation for the field of online learning and, in particular, for the development of no-regret learning [23]. Several influential learning algorithmssuch as multiplicative weights update (MWU) [7], follow-the-regularized-leader [54], and follow-theperturbed-leader [37]-have been shown to satisfy the no-regret property. These algorithms typically maintain a probability distribution (commonly referred to as a 'policy') over actions and update it iteratively, with per-iteration complexity that is polynomial in the number of actions.

Remarkably, no-regret algorithms can be used as a black-box in repeated games under the fullinformation setting, where each player observes the cost of all available actions, to recover wellestablished equilibrium concepts, such as coarse correlated equilibria (CCE). The no-regret property is of great importance for learning in games, as it guarantees that the time-average cost of any player using such an algorithm is no worse than the cost of the best fixed action in hindsight-regardless of

* Equal contribution.

Gabriele Farina MIT

Ioannis Panageas

UC Irvine Archimedes / Athena RC

how the other players choose their actions. Consequently, if all players adopt no-regret algorithms, the learning dynamics converge to CCE.

Polyhedral Games: motivation and challenges. In this paper, we focus on the problem of learning CCE in multi-player games with combinatorial structure and large action spaces where the players simultaneously use no-regret learning dynamics for T rounds. Specifically, we consider polyhedral games [32] (also dubbed linear hypergraph games [12]), a rich class of normal-form games where the actions per player are d -dimensional binary vectors with at most m ≤ d ones.

Polyhedral games capture important classes of games with large action sets, including the well-studied Colonel Blotto game [18], congestion games [52], extensive-form games [40], and dueling games [36]. For example, in multi-player Colonel Blotto games , each player must allocate n soldiers among k battlefields, where n is typically much larger than k . In this case, using the one-hot representation (see Section 4), we have that m = k , d = nk and N = ( n + k -1 k -1 ) , with the latter being of order n k . In graphic matroid congestion games , given a undirected graph G ( V, E ) , each player must choose a spanning tree, that is the basis of a graphic matroid of rank V -1 . In this case m = V -1 , d = | E | and N is of order | E | | V | . Similarly, in network congestion games , each player needs to choose a path from s → t , and the maximal path length is K . Here, m = K , d = | E | and N is of order | E | K .

In all the aforementioned examples, the number of actions N per player, grows exponentially with m (approximately of order d m ), and as a result the vanilla learning methods for finding CCE become computationally inefficient since their per-iteration complexity is polynomial in N and not polynomial in d, m . This computational challenge has recently been addressed in the full-information setting. Beaglehole et al. [12] demonstrated how to perform approximate fast sampling from the MWUdistribution in specific polyhedral games (including the Colonel Blotto and graphic matroid congestion games). However, their approach is somewhat restrictive beyond approximately sampling from MWU, and thus sub-optimal in convergence rate, as it is unknown how to use their techniques to efficiently deploy the near-optimal Optimistic MWU [30] in such game settings. In contrast, Farina et al. [32] proposed an efficient general methodology to simulate the exact MWU(which allows to use optimism) algorithm via kernelization , requiring only Θ( d ) kernel computations (see Section 2 for a formal definition) per iteration for any polyhedral game. In particular, the kernelization approach developed in [32] has led to state-of-the-art runtime to find CCE * in extensive-form games, as recently established in [31].

However, the applicability of kernelization to polyhedral games remains largely unexplored beyond the full-information setting-that is, in the bandit (and also semi-bandit ) feedback settings. These settings are of particular interest in practice, as the full-information assumption-where the costs of all available actions are revealed after each round-is often unrealistic. In contrast, bandit feedback reflects a more practical regime in which only the cost of the selected action is observed. For example, learning under full-information feedback in network congestion games would impractically require each player to be able to observe the cost of all paths of the network, rather than just the cost of the path she actually chose.

In order to obtain equilibrium convergence guarantees in a bandit setting, the learning dynamics of each player must satisfy no-realized-regret guarantees that hold with high probability against adaptive adversaries (i.e., assuming that the other players can potentially adjust their policies based on the player's past actions)-a stringent and technically demanding requirement. This stands in contrast to the more commonly studied expected regret from the online learning literature (e.g., see [38; 24; 26]). An even more challenging, but very practical, requirement is to ensure that the learning dynamics achieve an efficient runtime complexity to find CCE , with minimal dependence on the game parameters d and m , while still maintaining the no-regret property with favorable dependence on T as much as possible.

Many algorithms from the bandit linear optimization literature [41] can be leveraged to learn ε -CCE in polyhedral games. Bartlett et al. [11] provide an algorithm with high probability guarantees that achieves a √ T regret bound, albeit requiring a prohibitive per-iteration complexity of poly ( N ) . The well-established GEOMETRICHEDGE algorithm (also known as COMBAND [24], or EXP2 [22]) originally proposed by Dani et al. [28] has been shown to achieve T 2 / 3 regret with high probability [53; 19]. Despite GEOMETRICHEDGE being a classical algorithm in the literature, how to efficiently

* The runtime of an algorithm for finding an equilibrium is defined as the product between the number of iterations T needed to compute the equilibrium and the algorithm's per-iteration complexity.

implement it remains generally unclear, with path planning being the only setting where efficient implementations (e.g., via weight pushing [58; 62; 63]) were known prior to our work. Recently, Lee et al. [42] and Zimmert and Lattimore [65] proposed algorithms for continuous action spaces-which can be extended to polyhedral games using the same techniques as Abernethy et al. [1]-achieving a regret bound of O ( md 7 / 2 √ T ) and O ( md 2 √ T ) , respectively. However, the above bounds combined with a per-iteration complexity † , which suboptimally depends on d , result in impractical runtime complexity results for learning ε -CCE. In particular, the runtime of the algorithm in [42] to find ε -CCE scales as d 10 , while that of [65] scales as d 9 -both exhibiting impractically large dependence on the game parameters. Even more recently, the concurrent work of [46] proposed an algorithm for online shortest paths in DAGs with a near-optimal regret bound of O ( K 3 / 2 √ | E | T ) . However, their algorithm also comes with a polynomial yet impractical runtime complexity to find an approximate CCE stemming from ellipsoid method calls and other costly procedures.

Given the impractical runtime complexity results of the aforementioned approaches for learning CCE in polyhedral games, in this paper, we aim to address the following question:

Can kernelization techniques be extended beyond the full-information setting to design no-regret learning dynamics for computing CCE with state-of-the-art runtime complexity-achieving minimal dependence on the game parameters d and m ?

Main Contributions and Techniques. In this paper, we answer the above question affirmatively. Due to the exponentially large (in m ) per-player action sets in polyhedral games, designing efficient payoff-based learning algorithms involves addressing three primary challenges: (a) fast calculating the loss estimators which are used to update each player's policy, (b) fast sampling from each player's policy, and (c) ensuring that each player achieves efficient no-realized-regret guarantees, which imply efficiently learning ε -CCE.

To face the above challenges, we propose a kernelization -based framework, which allows us to efficiently implement standard loss estimators from bandit linear optimization. Specifically, in the bandit setting (Section 3.1), we propose a kernelized customization of the well-established GEOMETRICHEDGE algorithm [28] (see Algorithm 1). In contrast to the full-information setting, where the approach of [32] required the first moments of a MWU distribution, in the bandit setting, we require the second moments of MWU, needed to construct the unbiased combinatorial bandit estimator [28; 24]. Importantly, we show that we can efficiently calculate such second moments via only Θ( d 2 ) kernel computations (Theorem 3.1). In the semi-bandit setting, our approach (see Section 3.2) utilizes the implicit exploration loss estimator [48], which, we show that it is compatible with the kernels used for the first moment of MWU. In addition, we propose a general efficient sampling scheme (Procedure SAMPLING in Algorithm 1), based on kernelization, which only requires extra Θ( d ) kernel computations.

Apart from improvements in the per-iteration complexity, our analysis provides the following noregret results for learning in polyhedral games: In the bandit setting , we achieve ˜ O ( d 2 / 3 m 4 / 3 T 2 / 3 ) regret with high probability (Theorem 3.2), improving upon baselines [53; 19] in the dependence on the game parameters. Moreover, we achieve better regret than [65] in the realistic regime where T ≤ d 6 . Regarding the semi-bandit setting , we achieve ˜ O ( m √ Td ) regret with high probability (Theorem 3.4), which is a factor √ m worse than the optimal expected regret guarantee [8]. To the best of our knowledge, this is the first high probability result on the general setting.

To showcase the power of our general framework, we study three important classes of polyhedral games: the multi-player Colonel Blotto, graphic matroid and network congestion games.

In Colonel Blotto games , we use kernelization techniques based on the generator function induced by the game's combinatorial structure, in order to efficiently compute the required kernels. Remarkably, our kernelization-based approach operates directly on the geometry of the Colonel Blotto game, by allowing us to leverage an efficient Θ( nk ) -representation. Prior work had only been able to operate with DAG representations of the set, leading to suboptimal formulations with O ( n 2 k ) edges. As shown in Table 1 (and stated in Theorem 4.4), in the bandit setting, our approach learns an ε -CCE in time ˜ O ( n 2+ ω k 6+ ω /ε 3 ) -where ω is the multiplication exponent (currently the best known is

† The per-iteration complexity of the algorithm in [42] is ˜ O ( d 3 ) due to the fact that the optimization step is solved via an interior point method (see [2]), while that of [65] is ˜ O ( d 5 ) due to the pre-processing step needed to sample from a log-concave distribution (see [44]).

| Algorithm                  | Runtime to ε -CCE                               | Representation   | Feedback    |
|----------------------------|-------------------------------------------------|------------------|-------------|
| Beaglehole et al. [12]     | ˜ O ( nk 4 /ε 2 )                               | O ( k log n )    | Full-Info   |
| Our Work                   | ˜ O ( &#124;P&#124; nk 3 /ε )                   | O ( nk )         | Full-Info   |
| Our Work                   | ˜ O ( n 2 k 4 /ε 2 )                            | O ( nk )         | Semi-Bandit |
| Leon et al. [43]           | ˜ O ( n 4 k 5 ε 3 ( max { 1 λ min ,n 2 }) 3 2 ) | O ( n 2 k )      | Bandit      |
| Zimmert and Lattimore [65] | ˜ O ( n 18 k 11 /ε 2 ) †                        | O ( n 2 k )      | Bandit      |
| Our Work                   | ˜ O ( n 2+ ω k 6+ ω /ε 3 )                      | O ( nk )         | Bandit      |

Table 1: Comparison of results in Colonel Blotto games , split by feedback type (full-information, semi-bandit, and bandit). † : The approach of [65] is evaluated using the layered graph polytope [61] of size n 2 k . ‡ : The runtime of [43] depends on the arbitrarily large 1 /λ min - that is, the inverse of the minimum eigenvalue of E [ vv T ] under the exploration distribution.

Table 2: Comparison in Graphic Matroid Congestion Games . To assess [65], we used the polytope representation of [47] that uses d = | V | 3 and has a small number of constraints.

| Algorithm   | Runtime to ε -CCE                                                                            | Feedback    |
|-------------|----------------------------------------------------------------------------------------------|-------------|
| [12]        | ˜ O ( &#124; V &#124; 5 /ε 2 )                                                               | Full-Info   |
| Our Work    | ˜ O ( &#124; P &#124;&#124; V &#124; 4 ( &#124; V &#124; ω - 1 + &#124; E &#124; ) /ε )      | Full-Info   |
| Our Work    | ˜ O ( &#124; E &#124; 2 &#124; V &#124; 2+ ω /ε 2 )                                          | Semi-Bandit |
| [65]        | ˜ O ( &#124; V &#124; 29 /ε 2 )                                                              | Bandit      |
| Our Work    | ˜ O ( &#124; E &#124; 3 &#124; V &#124; 6 ( &#124; V &#124; ω - 1 + &#124; E &#124; ) /ε 3 ) | Bandit      |

Table 3: Comparison of results in Network Congestion Games . ∗ : The algorithm proposed in [49] also achieves convergence to Nash equilibria, albeit with slower rates.

| Algorithm   | Runtime to ε -CCE                     | Feedback    |
|-------------|---------------------------------------|-------------|
| [35]        | ˜ O ( &#124; E &#124; 1+ ω K 3 /ε 2 ) | Semi-Bandit |
| [49]        | ˜ O ( &#124; E &#124; 9 /ε 4 ) ∗      | Semi-Bandit |
| Our Work    | ˜ O ( &#124; E &#124; 1+ ω K 2 /ε 2 ) | Semi-Bandit |
| [65]        | ˜ O ( &#124; E &#124; 9 K 10 /ε 2 )   | Bandit      |
| Our Work    | ˜ O ( &#124; E &#124; 2+ ω K 4 /ε 3 ) | Bandit      |

≈ 2 . 372 [5])-thereby significantly improving over [65] in the dependence on the game parameters by a factor ≈ n 13 k 2 . In the semi-bandit setting, our approach learns an ε -CCE in time ˜ O ( n 2 k 4 /ε 2 ) .

In graphic matroid congestion games , we design kernelization techniques based on the celebrated Matrix-Tree Theorem [59]. To reduce the amortized kernel computation time, we use fast rank-1 updates of the LU decomposition of Laplacian matrices based on the structure of the required kernels. Moreover, we perform efficient exact sampling via an incremental kernelization approach. As shown in Table 2 (and also in Theorem 5.2), in the bandit setting, our approach learns an ε -CCE in time ˜ O ( | E | 3 | V | 6 ( | V | ω -1 + | E | ) /ε 3 ) , significantly improving upon the very impractical dependence on | V | 29 of [65]. In the semi-bandit setting, our approach learns an ε -CCE in time ˜ O ( | E | 2 | V | 2+ ω /ε 2 ) .

Remark 1.1. We can combine our kernelization results for Colonel Blotto and graphic matroid congestion games with the full-information framework developed in [32] to yield 1 /ε convergence to ε -CCE in these games-thus addressing an open question of Beaglehole et al. [12]

Remark 1.2. Our kernelization techniques developed for graphic matroids (see Lemma 5.1) allow us to efficiently implement GEOMETRICHEDGE over spanning trees-thus, to the the best of our knowledge, resolving an open question posed by Cesa-Bianchi and Lugosi [24].

In network congestion games , our framework improves upon [35; 27; 49; 65]. For a summary of our results, we refer to Table 3. Due to space constraints, our formal results can be found in Appendix I. Further details on existing approaches for each of the above games can be found in Appendix A.

## 2 Preliminaries

Polyhedral Games. In this paper, we consider Polyhedral Games , a structured class of normalform games with exponentially large action sets, where each action can be represented as a binary

d -dimensional vector of at most m ones and the incurred cost is linear in the action vector. For simplicity, here we assume that all players have the same action sets. Formally, we represent a polyhedral game as a tuple G = ( P , V , { L i } ) . The set P defines the set of players, each of which is assigned a unique player identifier in [ |P| ] := { 1 , 2 , . . . , |P|} . The finite set V ⊂ R d of size N represents the actions available to each player i ∈ [ |P| ] , such that for any v ∈ V , ∥ v ∥ 1 ≤ m . We denote by -i all agents except i . We define the loss vector function ℓ i : V | P | → R d + . L i : V | P | → R + is the cost function, which is linear in v i ; that is, L i ( v i ; v -i ) = ℓ ( v i ; v -i ) · v i .

Online Learning Setup in Polyhedral Games. In polyhedral game dynamics under partialinformation feedback, each player iteratively updates her strategies based on the feedback she receives about the loss. We consider the bandit and semi-bandit settings. In the semi-bandit setting, each player i selects an action v i ∈ V i and receives the losses ℓ i ( j ) of the loss vector ℓ i = ℓ i ( v i ; v -i ) for all j such that v i ( j ) = 1 . In the bandit setting, each player receives only L i ( v i ; v -i ) .

However, it is not clear how a selfish player i should update her strategy in order to minimize her overall loss, since the strategies of the other players can arbitrarily change over time. Thus, player i tries to minimize her experienced loss under the worst-case assumption that the loss of each coordinate is selected by a malicious adversary.

Based on the above, we focus on the single-player's perspective and examine an abstract-online learning-model, where each player is a decision maker interacting with an unknown and potentially adversarial environment. At each round t = 1 , 2 , . . . , T of the online learning process, the decision maker samples an action v t ∈ V from a probability distribution p t ∈ ∆( V ) . Subsequently, the environment chooses a loss vector ℓ t ∈ R d , potentially in an adversarial fashion. This is the same setup adopted in [49; 27]. Given any round T , we define the regret up to round T as follows:

<!-- formula-not-decoded -->

We note that the above notion measures realized regret, that is, it measures the performance of the algorithm based on the actions sampled from the distribution p t . We say that players are playing no-regret learning in the game if each one of them achieves sublinear regret.

A prominent result in the theory of learning in games establishes a celebrated connection between no-regret learning and CCE of the game (which, in two-player zero-sum games, are Nash equilibria).

Theorem 2.1 (Informal, [34]) . Suppose | P | players are playing no-regret learning in the game. Let σ ∗ := 1 T ∑ T t =1 v ( t ) 1 ⊗··· ⊗ v ( t ) |P| be the time-average joint actions over T rounds. Then, σ ∗ forms an T -1 max( R T, 1 , . . . , R T, |P| ) -approximate CCE of the game, where R T,i is the regret for the i -th player at the T -th round.

Kernelized MWU. Multiplicative Weights Update (MWU) is an online learning algorithm that iteratively updates a distribution p t over actions in V . Let p 0 := 1 |V| 1 ∈ ∆( V ) . The MWU rule at each time step t is p t ( v ) ∝ p t -1 ( v ) · e -η t w t ( v ) , ∀ v ∈ V . In the standard MWU variant we set w t := ℓ t -1 , where ℓ t -1 is the loss vector observed at t -1 . By setting w t := 2 ℓ t -1 -ℓ t -2 we derive the Optimistic MWU (OMWU) algorithm [30], which achieves constant regret (up to logarithmic factors), and thus 1 /ε convergence to CCE in the context of learning in games.

In polyhedral games, we are interested in the efficient calculation of moments of the MWU distribution. For this aim, a useful tool introduced in [58; 32] is the kernel function R d × R d → R defined as follows: K V ( x, y ) := ∑ v ∈V ∏ j : ∈ v ( j )=1 x ( j ) y ( j ) . The next theorem shows how to compute the first moment of p t via d +1 kernel computations.

Theorem 2.2 (First Moment Calculation, [32]) . At all rounds t ≥ 0 , the first moment of the MWU distribution, p t , can be calculated as follows:

<!-- formula-not-decoded -->

̸

where C t ( j ) := exp { -∑ t τ =1 η τ w τ ( j ) } and ¯ e j ( h ) := ✶ { h = j } , for h, j ∈ [ d ] .

## 3 Kernelized Payoff-based Learning in Polyhedral Games

In this section, we design a framework for efficient payoff-based learning in polyhedral games, under bandit and semi-bandit feedback. Upon this framework, we build learning algorithms, which achieve efficient no-realized-regret learning with high probability guarantees against adaptive adversaries-a key requirement to show convergence to CCE. For our algorithms to be efficiently implementable, as we will see, it suffices that the kernels used for constructing the loss estimators and for sampling (highlighted in orange in Algorithm 1) can be computed efficiently. This can be achieved by effectively leveraging the game's combinatorial structure, as we will explain in depth later in the paper. For the remainder of this section, we assume that we have oracles for calculating the required kernels. In the next sections, we will demonstrate how our algorithms can be implemented efficiently in prominent examples of polyhedral games.

## 3.1 Kernelized GEOMETRICHEDGE for Bandit No-Regret Learning

## Algorithm 1: Kernelized GEOMETRICHEGDE

```
Data: d , m , η > 0 , γ ∈ [0 , 1] 1 Compute a 2 -approximate-barycentric-spanner B 2 Initialize q 0 = [1 /N,. . . , 1 /N ] ∈ ∆( V ) , µ = 1 d ✶ { v ∈ B } , c 0 ( j ) = 0 and C 0 ( j ) = 1 , ∀ j ∈ [ d ] 3 for t = 1 , . . . , T do 4 Mixing: p t = (1 -γ ) q t + γµ , where q t = MWU ( C t ) 5 Compute the kernels: K V ( C t -1 , 1 ) and { K V ( C t -1 , ¯ e j,j ′ ) } , ∀ j, j ′ ∈ [ d ] 6 Sample v t ∼ (1 -γ ) SAMPLING ( V , C t -1 ) + γµ 7 Observe the bandit loss L t = ℓ t · v t 8 Compute Σ t ( q t ) using Theorem 3.1 and set Σ t = (1 -γ )Σ t ( q t ) + γ d BB T 9 Compute the unbiased loss estimator: ̂ ℓ t = L t Σ -1 t v t 10 Update the aggregated loss estimators: c t ( j ) = c t -1 ( j ) + ̂ ℓ t ( j ) , ∀ j ∈ [ d ] 11 Update the exponential cumulative loss estimators: C t ( j ) = exp( -ηc t ( j )) , ∀ j ∈ [ d ] 12 13 Procedure: SAMPLING 14 Input: V , C 15 Sample v [1] ∼ Be ( 1 -K V ( C, ¯ e 1 ) K V ( C, 1 ) ) 16 for j = 2 , ..., d do 17 Compute the kernel: K V ( j ) 18 Set V ( j ) = { v ′ ∈ V : v ′ [ i ] = v [ i ] , ∀ i ∈ [ j -1] } and p j = 1 -K V ( j ) ( C, ¯ e j ) K V ( j ) ( C, 1 ) 19 Sample v [ j ] ∼ Be ( p j ) 20 Return: v
```

Wepresent our first algorithm, which establishes efficient no-regret learning in polyhedral games under bandit feedback. Our algorithm (Algorithm 1) is a kernelized customization of GEOMETRICHEDGE [28], a classical algorithm in the study of combinatorial bandits [41]. Despite GEOMETRICHEDGE being an algorithm with a well-studied expected regret analysis, how to efficiently implement it remains largely unclear. The primary challenges in applying the vanilla method are the following:

1. Calculating Σ = E [ vv T ] - needed to construct the unbiased loss estimates which will be used by a MWU routine - in poly ( d, m ) time.
2. Sampling from MWU in poly ( d, m ) time.

In this paper, we tackle the above challenges in the context of polyhedral games (however, our approach can also be applied to the well-studied combinatorial settings discussed in [24]). The main idea of our approach is to utilize a loss estimate for each coordinate j ∈ [ d ] , which can be

kernelized efficiently, and simulate MWU using a fast sampling schema based on the computed kernels. Subsequently, we present the main components of our approach.

Second Moment Kernelization. Algorithm 1 uses a distribution p t which is the mixture between a MWU distribution q t and the uniform distribution, µ , over a 2-approximate barycentric spanner of V . Due to space constraints, we prompt the interested reader to Appendix E for background on barycentric spanners. In contrast to the full-information setting, where kernelized MWU [32] requires the first moment of the MWU distribution to simulate the update rule, our algorithm also requires the second moment of p t (i.e., the autocorrelation matrix, denoted by Σ t ) to construct the standard unbiased estimator of GEOMETRICHEDGE (Step 9). Through Step 8, it suffices to efficiently calculate Σ t ( q t ) , that is the autocorrelation matrix under the law of q t , which in general was not known how to efficiently compute prior to our work (with the only exception being path planning problems where weight pushing techniques [58; 53] can be applied).

For this purpose, we will make use of kernelization. The next theorem shows that we can efficiently calculate the second moment of q t using only d 2 +1 kernel computations ‡ .

Theorem 3.1 (Second Moment Calculation) . Let Σ t ( q t ) := ∑ v ∈V q t ( v ) · ( vv T ) be the autocorrelation matrix under the law of a MWU distribution q t . Then, for all j, j ′ ∈ [ d ] ,

<!-- formula-not-decoded -->

̸

where ¯ e jj ′ ( h ) := ✶ { h = j and h = j ′ } , for h, h ′ , j ∈ [ d ] and ¯ e j ( h ) := ✶ { h = j } , for h, j ∈ [ d ] . Kernelization implies efficient exact sampling. Based on kernelization, we propose an efficient sampling scheme (Procedure SAMPLING in Algorithm 1) that only requires extra d kernel computations. We are interested in sampling v ∼ p t . By using the chain rule on the probability of the intersection events, we derive the following:

̸

̸

<!-- formula-not-decoded -->

It is easy to see that the j -th term of the above product equals the j -th coordinate of the first moment kernelization (see Theorem 2.2 and Observation 3.3) of the conditional polytope V ( j ) -i.e., the polytope which has the first j -1 coordinate values equal to the j -1 sampled values (Step 18). Based on this, we iteratively sample each coordinate j ∈ [ d ] from a Bernoulli distribution (Step 19), which has probability equal to p j = Pr[ v ( j ) | v (1) , . . . , v ( j -1)] = 1 -K V ( j ) ( C, ¯ e j ) K V ( j ) ( C, 1 ) (Step 19).

Improved regret dependence on the game parameters. Our analysis differs from that of the original paper of GEOMETRICHEDGE [28], which studied expected regret. Importantly, we improve upon prior analyses [53; 19], which also studied the realized regret of the algorithm, by reducing the regret's dependence on d and m , while avoiding dependence on the possibly exponentially small minimum eigenvalue, λ min, of the autocorrelation matrix under the law of the initial distribution. In particular, we achieve this by using a more careful analysis on the effect of the barycentric spanner to the variance of the estimator. The following theorem shows that Algorithm 1 is no-regret.

Theorem 3.2 (No-Regret under Bandit Feedback) . For T ≥ 8 d 2 m , by setting γ = d 2 / 3 m 1 / 3 T 1 / 3 and η = 1 4 d 4 / 3 m 2 / 3 T 1 / 3 , Algorithm 1 achieves regret R T ≤ ˜ O ( d 2 / 3 m 4 / 3 T 2 / 3 ) with high probability.

## 3.2 The Semi-Bandit Feedback Case: Kernelizing Implicit Exploration

Now, we discuss our second learning algorithm for polyhedral games which establishes efficient no-regret learning under semi-bandit feedback. The main idea is similar to that of the bandit settingthat is, we utilize a loss estimator for each coordinate j ∈ [ d ] , which can be kernelized efficiently, and use the SAMPLING procedure to fast sample from a MWU routine using the computed kernels. Due to the exponentially large action set V , one challenge here is that it is intractable to brute-force over V in order to compute the unconditional probabilities, Pr[ v t ( j ) = 1] , of selecting j ∈ [ d ] as an active coordinate, needed to compute the standard loss estimators used in adversarial multi-armed bandits (MABs) [41]. The following observation suggests that we can kernelize such loss estimators.

‡ Related results were concurrently shown in [55] to compute the Hessian of a self-concordant function, which is needed to implement Newton's method.

Observation 3.3. Using Theorem 2.2, we can compute the first moment x t , which it turns out to provide the probabilities needed to compute the standard loss estimators used in adversarial MABs, since for any j ∈ [ d ] , we have that x t ( j ) := E v ∼ p t [ { v ( j ) = 1 } ] = Pr[ v ( j ) = 1] .

✶ Our algorithm (Algorithm 2 in Appendix B) kernelizes the implicit exploration (IX) loss estimator, ˜ ℓ t ( j ) = ℓ t ( j ) x t ( j )+ γ ✶ { v t ( j ) = 1 } , proposed in [48], which ensures sufficient exploration for each coordinate j ∈ [ d ] with low variance. Despite the fact that the implicit exploration loss estimator is biased, it satisfies the important property that, with high probability, the aggregated estimator losses are upper bounded by the realized losses plus a factor of ˜ O (1 /γ ) . The following theorem shows that the proposed algorithm is no-regret.

Theorem 3.4 (No-Regret under Semi-Bandit Feedback) . By setting γ = m/ √ dT and η = 1 / √ dT , Algorithm 2 achieves regret R T ≤ ˜ O ( m √ Td ) with high probability.

## 4 Efficient Kernelization in Colonel Blotto Games (CBGs)

We consider the setting proposed in [4] for the multiplayer Colonel Blotto game [17]. Each player i ∈ [ |P| ] must allocate n soldiers among k battlefields. Let variable s i,h denote the number of soldiers allocated by the i -th player to the h -th battlefield. Given the soldier assignments of all players, a per-battlefield loss is defined for each player i ∈ [ | P | ] , and and the incurred cost of player i is given by the sum of her per-battlefield losses.

Θ( nk ) - representation. We aim to find succinct vector representations of each player's actions and the loss. One challenge here is that we need the action and loss representations to satisfy the definition of polyhedral games- that is, the cost of each action, given the actions of other players, must equal the dot product of the action and loss representations. One such representation is through the layered graph [13] (also used in [60; 61; 43]), which implies a representation dimensionality of Θ( n 2 k ) that can be a bottleneck for efficiently learning CCE, as shown in Table 1.

Without loss of generality, we focus on player i and drop the subscript i . We use the notation [ b ] 0 = { 0 , 1 , ..., b } for b ∈ N . Let d = ( n +1) k . For any action a ∈ A , we consider its succinct representation v ∈ V ⊂ { 0 , 1 } d such that for all h ∈ [ k ] and s ∈ [ n ] 0 , v [ h, s ] = 1 iff a assigns s soldiers to the h -th battlefield. We similarly define the representation of the loss ℓ , such that for all h ∈ [ k ] and s ∈ [ n ] 0 , ℓ [ h, s ] is the h -th battlefield loss observed when assigning s soldiers to the h -th battlefield, given the assignments of the other players in h .

Remark 4.1. Although the Θ( nk ) -representation is more straightforward to design and more succinct than the Θ( n 2 k ) -graph-representation [61], it is not known how to derive a polytope description in the form of a small number of linear inequalities with the pure actions as corners-thus common techniques, such as Carathéodory decomposition (e.g., [26]) and barrier methods (e.g. [42; 65]) cannot be used. Kernelization overcomes this obstacle by operating directly on the game's geometry.

Kernelization. With the succinct representation established above, we are now ready to describe how fast kernel computations are achieved in Colonel Blotto games.

Given weight vectors x, y ∈ { 0 , 1 } d we define the polynomial P x,y ( z ) as follows:

<!-- formula-not-decoded -->

The key point is that the n -th coefficient of z in P x,y ( z ) generates kernel K V ( x, y ) . The main idea is as follows. To compute the n -th coefficient of 3, we execute a running product over the factors of the polynomial. This process involves k updates of the partial product. After each update, the partial product is truncated down to degree n . Thus, inductively we ensure that all k multiplications involve polynomials of degree at most n . Based on the above, we derive the following proposition.

Proposition 4.2. For given x, y ∈ { 0 , 1 } d , kernel K V ( x, y ) can be computed in time O ( nk log n ) .

To construct the loss estimators of the proposed algorithms, we need to compute d kernels K V ( C t , ¯ e j ) , for j ∈ [ d ] , for the semi-bandit setting and d 2 kernels K V ( C t , ¯ e j,j ′ ) , for j, j ′ ∈ [ d ] , for the bandit

setting, as well as the kernel K V ( C t , 1 ) . A naive approach is to use Proposition 4.2 separately for each kernel, resulting in a total kernel computation time O ( n 3 k 3 log n ) for the bandit setting and O ( n 2 k 2 log n ) for the semi-bandit.

We provide two algorithms, namely Algorithm 3 and 4 (Appendix G), which speed up the process of computing all required kernels by leveraging the above ideas and appropriate precomputing.

Lemma 4.3. At each round t ∈ [ T ] , all kernels K V ( C t , ¯ e j ) , for j ∈ [ d ] , can be computed in time O ( nk log n ) . Moreover, all kernels K V ( C t , ¯ e j,j ′ ) , for j, j ′ ∈ [ d ] , can be computed in time O ( n 2 k 2 ) .

Combining the above with the exact sampling procedure provided in [12] (see Algorithm 5, Appendix G.3), based on which we can calculate the required kernels of our SAMPLING procedure in time O ( nk log n ) , the per-iteration complexities for the bandit and semi-bandit are O ( n ω k ω log n ) and O ( nk log n ) , respectively. Putting everything together, we derive the following main result.

Theorem 4.4 (Runtime to learn ε -CCE) . In a Colonel Blotto game, under bandit feedback, if all players adopt Algorithm 1, then the total runtime for finding an ε -CCE, with high probability, is ˜ O ( n 2+ ω k 6+ ω /ϵ 3 ) . Under semi-bandit feedback, if all players adopt Algorithm 2 (Appendix B), then the total runtime for finding an ε -CCE, with high probability, is ˜ O ( n 2 k 4 /ϵ 2 ) .

Remark 4.5. If the Colonel Blotto game is two-player zero-sum (a more traditional setting which has received much attention [14; 15; 4]), then our algorithm learns an ϵ -Nash equilibrium.

## 5 Efficient Kernelization in Graphic Matroid Congestion Games (GMCGs)

In a graphic matroid congestion game (GMCG), players compete for the edges of a connected undirected graph G = ( V, E ) , with the actions of each player being spanning trees in G [64; 3; 33]. We use the incidence vector representation of actions v ∈ { 0 , 1 } | E | and denote by V the set of all these incidence vectors. Given an action profile ( v i , v -i ) , the total loss of player i is the sum of the losses of the selected edges of v i . Typically the cost of each edge is equal to the number of players using it but our framework can also handle arbitrary edge cost functions. Next, we will show how to efficiently compute the required kernels and perform efficient sampling in GMCGs.

̸

Kernelization. Given an edge weight vector C ∈ R | E | to compute the kernel K V ( C, 1 ) , we make use of the weighted Matrix-Tree Theorem [59; 45], which states that the value of ∑ T ∈V ∏ e ∈ T C ( e ) equals the value of a cofactor of the weighted Laplacian A of the graph, where A u,u = ∑ e ′ ∈ E incident to u C ( e ′ ) and A u,v = -C ( e ) · { e ∈ E } for u = v and edge e = ( u, v ) .

✶ A naive approach is to use the Matrix-Tree Theorem for each kernel separately, taking total time O ( | E | 2 | V | ω ) for the kernel computations in the bandit and O ( | E || V | ω ) in the semi-bandit setting.

We provide an algorithm (see Appendix H) that reduces the amortized time per kernel computation. Notably, the Matrix-Tree Theorem holds for any cofactor of the Laplacian matrix. We leverage this property by making a strategic choice of which row and column to delete. For each edge j ∈ [ | E | ] consider the Laplacian used for the computation of the kernel K V ( C, ¯ e j ) . The Laplacian for this kernel is constructed in the same way as the one we described above for K V ( C, 1 ) but with the difference that C ( j ) is set to zero. The main idea is that for each node v ∈ V , we can precompute the LU decomposition of A -v, -v , that is the submatrix of A derived by deleting row v and column v , and then for each j = ( u, u ′ ) ∈ E , we can fast compute kernel K ( C, ¯ e j ) by computing the determinant of that kernel's Laplacian via recursive LU updating [56] in O ( | V | 2 ) . The key point in this analysis is that we can always select a submatrix of the kernel's Laplacian that only differs in one element than A -u, -u . Similar arguments can also be used for fast computing K V ( C, ¯ e j,j ′ ) .

Sampling. We provide an efficient implementation of the SAMPLING procedure of Algorithm 1 for GMCGs (see Appendix H, Algorithm 8). Our approach is based on an iterative kernelization process where we sample each coordinate incrementally. The challenge here is how to perform kernelization on the conditional action set V ( j ) , for j ∈ [ | E | ] , induced by the so far sampled coordinates up to j . Interestingly, V ( j ) operates on an underlying multi-graph. The main idea is to transform this multi-graph into a meta-graph, where a meta-node merges the nodes of the j -th edge of the initial graph and a meta-edge accumulates the weights of parallel edges connecting the same nodes. First,

we show that it suffices to perform kernelization on the meta-graph (Proposition H.1) and, based on that, we show that our approach is efficient via an induction argument. Importantly, we derive the following lemma.

Lemma 5.1. At each round t ∈ [ T ] , all kernels K V ( C t , ¯ e j ) , for j ∈ [ | E | ] , can be computed in time O ( | V | ω +1 + | E || V | 2 ) and all kernels K V ( C t , ¯ e j,j ′ ) , for j, j ′ ∈ [ d ] , can be computed in time O ( | E || V | ω +1 + | E | 2 | V | 2 ) . Moreover, SAMPLING ( V , C t ) can be implemented in time O ( | E || V | ω ) .

Putting everything together, we derive the following main result.

Theorem 5.2 (Runtime to learn ε -CCE) . In a graphic matroid congestion game, under bandit feedback, if all players adopt Algorithm 1, then the total runtime for finding an ε -CCE, with high probability, is ˜ O ( | E | 3 | V | 6 ( | V | ω -1 + | E | ) /ε 3 ) . Under semi-bandit feedback, if all players adopt Algorithm 2 (Appendix B), then the total runtime for finding an ε -CCE, with high probability, is ˜ O ( | E | 2 | V | 2+ ω /ε 2 ) .

## 6 Conclusion

In this paper, we focused on the problem of efficiently learning coarse correlated equilibrium (CCE) in polyhedral games via kernelization-beyond full-information feedback. In particular, we proposed kernelized no-regret learning algorithms that improve the runtime of state-of-the-art methods in three important classes of polyhedral games, namely Colonel Blotto, graphic matroid and network congestion games.

There are several important open questions for follow-up research:

- Most important of all is whether we can design an FPTAS algorithm for efficiently learning correlated equilibria (CE) in polyhedral games; a stronger equilibrium notion than CCE.
- Another interesting open question is whether we can further leverage kernelization to achieve 1 /ε 2 dependence in the bandit setting, with a better dependence on d than Zimmert and Lattimore [65].
- Computing Nash equilibria in the general setting of the Colonel Blotto games we have studied in this paper is PPAD-hard, as any m-player normal-form game can be polynomially reduced to an m-player Colonel Blotto game. However, what can be said about the computational complexity of computing Nash equilibria in Colonel Blotto games with monotone piecewise-constant utility functions (e.g., see [12])?

We leave these as open questions for future work on the topic.

## Acknowledgments

Gabriele Farina was supported in part by the National Science Foundation award CCF-2443068, ONR grant N000142512296, and an AI2050 Early Career Fellowship. Ioannis Panageas was supported by NSF grant CCF-2454115. Most of this work was done while all authors were visiting Archimedes Research Unit, Athens, Greece. This work was also supported by the French National Research Agency (ANR) in the framework of the PEPR IA FOUNDRY project (ANR-23-PEIA-0003), the grant IRGA-SPICE (G7H-IRG24E90), and project MIS 5154714 of the National Recovery and Resilience Plan Greece 2.0 funded by the European Union under the NextGenerationEU Program.

## References

- [1] Jacob D Abernethy, Elad Hazan, and Alexander Rakhlin. Competing in the dark: An efficient algorithm for bandit linear optimization. In COLT , pages 263-274. Citeseer, 2008.
- [2] Jacob D Abernethy, Elad Hazan, and Alexander Rakhlin. Interior-point methods for fullinformation and bandit online learning. IEEE Transactions on Information Theory , 58(7): 4164-4175, 2012.

- [3] Heiner Ackermann, Heiko Röglin, and Berthold Vöcking. On the impact of combinatorial structure on congestion games. Journal of the ACM (JACM) , 55(6):1-22, 2008.
- [4] AmirMahdi Ahmadinejad, Sina Dehghani, MohammadTaghi Hajiaghayi, Brendan Lucier, Hamid Mahini, and Saeed Seddighin. From duels to battlefields: Computing equilibria of blotto and other games. Mathematics of Operations Research , 44(4):1304-1325, 2019.
- [5] Josh Alman and Virginia Vassilevska Williams. A refined laser method and faster matrix multiplication. TheoretiCS , 3, 2024.
- [6] Ioannis Anagnostides, Gabriele Farina, Christian Kroer, Chung-Wei Lee, Haipeng Luo, and Tuomas Sandholm. Uncoupled learning dynamics with O(log T) swap regret in multiplayer games. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 December 9, 2022 , 2022.
- [7] Sanjeev Arora, Elad Hazan, and Satyen Kale. The multiplicative weights update method: a meta-algorithm and applications. Theory of computing , 8(1):121-164, 2012.
- [8] Jean-Yves Audibert, Sébastien Bubeck, and Gábor Lugosi. Regret in online combinatorial optimization. Mathematics of Operations Research , 39(1):31-45, 2014.
- [9] Baruch Awerbuch and Robert Kleinberg. Online linear optimization and adaptive routing. Journal of Computer and System Sciences , 74(1):97-114, 2008.
- [10] Baruch Awerbuch and Robert D Kleinberg. Adaptive routing with end-to-end feedback: Distributed learning and geometric approaches. In Proceedings of the thirty-sixth annual ACM symposium on Theory of computing , pages 45-53, 2004.
- [11] Peter Bartlett, Varsha Dani, Thomas Hayes, Sham Kakade, Alexander Rakhlin, and Ambuj Tewari. High-probability regret bounds for bandit online linear optimization. In Proceedings of the 21st Annual Conference on Learning Theory-COLT 2008 , pages 335-342. Omnipress, 2008.
- [12] Daniel Beaglehole, Max Hopkins, Daniel Kane, Sihan Liu, and Shachar Lovett. Sampling equilibria: Fast no-regret learning in structured games. In Proceedings of the 2023 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA) , pages 3817-3855. SIAM, 2023.
- [13] Soheil Behnezhad, Sina Dehghani, Mahsa Derakhshan, MohammadTaghi HajiAghayi, and Saeed Seddighin. Faster and simpler algorithm for optimal strategies of blotto game. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 31, 2017.
- [14] Soheil Behnezhad, Avrim Blum, Mahsa Derakhshan, MohammadTaghi HajiAghayi, Mohammad Mahdian, Christos H Papadimitriou, Ronald L Rivest, Saeed Seddighin, and Philip B Stark. From battlefields to elections: Winning strategies of blotto and auditing games. In Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms , pages 2291-2310. SIAM, 2018.
- [15] Soheil Behnezhad, Avrim Blum, Mahsa Derakhshan, MohammadTaghi Hajiaghayi, Christos H Papadimitriou, and Saeed Seddighin. Optimal strategies of blotto games: Beyond convexity. In Proceedings of the 2019 ACM Conference on Economics and Computation , pages 597-616, 2019.
- [16] David Blackwell. An analog of the minimax theorem for vector payoffs. 1956.
- [17] Enric Boix-Adserà, Benjamin L Edelman, and Siddhartha Jayanti. The multiplayer colonel blotto game. In Proceedings of the 21st ACM Conference on Economics and Computation , pages 47-48, 2020.
- [18] Emile Borel. The theory of play and integral equations with skew symmetric kernels. Econometrica: journal of the Econometric Society , pages 97-100, 1953.
- [19] Gábor Braun and Sebastian Pokutta. An efficient high-probability algorithm for linear bandits. arXiv preprint arXiv:1610.02072 , 2016.

- [20] EO Brigham. The fast fourier transform and its applications, 1988.
- [21] George W Brown. Iterative solution of games by fictitious play. Act. Anal. Prod Allocation , 13 (1):374, 1951.
- [22] Sébastien Bubeck, Nicolo Cesa-Bianchi, and Sham M Kakade. Towards minimax policies for online linear optimization with bandit feedback. In Conference on Learning Theory , pages 41-1. JMLR Workshop and Conference Proceedings, 2012.
- [23] Nicolo Cesa-Bianchi and Gábor Lugosi. Prediction, learning, and games . Cambridge university press, 2006.
- [24] Nicolo Cesa-Bianchi and Gábor Lugosi. Combinatorial bandits. Journal of Computer and System Sciences , 78(5):1404-1422, 2012.
- [25] Xi Chen and Binghui Peng. Hedging in games: Faster convergence of external and swap regrets. In Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual , 2020.
- [26] Richard Combes, Mohammad Sadegh Talebi Mazraeh Shahi, Alexandre Proutiere, et al. Combinatorial bandits revisited. Advances in neural information processing systems , 28, 2015.
- [27] Leello Dadi, Ioannis Panageas, Stratis Skoulakis, Luca Viano, and Volkan Cevher. Polynomial convergence of bandit no-regret dynamics in congestion games. arXiv preprint arXiv:2401.09628 , 2024.
- [28] Varsha Dani, Sham M Kakade, and Thomas Hayes. The price of bandit information for online optimization. Advances in Neural Information Processing Systems , 20, 2007.
- [29] Constantinos Daskalakis, Alan Deckelbaum, and Anthony Kim. Near-optimal no-regret algorithms for zero-sum games. In Dana Randall, editor, Proceedings of the Twenty-Second Annual ACM-SIAM Symposium on Discrete Algorithms, SODA 2011, San Francisco, California, USA, January 23-25, 2011 , pages 235-254. SIAM, 2011.
- [30] Constantinos Daskalakis, Maxwell Fishelson, and Noah Golowich. Near-optimal no-regret learning in general games. Advances in Neural Information Processing Systems , 34:2760427616, 2021.
- [31] Zhiyuan Fan, Christian Kroer, and Gabriele Farina. On the optimality of dilated entropy and lower bounds for online learning in extensive-form games. Advances in Neural Information Processing Systems , 37:88373-88409, 2024.
- [32] Gabriele Farina, Chung-Wei Lee, Haipeng Luo, and Christian Kroer. Kernelized multiplicative weights for 0/1-polyhedral games: Bridging the gap between learning in extensive-form and normal-form games. In International Conference on Machine Learning , pages 6337-6357. PMLR, 2022.
- [33] Wouter Fokkema, Ruben Hoeksma, and Marc Uetz. Price of anarchy for graphic matroid congestion games. In International Symposium on Algorithmic Game Theory , pages 371-388. Springer, 2024.
- [34] Yoav Freund and Robert E Schapire. Adaptive game playing using multiplicative weights. Games and Economic Behavior , 29(1-2):79-103, 1999.
- [35] András György, Tamás Linder, Gábor Lugosi, and György Ottucsák. The on-line shortest path problem under partial monitoring. Journal of Machine Learning Research , 8(10), 2007.
- [36] Nicole Immorlica, Adam Tauman Kalai, Brendan Lucier, Ankur Moitra, Andrew Postlewaite, and Moshe Tennenholtz. Dueling algorithms. In Proceedings of the forty-third annual ACM symposium on Theory of computing , pages 215-224, 2011.
- [37] Adam Kalai and Santosh Vempala. Efficient algorithms for online decision problems. Journal of Computer and System Sciences , 71(3):291-307, 2005.

- [38] Satyen Kale, Lev Reyzin, and Robert E Schapire. Non-stochastic bandit slate problems. Advances in Neural Information Processing Systems , 23, 2010.
- [39] Terry Koo, Amir Globerson, Xavier Carreras Pérez, and Michael Collins. Structured prediction models via the matrix-tree theorem. In Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP-CoNLL) , pages 141-150, 2007.
- [40] Harold W Kuhn. Extensive games and the problem of information. Contributions to the Theory of Games , 2(28):193-216, 1953.
- [41] Tor Lattimore and Csaba Szepesvári. Bandit algorithms . Cambridge University Press, 2020.
- [42] Chung-Wei Lee, Haipeng Luo, Chen-Yu Wei, and Mengxiao Zhang. Bias no more: highprobability data-dependent regret bounds for adversarial bandits and mdps. Advances in neural information processing systems , 33:15522-15533, 2020.
- [43] Vincent Leon and S Rasoul Etesami. Bandit learning for dynamic colonel blotto game with a budget constraint. In 2021 60th IEEE Conference on Decision and Control (CDC) , pages 3818-3823. IEEE, 2021.
- [44] László Lovász and Santosh Vempala. The geometry of logconcave functions and sampling algorithms. Random Structures &amp; Algorithms , 30(3):307-358, 2007.
- [45] Russell Lyons and Yuval Peres. Probability on trees and networks , volume 42. Cambridge University Press, 2017.
- [46] Arnab Maiti, Zhiyuan Fan, Kevin Jamieson, Lillian J Ratliff, and Gabriele Farina. Efficient near-optimal algorithm for online shortest paths in directed acyclic graphs with bandit feedback against adaptive adversaries. In Conference on Learning Theory (COLT) , 2025.
- [47] R Kipp Martin. Using separation algorithms to generate mixed integer model reformulations. Operations Research Letters , 10(3):119-128, 1991.
- [48] Gergely Neu. Explore no more: Improved high-probability regret bounds for non-stochastic bandits. Advances in Neural Information Processing Systems , 28, 2015.
- [49] Ioannis Panageas, Stratis Skoulakis, Luca Viano, Xiao Wang, and Volkan Cevher. Semi bandit dynamics in congestion games: Convergence to nash equilibrium and no-regret guarantees. In International Conference on Machine Learning , pages 26904-26930. PMLR, 2023.
- [50] Sasha Rakhlin and Karthik Sridharan. Optimization, learning, and games with predictable sequences. Advances in Neural Information Processing Systems , 26, 2013.
- [51] Julia Robinson. An iterative method of solving a game. Annals of mathematics , 54(2):296-301, 1951.
- [52] Robert W Rosenthal. A class of games possessing pure-strategy nash equilibria. International Journal of Game Theory , 2:65-67, 1973.
- [53] Shinsaku Sakaue, Masakazu Ishihata, and Shin-ichi Minato. Efficient bandit combinatorial optimization algorithm with zero-suppressed binary decision diagrams. In International Conference on Artificial Intelligence and Statistics , pages 585-594. PMLR, 2018.
- [54] Shai Shalev-Shwartz et al. Online learning and online convex optimization. Foundations and Trends® in Machine Learning , 4(2):107-194, 2012.
- [55] Ashkan Soleymani, Georgios Piliouras, and Gabriele Farina. Faster rates for no-regret learning in general games via cautious optimism. In Proceedings of the 57th Annual ACM Symposium on Theory of Computing , pages 518-529, 2025.
- [56] Peter Stange, Andreas Griewank, and Matthias Bollhöfer. On the efficient update of rectangular lu factorizations subject to low rank modifications. 2006.

- [57] Vasilis Syrgkanis, Alekh Agarwal, Haipeng Luo, and Robert E. Schapire. Fast convergence of regularized learning in games. In Corinna Cortes, Neil D. Lawrence, Daniel D. Lee, Masashi Sugiyama, and Roman Garnett, editors, Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems 2015, December 7-12, 2015, Montreal, Quebec, Canada , pages 2989-2997, 2015.
- [58] Eiji Takimoto and Manfred K Warmuth. Path kernels and multiplicative updates. The Journal of Machine Learning Research , 4:773-818, 2003.
- [59] William Thomas Tutte. Graph theory , volume 21. Cambridge university press, 2001.
- [60] Dong Quan Vu, Patrick Loiseau, and Alonso Silva. Combinatorial bandits for sequential learning in colonel blotto games. In 2019 IEEE 58th conference on decision and control (CDC) , pages 867-872. IEEE, 2019.
- [61] Dong Quan Vu, Patrick Loiseau, Alonso Silva, and Long Tran-Thanh. Path planning problems with side observations-when colonels play hide-and-seek. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 34, pages 2252-2259, 2020.
- [62] Dong Quan Vu, Kimon Antonakopoulos, and Panayotis Mertikopoulos. Fast routing under uncertainty: Adaptive learning in congestion games with exponential weights. In NeurIPS '21: Proceedings of the 35th International Conference on Neural Information Processing Systems , 2021.
- [63] Dong Quan Vu, Kimon Antonakopoulos, and Panayotis Mertikopoulos. Routing in an uncertain world: Adaptivity, efficiency, and equilibrium. https://arxiv.org/abs/2201.02985, January 2022.
- [64] Renato Werneck, Joao Setubal, and Arlindo da Conceicao. Finding minimum congestion spanning trees. Journal of Experimental Algorithmics (JEA) , 5:11-es, 2000.
- [65] Julian Zimmert and Tor Lattimore. Return of the bias: Almost minimax optimal high probability bounds for adversarial linear bandits. In Conference on Learning Theory , pages 3285-3312. PMLR, 2022.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We have provided a summary of the main focus of the paper and the main results in the abstract and intro.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Our rates of convergence to CCE reveal the dependence of the parameters of interest which are not claimed to be tight.

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

Justification: The proofs of our claims can be found in the appendix due to space limitations.

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

Justification: Our work deals with computing coarse correlated equilibrium in various game settings and is theoretical. As a result, there is no social impact.

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

Justification: The paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

## Answer: [NA]

Justification: The core method development in this research does not involve LLMs.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Contents

| 1 Introduction                                                        | 1 Introduction                                                                                                                                                       |   1 |
|-----------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
| 2 Preliminaries                                                       | 2 Preliminaries                                                                                                                                                      |   4 |
| 3 Kernelized Payoff-based Learning in Polyhedral Games                | 3 Kernelized Payoff-based Learning in Polyhedral Games                                                                                                               |   6 |
| 3.1                                                                   | Kernelized GEOMETRICHEDGE for Bandit No-Regret Learning . . . . . . . . .                                                                                            |   6 |
| 3.2                                                                   | The Semi-Bandit Feedback Case: Kernelizing Implicit Exploration . . . . . . . .                                                                                      |   7 |
| 4 Efficient Kernelization in Colonel Blotto Games (CBGs)              | 4 Efficient Kernelization in Colonel Blotto Games (CBGs)                                                                                                             |   8 |
| 5 Efficient Kernelization in Graphic Matroid Congestion Games (GMCGs) | 5 Efficient Kernelization in Graphic Matroid Congestion Games (GMCGs)                                                                                                |   9 |
| 6 Conclusion                                                          | 6 Conclusion                                                                                                                                                         |  10 |
| A Extended Related Work                                               | A Extended Related Work                                                                                                                                              |  24 |
| B Semi-bandit No-Regret Learning: Analysis of Algorithm               | 2                                                                                                                                                                    |  25 |
| C Second Moment Calculation via Kernelization                         | C Second Moment Calculation via Kernelization                                                                                                                        |  28 |
| D Layered Graph Representation in Colonel Blotto                      | D Layered Graph Representation in Colonel Blotto                                                                                                                     |  29 |
| E Barycentric Spanners                                                | E Barycentric Spanners                                                                                                                                               |  29 |
| E.1                                                                   | Computing an Approximate-Barycentric Spanner for Colonel Blotto Games . . .                                                                                          |  29 |
| E.2                                                                   | Computing an Approximate-Barycentric Spanner for Graphic Matroid Congestion Games . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    |  30 |
| F Bandit No-Regret Learning: Analysis of Algorithm 1                  | F Bandit No-Regret Learning: Analysis of Algorithm 1                                                                                                                 |  31 |
| G Kernelization in Colonel Blotto games                               | G Kernelization in Colonel Blotto games                                                                                                                              |  36 |
| G.1                                                                   | Proof of Proposition 4.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                               |  37 |
| G.2                                                                   | Proof of Lemma 4.3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                 |  38 |
| G.3                                                                   | Alternative Proof of Lemma 4.3 . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                   |  38 |
| G.4                                                                   | Proof of Theorem 4.4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                 |  42 |
| G.5                                                                   | Similar Techniques for Efficient Implementation of Kernelized GEOMETRICHEGDE in m-sets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |  43 |
|                                                                       | G.5.1 Comparison with a DAG approach . . . . . . . . . . . . . . . . . . . . .                                                                                       |  46 |
| H Kernelization in Graphic Matroid Congestion Games                   | H Kernelization in Graphic Matroid Congestion Games                                                                                                                  |  47 |
| H.1                                                                   | Proof of Proposition H.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                 |  47 |
| H.2                                                                   | Proof of Lemma 5.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                 |  48 |
| H.3                                                                   | Proof of Theorem 5.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                 |  49 |
| Kernelization in Network Congestion Games                             | Kernelization in Network Congestion Games                                                                                                                            |  50 |

- J Efficient Uniform Random Path Sampling from a DAG 50

## A Extended Related Work

Online learning in games. The connection between no-regret learning and the computation of approximate CCE in games has been well-known since the work of Freund and Schapire [34] (see also [23]); assuming that all players use no-regret learning algorithms with regret O ( √ T ) , the time- averaged history of joint-play consists a O ( 1 √ T ) -CCE. In [29], it was shown for two player zero-sum √

games that rate of convergence ˜ O (1 /T ) can be achieved, improving the standard O ( 1 / T ) that can be derived from black-box regret analysis. Further improvements using the idea of optimism were shown in [50] and for general-sum games appeared in later works [57; 25; 30; 6] (see also references therein as there is a vast literature in learning in games and it is impossible to cite properly all works).

Colonel Blotto games. The classical Colonel Blotto game introduced by Borel [18] dates back to 1953. Some notable works about computing NE in two-player zero-sum games include [4; 14; 15], and for learning CCE in multi-player games include [12; 43]. Moreover, [60; 61] show no-expectedregret learning in Colonel Blotto games which does not suffice for convergence to CCE. Our work improves upon previous works the runtime time to learn a CCE (see also Table 1).

Congestion games. Congestion games are potential games [52] and always admit a pure Nash Equilibrium (NE); i.e, a state in which no agent has an incentive to unilaterally deviate. In fullinformation feedback, a long line of research studies the convergence properties to NE of game dynamics (e.g. best/better response play or no-regret). The seminal work of Takimoto and Warmuth [58], which studies online shortest paths, provides an efficient learning algorithm for network congestion games. Regarding the semi-bandit and bandit feedback settings, György et al. [35], based on [58], provide efficient algorithms for online shortest paths, which can also be applied to network congestion games. Moreover, efficient algorithms based on online gradient descent have also been established [49; 27]. The regret rate and per-iteration complexity, i.e., the total running time to reach a CCE of the aforementioned works is inferior to ours (see also Table 3). Regarding learning on spanning trees, Koo et al. [39] used the Matrix-Tree Theorem in the context of directed spanning trees for calculating the normalization factor of exponentiated gradient algorithm. However, their approach does not provide exact sampling, nor kernelization of bandit estimators.

## Algorithm 2: Kernelized Algorithm based on IX under semi-bandit feedback

```
Data: d , m , η > 0 and γ ∈ [0 , 1] 1 Initialize c 0 ( j ) = 0 and C 0 ( j ) = 0 for all j ∈ [ d ] , p 0 = [1 /N,. . . , 1 /N ] ∈ ∆( V i ) 2 for t = 1 , ..., T do 3 Compute the kernels: K V ( C t -1 , 1 ) and { K V ( C t -1 , ¯ e j ) } for j ∈ [ d ] 4 Sample an action v t ∼ p t (MWU) using v t = SAMPLING ( V , C t -1 ) 5 Observe semi-bandit losses ℓ t ∈ R d 6 Compute the unconditional probabilities: x t = ( 1 -K V ( C t -1 , ¯ e 1 ) K V ( C t -1 , 1 ) , . . . , 1 -K V ( C t -1 , ¯ e d ) K V ( C t -1 , 1 ) ) 7 Compute the IX loss estimators: ˜ ℓ t ( j ) = ℓ t ( j ) x t ( j )+ γ ✶ { v t ( j ) = 1 } , ∀ j ∈ [ d ] 8 Update the aggregated loss estimators: c t ( j ) = c t -1 ( j ) + ˜ ℓ t ( j ) , ∀ j ∈ [ d ] 9 Update the exponential cumulative loss estimators: C t ( j ) = exp ( -ηc t ( j )) , ∀ j ∈ [ d ] 10 11 Procedure: SAMPLING 12 Input: V , C 13 Sample v [1] ∼ Be ( 1 -K V ( C, ¯ e 1 ) K V ( C, 1 ) ) 14 for j = 2 , ..., d do 15 Compute the kernel: K V ( j ) 16 Set V ( j ) = { v ′ ∈ V : v ′ [ i ] = v [ i ] , ∀ i ∈ [ j -1] } and p j = 1 -K V ( j ) ( C, ¯ e j ) K V ( j ) ( C, 1 ) 17 Sample v [ j ] ∼ Be ( p j ) 18 Return: v
```

## B Semi-bandit No-Regret Learning: Analysis of Algorithm 2

Lemma B.1 (Corollary 1, [48]) . Let γ t = γ ≥ 0 for all t . With probability at least 1 -δ ′ ,

<!-- formula-not-decoded -->

simultaneously holds for all i ∈ [ d ] .

Theorem B.2 (Theorem 3.4 restated) . For any δ ∈ (0 , 1) , the sequence v 1 , ..., v T of actions played by Algorithm 2 with γ = m √ dT and η = 1 √ dT satisfies

<!-- formula-not-decoded -->

with probability at least 1 -δ .

Proof. Let ˆ L t ( v ) = ∑ i ∈V ˆ ℓ t ( i ) be the loss estimator of selecting pure action v at time step t and L t ( v ) = ∑ i ∈V ℓ t ( i ) the corresponding true loss. Moreover, let ˆ C t ( v ) = ∑ t t ′ =1 ˆ L t ′ ( v ) be the cumulative loss estimator of selecting pure action v for the first t time steps. Also, let w t ( v ) = exp( -η ˆ C t ( v )) , where η is the learning rate of MWU, and let W t = ∑ v ∈V w t ( v ) , with W 0 = |V| = N .

As in the standard analysis of MWU, we will upper and lower bound the quantity log W T W 0 . First, we fix a pure action v ∗ ∈ V and using the fact that W t ( v ) ≥ w t ( v ∗ ) , for all t , we have the following lower bound:

<!-- formula-not-decoded -->

On the other hand, assuming η ˆ L t ( v ) ≤ 1 for all A ∈ A (this condition will be verified later), using the elementary inequalities e x ≤ 1 + x + x 2 for | x | ≤ 1 and ln(1 + y ) ≤ y for y &gt; -1 , we get the standard following upper bound:

<!-- formula-not-decoded -->

Now, we will upper bound the first term and lower bound the second term of 7, as follows:

<!-- formula-not-decoded -->

where in 9 we have used the property that the arithmetic mean is less or equal than the quadratic mean, in 13 we have used the fact that ℓ t ( i ) ≤ 1 and we defined q t ( i ) = x t ( i ) x t ( i )+ γ , and in 14 we have the fact that q t ( i ) ≤ 1 .

Similarly, we lower bound the second term as follows:

<!-- formula-not-decoded -->

Now, summing for t = 1 , 2 , ..., T , we get:

<!-- formula-not-decoded -->

Combining the above with the lower bound of 4, we get the following:

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

Then, using Lemma B.1, with probability at least 1 -δ ′ , we get the following:

<!-- formula-not-decoded -->

Now, we can apply Lemma B.1 to the term ∑ i ∈ [ d ] ∑ T t =1 ˆ ℓ t ( i ) . Then with probability at least 1 -2 δ ′ we obtain:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in the last inequality we have used the fact that N ≤ ∑ m i =0 ( d i ) ≤ md m ⇒ log N ≤ 2 m log d and that ∑ i ∈ v t ℓ t ( i ) ≤ m , which holds because | v t | 1 ≤ m .

Next we will optimize over parameters γ and η to minimize the RHS in 27. We have one constraint over the parameters. In the proof of 7 we used the condition η ˆ L t ( v t ) ≤ 1 for all t ∈ [ T ] . It is easy to verify that if ηm ≤ γ then the above condition is satisfied. We set γ = m √ dT and η = 1 √ dT to balance the dominating terms in the regret bound. Moreover, we set δ = 2 δ ′ . Therefore, using the above and plug them in 27, we get the following:

<!-- formula-not-decoded -->

Finally, we obtain our result by setting v ∗ = arg min v ∈V ∑ T t =1 L t ( v ) .

## C Second Moment Calculation via Kernelization

Theorem C.1 (Theorem 3.1 restated) . Let Σ t ( q t ) := ∑ v ∈V q t ( v ) vv T be the autocorrelation matrix under the law of q t . Then, for all j, j ′ ∈ [ d ] , it holds that:

<!-- formula-not-decoded -->

Proof. We observe that for all j, j ′ ∈ [ d ] , the feature map ϕ (¯ e j,j ′ ) satisfies for all v ∈ V

̸

̸

Using the fact that ϕ ( 1 ) = 1 and ϕ (¯ e j )[ v ] = j / ∈ v , we conclude that for all j, j ′ ∈ [ d ] , v ∈ V

<!-- formula-not-decoded -->

Therefore, for all j, j ′ ∈ [ d ] , we obtain for the autocorrelation matrix

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the third equation follows from (29), the fourth from Theorem 2.2, and the fifth from the definition of K ( · , · ) .

## D Layered Graph Representation in Colonel Blotto

Definition D.1 (Layered Graph [13]) . The layered graph has k +1 layers and n +1 vertices in each layer. Let v i,j denote the j -th vertex in the i -th layer ( 0 ≤ i ≤ k and 0 ≤ j ≤ n ). For any 0 ≤ i ≤ k there exists a directed edge from v i -1 ,j to v i,l iff 0 ≤ j ≤ l ≤ n .

Lemma D.2 (Pure actions in a Layered Graph [13]) . Each directed path in the layered graph starting from v 0 , 0 and ending at v n,k is equivalent to exactly one pure action of the Colonel Blotto game, and vice versa. For each pure action, the reward for each battlefield is associated with a unique edge of the directed path.

However, the layered graph, which has been used to succinctly represent the action space for learning in Colonel Blotto games, see [60; 61; 43], implies a representation complexity of Θ( n 2 k ) that can be a bottleneck for efficient no-regret learning and convergence to CCE.

## E Barycentric Spanners

Before proceeding with the proposed algorithm for the bandit setting, we introduce the important notion of barycentric spanners [10]. We will use barycentric spanners to ensure adequate exploration of each coordinate j ∈ [ d ] , sufficient to guarantee low variance of the loss estimators.

Definition E.1. A subset of independent vectors { b 1 , . . . , b d } ⊆ V is said to be C -approximate barycentric spanner of V i , with C &gt; 1 , if, for all v ∈ V , there exists α ∈ R d such that

<!-- formula-not-decoded -->

We define B to be the matrix whose columns are the barycentric spanners { b 1 , . . . , b d } .

The following proposition ensures that, if specific conditions hold, there exists an efficient algorithm for computing a C -approximate barycentric spanner.

Proposition E.2 (Proposition 2.5, [9]) . Suppose S ⊆ R d is a compact set not contained in any proper linear subspace. Given an oracle for optimizing linear functions over S, for any C &gt; 1 there exists an algorithm that computes a C -approximate barycentric spanner for S in polynomial time, using O ( d 2 log C ( d ) ) calls to the optimization oracle.

## E.1 Computing an Approximate-Barycentric Spanner for Colonel Blotto Games

Proposition E.3 (Oracle for finding best-response in polynomial-time) . Given a reward vector r , the following linear optimization problem

<!-- formula-not-decoded -->

can be solved in time O ( n 2 k ) .

Proof. We will solve the following linear optimization problem (which corresponds to playing best-response with respect to the reward vector r ):

<!-- formula-not-decoded -->

The above problem is equivalent to the problem of finding the longest path from a directed weighted DAG (with | V | nodes and | E | edges), which can be solved via Dynamic Programming in time | V | · | E | . To do so, we leverage the Layered Graph representation (see Section D), which is a DAG with Θ( nk ) nodes and Θ( n 2 k ) edges. More specifically, in the Layered Graph, in layer h ∈ [ k ] the edge e h = ( u h,i , u h +1 ,j ) for i ≤ j and i, j ∈ [ n ] 0 corresponds to assigning j -i soldiers on battlefield h +1 . On each edge, we use as edge weight the battlefield reward taken by assigning the corresponding number of soldiers on the corresponding battlefield, and the longest path of this graph,

denoted by x ∗ ∈ R n 2 k , represents the best response with respect to r . Thus, we can solve the linear optimization problem in time O ( n 2 k ) .

The only thing left to do is to get V ∗ = arg max V ∈V i r T V from x ∗ . It is straightforward that there exists an one-to-one correspondence between these two vectors. To get V ∗ , one must do the following:

- Initialize V ∗ = 0 ∈ R d .
- For each layer h ∈ [ k ] , given the selected edge e h = ( u h,i , u h +1 ,j ) in x ∗ , assign V ∗ [ h, j -i ] = 1 .

Lemma E.4 (Polynomial-time algorithm for C -barycentric spanner in Colonel Blotto) . In Colonel Blotto, for C &gt; 1 , there exists a polynomial-time algorithm that computes a C -approximate barycentric spanner for V i in time O ( n 4 k 3 log C ( nk )) .

Proof. To prove Lemma E.4, first observe that V i satisfies Proposition E.2 because it is compact and is not contained in any proper linear subspace of R d . Now, we need to have access to an oracle for optimizing linear functions over V i . To do so, we utilize the oracle for finding a best response given a reward vector from Proposition E.3.

Therefore, each oracle call needs time O ( n 2 k ) . Now, using Proposition E.2, to compute the barycentric spanner for V i , one may use the algorithm defined in [9] that computes an approximate C -spanner (with C &gt; 1 ) in time O ( n 4 k 3 log C ( nk )) .

## E.2 Computing an Approximate-Barycentric Spanner for Graphic Matroid Congestion Games

The idea is similar to above but using the Kruskal algorithm as the oracle. To compute a C -barycentric spanner here we need O ( | E | 2 log C ( | E | ) .

## F Bandit No-Regret Learning: Analysis of Algorithm 1

We have the following:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where B ∈ R d × d is a full rank matrix that has the approximate spanners as columns. In 40, we have used the fact that B is invertible because it is full rank, and also that Σ t is non-singular (see [28]) which implies that Σ -1 t exists and thus the pseudo-inverse matrix Σ + t equals the inverse matrix, i.e., Σ + t = Σ -1 t . Moreover, in 42, we used the Weyl's inequality.

Let C t be the autocorrelation matrix under the law of the exploration distribution on the barycentric spanner. We aim to bound the minimum non-zero eigenvalue of C t . Similarly to above, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let E t [ · ] denote expectation conditioned on the past events; i.e. the realized rewards received and the actions taken by player i up to time step t -1 . Also, let 1 be the ones vector. We define L t ( v ) = ℓ t · v , and similarly ̂ L t ( v ) = ̂ ℓ t · v . In the following analysis, we drop the superscript i and sometimes write q t and p t for the distributions of player i . For now, we assume that T ≥ 8 d 2 m , an assumption that will be verified later by the average regret guarantee for the convergence to CCE.

Using the above analysis, along with the basic lemma of [11], we can easily get the following lemma with the basic properties of the algorithm.

Lemma F.1. For any v ∈ V i and t ∈ [ T ] , the following hold:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, using Lemma F.1 and selecting η = γ d 2 mC 2 = 1 d 4 / 3 m 2 / 3 C 2 T 1 / 3 , we have

<!-- formula-not-decoded -->

Lemma F.2 (Bernstein's inequality for martingales) . Let Y 1 , ..., Y T be a martingale difference sequence. Suppose that Y t ∈ [ a, b ] and

<!-- formula-not-decoded -->

for all t ∈ { 1 , . . . , T } . Then for all ε &gt; 0 ,

<!-- formula-not-decoded -->

Lemma F.3. Simultaneously for any v ∈ V i , with probability at least 1 -δ , it holds that

<!-- formula-not-decoded -->

Proof. Fix any v ∈ V i , we define Y t ( v ) = ̂ L t ( v ) -L t ( v ) . Y t is a martingale difference sequence. Using Lemma F.1, the following hold:

<!-- formula-not-decoded -->

where in 53 we used the Cauchy-Schwarz inequality and in 55 we used Lemma F.1.

<!-- formula-not-decoded -->

Now by applying the Bernstein's inequality (Lemma F.2), with probability at least 1 -δ |V i | we obtain

<!-- formula-not-decoded -->

Taking the union bound, we obtain the desired result.

Lemma F.4. With probability at least 1 -δ ,

<!-- formula-not-decoded -->

Proof. Using Lemma F.3, with probability at least 1 -δ , we have, simultaneously for all v ∈ B ,

<!-- formula-not-decoded -->

Summing over the d elements of the spanner, and using the fact that L t ( v ) ≤ m , we get the result of the statement.

Lemma F.5. With probability at least 1 -δ ,

<!-- formula-not-decoded -->

Proof. The proof follows directly from the proof of Lemma 6 in [11], using | Y t | ≤ d 2 mC 2 γ + m , and Var t Y t ≤ m √ d + m .

Lemma F.6. With probability at least 1 -δ ,

<!-- formula-not-decoded -->

Proof. The proof directly follows the proof of Lemma 8 from [11], by using that the summands v T t Σ -1 t v t are bounded by d 2 C 2 γ .

Theorem F.7 (Theorem 3.2 restated) . For T ≥ 8 d 2 m and for any δ ∈ (0 , 1) , the sequence v 1 , . . . , v T of actions played by Algorithm 2 with γ = d 2 / 3 m 1 / 3 T 1 / 3 and η = 1 4 d 4 / 3 m 2 / 3 T 1 / 3 satisfies

<!-- formula-not-decoded -->

Proof. Following the standard analysis of MWU (also similar to our analysis in the semi-bandit setting), we have that,

<!-- formula-not-decoded -->

since by definition of p t ,

Fix any v ∗ ∈ V i . We have that,

<!-- formula-not-decoded -->

where in 66 we used Lemma F.3, and in 67 we used the fact that γ = d 2 / 3 m 1 / 3 T 1 / 3 . Putting these together, using Lemmas F.4, F.5 and F.6 in 64, we have

<!-- formula-not-decoded -->

Taking logs, using the fact that ln(1 + x ) ≤ x , and also the fact that η 1 -γ ≤ 2 η , because we have assumed that T ≥ 8 d 2 m , and summing over t , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in 70 we used the definitions of γ and η .

Finally, using 67 and 70, rearranging terms, dividing with η , using the fact that ln d/η = d 4 / 3 m 2 / 3 T 1 / 3 ln d , and rescaling δ = 4 δ , with probability at least 1 -δ , simultaneously for all u ∗ ∈ V i , we have that,

<!-- formula-not-decoded -->

## G Kernelization in Colonel Blotto games

Algorithm 3: Efficient First-Moment Kernel Computations in Colonel Blotto games

<!-- formula-not-decoded -->

Algorithm 4 Efficient Second-Moment Kernel Computations in Colonel Blotto games

## Require: C ( t )

```
1: # Compute the interval products 2: for h, h ′ in [ k +1] 0 × [ k +1] 0 do 3: P int [ h, h ′ ] = 1 4: for h = 1 , ..., k -1 do 5: P int [ h, h ]( z ) = n ∑ s =0 C ( t ) h,s · z s 6: for h ′ = h, ..., k -1 do 7: P ( z ) = P int [ h, h ′ ]( z ) · n ∑ s =0 C ( t ) h ′ +1 ,s · z s 8: P int [ h, h ′ +1]( z ) = truncate P ( z ) to degree n 9: # P int [ h, h ′ ]( z ) = h ′ ∏ i = h n ∑ s =0 C ( t ) i,s · z s 10: # Compute the d 2 kernels 11: for h = 1 , ..., k do 12: # case 1: h ′ = h 13: P -h ( z ) = P int [1 , h -1]( z ) · P int [ h +1 , k ]( z ) 14: n ∑ s =0 α s · z s = truncate P -h to degree n 15: for s = 0 , ..., n do 16: for s ′ = 0 , ..., n do 17: K V ( C ( t ) , ¯ e h,h,s,s ′ ) = K V ( C ( t ) , 1 ) -α n -s · C ( t ) h,s -α n -s ′ · C ( t ) h,s ′ · ✶ { s = s ′ } 18: # case 2: h ′ > h 19: if h < k then 20: for h ′ = h +1 , ..., k do 21: P h ( z ) = n ∑ s =0 C ( t ) h,s · z s 22: for s = 0 , ..., n do 23: P h,s ( z ) = P h ( z ) -C ( t ) h,s · z s 24: P -h,h ′ ( z ) = P int [1 , h -1]( z ) · P int [ h +1 , h ′ -1]( z ) · P int [ h ′ +1 , k ]( z ) 25: P -h ′ ( z ) = P -h,h ′ ( z ) · P h,s ( z ) 26: n ∑ s =0 α s · z s = truncate P -h ′ to degree n 27: for s ′ = 0 , ..., n do 28: K V ( C ( t ) , ¯ e h,h ′ ,s,s ′ ) = K V ( C ( t ) , 1 ) -α n -s ′ · C ( t ) h ′ ,s ′
```

## G.1 Proof of Proposition 4.2

Proposition G.1. For given x, y ∈ { 0 , 1 } d , there exists an algorithm that computes the kernel K ( x, y ) in time O ( nk log n ) .

Proof. To compute the n -th coefficient of 3, we execute a running product over the factors of the polynomial. This process involves k updates of the partial product. After each update, the partial product is truncated down to degree n . Thus, inductively we ensure that all k multiplications involve polynomials of degree at most n . Each multiplication can be implemented with FFT [20] in O ( n log n ) time. The overall complexity over the k multiplications is O ( nk log n ) and after the truncated product is computed the target coefficient is obtained in O (1) .

̸

## G.2 Proof of Lemma 4.3

The proof is based on Algorithm 3. We define the running product P ( t ) l [ i ] from left to right, which is the sum of the degree 0 to n terms of the polynomial i ∏ i ′ =1 n ∑ j =0 C t [ i ′ , j ] · z j . We can compute all polynomials P ( t ) l [ i ] , for i = 1 , ..., k in total time nk log n using the following induction argument:

Given P ( t ) l [ i ] , we compute P ( t ) l [ i + 1] by performing the polynomial multiplication P ( t ) l [ i ]( z ) · n ∑ j =0 C t [ i +1 , j ] · z j and truncating all terms of degree greater than n . The two multiplied polynomials have degree n , so the multiplication can be done in time n log n using FFT, while the truncation of the higher degree terms can be done in time n since the product polynomial has degree 2 n . Repeating this procedure for i = 1 , ..., k -1 we get all left-to-right partial products in total time nk log n .

( t )

Similarly, we define the running product P r [ i ] from right to left as the sum of the degree 0 to n terms of the polynomial k ∏ i ′ = k -i +1 n ∑ j =0 C t [ i ′ , j ] · z j . Similarly to P ( t ) l , we can compute all right-to-left partial products P ( t ) r [ i ] , for i = 1 , ..., k in total time nk log n .

Now, using the above partial products, we compute all the kernels required for (O)MWU at time step t . All polynomial multiplications in the Algorithm are performed using FFT so that each of them takes time n log( n ) .

Following similar logic as above, via Algorithm 4 we get the desired result.

## G.3 Alternative Proof of Lemma 4.3

Efficient sampling of MWU in CBGs has been studied in Beaglehole et al. [12]. A useful tool for this purpose is the partition function defined in equation 100. Here we describe their method with details and we extend their ideas to the efficient calculation of first and second order moments of the MWU distribution.

Remark G.2. We give an algorithm (Algorithm 4) that performs the second moment computation in terms of kernels. The algorithm follows a similar logic to Algorithm 3, but is somewhat more complicated, due to the nature of the problem. In steps where polynomial multiplication is performed, we imply that the multiplication is implemented efficiently through FFT.

We remind that the calculation of first order moments was used in our method for learning in the semi-bandit setting and the second moments appear in the calculation of the autocorrelation matrix which is used in the bandit setting. Next we proceed to the technical details of our methods.

Focusing on a single player, at time step t , let ℓ t [ h, s ] be the loss observed by the player when assigning s soldiers to the h -th battlefield, given the assignments of the other players in this battlefield. Moreover, let

<!-- formula-not-decoded -->

We define the partition function

<!-- formula-not-decoded -->

We also define the partition function g h ( y ) , which is similar to f h ( y ) but aggregates battlefields in the reverse order.

<!-- formula-not-decoded -->

Let L ( t ) ( x 1 , ..., x k ) be the cumulative loss at timestep t . L ( t ) can be decomposed into the cumulative losses per battlefield as follows:

<!-- formula-not-decoded -->

Under MWU the probability of some assignment x 1 , ..., x k at timestep t can be written as

<!-- formula-not-decoded -->

Marginal probabilities of soldier assignments at a single battlefield can be written as follows:

<!-- formula-not-decoded -->

Moreover, we can compute the conditional probability of each soldier assignment at a single battlefield, given a set of soldier assignments at other battlefields:

<!-- formula-not-decoded -->

Similarly, in terms of the partition function g h ( y ) , we derive

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The conditional probabilities can be used to implement an efficient sampling procedure for the MWU distribution, as was proposed in [12]. For completeness we write the algorithm below.

## Algorithm 5 Sampling from the MWU distibution in Colonel Blotto games

Require: Soldiers n ≥ 0 , battlefields k ≥ 1 and cumulative loss c ( t ) h ( s ) for h, s ∈ [ k ] × [ n ] 0 1: f 0 ( s ) = 1 for all s ∈ [ n ] 0

- 2: for h = 1 , ..., k -1 do
- 3: Using FFT, calculate the convolution ( a ∗ b )( s ) where a ( s ) = exp ( -ηc ( t ) h ( s ) ) and b ( s ) = f h -1 ( s ) , s ∈ [ n ] 0 .
- 4: ∀ s ∈ [ n ] 0 , calculate the partition function for battlefield h :

<!-- formula-not-decoded -->

- 5: Sample the number s k of soldiers at the last battlefield:

<!-- formula-not-decoded -->

- 6: for h = 1 , ..., k -1 do
- 7: Sample the number s k -h of soldiers at battlefield k -h given the numbers of soldiers, s k -h +1 , . . . , s k , assigned to battlefields k -h +1 , . . . , k as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark G.3. Algorithm 5 implements the SAMPLING procedure of Algorithms 1 and 2. The key point is that instead of explicitly calculating the required kernels, it directly computes the conditional probabilities via a partition function, which corresponds to kernelizing the conditional polytope.

The unconditional marginals that constitute the first moment are calculated as follows:

̸

<!-- formula-not-decoded -->

̸

We can precompute the partition functions f h ( y ) , g h ( y ) , for all h ∈ [ k ] and y ∈ [ n ] 0 in total time kn log n utilizing the self reducible structure of the partition function (see algorithm 6 lines 1-5 for details). Then we compute f h -1 ∗ g h +1 for all h ∈ [ k ] and y ∈ [ n ] 0 in total time kn log n with FFT. Using these calculations each term Pr[ s h = s ] computation takes constant time. Note that this method is essentially equivalent to the kernel method we describe in the main paper (Algorithm 3).

For the calculation of the second-order marginals that constitute the second moment, we will make use of the interval partition function f h,h ′ ( y ) , that aggregates possible assignments between the h and the h ′ battlefields.

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

̸

̸

<!-- formula-not-decoded -->

We observe that the marginal probabilities only depend on the interval partition function and the cumulative loss per battlefield. We can precompute the partition function f h,h ′ ( y ) , for all h, h ′ ∈ [ k ] 2 : h ≤ h ′ and y ∈ [ n ] 0 in total time nk 2 log n utilizing the self reducible structure of the partition function (see Algorithm 5). Then we compute the convolutions ( f 1 ,h -1 ∗ f h +1 ,h ′ -1 ∗ f h ′ +1 ,k )( y ) for all h, h ′ ∈ [ k ] 2 : h &lt; h ′ and y ∈ [ n ] 0 in total time nk 2 log n with FFT. Using these calculations each term Pr[ s h , s h ′ = s, s ′ ] computation takes constant time.

## G.4 Proof of Theorem 4.4

Proof. Using Lemma 4.3 and the exact sampling procedure provided in [12] (see Algorithm 5), based on which we can calculate the required kernels of our SAMPLING procedure in time O ( nk log n ) , the per-iteration complexity for the bandit and semi-bandit algorithms is O ( n ω k ω log n ) and O ( nk log n ) , respectively. By combining Theorems 3.2 and 3.4 with Theorem 2.1, we can achieve the desired results.

## G.5 Similar Techniques for Efficient Implementation of Kernelized GEOMETRICHEGDE in m-sets

Algorithm 6 Sampling the MWU distribution in m-sets

Require: Soldiers n ≥ 0 , battlefields d ≥ 1 and cumulative loss c h · b for h, s ∈ [ d ] × [ n ] 0

- 1: f 0 ( y ) = 1 for all y ∈ [ m ] 0
- 2: for h = 1 , ..., d -1 do
- 3: ∀ y ∈ [ m ] 0 , calculate the partition function for item h :

<!-- formula-not-decoded -->

- 4: Sample the selection of the last item:

<!-- formula-not-decoded -->

- 5: for h = 1 , ..., d -1 do
- 6: Sample the selection v d -h of item d -h given the selections, v d -h +1 , . . . , v d , of items d -h +1 , . . . , d as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Summary: We can apply similar techniques to efficiently compute the second moment used in Algorithm 1 for the classic m-sets setting. In particular, our approach requires time ˜ O ( md 2 ) , improving upon the DAG formulation approach of [24; 58] which requires time O ( m 2 d 2 ) .

A classic setting in combinatorial bandits which is also considered in [32] are m-sets, where actions are selections of m out of d items. Its binary representation the action set can be written as V = { v ∈ { 0 , 1 } d | ∑ i v i = m } .

We will show how to perform efficient exact sampling and autocorrelation matrix calculation in m-sets. For this purpose we will use the partition function defined in equation 100, similarly to Blotto. The partition function resembles the kernels used in kernelized MWU. Next we proceed to the technical details of our methods.

At time step t , let ℓ t [ i ] be the loss observed by the player when selecting the i -th item. Moreover, let

<!-- formula-not-decoded -->

be the cumulative loss of the i -th item over the first t time steps. We define the partition function

<!-- formula-not-decoded -->

where x i ∈ { 0 , 1 } , h ∈ [ d ] and y ∈ [ m ] .

We also define the partition function g h ( y ) , which is similar to f h ( y ) but aggregates the set items in the reverse order.

<!-- formula-not-decoded -->

( t )

Let L ( t ) ( x 1 , ..., x d ) be the cumulative loss at timestep t . L ( t ) can be decomposed into the cumulative losses per item as follows:

<!-- formula-not-decoded -->

Under MWU the probability of some assignment x 1 , ..., x d at timestep t can be written as

<!-- formula-not-decoded -->

Marginal probabilities over assignments can be written as follows:

<!-- formula-not-decoded -->

Conditional probabilities over assignments are calculated as follows:

<!-- formula-not-decoded -->

Similarly we derive

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The conditional probabilities can be used to implement an efficient sampling procedure for the MWU distribution. We write the algorithm below.

For the calculation of second-order marginals we will make use of the interval partition function f h,h ′ ( y ) , that aggregates possible assignments between the h and the h ′ battlefields.

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

We observe that the marginal probabilities only depend on the interval partition function and the cumulative loss per battlefield. We can precompute the partition function f h,h ′ ( y ) , for all h, h ′ ∈ [ d ] 2 : h ≤ h ′ and y ∈ [ n ] 0 in total time md 2 log m utilizing the self reducible structure of the partition function (see algorithm 6 lines 1-5 for details). Then we compute the convolutions ( f 1 ,h -1 ∗ f h +1 ,h ′ -1 ∗ f h ′ +1 ,d )( y ) for all h, h ′ ∈ [ d ] 2 : h &lt; h ′ and y ∈ [ n ] 0 in total time md 2 log m with FFT. Using these calculations, each term Pr[ v h , v h ′ = b, b ′ ] computation takes constant time.

We remind that the autocorrelation matrix, which is used to construct the loss estimator in GEOMETRICHEGDE, has the probabilities Pr[ v h , v h ′ = b, b ′ ] as entries. At this point we have shown how to efficiently sample from MWU and how to compute the autocorrelation matrix and thus that GEOMETRICHEGDE can be efficiently implemented.

## G.5.1 Comparison with a DAG approach

One can easily see that online learning in m-sets can be modeled as online path planning in an appropriately constructed DAG with E = O ( d ∗ m ) edges. In this graph, nodes are parameterized by two indices i, j ∈ [ d +1] × [ m ] . The source is node N (0 , 0) and the sink is N ( d +1 , m ) . At node N ( i, j ) , 1 ≤ i ≤ d we have considered items 1 to i -1 and we have selected j of them. If we select item i we make a transition from N ( i, j ) to N ( i +1 , j +1) , otherwise we make a transition to N ( i +1 , j ) . Transitions that lead to selecting more than m items are illegal and at the sink node N ( d +1 , m ) we should have selected exactly m items. This way, there is an equivalence between paths in the constructed DAG and selections of m out of d items and in both cases the reward is linear to the components.

For m-sets over d items the corresponding DAG has | E | = Θ( md ) edges. Then, sampling can be performed through weight pushing [58] in O ( E ) = O ( md ) , which is similar to complexity of sampling via the partition function. For the calculation of the autocorrelation matrix the approach of path planning in the m-set DAG would need O ( E 2 ) = O ( m 2 d 2 ) using the techniques of [58]. Compared to the above our approach saves an m factor. Thus, along with partitions (which is the action set in Blotto) m-sets is another application, where kernelization is beneficial compared to standard techniques such as weight pushing.

## H Kernelization in Graphic Matroid Congestion Games

Algorithm 7: Efficient First-Moment Kernel Computations in Graphic Matroid Congestion games

```
Data: C ∈ R E 1 / ∗ Compute the weighted Laplacian A ∈ R | V |×| V | ∗ / 2 A [ u, v ] =    ∑ e ∈ E incident to u C ( e ) if u = v -C ( e ) if e = ( u, v ) ∈ E 0 otherwise. 3 / ∗ Compute the LU decompositions of the submatrices A -u, -u ∗ / 4 for u = 1 , ..., | V | do 5 A -u, -u = the submatrix of A derived by deleting row u and column u 6 Compute the LU decomposition ( L u , U u ) of A -u, -u , that is lower triangular L u and upper triangular U u such that A -u, -u = L u · U u 7 / ∗ Compute the d kernels ∗ / 8 for j = 1 , ..., | E | do 9 Let u j , v j be the two nodes connected by edge j 10 Let E -j = E \ { j } be the subgraph that does not have edge j 11 / ∗ Compute the weighted Laplacian A ( j ) in the subgraph where edge j is missing ∗ / 12 A ( j ) [ u, v ] =      ∑ e ∈ E -j incident to u C ( e ) if u = v -C ( e ) if e = ( u, v ) ∈ E -j 0 otherwise. 13 A ( j ) -u j , -u j = the submatrix of A ( j ) derived by deleting row u j and column u j 14 Compute the LU decomposition ( L , U ) of A ( j ) -u j , -u j in O ( | V | 2 ) using the precomputed matrices L u j , U u j and the technique of [56] 15 Compute the kernel K V ( C, ¯ e j ) = det( L · U ) in O ( | V | 2 )
```

## H.1 Proof of Proposition H.1

Let ¯ G = ( ¯ V , ¯ E ) be a connected multigraph, and let ˆ G = ( ˆ V , ˆ E ) be the meta-graph associated with ¯ G , defined as follows:

1. The vertex sets coincide:

<!-- formula-not-decoded -->

2. For an edge e = ( u, v ) ∈ ¯ E with weight ¯ w ( e ) :
- If e is the unique edge between u and v , then e ∈ ˆ E with the same weight:

<!-- formula-not-decoded -->

- Otherwise, let { e ′ } ⊂ ¯ E be all parallel edges connecting u and v . Then, we define a single meta-edge ˆ e ∈ ˆ E with weight equal to the total weight of the merged edges:

<!-- formula-not-decoded -->

We say that ˆ e is a merged meta-edge , and write e ′ ⊂ ˆ e if the edge e ′ ∈ ¯ E participates in the construction of ˆ e ∈ ˆ E .

3. Let ¯ T denote the set of spanning trees of ¯ G , and let ˆ T denote the set of spanning trees of ˆ G .

We derive the following proposition.

Proposition H.1. It holds that K ˆ V ( ˆ w, 1 ) = K ¯ V ( ¯ w, 1 ) .

Proof. It holds that:

<!-- formula-not-decoded -->

## H.2 Proof of Lemma 5.1

## Proof. Kernelization.

The above algorithm (Algorithm 7) shows how to compute the first-moment kernel computations in Graphic Matroid Congestion games. The time complexity and correctness of the algorithm are discussed below.

We will use the following:

- We need O ( | V | ω ) time for computing an LU decomposition.
- We need O ( | V | ω +1 ) time for precomputing the LU decompositions of the minors.

We leverage the property of the Matrix-Tree Theorem which allows us to use any submatrix to compute the determinant of the Laplacian matrix. Therefore, we can always make a strategic choice of which row and column to delete. For each edge j ∈ [ | E | ] consider the Laplacian used for the computation of the kernel K V ( C, ¯ e j ) . The Laplacian for this kernel is constructed in the same way as the one we described above for K V ( C, 1 ) but with the difference that C ( j ) is set to zero. For each node v ∈ V , we precompute the LU decomposition of the minors A -v, -v -that is the submatrix of A derived by deleting row v and column v -and then for each j = ( u, u ′ ) ∈ E , we fast compute kernel K ( C, ¯ e j ) by computing the determinant of that kernel's Laplacian via recursive LU updating [56] in O ( | V | 2 ) . The latter is due to the fact that we can always select a submatrix of the kernel's Laplacian that only differs in one element from A -u, -u , so we can apply the techniques of [56]. Similar arguments can also be used for fast computing K V ( C, ¯ e j,j ′ ) to derive the desired results.

## Per-Iteration complexity of SAMPLING.

We implement the SAMPLING procedure of Algorithms 1 and 2 based on the above algorithm (Algorithm 8). Since we have guaranteed that the SAMPLING procedure performs exact sampling from a MWU( V , C ), what remains to prove is that the implementation we propose correctly computes the conditional kernels. We will prove this using an induction argument on the iterations j ∈ [ | E | ] of the algorithm. We will show that the algorithm correctly computes the new Bernoulli probability p j +1 .

- Basis : The meta-graph is initialized as the initial graph. From Theorem 2.2 and Observation 3.3, we get the unconditional probability p 1 . The algorithm samples v (1) ∼ p 1 . If the first edge ( u, v ) is not selected then the algorithm removes it from the new meta-graph and computes K V (2) via the cofactor of the Laplacian of the new meta-graph. If the first edge is selected then: (a) if the first edge is not a merging meta-edge (that is, nodes u

```
Algorithm 8: Efficient Exact Sampling of MWU in Graphic Matroids Data: C ∈ R | E | 1 Initialize Meta-Graph = G ( V, E ) and assign weight C ( e ) to each edge e 2 Compute kernels K V ( C, ¯ e 1 ) and K V ( C, 1 ) using the Matrix-Tree Theorem 3 Sample v (1) ∼ Be ( 1 -K V ( C, ¯ e 1 ) K V ( C, 1 ) ) 4 Initialize the cumulative weight w = 1 5 for j = 2 , ..., d do 6 if v ( j -1) = 0 then 7 Find the meta-edge of Meta-Graph containing edge j -1 and reduce its weight by C ( j -1) 8 else 9 Find the meta-edge e of Meta-Graph containing edge j -1 10 Update w = w · weight ( e ) 11 Merge the two meta-nodes connected by the meta-edge e . 12 If parallel edges are created then merge them into a single meta-edge containing all the parallel edges and assign to the new meta edge weight equal to the sum of the weights of the parallel edges 13 / ∗ Compute the kernel K V ( j ) ∗ / 14 Compute a cofactor c of the Laplacian of the Meta-Graph 15 K V ( j ) ( C, 1 ) = w · c 16 Find the meta-edge of Meta-Graph containing the edge j and reduce its weight by C ( j ) 17 Compute a cofactor c ′ of the Meta-Graph Laplacian using the weights of the meta-edges 18 K V ( j ) ( C, ¯ e j ) = w · c ′ 19 Find the meta-edge of Meta-Graph containing the edge j and increase its weight by C ( j ) 20 p j = 1 -K V ( j ) ( C, ¯ e j ) K V ( j ) ( C, 1 ) 21 Sample v ( j ) ∼ Be ( p j )
```

and v do not have common neighbors in the meta-graph) then the algorithm removes this edge from the graph, merges the two associated nodes of this edge in the new meta-graph, updates the cumulative weight with the weight of this edge, and computes K V (2) , (b) if the first edge is a merging meta-edge (i.e., nodes u and v do have common neighbors in the meta-graph), then the algorithm makes the above steps, but now the meta-graph is a multi-graph. In this case, the algorithm also merges the resulted parallel edges connecting the associated nodes into a meta-edge with weight equal with the sum the weights of the merged edges. The computation of K V (2) is correct, due to Proposition H.1, because the kernel computation on a multi-graph (that is, the meta-graph after merging the nodes u and v , but before merging the resulted parallel edges) equals the kernel computation on the corresponding new meta-graph.

- Induction Step : We use similar arguments with the basis, with the only difference when removing an edge. Now, if edge j is not selected by the Bernoulli distribution p j but j is part of a merged meta-edge (i.e., a meta-edge consisting of many edges of the initial graph), then the algorithm removes its weight from this meta-edge and computes the new meta-graph. Again the computation of K V ( j +1) is correct due to Proposition H.1.

Therefore, SAMPLING ( V , C t ) can be implemented in time O ( | E || V | ω ) , where | V | ω is due to the time we need to compute a single kernel.

## H.3 Proof of Theorem 5.2

Proof. We directly derive the statement of the theorem by combining Theorems 3.2 and 3.4 with Theorem 2.1 and Lemma 5.1.

## I Kernelization in Network Congestion Games

We consider the setting used in [49; 27]; that is, the network congestion game takes place on a DAG, consisting of nodes V and edges E , and thus an action of each player is a path of a DAG. We assume that the maximal path length is K . Following [24], we represent an action of each player i ∈ [ |P| ] , as the incidence vector v ∈ { 0 , 1 } | E | of the corresponding path: for all j ∈ [ | E | ] , v ( j ) = 1 if and only if the corresponding edge is present in the path. We denote the action set (i.e., a set of path vectors) of player i ∈ [ |P| ] by V i . Given an action profile ( v i , v -i ) the total loss of player i , L i , is the sum of the losses of the selected edges of v i . Based on the above, it is easy to check that a network congestion game is a combinatorial game with |P| players, actions sets {V i } and losses {L i } , where the action vectors are | E | -dimensional and their L 1 -norm is at most K .

To perform efficient sampling in DAGs and compute all kernels needed by Algorithms 2 and 1, we utilize the methodology based on DP developed by [58]. For sampling we need time O ( | E | ) . For the first moment calculation we need time O ( | E | ) , while for the second moment calculation we need time O ( | E | 2 ) . Using also the fact that we can compute a 2-approximate barycentric spanner in time ˜ O (( | E | + | V | ) 3 ) , we obtain the following CCE convergence results.

Theorem I.1 (Semi-bandit Convergence to CCE) . In a network congestion game, under the semibandit online learning setup, if all players adopt Algorithm 2, then after ˜ O ( | E | 1+ ω K 2 /ϵ 2 ) runtime, with T ≥ | E | K 2 /ε 2 , the time-average joint actions, σ ∗ := 1 T ∑ T t =1 v ( t ) 1 ⊗ · · · ⊗ v ( t ) |P| , forms an ε -CCE of the game with high probability.

Theorem I.2 (Bandit Convergence to CCE) . In a network congestion game, under the bandit online learning setup, if all players adopt Algorithm 1, then after ˜ O ( | E | 2+ ω K 4 /ϵ 3 ) runtime, with T ≥ | E | 2 K 4 /ε 3 , the time-average joint actions, σ ∗ := 1 T ∑ T t =1 v ( t ) 1 ⊗··· ⊗ v ( t ) |P| , forms an ε -CCE of the game with high probability.

## J Efficient Uniform Random Path Sampling from a DAG

In this section, we describe a method for efficiently and exactly sampling paths uniformly at random from a Directed Acyclic Graph (DAG). This process is essential for the initialization phase of MWU. We present the algorithm's pseudocode and analyze its computational complexity as well as its correctness.

̸

```
Algorithm 9 Uniform Random Path Sampling from a DAG Require: A DAG G = ( V, E ) , source node s , target node t Ensure: A uniformly random path P from s to t 1: / ∗ Path Count Precomputation: ∗ / 2: Perform a topological sort of the nodes in G . 3: Set C ( v ) ← 0 for all v ∈ V , and C ( t ) ← 1 . 4: for each node v in reverse topological order do 5: C ( v ) ← ∑ ( v,u ) ∈ E C ( u ) 6: / ∗ Path Sampling: ∗ / 7: Initialize P ← [ s ] and set v ← s . 8: while v = t do 9: Calculate probabilities P ( u ) ← C ( u ) ∑ ( v,w ) ∈ E C ( w ) for all ( v, u ) ∈ E . 10: Select the next node u based on probabilities P ( u ) . 11: Add u to P and update v ← u . 12: return P
```

Computationally Complexity. The precomputation step requires O ( V + E ) for the topological sort and another O ( V + E ) for calculating the path counts. Therefore, the overall complexity of the precomputation step is O ( V + E ) . During the sampling phase, there are at most O ( V ) iterations, one for each node in the path. Computing transition probabilities takes O (deg( v )) for each node v , leading to a total of O ( E ) operations. Thus, the complexity of the sampling phase is O ( E ) .

Combining both steps, the total complexity of the algorithm is O ( V + E ) .

Correctness Proof. To demonstrate correctness, we prove that every path P from s to t is selected with equal probability.

The dynamic programming step calculates C ( v ) , the number of paths from node v to t . Using the recurrence relation:

<!-- formula-not-decoded -->

we ensure that C ( s ) represents the total number of paths from s to t , and C ( v ) indicates the number of paths passing through v . At each node v , the transition probability to a neighboring node u is:

<!-- formula-not-decoded -->

For any path P = s → v 1 → v 2 → ··· → t in G the probability of selecting it is given by the product of transition probabilities:

<!-- formula-not-decoded -->

By substituting P ( u | v ) = C ( u ) C ( v ) , we get:

<!-- formula-not-decoded -->

Since C ( s ) equals the total number of paths from s to t , every path is selected with an equal probability of 1 C ( s ) .