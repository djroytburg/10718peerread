## Precise Asymptotics and Refined Regret of Variance-Aware UCB

## Yingying Fan

University of Southern California fanyingy@marshall.usc.edu

## Jinchi Lv

## Yuxuan Han

New York University yh6061@stern.nyu.edu

## Xiaocong Xu

University of Southern California jinchilv@marshall.usc.edu

University of Southern California xuxiaoco@marshall.usc.edu

## Zhengyuan Zhou

New York University zzhou@stern.nyu.edu

## Abstract

In this paper, we study the behavior of the Upper Confidence Bound-Variance (UCB-V) algorithm for the Multi-Armed Bandit (MAB) problems, a variant of the canonical Upper Confidence Bound (UCB) algorithm that incorporates variance estimates into its decision-making process. More precisely, we provide an asymptotic characterization of the arm-pulling rates for UCB-V, extending recent results for the canonical UCB in [21] and [23]. In an interesting contrast to the canonical UCB, our analysis reveals that the behavior of UCB-V can exhibit instability, meaning that the arm-pulling rates may not always be asymptotically deterministic. Besides the asymptotic characterization, we also provide non-asymptotic bounds for the arm-pulling rates in the high probability regime, offering insights into the regret analysis. As an application of this high probability result, we establish that UCB-V can achieve a more refined regret bound, previously unknown even for more complicate and advanced variance-aware online decision-making algorithms. A matching regret lower bound is also established, demonstrating the optimality of our result.

## 1 Introduction

The Multi-Armed Bandit (MAB) problem is a fundamental framework capturing the explorationexploitation trade-off in sequential decision-making. Over decades, it has been rigorously studied and widely applied across fields like dynamic pricing, clinical trials, and online advertising [32, 35, 27].

In the classic K -armed MAB problem, a learner is faced with K arms, each associated with an unknown reward distribution P a for a ∈ [ K ] , supported on [0 , 1] with mean µ a and variance σ 2 a . At each time step t ∈ [ T ] , the learner selects an arm a t and observes a reward X t drawn independently from P a t . The learner's goal is to maximize the cumulative reward by striking an optimal balance between exploration (sampling less-known arms to improve estimates) and exploitation (selecting arms with high estimated rewards). This objective is commonly framed as a regret minimization problem, where the regret over a time horizon T is defined as Reg( T ) ≡ ∑ T t =1 ( µ a ⋆ -µ a t ) with a ⋆ ≡ arg max a ∈ [ K ] µ a the optimal arm with the highest expected reward.

Table 1: Summary of stability and regret results for UCB-V under different σ 1 , σ 2 regimes.

| Regime                         | Stability   | Regret Upper Bound            | Regret Lower Bound         |
|--------------------------------|-------------|-------------------------------|----------------------------|
| σ 1 = o ( σ 2 )                | No ∗        | ˜ O ( σ 2 √ T )               | ˜ Ω( σ 2 √ T )             |
| σ 1 = Ω( σ 2 ) σ 2 = o ( σ 1 ) | Yes         | ˜ O (( σ 2 /σ 1 ) · σ 2 √ T ) | Ω(( σ 2 /σ 1 ) · σ 2 √ T ) |

∗ There exists an instance such that the arm pulling rate is unstable.

The minimax-optimal regret for the K -armed bandit problems is known to be Θ( √ KT ) up to logarithmic factors. This rate is achievable by several well-established algorithms, including the Upper Confidence Bound (UCB) [5, 4], Thompson Sampling [1, 32], and Successive Elimination [13], among others. Beyond the regret minimization, increasing attention has been devoted to analyzing finer properties and largeT behaviors of several typical algorithms, including the regret tail distributions [15, 16], diffusion approximations [14, 21, 2, 24], and arm-pulling rates [21, 23]. Among these works, [21] and [23] introduced the concept of stability for the canonical UCB algorithm, enabling the precise characterization of arm-pulling behaviors and facilitating statistical inference for adaptively collected data, a task traditionally considered challenging for the general bandit algorithms.

In more structured settings, sharper regret bounds are achievable, particularly when arm variances are small. For instance, if all arms have zero variance (i.e., deterministic rewards), a single pull of each arm is sufficient to identify the optimal one. This observation has motivated active research into developing variance-aware algorithms for bandit problems [5, 3, 4, 20, 28, 40, 41, 33]. Among these algorithms, the Upper Confidence Bound-Variance (UCB-V) algorithm [3, 4], detailed in Algorithm 1, adapts the classic UCB algorithm by incorporating variance estimates of each arm.

While regret minimization for variance-aware algorithms has been well-studied, their precise armpulling behavior remains less explored. The main challenge here is that incorporating variance information introduces a new quantity, in addition to the reward gaps ∆ a ≡ µ a ⋆ -µ a , a ∈ [ K ] , that influences the arm-pulling rates. In fact, the behavior of the algorithm can differ significantly depending on whether variance information is incorporated or not. In Figures 1 and 2b, we compare the empirical arm-pulling rates between UCB-V and the canonical UCB in a two-armed setting. Compared to the canonical UCB, its variance-aware version exhibits significantly greater fluctuations in arm-pulling numbers as the reward gap changes, and its arm-pulling distribution is more heavytailed. This highlights the significant differences in the variance-aware setting and introduces additional challenges.

## Algorithm 1 UCB-V Algorithm

- 1: Input: Arm number K, time horizon T , and exploration coefficient ρ .
- 2: Pull each of the K arms once in the first K iterations.
- 3: for t = K +1 to T do
- 4: Compute arm pulls for arm a up to time t by n a,t ≡ ∑ s ∈ [ t -1] 1 { A s = a } , a ∈ [ K ] .
- 5: Compute the empirical means and variances

<!-- formula-not-decoded -->

- 6: Compute the optimistic rewards UCB( a, t ) ≡ ¯ X a,t + ( ̂ σ a,t √ ρ log T ∨ 1 √ n a,t ) ρ log T √ n a,t for a ∈ [ K ] .
- 7: Choose arm A t given by A t = arg max a ∈ [ K ] UCB( a, t ) .
- 8: end for

## 1.1 Our Contributions

In this work, we try to close this gap by presenting a precise asymptotic analysis of UCB-V. As in [21] and [23], our main focus is on the arm-pulling numbers and stability. For clarity, we concentrate on the two-armed bandit setting, specifically K = 2 , in the main part of our paper, as it effectively captures the core exploration-exploitation trade-off while allowing a precise exposition of results [31, 18, 22, 21]. The extension to the K -armed case is discussed in Appendix B.

Figure 1: The distributions of n 1 ,T for UCB-V and UCB with T = 50 , 000 over 5000 repetitions. (a) σ 1 = σ 2 = 1 / 4 , ∆ 2 = 0 (b) σ 1 = 0 , σ 2 = 1 / 4 , ∆ 2 = σ 2 √ log T/T . A more detailed numerical setting is provided in Appendix H.

<!-- image -->

More precisely, by providing a general concentration result, we establish both precise asymptotic characterizations and high-probability bounds for the arm-pulling numbers of UCB-V. When σ 1 , σ 2 = Ω(1) , our results provide a straightforward generalization of those obtained for the canonical UCB. In contrast, when σ 1 ≪ σ 2 , we show that, unlike UCB, the stability result may not hold for UCB-V and that it reveals a phase transition phenomenon in the optimal arm-pulling numbers, as illustrated in Figure 2. Finally, as an application of our sharp characterization of arm-pulling rates, we present a refined regret result for UCB-V that is previously unknown for any other variance-aware decisionmaking algorithms. We summarize our contributions in more detail below (see also Table 1). For notational simplicity, we assume that the optimal arm is a ⋆ = 1 and denote by ∆ = ∆ 2 .

Precise asymptotic behavior for UCB-V. For fixed values of ∆ , σ 1 , σ 2 , and T , we introduce a pair of deterministic equations whose unique solution { n ⋆ a,T ≡ n ⋆ a,T (∆ , σ 1 , σ 2 , T ) } a =1 , 2 predicts the arm-pulling numbers of UCB-V. We show that for possibly T -dependent (∆ , σ 1 , σ 2 ) ,

<!-- formula-not-decoded -->

except when σ 1 ≪ σ 2 and σ 2 √ ρ log T/ ( √ T ∆) = 1 hold simultaneously; see Theorem 1 for a precise statement. This phenomenon, referred to as the stability of arm-pulling numbers, is formally defined in Definition 1. Consequently, the solutions of these deterministic equations can be used to predict the behavior of UCB-V in the largeT regime, as illustrated in Figure 2. Our results reveal several notable differences between UCB-V and the canonical UCB, as analyzed in [21] and [23], due to the incorporation of variance information:

- When σ 1 , σ 2 = Ω(1) , the optimal arm will always be pulled a linear number of times, with n 1 ,T ∼ σ 2 1 σ 2 1 + σ 2 2 T as ∆ → 0 . This indicates that, in the small-gap regime, the UCB-V algorithm tends to allocate more pulls to the arm with higher reward variance, generalizing the result for the canonical UCB in which both arms are pulled T/ 2 times in the small-∆ regime.
- When σ 1 = o ( σ 2 ) , UCB-V may pull the optimal arm sublinearly in the small-∆ regime. In contrast, the canonical UCB pulls the optimal arm linearly in T , as illustrated in Figure 2b. This behavior is governed by the ratio Λ T ≡ σ 2 √ ρ log T/ ( √ T ∆) . More precisely, given σ 1 ≤ √ ρ log T/T and σ 2 = Ω(1) ,
1. When lim T →∞ Λ T &lt; 1 , we have n 1 ,T /n ⋆ 1 ,T p - → 1 with n ⋆ 1 ,T = Ω( T ) .
2. When lim T →∞ Λ T &gt; 1 , we have n 1 ,T /n ⋆ 1 ,T p - → 1 with n ⋆ 1 ,T = O ( √ T/σ 2 ) .
3. When lim T →∞ Λ T = 1 , there exists a bandit instance where, for large T , P ( n 1 ,T ≳ T/ √ log T ) ∧ P ( n 1 ,T ≲ √ T log T ) ≳ 1 .

The above results completely characterize the behavior of n 1 ,T in the prescribed regime. Notably, we establish a phase transition in the optimal-arm pulling times n 1 ,T , shifting from O ( √ T/σ 2 ) to Ω( T ) at the critical point Λ T = 1 . We also note that the existence of an unstable instance when Λ T ∼ 1 highlights a stark contrast between UCB-V and the

canonical UCB, where, for the latter, it has been shown that the behavior of n 1 ,T for any ∆ &gt; 0 can be asymptotically described by a deterministic sequence { n ⋆ 1 ,T } as T →∞ .

High probability bounds and confidence region for arm pulling numbers. While our asymptotic theory provides the precise limiting behavior of UCB-V, it has a drawback similar to those found in [21] and [23]: the convergence rate in probability is quite slow. Specifically, the probability that the uncontrolled event occurs decays at a rate of O ((log T ) -1 ) . This slow rate is inadequate for providing insight or guarantees in the popular high probability regime for studying bandit algorithms, where the probability of an uncontrolled event should be in the order of O ( T -1 ) . To address this gap , we demonstrate that, starting from our unified concentration result in Proposition 1, one can derive non-asymptotic bounds for the arm pulling numbers in the high probability regime. Such result provides a high probability confidence region for arm pulling numbers, as illustrated in Figure 3a in Appendix D.1.

Refined regret for variance-aware decision making. As an application of our high probability arm-pulling bounds, we demonstrate in Theorem 2 that the UCB-V algorithm achieves a refined regret bound of form

<!-- formula-not-decoded -->

This result improves upon the best-known regret 1 for UCB-V [4, 36] and surpasses the regrets for all known variance-aware bandit algorithms [41, 40, 33, 9, 10], which are of form O ( σ 2 √ T ) in the two-armed setting. Our result is the first to reveal the effect of the optimal arm's variance: When σ 1 = o ( σ 2 ) , the regret matches the previously known O ( σ 2 √ T ) bound, and the optimal arm's variance does not affect performance; as σ 1 surpasses σ 2 , the regret decreases as σ 1 increases, following the form O ( σ 2 2 σ 1 √ T ) . Simulations presented in Figure 2 confirm our theoretical predictions, demonstrating that UCB-V exhibits improved performance in highσ 1 scenarios, which was previously unknown . A matching regret lower bound is established in Theorem 3, demonstrating the optimality of our result.

Notation. For any positive integer n , denote by [ n ] = [1 : n ] set { 1 , . . . , n } . For a, b ∈ R , a ∨ b ≡ max { a, b } and a ∧ b ≡ min { a, b } . For any a ∈ R , a + ≡ max { a, 0 } . Throughout this paper, we regard T as our fundamental large parameter. For any possibly T -dependent quantities a ≡ a ( T ) and b ≡ b ( T ) , we say that a = o ( b ) or b = ω ( a ) if lim T →∞ a/b = 0 . Similarly, a = O ( b ) or b = Ω( a ) if lim T →∞ | a/b | ≤ C for some constant C . If a = O ( b ) and a = Ω( b ) hold simultaneously, we say a = Θ( b ) , or a ≍ b , and write a ∼ b in the special case when lim T →∞ a/b = 1 . We write a ≳ b (resp. a ≲ b ) to mean a ≥ Cb (resp. a ≤ Cb ) for some absolute constant C &gt; 0 . If either sequence a or b is random, we say a = o P ( b ) if a/b p - → 0 as T →∞ .

## 2 Implicit bounds for arm-pulling numbers

Additional notations. For ρ &gt; 1 and T ∈ N + , let

<!-- formula-not-decoded -->

Whenever there is no confusion, we write σ a ≡ σ a ( ρ ; T ) and ∆ a ≡ ∆ a ( ρ ; T ) . We also allow σ a and ∆ a to depend on T . For any σ ≥ 0 and n ∈ N + , define φ ( n ; σ ) ≡ σ ∨ n -1 / 2 / √ n . For each fixed σ ≥ 0 , the map φ ( · ; σ ) : R ≥ 0 → R ≥ 0 is monotone nonincreasing in n , and its (piecewise) inverse n ( · ; σ ) : R ≥ 0 → R ≥ 0 is n ( φ ; σ ) ≡ ( σ 2 ∨ φ ) /φ 2 . Consequently, studying n a,T reduces to analyzing

<!-- formula-not-decoded -->

Here the two pieces in φ match the variance-driven term ( σ/ √ n ) and the boundedness-driven term ( 1 /n ) in Bernstein's inequality ; the rescalings σ a and ∆ a place variance and gap on the same Bernstein-type exploration scale used by our UCB-V bonus.

̸

1 [4] does not directly provide the worst-case regret bound for UCB-V, and the best known worst-case regret bound for UCB-V is claimed as O ( √ KT log T ) in [28]. In Appendix E.2, we show that the O ( √∑ a = a ⋆ σ 2 a T log T ) regret for UCB-V can be derived based on results in [4].

Figure 2: (a): The regrets of UCB-V with σ 2 ≍ T -1 / 4 , ∆ 2 ≍ 1 / √ T fixed and σ 1 ≍ T -1 / 2 , T -1 / 4 , 1 , each instance with 10 repetitions. (b): The median and 30% quantile of n 1 ,T (optimal arm-pulling count) for UCB and UCB-V, under varying Λ T in the σ 1 = o ( σ 2 ) regime, with T = 1 , 000 , 000 over 30 repetitions for each ∆ 2 . The red dotted line is the predicted n 1 ,T of UCB-V using (6). A more detailed numerical setting is provided in Appendix H.

<!-- image -->

## 2.1 Implicit bounds for arm-pulling numbers

Consider Algorithm 1 with K = 2 . We start with the following concentration result for φ a,T .

Proposition 1. Recall that σ a ≡ σ a ( ρ ; T ) and ∆ a ≡ ∆ a ( ρ ; T ) for a ∈ [2] . Fix any δ &gt; 0 , ρ &gt; 1 , and any positive integer T ≥ 4 such that ε ≡ ε ( δ ; ρ ; T ) ≡ 5 ( √ 48 log(log( T/δ ) /δ ) ρ log T + 128 log(1 /δ ) ρ log T ) 1 / 2 + 200 √ T ≤ 1 2 . Then, with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof of Proposition 1 above is inspired by the delicate analysis of bonus terms for the canonical UCB [23] in the largeT regime and is presented in Appendix C.1. Proposition 1 establishes a sandwich inequality, demonstrating that as term ε ( δ ; ρ ; T ) approaches 0 , the concerned ratio converges to 1 . This can be viewed as a non-asymptotic and variance-aware extension of the results found in [21] and [23] for the canonical UCB. To incorporate variance information, we have also developed a Bernstein-type non-asymptotic law of iterated logarithm result in Lemma 8 in Appendix G, which is of independent interest. We now make several comments regarding Proposition 1:

Deriving the asymptotic equation. When selecting δ = (log T ) -1 , we find that for any ρ &gt; 0 , the ε ( δ ; ρ ; T ) term approaches 0 at a rate of O ( log log T log T ) . Under this selection, Proposition 1 yields the following probability convergence guarantee

<!-- formula-not-decoded -->

It is noteworthy that in this asymptotic result, the o P (1) term converges to 0 at a relatively slow rate, both in probability and magnitude. For each T , it has a probability of at least 1 -O ( 1 log T ) of being bounded by O ( log log T log T ) . This slow rate appears not only in our result but also in [21] and [23], necessitating a considerably large T to observe theoretical predictions clearly in experiments, as illustrated in Figure 2b.

Permissible values of ρ for high probability bounds. One popular scale selection of δ is δ ≍ 1 /T 2 , where Proposition 1 becomes a high-probability guarantee widely adopted in the pure-exploration and regret analysis literature [5, 3]. With such selection, the requirement ε ( δ ; ρ ; T ) &lt; 1 / 2 translates to ρ &gt; c ′ for some universal and sufficiently large constant c ′ . Thus, we have that with probability at

least 1 -O (1 /T 2 ) ,

<!-- formula-not-decoded -->

In particular, since the centered term is a monotone increasing function of φ 1 ,T , this result provides a high-probability confidence region for φ 1 ,T and consequently for n 1 ,T , as shown in Figure 3a in Appendix D.1. These high probability bounds of n 1 ,T enable us to derive the refined regret bound for UCB-V in Section 4.

The implications outlined above will be made precise in the subsequent two sections.

## 3 Asymptotic characterization of arm-pulling numbers

## 3.1 Stability of the asymptotic equation

Consider the deterministic equation

<!-- formula-not-decoded -->

Recall that σ a ≡ σ a / √ ρ log T and ∆ a ≡ ∆ a / ( ρ log T ) for a = 1 , 2 , so the equation actually depends on both ρ and T . For any fixed T ∈ N + and ρ &gt; 1 , Proposition 1 indicates that φ 1 ,T satisfies f ( φ 1 ,T ) = 1 + ζ , where ζ represents a perturbation term. Given the asymptotic equation derived in (1), it is reasonable to conjecture that the behavior of φ 1 ,T aligns with the solution of the above deterministic equation. To formally connect φ 1 ,T to the solution of (3), we conduct a perturbation analysis of this equation. Specifically, we have the following result.

Lemma 1. Fix any T ∈ N + and ρ &gt; 1 . It holds that:

1. The fixed-point equation (3) admits a unique solution φ ⋆ ∈ (1 /T, 1) .
2. Assume that there exist some ζ ∈ ( -1 / 2 , 1 / 2) and φ ζ such that f ( φ ζ ) = 1 + ζ . Then

<!-- formula-not-decoded -->

The proof of Lemma 1 above is provided in Appendix D.1. Intuitively, Lemma 1 asserts that in either the homogeneous variance case ( σ 2 ≍ σ 1 ) or when the ratio σ 2 2 / ( T ∆ 2 2 ) is bounded away from 1 , the solution φ ζ of the ζ -perturbed equation is provably bounded by ( 1 ±O ( ζ ) ) φ ⋆ . However, when σ 1 = o ( σ 2 ) and the ratio σ 2 2 / ( T ∆ 2 2 ) approaches 1 , the stability guarantee of the lemma breaks down. Although this result may seem pessimistic-as σ 2 2 / ( T ∆ 2 2 ) ∼ 1 causes the right-hand side of (4) to diverge-it accurately reflects the behavior of the perturbed solution. This is illustrated in Figure 3b in Appendix D.1, where instability arises when σ 2 2 / ( T ∆ 2 2 ) = 1 .

## 3.2 Asymptotic stability

One important consequence of the perturbation bound is the following asymptotic stability result.

Theorem 1. Consider Algorithm 1 with K = 2 . Assume that

<!-- formula-not-decoded -->

Then for a ∈ [2] , we have n a,T /n ⋆ a,T p - → 1 , where n ⋆ 1 ,T is the unique solution to equation

<!-- formula-not-decoded -->

and n ⋆ 2 ,T ≡ σ 2 2 ∨ ( φ ( n ⋆ 1 ,T ; σ 1 ) + ∆ 2 ) / ( φ ( n ⋆ 1 ,T ; σ 1 ) + ∆ 2 ) 2 .

The proof of Theorem 1 above is provided in Appendix D.2, relying on Proposition 1 and Lemma 1. Lemma 1 shows that the boundedness condition in (5) is essential for ensuring the stability of the arm-pulling process, as defined in Definition 1. Specifically, when σ 1 = σ 2 = 1 , condition (5) is always satisfied, and the asymptotic equation (6) reduces to the canonical UCB setting studied in [21] and [23], where stability is guaranteed. The key insight behind such stability in an even more general homogeneous setting σ 1 ≍ σ 2 is that the optimal arm's pulling number grows linearly with T , as noted in [23, Eqn. (22)]. In the inhomogeneous case where σ 1 = o ( σ 2 ) , the boundedness of | σ 2 2 / ( T ∆ 2 2 ) -1 | -1 becomes crucial for stability and appears to be novel , ensuring stability even when n 1 ,T grows sub-linearly in T . The instability result in the next subsection complements Theorem 1 by presenting a counterexample where the boundedness condition in (5) fails. An extension to the K -armed setting is discussed in Appendix B.

Asymptotic behavior of arm-pulling numbers. Let us write φ ⋆ ≡ φ ( n ⋆ 1 ,T ; σ 1 ) for simplicity. From Theorem 1, the asymptotic behavior of n 1 ,T can be derived through n ⋆ 1 ,T , which, in turn, can be fully determined by φ ⋆ , provided that the boundedness condition in (5) holds. Thus, understanding the analytical properties of φ ⋆ is sufficient to determine the asymptotic behavior of n 1 ,T .

The function f ( φ ⋆ ) is known to be monotonic increasing in φ ⋆ , and the solution to f ( φ ⋆ ) = 1 lies within the interval [0 , 1] . This allows us to efficiently compute the numerical behavior of φ ⋆ to predict both asymptotic and finite-time arm-pulling behavior, as illustrated in Figure 3a in Appendix D.1. However, due to the presence of a maximum operation and the complexity of the underlying equation, obtaining a closed-form solution for φ ⋆ remains difficult. Below, we present several extreme cases that highlight new phenomena arising from the incorporation of variance information, which differs from the classical UCB algorithm. A full analytical characterization of φ ⋆ is left for future research.

Example 1. When σ 1 , σ 2 = Ω(1) , Eqn. (6) simplifies to

<!-- formula-not-decoded -->

In the moderate gap regime ∆ 2 ∼ σ 2 √ θ log T/T for some fixed θ ∈ R ≥ 0 , we have n ⋆ 1 ,T /T = λ ⋆ ( θ ) ( 1 + o (1) ) , with λ ⋆ ( θ ) being the unique solution to equation λ + ( σ 1 σ 2 · 1 √ λ + √ θ/ρ ) -2 = 1 . Moreover, we can compute the limits lim θ → + ∞ λ ⋆ ( θ ) = 1 and lim θ → 0 λ ⋆ ( θ ) = σ 2 1 2 2 .

σ 1 + σ 2

This result recovers the asymptotic equation for the canonical UCB in [21] and [23] when σ 1 = σ 2 = 1 . The λ ⋆ equation derived here can be viewed as an extension of those in [21]. Notably, in the small-gap limit θ → 0 , the arm-pulling allocation for UCB-V becomes proportional to the variance instead of being equally divided.

Example 2. When σ 1 ≤ √ ρ log T/T,σ 2 = Ω(1) , Eqn. (6) simplifies to

<!-- formula-not-decoded -->

In the moderate gap regime where ∆ ∼ σ 2 √ θ log T/T for some fixed θ ∈ R ≥ 0 \ { ρ } , we have n ⋆ 1 ,T /T = λ ⋆ ( θ ) ( 1 + o (1) ) , with λ ⋆ ( θ ) being the unique solution to equation λ + ( 1 σ 2 √ ρ log T T · 1 λ + √ θ/ρ ) -2 = 1 . Moreover, we can compute the limits lim θ → + ∞ λ ⋆ ( θ ) = 1 and lim θ → 0 λ ⋆ ( θ ) = 2 √ 1+4 σ 2 2 T +1 .

This limit indicates a distinct behavior between UCB-V and UCB: as θ → 0 , the number of pulls for the optimal arm becomes sub-linear in T , specifically O ( √ T/σ 2 ) , while for UCB, it approaches T/ 2 . Additionally, we have a more detailed description of the transition between sub-linear and linear pulling times: (i) When θ &lt; ρ , λ ⋆ ( θ ) ≤ 1 / ( σ 2 (1 -√ θ/ρ ) √ T ) and (ii) When θ &gt; ρ , λ ⋆ ( θ ) ≥ 1 -ρ/θ . This result indicates that the transition from O ( √ T ) pulling time to Ω( T ) pulling time for the optimal arm occurs at θ = ρ .

Inference with UCB-V. Another key implication of Theorem 1 is its relevance to the post-policy inference in the UCB-V algorithm. To begin, we recall the notion of arm stability as defined in [23, Definition 2.1].

Definition 1. An arm a ∈ [ K ] is stable if there exists a deterministic sequence n ⋆ a,T such that

<!-- formula-not-decoded -->

where n ⋆ a,T may depend on T, { µ a } a ∈ [ K ] , and { σ 2 a } a ∈ [ K ] .

This notion of stability guarantees that, as T →∞ , the number of times that an arm is pulled becomes predictable, thus enabling valid statistical inference for each arm's reward distribution. Specifically, under the stability condition, the following Z -statistic converges in distribution to a standard normal

<!-- formula-not-decoded -->

This result is crucial for constructing valid hypothesis tests and confidence intervals in the adaptive sampling settings [29, 39]. For space efficiency, we provide a more detailed discussion along with simulation results in Appendix I.

## 3.3 Unstable results: closer look and hard instances

Recall that Theorem 1 establishes the stability result, except for the case when σ 1 = o ( σ 2 ) and σ 2 2 / ( T ∆ 2 2 ) = 1 hold simultaneously. In this section, we provide a hard instance in this setting and show the instability result. More precisely, let us consider the setting similar to Example 2, with σ 2 = 1 , σ 1 = 0 , and gap ∆ 2 = σ 2 √ θ log T/T .

In this scenario, the behavior of the solution to the asymptotic equation is described by n ⋆ 1 ,T ∼ λ ⋆ ( θ ) T , where λ ⋆ ( θ ) solves λ + (√ ρ log T/T/λ + √ θ/ρ ) -2 = 1 or equivalently,

<!-- formula-not-decoded -->

To heuristically explain why θ = ρ (or equivalently σ 2 2 / ( T ∆ 2 2 ) = 1 ) acts as a phase transition point and leads to instability, we formally take the first-order expansion 1 / √ 1 -λ ≈ 1 + λ/ 2 in (9) and multiply both sides by λ . This yields a quadratic equation in λ , 1 2 λ 2 +(1 -√ θ/ρ ) λ -√ ρ log T T = 0 . Notably, the order of the solution in T depends crucially on the sign of 1 -√ θ/ρ . For any ε &gt; 0 , by observing the analytical solution to the quadratic equation, we have

<!-- formula-not-decoded -->

In particular, a perturbation in θ around θ = ρ of magnitude ε with different signs can lead to a fluctuation in n ⋆ 1 ,T from O ( ε -1 √ T log T ) to Ω( εT ) .

Based on the above intuition and informal argument, we can rigorously demonstrate the instability result by constructing a Bernoulli bandit instance and establishing a time-uniform anti-concentration result for the Bernoulli reward process via Donsker's principle. Intuitively, our anti-concentration result shows that, with constant probability, θ can be perturbed by a magnitude of O ((log T ) -1 ) with different signs, which then implies instability in n 1 ,T even when we play UCB-V with the oracle information of the variance (i.e., with ̂ σ a,t = σ a ). We summarize the instability result in the following proposition and provide its proof in Appendix D.3.

Proposition 2. Consider the following two-armed Bernoulli bandit instance: for µ = 1 / 2 and ∆ 2 = µ (1 -µ ) ρ log T/T , arm 1 has reward µ +∆ and variance 0, and arm 2 has reward µ and variance µ (1 -µ ) . Assume that σ a 's are known and let n a,T be computed by Algorithm 1 with ̂ σ a,t = σ a . Then for sufficiently large T , there exist some constants c 0 , c 1 &gt; 0 such that

<!-- formula-not-decoded -->

Remark 1. In the example above we set σ 1 = 0 for simplicity. The instability analysis in Proposition 2 also extends to cases where the optimal arm has variance σ 1 = O ( T -1 / 2 ) , since in this regime the

UCB-V bonus for the optimal arm matches the σ 1 = 0 case. By contrast, constructing an instance with σ 1 = T -α for some α ∈ (0 , 1 / 2) is more delicate and appears to require new ideas: the associated asymptotic equation becomes substantially more involved, and a complete analysis is currently unclear.

̸

The existence of an unstable instance at Λ T = 1 complements our previous finding, where UCB-V's stability was shown for Λ T = 1 . This result highlights a contrast between the UCB-V and canonical UCB: as shown in [21] and [23], the canonical UCB is stable for all ∆ , which corresponds to the case in our setting when σ 1 = σ 2 = 1 . Our result demonstrates that, in a heterogeneous variance environment, UCB-V may exhibit significant fluctuations. Another implication of this instability is that, unlike the canonical UCB, the CLT for the Z -statistic may not hold for data collected by UCB-V, as shown in Figure 4b in Appendix I. This necessitates the developments of new statistical inference methods for UCB-V collected data or new variance-aware decision-making algorithms with stronger stability guarantees than UCB-V, which we leave as a valuable future direction.

## 4 Refined regret for variance-aware decision making

In this section, we show that a refined regret can be achieved by UCB-V based on our arm-pulling number bounds presented in Section 2. Previously, the best-known regret for UCB-V, shown in [4], is given by 2

<!-- formula-not-decoded -->

As mentioned earlier, the regret in (11) does not account for the effect of σ 1 , which contradicts the empirical performance of UCB-V and is conservative in the large σ 1 , small σ 2 regime, as shown in Figure 2a. On the other hand, the derived asymptotic equation for the arm-pulling times naturally indicates a dependency of n 2 ,T on σ 1 , opening the possibility for refined worst-case regret bounds. More precisely, by adopting the high-probability bound in (2), we can show the following proposition.

Proposition 3. There exists some universal constant C 0 such that with ρ ≥ C 0 and T ≥ 3 ,

<!-- formula-not-decoded -->

By applying the upper bound on the number of pulls for sub-optimal arms, we are able to derive the refined worst-case regret for UCB-V.

Theorem 2. There exists some universal constant C 0 &gt; 0 such that for ρ ≥ C 0 and T ≥ 3 , we have that with probability at least 1 -6 /T 2 ,

<!-- formula-not-decoded -->

The proof of the above results can be found in Appendix E. The regret bound in Theorem 2 above improves upon the bound in (11) for the large σ 1 , small σ 2 regime, while recovering (11) in the small σ 1 regime. We note that sharper bounds may be derived in special instances based on Proposition 3. For example, in the Bernoulli setting where µ 1 is close to 1 , the variance inequality σ 2 2 ≲ σ 2 1 +∆ 2 holds. In this case, Proposition 3 implies a regret bound of the form O ( σ 1 √ T log T ) , which matches the result established in [30].

Regret results for the K -armed setting. Since most works consider the general K -armed setting, for ease of comparison we now presenting the K -armed extension of Theorem 2, we leave the

2 Rather than presenting an equation of the form (11), [4] established the following gap-dependent upper bound on the number of times a suboptimal arm is pulled:

<!-- formula-not-decoded -->

However, it is straightforward to show (11) by combining (10) with the trivial bound Reg( T ) ≤ ∆ 2 E ( n 2 ,T ) .

rigorous statements in Theorem 5 in Appendix B. For a K -armed MAB problem with variances { σ 2 a } K a =1 (assuming again W.L.O.G. arm 1 is optimal), we have:

<!-- formula-not-decoded -->

√

The best known worst-case regret of UCB-V was previously known as O ( KT log T ) , as discussed in [28]. In comparison, our result shows that UCB-V can achieve regret adaptive to both the variances of sub-optimal and optimal arms. Beyond UCB-V, for Thompson sampling algorithms, the O ( √∑ a ∈ [2: K ] σ 2 a T log T ) regret was proved in a Bayesian setting, where each arm's reward distribution follows a known prior distribution. Especially, even when working with the Bayesian regret, the dependency on σ 1 is previously unrevealed . We believe that our results for UCB-V can be extended to these posterior sampling algorithms.

Optimality of Theorem 2. Now we establish a matching lower bound to show the optimality of Theorem 2. To describe the lower bound result, consider any given distributions P 1 and P 2 . Let v = ( P 1 , P 2 ) denote a 2-armed bandit instance, where P i represents the distribution of the i -th arm. To emphasize the dependency on the means µ i ≡ E X ∼ P i [ X ] and variances σ 2 i ≡ Var X ∼ P i [ X ] of the arms, we redundantly express this instance as v = ( P 1 ( µ 1 , σ 1 ) , P 2 ( µ 2 , σ 2 )) .

Given any instance v , we use the notation σ 2 opt ( v ) , σ 2 sub ( v ) to denote variances of optimal and suboptimal arms under v for clarity. For any policy π and any instance v , we denote E π v and P π v as the expectation and probability with respect to the reward distribution induced by π under v . More precisely, we establish the following regret lower bounds. The proof is deferred to Appendix E.3.

Theorem 3. Given any sufficiently large T &gt; 0 and 0 &lt; σ 2 &lt; 1 / 3 , consider the following class of problems with σ -bounded variances:

<!-- formula-not-decoded -->

Then for every policy π , there exists some v ∈ V such that,

√

<!-- formula-not-decoded -->

Moreover, the following trade-off lower bound holds: Given any 0 &lt; β &lt; 1 / 2 and c T = O ( poly (log T )) , consider

<!-- formula-not-decoded -->

and the good policy class

<!-- formula-not-decoded -->

Then for any π ∈ Π good , there exists some v ′ ∈ V ′ β such that

<!-- formula-not-decoded -->

Theorem 3 states that, in the general scenario with σ -bounded variances, no algorithm can achieve a regret better than σ sub ( v ) √ T . This reveals the optimality of Theorem 2 in the regime where σ opt ( v ) ≤ σ sub ( v ) . Furthermore, we show that in σ opt &gt; σ sub regime, any reasonably good algorithm that performs nearly optimal in worst-case over V (i.e., the algorithms lie within Π good ) cannot achieve a regret better than σ 2 sub σ opt √ T . In particular, the regret upper bound in Theorem 2 demonstrates that UCB-V matches such trade-off lower bound in the regime where σ opt &gt; σ sub , illustrating its optimality.

Conclusion. In this paper, we provide a refined analysis of the UCB-V algorithm, including a precise characterization of its asymptotic arm-pulling behavior and high-probability, non-asymptotic bounds that lead to a sharper and optimal regret upper bound. Several valuable future directions remain open. First, the instability result is established only under the regime σ 1 = o ( √ log T/T ) and σ 2 = Ω(1) , while the stability condition in the more general case σ 1 = o ( σ 2 ) is not yet sharply understood. Second, as discussed in Section B, our stability condition in the K -armed setting requires a uniform-type separation condition min a ¯ ∆ a ≥ ¯ σ a √ ( K -1) /T , which we believe can be further refined.

Acknowledgments. This work is generously supported by NSF CCF-2312205, ONR 13983263, and 2026 New York University Center for Global Economy and Business grant.

## References

- [1] Shipra Agrawal and Navin Goyal. Analysis of thompson sampling for the multi-armed bandit problem. In Conference on Learning Theory , pages 39-1. JMLR Workshop and Conference Proceedings, 2012.
- [2] Victor F Araman and René A Caldentey. Diffusion approximations for a class of sequential experimentation problems. Management Science , 68(8):5958-5979, 2022.
- [3] Jean-Yves Audibert, Rémi Munos, and Csaba Szepesvári. Tuning bandit algorithms in stochastic environments. In International Conference on Algorithmic Learning Theory , pages 150-165. Springer, 2007.
- [4] Jean-Yves Audibert, Rémi Munos, and Csaba Szepesvári. Exploration-exploitation tradeoff using variance estimates in multi-armed bandits. Theoretical Computer Science , 410(19):18761902, 2009.
- [5] Peter Auer. Using confidence bounds for exploitation-exploration trade-offs. Journal of Machine Learning Research , 3(Nov):397-422, 2002.
- [6] Patrick Billingsley. Convergence of Probability Measures . John Wiley &amp; Sons, 2013.
- [7] Haoyu Chen, Wenbin Lu, and Rui Song. Statistical inference for online decision making: In a contextual bandit setting. Journal of the American Statistical Association , 116(533):240-255, 2021.
- [8] Xi Chen, Zehua Lai, He Li, and Yichen Zhang. Online statistical inference for contextual bandits via stochastic gradient descent. arXiv preprint arXiv:2212.14883 , 2022.
- [9] Yan Dai, Ruosong Wang, and Simon S Du. Variance-aware sparse linear bandits. arXiv preprint arXiv:2205.13450 , 2022.
- [10] Qiwei Di, Tao Jin, Yue Wu, Heyang Zhao, Farzad Farnoud, and Quanquan Gu. Variance-aware regret bounds for stochastic contextual dueling bandits. arXiv preprint arXiv:2310.00968 , 2023.
- [11] Maria Dimakopoulou, Zhimei Ren, and Zhengyuan Zhou. Online multi-armed bandits with adaptive inference. Advances in Neural Information Processing Systems , 34:1939-1951, 2021.
- [12] Congyuan Duan, Jingyang Li, and Dong Xia. Online policy learning and inference by matrix completion. arXiv preprint arXiv:2404.17398 , 2024.
- [13] Eyal Even-Dar, Shie Mannor, Yishay Mansour, and Sridhar Mahadevan. Action elimination and stopping conditions for the multi-armed bandit and reinforcement learning problems. Journal of Machine Learning Research , 7(6), 2006.
- [14] Lin Fan and Peter W Glynn. Diffusion approximations for thompson sampling. arXiv preprint arXiv:2105.09232 , 2021.
- [15] Lin Fan and Peter W Glynn. The fragility of optimized bandit algorithms. arXiv preprint arXiv:2109.13595 , 2021.
- [16] Lin Fan and Peter W Glynn. The typical behavior of bandit algorithms. arXiv preprint arXiv:2210.05660 , 2022.
- [17] Louis Faury, Marc Abeille, Clément Calauzènes, and Olivier Fercoq. Improved optimistic algorithms for logistic bandits. In International Conference on Machine Learning , pages 3052-3060. PMLR, 2020.
- [18] Alexander Goldenshluger and Assaf Zeevi. A linear response bandit problem. Stochastic Systems , 3(1):230-261, 2013.

- [19] Vitor Hadad, David A Hirshberg, Ruohan Zhan, Stefan Wager, and Susan Athey. Confidence intervals for policy evaluation in adaptive experiments. Proceedings of the National Academy of Sciences , 118(15):e2014602118, 2021.
- [20] Junya Honda and Akimichi Takemura. Optimality of thompson sampling for gaussian bandits depends on priors. In Artificial Intelligence and Statistics , pages 375-383. PMLR, 2014.
- [21] Anand Kalvit and Assaf Zeevi. A closer look at the worst-case behavior of multi-armed bandit algorithms. Advances in Neural Information Processing Systems , 34:8807-8819, 2021.
- [22] Emilie Kaufmann, Olivier Cappé, and Aurélien Garivier. On the complexity of best-arm identification in multi-armed bandit models. Journal of Machine Learning Research , 17(1):142, 2016.
- [23] Koulik Khamaru and Cun-Hui Zhang. Inference with the upper confidence bound algorithm. arXiv preprint arXiv:2408.04595 , 2024.
- [24] Xu Kuang and Stefan Wager. Weak signal asymptotics for sequentially randomized experiments. Management Science , 2023.
- [25] Tze Leung Lai and Herbert Robbins. Asymptotically efficient adaptive allocation rules. Advances in Applied Mathematics , 6(1):4-22, 1985.
- [26] Tor Lattimore. Refining the confidence level for optimistic bandit strategies. Journal of Machine Learning Research , 19(20):1-32, 2018.
- [27] Tor Lattimore and Csaba Szepesvári. Bandit Algorithms . Cambridge University Press, 2020.
- [28] Subhojyoti Mukherjee, KP Naveen, Nandan Sudarsanam, and Balaraman Ravindran. Efficientucbv: An almost optimal algorithm using variance estimates. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 32, 2018.
- [29] Xinkun Nie, Xiaoying Tian, Jonathan Taylor, and James Zou. Why adaptively collected data have negative bias and how to correct for it. In International Conference on Artificial Intelligence and Statistics , pages 1261-1269. PMLR, 2018.
- [30] Hao Qin, Kwang-Sung Jun, and Chicheng Zhang. Kullback-leibler maillard sampling for multiarmed bandits with bounded rewards. Advances in Neural Information Processing Systems , 36:60514-60526, 2023.
- [31] Philippe Rigollet and Assaf Zeevi. Nonparametric bandits with covariates. COLT 2010 , page 54, 2010.
- [32] Daniel J Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, Zheng Wen, et al. A tutorial on thompson sampling. Foundations and Trends® in Machine Learning , 11(1):1-96, 2018.
- [33] Aadirupa Saha and Branislav Kveton. Only pay for what is uncertain: Variance-adaptive thompson sampling. In The Twelfth International Conference on Learning Representations , 2024.
- [34] David Simchi-Levi, Zeyu Zheng, and Feng Zhu. Regret distribution in stochastic bandits: Optimal trade-off between expectation and tail risk. arXiv preprint arXiv:2304.04341 , 2023.
- [35] Aleksandrs Slivkins et al. Introduction to multi-armed bandits. Foundations and Trends® in Machine Learning , 12(1-2):1-286, 2019.
- [36] Mohammad Sadegh Talebi and Odalric-Ambrym Maillard. Variance-aware regret bounds for undiscounted reinforcement learning in mdps. In Algorithmic Learning Theory , pages 770-805. PMLR, 2018.
- [37] Ruitu Xu, Yifei Min, and Tianhao Wang. Noise-adaptive thompson sampling for linear contextual bandits. Advances in Neural Information Processing Systems , 36, 2024.

- [38] Ruohan Zhan, Vitor Hadad, David A Hirshberg, and Susan Athey. Off-policy evaluation via adaptive weighting with data from contextual bandits. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery &amp; Data Mining , pages 2125-2135, 2021.
- [39] Kelly Zhang, Lucas Janson, and Susan Murphy. Inference for batched bandits. Advances in Neural Information Processing Systems , 33:9818-9829, 2020.
- [40] Zihan Zhang, Jiaqi Yang, Xiangyang Ji, and Simon S Du. Improved variance-aware confidence sets for linear bandits and linear mixture mdp. Advances in Neural Information Processing Systems , 34:4342-4355, 2021.
- [41] Heyang Zhao, Jiafan He, Dongruo Zhou, Tong Zhang, and Quanquan Gu. Variance-dependent regret bounds for linear bandits and reinforcement learning: Adaptivity and computational efficiency. In The Thirty Sixth Annual Conference on Learning Theory , pages 4977-5020. PMLR, 2023.

## A Additional related works

## Variance-aware decision making

In the multi-armed bandit setting, leveraging the variance information to balance the explorationexploitation trade-off was first studied by [5] and [3, 4], where the UCB-V algorithm was proposed and analyzed. Beyond the optimistic approach, [28] examined elimination-based algorithms that utilize variance information, while [20] and [33] focused on analyzing the performance of Thompson sampling in variance-aware contexts.

Beyond the MAB model, variance information has been utilized in contextual bandit and Markov Decision Process settings [36, 40, 9, 41, 10, 37]. Among these works, [40] and [41] are most relevant to our study, especially regarding the linear contextual bandits. Instead of assuming that each arm a has a fixed variance σ a , they assumed that at each round t , all arms have the same variance σ t depending on the time-step t . In the homogeneous case - i.e. when σ a ≡ σ, σ t ≡ σ for some σ &gt; 0 -their settings coincides with ours, and both algorithms achieve a regret of √ σ 2 KT regret. However, in the general case, the regret results are not directly comparable.

## Asymptotic behavior analysis in multi-armed bandits

Our investigation into the asymptotic behaviors of UCB-V is inspired by recent advancements in the precise characterization of arm-pulling behavior [21, 23], which focused on the canonical UCB algorithm. Beyond the canonical UCB algorithm, another line of research explores the asymptotic properties of bandit algorithms within the Bayesian frameworks, particularly under diffusion scaling [14, 2, 24], where reward gaps scale as 1 / √ T . Additionally, a noteworthy body of work [15, 16, 34] conducts asymptotic analyses in the regime described by [25], where the reward gaps ∆ remain constant as T increases.

## Inference with adaptively collected data

In addition to the regret minimization, there is a growing amount of interest in statistical inference for the bandit problems. A significant body of research addresses the online debiasing and adaptive inference methods [11, 7, 8, 12], but our findings are more closely related to studies focused on the post-policy inference [29, 39, 19, 38]. These works aim to provide valid confidence intervals for reward-related quantities based on data collected from pre-specified adaptive policies. Among them, the most relevant works are [21] and [23]. The former addresses the inference guarantee of Z -estimators collected by UCB in the two-armed setting, while the latter extends this to the K -armed setting. In particular, these two works assert the stability of the pulling time for the optimal arm under arbitrary gap conditions ∆ , enabling the application of the martingale Central Limit Theorem (CLT) to obtain an asymptotically normal estimator of µ ⋆ . However, our results point out a critical regime where such stability breaks down , as shown in Figure 4b, highlighting the price of utilizing variance information to enhance the regret minimization performance.

## B Discussions on K -armed setting

In this section, we extend the results in Theorem 1, which addresses the 2 -armed setting, to the general K -armed case. Recall that σ a ≡ σ a ( ρ ; T ) and ∆ a ≡ ∆ a ( ρ ; T ) for a ∈ [ K ] . The formal statement of the result is given in the following theorem.

Theorem 4. Consider Algorithm 1. Assume that for any a ∈ [2 : K ] ,

<!-- formula-not-decoded -->

Then for a ∈ [ K ] , we have

<!-- formula-not-decoded -->

where n ⋆ 1 ,T is the unique solution to equation

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

When K = 2 , the condition in (4) simplifies to the stability condition for the 2 -armed setting as stated in Theorem 1. While instability occurs when (4) is violated in the 2 -armed case (cf. Proposition 2), there is no clear reason to expect that (4) is optimal for the K -armed setting. Indeed, as indicated in the proof, the fixed-point equation (15) may remain stable if there exists some a ∈ [ K ] \ { 1 } that satisfies the boundedness condition in (14). However, formally characterizing such an a poses a significant challenge in itself. As such, a full generalization to the K -armed setting is left for future work.

Asymptotic arm-pulling characterization (moderate gap, K -armed). To build intuition for (15), consider the K -armed setting with all variances { σ a } a ∈ [ K ] = Ω(1) and gaps in the moderate-gap regime ∆ a = σ a √ ( θ a log T ) /T with fixed θ a ≥ 0 . In this regime the variance branches of φ ( · ; · ) are active, and

<!-- formula-not-decoded -->

Plugging these into (15) and multiplying both sides by ρ log T yields

<!-- formula-not-decoded -->

As θ a → 0 (shrinking gaps), the normalized pulling rates converge to

<!-- formula-not-decoded -->

That is, when the gaps vanish, each arm's asymptotic pulling rate becomes proportional to its variance, generalizing the two-armed intuition and the homogeneous-variance case.

Besides the stability results, the high-probability bounds for n a,T in the K -armed setting, as in (2), still hold when (6) is replaced with (15), as shown in Appendix C. The following corollary provides a refined regret bound in the K -armed setting.

Theorem 5. There exists some universal constant C 0 &gt; 0 such that for ρ ≥ C 0 and T ≥ 3 , we have that with probability at least 1 -3 K/T 2 ,

<!-- formula-not-decoded -->

When K = 2 , Theorem 5 above aligns with Theorem 2. A comparison with previous findings is provided at the end of the previous section. Notably, in the small σ 2 a /σ 1 regime, the bound in Theorem 5 grows linearly with the number of arms K . We conjecture that the result in Theorem 5 is optimal up to logarithmic factors for variance-aware decision-making, even for more complicated algorithms beyond UCB-V. Establishing the corresponding minimax lower bound is left for future work.

and for a ∈ [2 : K ] ,

## C Proofs for Section 2

For any T ∈ N + and δ &gt; 0 , we define events

<!-- formula-not-decoded -->

where e ( δ ; ρ ; T ) ≡ √ 48 ρ log T log(log( T/δ ) /δ ) + 128 log(1 /δ ) . The majority of the technical analysis in this paper will be done on event M ( δ ; ρ ; T ) ∩V ( δ ; ρ ; T ) . It is worth noting that, as shown in Proposition 7 below, we have P ( M ( δ ; ρ ; T ) ∩V ( δ ; ρ ; T )) ≥ 1 -3 Kδ for any δ &lt; 1 / 2 . Although the statements in Section 2 are presented for the 2 -armed setting, the proof in this section applies to the K -armed setting as well.

## C.1 Proof of Proposition 1

Recall that φ a,t ≡ φ ( n a,t ; σ a ) . For simplicity, denote by ̂ φ a,t ≡ φ ( n a,t ; ̂ σ a,t / √ ρ log T ) for a ∈ [ K ] and t ∈ [ T ] . The following lemma provides a quantified error between ̂ φ a,t and φ a,t .

Lemma 2. On event M ( δ ; ρ ; T ) ∩ V ( δ ; ρ ; T ) , we have that for any a ∈ [ K ] and t ∈ [ T ] ,

<!-- formula-not-decoded -->

Proof. By the elementary inequality that | √ x - √ y | ≤ √ | x -y | for any x, y ≥ 0 , we can derive on event M ( δ ; ρ ; T ) ∩ V ( δ ; ρ ; T ) that

<!-- formula-not-decoded -->

which completes the proof.

For any δ, ρ &gt; 0 , let I ± ( δ ; ρ ; T ) ≡ 1 ± Err 1 / 2 ( δ ; ρ ; T ) ± Err ( δ ; ρ ; T ) with Err ( δ ; ρ ; T ) ≡ e ( δ ; ρ ; T ) / ( ρ log T ) . Recall that φ a,T = φ ( n a,T ; σ a ) . The following proposition provides (tight) upper and lower bounds for φ a,T .

Proposition 4. Fix any T ∈ N + and any δ, ρ ∈ R + such that I ± ( δ ; ρ ; T ) &gt; 0 . Then on event M ( δ ; ρ ; T ) ∩ V ( δ ; ρ ; T ) , we have that for any a ∈ [ K ] ,

<!-- formula-not-decoded -->

Proof. The proof primarily follows the framework in [26, 23]. In the proof, we write I ± ≡ I ± ( δ ; ρ ; T ) , M≡M ( δ ; ρ ; T ) , and V ≡ V ( δ ; ρ ; T ) for simplicity. For any a ∈ [ K ] , denote by T a the last timestep such that arm a was pulled.

( Step 1 ). In this step, we provide an upper bound for φ -1 a,T . By the UCB rule, we have

<!-- formula-not-decoded -->

By Lemma 2, it holds on event M∩V that

<!-- formula-not-decoded -->

Using further n a,T a = n a,T -1 , n 1 ,T a ≤ n 1 ,T , and ∆ a ≥ 0 , we can show that

<!-- formula-not-decoded -->

As φ ( n a,T -1; σ a ) ≤ φ a,T n a,T / ( n a,T -1) , we can obtain an upper bound for φ -1 a,T

<!-- formula-not-decoded -->

( Step 2 ). In this step, we provide a lower bound for φ -1 a,T . Consider the last time-step such that arm 1 was pulled. In view of the UCB rule, it holds that

<!-- formula-not-decoded -->

By Lemma 2, we have that on event M∩V ,

<!-- formula-not-decoded -->

Then by n a,T 1 ≤ n a,T -1 , n 1 ,T 1 ≤ n 1 ,T -1 , and ∆ a ≥ 0 , it follows that

<!-- formula-not-decoded -->

As φ ( n 1 ,T -1; σ 1 ) ≤ φ 1 ,T n 1 ,T / ( n 1 ,T -1) , we can obtain a lower bound for φ -1 a,T

<!-- formula-not-decoded -->

Combining (17) and (20) above concludes the proof.

Proposition 5. Fix any T ∈ N + and any δ, ρ ∈ R + such that I ± ( δ ; ρ ; T ) &gt; 0 . Then on event M ( δ ; ρ ; T ) ∩ V ( δ ; ρ ; T ) , we have that for any a ∈ [ K ] ,

<!-- formula-not-decoded -->

Proof. In the proof below, let us write I ± ≡ I ± ( δ ; ρ ; T ) for simplicity. Denote by

<!-- formula-not-decoded -->

Note that for any σ &gt; 0 , function φ ( · ; σ ) : R + → R + is bijective and monotone decreasing, with its inverse map given by

<!-- formula-not-decoded -->

Starting from the sandwich inequality (16), the monotonicity of φ yields that

<!-- formula-not-decoded -->

As ξ -1 l &gt; 1 , we have

<!-- formula-not-decoded -->

Using further the fact that ψ ( z ; σ ) -1 ∨ σ 2 = z ∨ σ 2 , we arrive at

<!-- formula-not-decoded -->

Similar, from ξ -1 u &lt; 1 we can derive that

<!-- formula-not-decoded -->

Combining the above two bounds completes the proof.

Lemma 3. Fix any T ∈ N + and any δ, ρ ∈ R + such that 0 &lt; I + ( δ ; ρ ; T ) / I -( δ ; ρ ; T ) ≤ 2 and T &gt; 2 K . Then on event M ( δ ; ρ ; T ) ∩ V ( δ ; ρ ; T ) , we have that for any a ∈ [ K ] ,

̸

<!-- formula-not-decoded -->

Proof. In the proof below, we write I ± ≡ I ± ( δ ; ρ ; T ) for simplicity. Let us first consider the lower bound for n 1 ,T . Suppose by contradiction that (21) does not hold and we have

<!-- formula-not-decoded -->

which implies σ 1 ∨ n -1 / 2 1 ,T ≥ σ 1 ∨ ( σ -1 1 T -1 / 2 ∧ T -1 / 4 ) = σ 1 ∨ T -1 / 4 . Thus, it follows that

<!-- formula-not-decoded -->

On the other hand, as ∑ a ∈ [ K ] n a,T = T , there must exist some a ∈ [ K ] , a = 1 such that n a,T ≥ T/K . For such a , we have by (17) that

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which then leads to a contradiction. The lower bound for n 1 ,T follows.

Next, we prove the lower bound for n a,T , a ∈ [2 : K ] . Starting from (20), we have by the lower bound for n 1 ,T in the first part that

<!-- formula-not-decoded -->

which proves the claim.

We are now ready to prove Proposition 1.

Proof Proposition 1. In the proof below, let us write I ± ≡ I ± ( δ ; ρ ; T ) and Err ≡ Err ( δ ; ρ ; T ) for simplicity. It follows from Propositions 4 and 5 that on event M ( δ ; ρ ; T ) ∩ V ( δ ; ρ ; T ) ,

<!-- formula-not-decoded -->

Applying the lower bounds for n a,T in Lemma 3, we can further bound the right-hand side above as

<!-- formula-not-decoded -->

where we have used √ Err ≥ 8 / ( ρ log T ) and the elementary inequality (1 -x )(1 -y )(1 -z ) ≥ 1 -x -y -z for ∀ x, y, z &gt; 0 .

Using the same argument for the left-hand side, we can show that

<!-- formula-not-decoded -->

Summing over a and using the fact that ∑ a ∈ [ K ] n a,T = T, we can obtain that

<!-- formula-not-decoded -->

as desired.

Figure 3: (a): The confidence region of n 1 ,T under different Λ T , with σ 1 = o ( σ 2 ) . The dotted lines represent the exact and perturbed solutions of (3), where the perturbed curves solve f ( φ ) = 1 ± 1 / log T . The UCB-V line shows the number of arm pulls under the UCB-V algorithm with 30% quantile, with T = 10 5 over 30 repetitions. (b): The ratio between the perturbed solution f ( φ ) = 1 ± 1 / log T and the exact solution f ( φ ) = 1 is shown for T = 10 5 , 10 7 , 10 9 under different Λ T . It can be seen that the ratio deviates from 1 as Λ T → 1 with increasing T .

<!-- image -->

## D Proofs for Section 3

## D.1 Proof of Lemma 1

Proof of Lemma 1. (1). First, we prove the existence and uniqueness of φ ⋆ . Note that f ( φ ) is continuously monotone decreasing on (0 , ∞ ) . On the other hand, as

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

there must exist a unique φ ⋆ ∈ (1 /T, 1) such that f ( φ ⋆ ) = 1 , proving the claim.

(2). We only show the proof for the case of ζ &gt; 0 , while the case of ζ &lt; 0 can be handled similarly. Note by the monotonicity, we have φ ζ ≤ φ ⋆ . Let φ ζ /φ ⋆ = 1 -¯ ζ for some ¯ ζ ≥ 0 depending on ζ . Our aim is to derive an upper bound for ¯ ζ .

It follows from (3) that at least one of the following holds:

<!-- formula-not-decoded -->

Case 1 : If σ 2 1 = σ 2 1 ∨ φ ⋆ , we have σ 2 1 = σ 2 1 ∨ φ ζ . Hence, it holds that

<!-- formula-not-decoded -->

For the case when φ ⋆ = σ 2 1 ∨ φ ⋆ , we can compute that

<!-- formula-not-decoded -->

By combining the above results, in Case 1, we can conclude that ¯ ζ ≤ 2 ζ .

Case 2 : Note that in this case, we have

<!-- formula-not-decoded -->

On the other hand, if φ ⋆ +∆ 2 = σ 2 2 ∨ ( φ ⋆ +∆ 2 ) , we can show that 1 / ( Tφ ⋆ ) ≥ 1 /T ( φ ⋆ +∆ 2 ) ≥ 1 / 2 . Thus, it must hold that φ ⋆ = T/ 2 and ∆ 2 = 0 . This implies that when φ ⋆ +∆ 2 = σ 2 2 ∨ ( φ ⋆ +∆ 2 ) ,

<!-- formula-not-decoded -->

For the case when σ 2 2 = σ 2 2 ∨ ( φ ⋆ +∆ 2 ) , using (22) and the fact that φ ⋆ ≥ T -1 , we can deduce that σ 2 2 T ( φ ⋆ +∆ 2 ) 2 ≥ 1 2 ≥ σ 2 1 ∨ φ ⋆ T ( φ ⋆ ) 2 ≥ σ 2 1 ∨ T -1 T ( φ ⋆ ) 2 , which implies that φ ⋆ / ( φ ⋆ +∆ 2 ) ≥ ( σ 1 ∨ T -1 / 2 ) /σ 2 . Hence, it follows that

<!-- formula-not-decoded -->

This gives the bound corresponding to σ 2 / ( σ 1 ∨ T -1 / 2 ) in (4).

Next, we will derive the bound that corresponds to | σ 2 2 / ( T ∆ 2 2 ) -1 | -1 in (4). The analysis will be further separated into two cases: (i) 1 -σ 2 2 / ( T ∆ 2 2 ) ≥ 0 and (ii) 1 -σ 2 2 / ( T ∆ 2 2 ) &lt; 0 . For case (i), we can show that

<!-- formula-not-decoded -->

For case (ii), let us recall from (23) that

<!-- formula-not-decoded -->

Note that when ∆ 2 /φ ⋆ &gt; 2 (or equivalently φ ⋆ / ∆ 2 &lt; 1 / 2 ),

<!-- formula-not-decoded -->

Using the condition that 1 -σ 2 2 / ( T ∆ 2 2 ) &lt; 0 , we can derive that

<!-- formula-not-decoded -->

As σ 2 2 / ( φ ⋆ +∆ 2 ) 2 ≤ T and φ ⋆ / ∆ 2 &lt; 1 / 2 , we can obtain that σ 2 2 / ( T ∆ 2 2 ) ≤ 9 / 4 , which then entails that ( φ ⋆ ) -1 ∆ 2 ≤ 27 / (2 σ 2 2 / ( T ∆ 2 2 ) -2) . Consequently, it holds that

<!-- formula-not-decoded -->

Plugging this back to (24) yields that

<!-- formula-not-decoded -->

By combining the above results, in Case 2, we can conclude that

<!-- formula-not-decoded -->

The claim then follows by combining the estimates in Cases 1 and 2 above.

## D.2 Proof of Theorem 1

Proposition 6. For any positive sequences { σ T } T ≥ 1 , { n T } T ≥ 1 , and { ¯ n T } T ≥ 1 , the following are equivalent:

<!-- formula-not-decoded -->

2. lim T →∞ n T / ¯ n T = 1 .

Proof. (1) ⇒ (2). First, noticing that by the elementary inequality,

<!-- formula-not-decoded -->

we can deduce from (1) that lim T →∞ ( φ ( n T ; σ T ) ∨ σ 2 T ) / ( φ (¯ n T ; σ T ) ∨ σ 2 T ) = 1 . Using further the identity n = ( σ 2 ∨ φ ( n ; σ )) /φ 2 ( n ; σ ) for any n, σ &gt; 0 , we have

<!-- formula-not-decoded -->

(2) ⇒ (1). In light of (25), we see that (2) implies lim T →∞ ( σ T ∨ n -1 / 2 T ) / ( σ T ∨ ¯ n -1 / 2 T ) = 1 . Consequently, it holds that

<!-- formula-not-decoded -->

The claim follows.

We are now ready to prove Theorem 1.

Proof of Theorem 1. Let us select δ = 1 / log T . In the proof, we will write I ± ≡ I ± ( δ ; ρ ; T ) , M ≡ M ( δ ; ρ ; T ) , and V ≡ V T ( δ ; ρ ; T ) for simplicity. We also write φ ⋆ ≡ φ ( n ⋆ 1 ,T ; σ 1 ) . From Proposition 7, we have P ( M∩V ) ≥ 1 -3 K/ log T .

The uniqueness of n 1 ,⋆ follows directly from Lemma 1(1) and the fact that φ ( · ; σ 1 ) is bijective. It follows from Propositions 4 and 5 that on event M∩V ,

<!-- formula-not-decoded -->

Applying Lemma 3 and using the facts that φ 1 ,T = o (1) on M∩V and ∆ a = o (1) , we can show that for any a ∈ [ K ] , lim T →∞ n a,T = + ∞ on M∩V . Hence, the inequality in the above display implies that for any a ∈ [ K ] ,

<!-- formula-not-decoded -->

Using further ∑ a ∈ [ K ] n a,T /T = 1 , we can arrive at

<!-- formula-not-decoded -->

Let us consider the case when K = 2 . By resorting to Lemma 1(2), we have that φ 1 ,T /φ ⋆ = 1 + o P (1) , which together with Proposition 6 implies that n 1 ,T /n ⋆ 1 ,T = 1 + o P (1) .

For arm 2 , with (26) at hand, by Proposition 6 it suffices to show that

<!-- formula-not-decoded -->

which is straightforward by (25).

<!-- formula-not-decoded -->

## D.3 Proof of Proposition 2

We need the following lemma to prove Proposition 2.

Lemma 4. Consider the Two-armed Bernoulli Bandit instance in Proposition 2. Recall that T i is the last time that arm i was pulled. Then for sufficiently large T , there exists some constant c 0 &gt; 0 such that

<!-- formula-not-decoded -->

Proof. ( Step 1 ). We first prove the following: On event M T ∩ V T , there exists some constant 0 &lt; c &lt; 1 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

holds on event M (1 / log T ; ρ ; T ) ∩V (1 / log T ; ρ ; T ) and using Lemma 3 with the fact that n 1 ,T 1 = n 1 ,T -1 , we can deduce that for sufficiently large T ,

<!-- formula-not-decoded -->

Given that n 2 ,T 2 = n 2 ,T -1 , a similar estimate applies to n 2 ,T 2 if we start from (19), which proves (28).

( Step 2 ). Let c be the constant in (28). It follows from Step 1 and Lemma 8 that for any ε &gt; 0 , there exists some T 0 = T 0 ( ε ) &gt; 0 such that for T ≥ T 0 , P ( n 2 ,T 1 ≥ cT ) ≥ 1 -ε . Using Lemma 9 and choosing ε = c anti / 2 , we can obtain that

<!-- formula-not-decoded -->

This concludes the proof for the probability estimate for U . The estimate for P ( L ) follows a similar argument, so will be omitted here to avoid repetitive details.

We are now ready to prove Proposition 2.

Proof of Proposition 2. (1). We will first prove the upper bound for n 1 ,T on event U ∩ M (1 / log T ; ρ ; T ) ∩ V (1 / log T ; ρ ; T ) . Consider time T 1 that arm 1 was last pulled. Using (i) the definition of U and (ii) the UCB rule, we can show that on event U ∩ M (1 / log T ; ρ ; T ) ∩ V (1 / log T ; ρ ; T ) ,

<!-- formula-not-decoded -->

Rearranging the terms, it holds on event U ∩ M (1 / log T ; ρ ; T ) ∩ V (1 / log T ; ρ ; T ) that

<!-- formula-not-decoded -->

By the facts that σ 2 2 = µ (1 -µ ) , ∆ 2 = µ (1 -µ ) ρ log T/T , n 2 ,T ≥ n 2 ,T 1 , n 1 ,T 1 = n 1 ,T 1 -1 , and n 1 ,T + n 2 ,T = T , we can arrive at

<!-- formula-not-decoded -->

To this end, recalling from (18) that

Further using 1 / √ T -n 1 ,T ≥ (1 + n 1 ,T / (2 T )) / √ T , the above inequality becomes

<!-- formula-not-decoded -->

Note that the above inequality is quadratic, we can then solve it to obtain that

<!-- formula-not-decoded -->

where for sufficiently large T ,

<!-- formula-not-decoded -->

Thus, we can conclude on event U ∩ M (1 / log T ; ρ ; T ) ∩ V (1 / log T ; ρ ; T ) that

<!-- formula-not-decoded -->

(2). The lower bound for n 1 ,T on event L∩M (1 / log T ; ρ ; T ) ∩V (1 / log T ; ρ ; T ) can be established using a similar strategy. Let us consider time T 2 that arm 2 was last pulled. Using (i) the definition of L and (ii) the UCB rule, we have that on event L∩M (1 / log T ; ρ ; T ) ∩ V (1 / log T ; ρ ; T ) ,

<!-- formula-not-decoded -->

Rearranging the terms yields that on event L∩M (1 / log T ; ρ ; T ) ∩ V (1 / log T ; ρ ; T ) ,

<!-- formula-not-decoded -->

It follows from the facts that ∆ 2 2 = σ 2 2 /T , n 1 ,T ≥ n 1 ,T 2 , n 2 ,T 2 = n 2 ,T -1 , and n 1 ,T + n 2 ,T = T that

<!-- formula-not-decoded -->

Observe that on event L∩M (1 / log T ; ρ ; T ) ∩V (1 / log T ; ρ ; T ) , it holds that c 1 √ T ≤ n 1 ,T +1 ≤ c 2 T for some constants c 1 &gt; 0 and 0 &lt; c 2 &lt; 1 , cf. Lemme 3 and (28). Then we can derive that for sufficiently large T ,

<!-- formula-not-decoded -->

where we have also used 1 / √ 1 -( n 1 ,T +1) /T ≤ 1 + 2 n 1 ,T / (1 -c 2 ) T . Therefore, with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we can solve the inequality (29) to obtain that

<!-- formula-not-decoded -->

The claim now follows from Lemma 4.

## E Proofs for Section 4

We would like to emphasize that, although the statements in Section 4 are presented for the 2 -armed setting, the proof in this section applies to the K -armed setting as well.

## E.1 Proof of Proposition 3

We will work on the following events

<!-- formula-not-decoded -->

The following lemma provides a probability estimate for event ˜ M T ∩ ˜ V T when ρ is large.

Lemma 5. There exists some universal constant C 0 &gt; 0 such that for ρ ≥ C 0 and T ≥ 3 ,

<!-- formula-not-decoded -->

Proof. By selecting δ = 1 /T 2 in equations (52) and (53), we achieve the desired result, provided that ρ ≥ C 0 with C 0 sufficiently large to satisfy √ 96 C 0 +256 ≤ C 0 / 4 .

Proof of Proposition 3. First, in view of the proof for Lemma 2, it holds that on event ˜ M T ∩ ˜ V T ,

<!-- formula-not-decoded -->

At time-step T a , by the UCB rule of ¯ X a,T a + ̂ φ a,T a · ρ log T ≥ ¯ X 1 ,T a + ̂ φ 1 ,T a · ρ log T , we have that on event ˜ M T ∩ ˜ V T ,

<!-- formula-not-decoded -->

Dividing ρ log T on both sides and rearranging the inequality, we can deduce that

<!-- formula-not-decoded -->

which then implies either n a,T ≤ 1 or φ a,T ≥ (∆ a + φ 1 ,T ) / 16 . Therefore, we have

<!-- formula-not-decoded -->

The claim now follows from Lemma 5.

## E.2 Proof of Theorem 2

Proof of Theorem 2. Observe that

<!-- formula-not-decoded -->

with A 1 ≡ { a ∈ [2 : K ] : 16 σ 2 a &gt; ∆ a + φ 1 ,T } and A 2 ≡ [2 : K ] \ A 1 . It follows from Proposition 3 that

<!-- formula-not-decoded -->

Then for the summation on A 2 , we can derive on event E T that

<!-- formula-not-decoded -->

For the summation on A 1 , we have that, on one hand,

<!-- formula-not-decoded -->

and on the other hand,

<!-- formula-not-decoded -->

Combining (32)-(34) above concludes the proof.

Remark 2. Using the high-probability arm-pulling bound from [4]

<!-- formula-not-decoded -->

inequalities (32) and (33) remain valid when the definitions of A 1 and A 2 are replaced with

<!-- formula-not-decoded -->

̸

respectively. This adjustment leads to a regret bound of O ( √∑ a =1 σ 2 a T log T ) .

## E.3 Lower bound proofs

In this section, we prove Theorem 3. For clarity, we first present the proof for the Gaussian case in Section E.3.1, then extend the construction of reward distributions to the bounded support setting in Section E.3.2.

## E.3.1 Proof with Gaussian bandit instances

For simplicity, we present the proof with Gaussian bandit instances first and then generalize it to the hard instances constructed over bounded reward distributions.

Theorem 6. Consider the following two classes of problems with any β &gt; 0 :

<!-- formula-not-decoded -->

Then for every policy π , there exists some v ∈ V such that

<!-- formula-not-decoded -->

Moreover, the following trade-off lower bound holds: Given any c T = O ( poly (log T )) , consider the good policy class

<!-- formula-not-decoded -->

then for any π ∈ Π good , there exists some v ′ such that

<!-- formula-not-decoded -->

Proof. We consider the proof of (35) first. Let ∆ ∈ [0 , 1 / 2] and σ 2 ≥ σ 1 , where the specific values will be determined later. We consider the following two instances:

<!-- formula-not-decoded -->

Fix a policy π . Recall the divergence decomposition Lemma for the reward distributions P π ν 1 , P π ν 2 (see e.g. Lemma 15.1 of [27]), we have by our construction,

<!-- formula-not-decoded -->

Here, P ν i ; j denotes the distribution of j th arm in instance ν i . Moreover, for instance ν 1 , we have E π ν 1 Reg( T ) = ∆ E π ν 1 n 2 ,T ≥ ∆ T P π ν 1 ( n 2 ,T ≥ T/ 2) / 2 and for instance ν 2 , we have E π ν 2 Reg( T ) = ∆ E π ν 2 n 1 ,T ≥ ∆ T P π ν 2 ( n 2 ,T &lt; T/ 2) / 2 . Combining the above estimates and applying Bretagnolle-Huber inequality, we can obtain that

<!-- formula-not-decoded -->

Taking ∆ = σ 2 / √ 2 T , we arrive at the lower bound

<!-- formula-not-decoded -->

This implies that either

<!-- formula-not-decoded -->

hold, which completes the proof of (35).

Next, we prove the trade-off lower bound in (36). With slight abuse of notations, we construct the following instances: with σ 1 ∈ [ T -β / 256 , 1] and ∆ ≡ 16 σ 1 c T / √ T ,

<!-- formula-not-decoded -->

We would like to argue that for any π ∈ Π good ,

<!-- formula-not-decoded -->

We will prove this claim by contradiction. Suppose (39) does not hold, we may derive that

<!-- formula-not-decoded -->

Therefore, similar to (37), we can obtain a refined upper bound for KL ( P π ν 1 ∥ P π ν 2 ) (instead of using the trivial upper bound n 2 ,T ≤ T ),

<!-- formula-not-decoded -->

Then we may proceed as (38) to derive

<!-- formula-not-decoded -->

By the upper bound of E π ν 1 Reg( T ) (hypothesis), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies that π / ∈ Π good , a contradiction. This proves (39) and therefore completes the proof of the trade-off lower bound in (36).

## E.3.2 Remark on Lower Bound over Bounded Distributions

In the proof of Theorem 6, we used the Gaussian distribution for notational convenience, owing to its analytical KL divergence bound. However, the upper bound result is derived for bounded bandit instances, which leads to a slight mismatch between the lower bound and upper bound environments. In this section, we show that the proof can be readily generalized to the hard instances constructed over bounded reward distributions.

Here we specify a two-point supported variable based construction: Given any positive µ, σ 2 , we consider distributions Q µ,σ supported over { 0 , µ 2 + σ 2 µ } with

<!-- formula-not-decoded -->

It can be seen that E X ∼ Q µ,σ [ X ] = µ and Var X ∼ Q µ,σ ( X ) = σ 2 .

Suppose, in addition, that µ &lt; σ and let Q ∆ µ,σ denote a ∆ -modification of Q µ,σ for any 0 &lt; ∆ &lt; µ 2 + σ 2 µ -2 µ , defined as

<!-- formula-not-decoded -->

where ˜ σ ≥ 0 the solution to

<!-- formula-not-decoded -->

Since m ( a ) ≡ ( µ +∆) 2 + a µ +∆ is an increasing function in a and

<!-- formula-not-decoded -->

it follows that ˜ σ is well-defined and satisfies ˜ σ &gt; σ.

Then we may compute the KL-divergence between Q µ,σ and Q ∆ µ,σ as follows:

<!-- formula-not-decoded -->

where in the last step, we have used the fact that log( σ 2 / ˜ σ 2 ) ≤ 0 .

Taking µ ≥ σ -∆ in the last inequality and using the elementary inequality log(1 + x ) ≤ x then leads to

<!-- formula-not-decoded -->

Now we can summarize the above construction and calculation into the following lemma:

Lemma 6. Given any µ, σ &gt; 0 , there exists a distribution Q µ,σ supported on { 0 , µ 2 + σ 2 µ } with

<!-- formula-not-decoded -->

Given any ∆ &gt; 0 and suppose that

<!-- formula-not-decoded -->

holds simultaneously. Then exists a ∆ -modification Q ∆ µ,σ of Q µ,σ supported on { 0 , µ 2 + σ 2 µ } such that

<!-- formula-not-decoded -->

In particular, for any σ &gt; 0 , with the selection 0 &lt; ∆ ≤ σ/ 2 , µ = σ -∆ , one can verify that ( µ, σ, ∆) satisfies the condition (41) and the corresponding Q µ,σ , Q ∆ µ,σ are supported inside [0 , 3 σ ] with variances lies in [ σ 2 , 3 σ 2 ] .

Based on Lemma 6, we now explain how to modify the distribution class constructed for proving (35) and (36) separately.

Proof of (12) . The proof follows the same as that of (35); we highlight the differences below. Given any σ 1 , σ 2 , ∆ satisfying σ 2 ≥ σ 1 ≥ 2∆ , we consider the following two instances:

<!-- formula-not-decoded -->

Note that with such selection, Lemma 6 ensures that the distributions of the second arm's reward in both instances are supported over [0 , 3 σ ] with their variances lied in [ σ 2 2 , 3 σ 2 2 ] . For the first arm, we have its variance is given by σ 2 1 and its support is bounded by

<!-- formula-not-decoded -->

thus selecting σ 2 &lt; 1 / 3 ensures that the constructed reward distributions are supported over [0 , 1] .

Next, we can proceed the same analysis as in section E.3.1 with the proof of (35). The only difference is that the bound in (37) under our new construction turns to

<!-- formula-not-decoded -->

and therefore, (38) becomes to

<!-- formula-not-decoded -->

Taking ∆ = σ 2 / √ 2 T (as T ≥ 2 , we must have that ∆ ≤ σ 2 / 2 ), we arrive at the lower bound

<!-- formula-not-decoded -->

This implies that either

<!-- formula-not-decoded -->

hold, which completes the proof of (12).

Proof of (13) . The proof follows the same as that of (36); we highlight the differences below. For any fixed 0 &lt; β &lt; 1 / 2 and σ 1 ≥ T -β / 256 , ∆ = 16 σ 1 c T / √ T , and σ 2 = √ σ 1 T -β , we set

<!-- formula-not-decoded -->

In particular, notice that

<!-- formula-not-decoded -->

which always holds under our selection for sufficiently large T . Thus the KL divergence upper bound between Q σ 2 -∆ ,σ and Q ∆ σ 2 -∆ ,σ 2 in Lemma 6 still holds under our construction. Now applying the same argument in Section E.3.1, we can arrive at

<!-- formula-not-decoded -->

when suppose in contradiction that

<!-- formula-not-decoded -->

Now by Lemma 6, we have

<!-- formula-not-decoded -->

Which then, together with (42) implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies that π / ∈ Π good , a contradiction. This disproves (42) and therefore completes the proof of the trade-off lower bound in (13).

## F Proofs for Section B

With slight abuse of notation, we define function f ( x ) for any x ∈ R ≥ 0 as

<!-- formula-not-decoded -->

Let us consider the following fixed-point equation in φ

<!-- formula-not-decoded -->

The following lemma extends Lemma 1 to the K -armed setting.

Lemma 7. It holds that:

1. The fixed-point equation (43) admits a unique solution φ ⋆ ∈ (1 /T, 1) for all T ∈ N + .
2. Assume that there exist some δ ∈ ( -1 / 2 , 1 / 2) and φ δ such that f ( φ δ ) = 1 + δ . Then there exists some constant c = c ( K ) &gt; 0 such that

<!-- formula-not-decoded -->

Proof. The proof is almost the same as that of Lemma 1. We include the proof here for the convenience of readers.

- (1). First we prove the existence and uniqueness of φ ⋆ . Note that f ( φ ) is continuously monotone decreasing on (0 , ∞ ) . On the other hand, as

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

there must exist a unique φ ⋆ ∈ (1 /T, 1) such that f ( φ ⋆ ) = 1 , proving the claim.

(2). We only show the proof for the case of δ &gt; 0 , while the case of δ &lt; 0 can be handled similarly. It follows from the monotonicity that φ δ ≤ φ ⋆ . Let φ δ /φ ⋆ = 1 -¯ δ for some ¯ δ = ¯ δ ( δ ) ≥ 0 . Our aim is to derive an upper bound for ¯ δ .

It follows from (43) that at least one of the following holds:

<!-- formula-not-decoded -->

Case 1 : If σ 2 1 = σ 2 1 ∨ φ ⋆ , we have σ 2 1 = σ 2 1 ∨ φ δ . Then it holds that

<!-- formula-not-decoded -->

For the case when φ ⋆ = σ 2 1 ∨ φ ⋆ , we can compute that

<!-- formula-not-decoded -->

By combining the above results, in Case 1, we can conclude that ¯ δ ≤ Kδ .

Case 2 : Note that in this case, we have

<!-- formula-not-decoded -->

This implies that when φ ⋆ +∆ a = σ 2 a ∨ ( φ ⋆ +∆ a ) ,

<!-- formula-not-decoded -->

For the case when σ 2 a = σ 2 a ∨ ( φ ⋆ +∆ a ) , using (45) and the fact that φ ⋆ ≥ T -1 leads to

<!-- formula-not-decoded -->

which entails that φ ⋆ / ( φ ⋆ +∆ a ) ≥ ( σ 1 ∨ T -1 / 2 ) / ( σ a √ K -1) . Hence, we have that

<!-- formula-not-decoded -->

This gives the bound that corresponds to σ a / ( σ 1 ∨ T -1 / 2 ) in (44).

Next, we will derive the bound corresponding to ( σ 2 a / ( T ∆ 2 a ) -1) -1 + and (1 / ( K -1) -σ 2 a / ( T ∆ 2 a )) -1 + in (44). The analysis will be further separated into two cases: (i) 1 / ( K -1) -σ 2 a / ( T ∆ 2 a ) ≥ 0 and (ii) 1 -σ 2 a / ( T ∆ 2 a ) &lt; 0 .

For case (i), notice that

Thus, we can arrive at

<!-- formula-not-decoded -->

For case (ii), let us recall from (46) that

<!-- formula-not-decoded -->

Observe that when ∆ a /φ ⋆ &gt; 2 (or equivalently φ ⋆ / ∆ a &lt; 1 / 2 ),

<!-- formula-not-decoded -->

Using the condition that 1 -σ 2 a / ( T ∆ 2 a ) &lt; 0 , we can deduce that

<!-- formula-not-decoded -->

As σ 2 a / ( φ ⋆ +∆ a ) 2 ≤ T and φ ⋆ / ∆ a &lt; 1 / 2 , we can obtain that σ 2 a / ( T ∆ 2 a ) ≤ 9 / 4 , which then leads to ( φ ⋆ ) -1 ∆ a ≤ 27 / (2 σ 2 a / ( T ∆ 2 a ) -2) . Hence, it follows that

<!-- formula-not-decoded -->

Plugging this back to (47) yields that

<!-- formula-not-decoded -->

By combining the above results, in Case 2, we can conclude that

<!-- formula-not-decoded -->

The claim then follows by combining the estimates in Cases 1 and 2 above.

Proof of Theorem 4. The proof of Theorem 4 closely mirrors that of Theorem 1, and we spell out some of the details for the convenience of readers. We start by noticing that most of the derivations in Section D.2 hold for the K -armed setting. Specifically, from (27) we have that

<!-- formula-not-decoded -->

Combining (48) above with the stability result in Lemma 7 concludes the proof.

Proof of Theorem 5. See the proof of Theorem 2 in Section E.2.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let a ⋆ ≡ arg max a ≥ 2 ( σ 2 a ∨ ( φ ⋆ +∆ a )) / ( T ( φ ⋆ +∆ a ) 2 ) . Then we can show that

<!-- formula-not-decoded -->

## G Auxiliary results

Lemma 8. Assume that X 1 , X 2 , . . . are i.i.d. bounded random variables in [ -1 , 1] with zero mean and variance σ 2 . Let ¯ X t ≡ ∑ s ∈ [ t ] X s /t be the empirical mean. Then we have that for any δ &lt; 1 / 2 ,

<!-- formula-not-decoded -->

Proof. As in previous works, our proof relies on constructing a supermartingale and applying Doob's inequality. Let F t be the filtration generated by X 1 , . . . , X t and S t ≡ ∑ s ∈ [ t ] X s . For any η ∈ [0 , 1] ,

<!-- formula-not-decoded -->

Here, in the first inequality above, we have used the fact that E [exp( ηX t )] ≤ 1 + η 2 σ 2 a ; see, e.g., [17, Lemma 7].

With γ i ≡ i -1 ( i +1) -1 and η i ∈ [0 , 1] to be determined, let us define

<!-- formula-not-decoded -->

Then in light of (49), it holds that

<!-- formula-not-decoded -->

implying that { Z t } t ≥ 0 is sequence of supermartingale. Note that by invoking a similar argument as in [23, Proof of Lemma 5.1],

<!-- formula-not-decoded -->

For any fixed δ &gt; 0 , set

<!-- formula-not-decoded -->

where i 1 ≡ inf { i ∈ Z ≥ 1 : 2 ¯ i ≥ σ -2 a (log( ¯ i ( ¯ i +1)) + log(1 /δ )) holds for ∀ ¯ i ≥ i } .

Given these parameters, the analysis is further divided into two cases: σ a ≥ 1 / √ T and σ a &lt; 1 / √ T .

Case 1: For σ a ≥ 1 / √ T , it holds that

<!-- formula-not-decoded -->

For those t satisfying ⌈ log 2 t ⌉ ≥ i 1 , we can choose i = ⌈ log 2 t ⌉ to obtain

<!-- formula-not-decoded -->

For those t satisfying log 2 t ≤ i 1 -1 , it follows from the definition of i 1 that t &lt; 2 i 1 and 2 i 1 -1 ≥ σ -2 (log( i 1 ( i 1 -1)) + log(1 /δ )) . Hence, we can deduce that

<!-- formula-not-decoded -->

Below we will derive an upper bound for i 1 . Let us consider functions f ( x ) = 2 x and g ( x ) = σ -2 log( x ( x +1)) + log(1 /δ ) . It is easy to show that

<!-- formula-not-decoded -->

Then we have f ′ ( x ) &gt; g ′ ( x ) as long as x &gt; log(2 T ) + 1 . On the other hand, for x 0 = 4log(1 /δ ) + 4 log T +1 , it follows from σ -2 ≤ T that

<!-- formula-not-decoded -->

which entails that f ( x ) &gt; g ( x ) holds for all x &gt; x 0 . Thus, we have i 1 ≤ x 0 = 4log( T/δ ) + 1 .

Combining this upper bound of i 1 with (51), we can obtain that

<!-- formula-not-decoded -->

Case 2: For σ &lt; 1 / √ T , we can deduce that

<!-- formula-not-decoded -->

The claim follows.

As a corollary of Lemma 8, we have the following results on the concentration of the empirical mean and variance: for any δ ≥ 0 , let

<!-- formula-not-decoded -->

Proposition 7. For all δ &lt; 1 / 2 and a ∈ [ K ] , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. The claim in (52) follows from a direct application of Lemma 8 with the following simplification for the error bound therein

<!-- formula-not-decoded -->

To prove (53), we first notice that by | µ a -¯ X a,t | ≤ 1 ,

<!-- formula-not-decoded -->

Then we can bound the second term above using (52). The first term above can be analyzed by the same argument as in proving (52), upon noting that Var(( X a,s -µ a ) 2 -σ 2 a ) ≤ E (( X a,s -µ a ) 4 ) ≤ σ 2 a . This concludes the proof.

Lemma 9. Assume that { X i } i ≥ 1 i.i.d. ∼ Bernoulli (1 / 2) and fix any c &gt; 0 . Then there exist some N 0 ∈ Z + and constant c anti = c anti ( c ) &gt; 0 such that for all N ≥ N 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Our proof relies on Donsker's Theorem, which states the convergence (in distribution) of the random walk to the continuous-time Brownian motion; see, e.g., [6, Section 14]. More precisely, for { X i } i ≥ 1 i.i.d. ∼ Bernoulli (1 / 2) , Donsker's Theorem states that for

<!-- formula-not-decoded -->

it holds uniformly in t ∈ [0 , 1] that lim N →∞ Ψ N ( t ) d = B ( t ) , where B ( t ) represents the standard Brownian motion. Equivalently, for any ε &gt; 0 , there exists some N 0 ∈ Z + such that for all N ≥ N 0 ,

<!-- formula-not-decoded -->

Using the fact that inf ⌊ N/ c ⌋≤ n ≤ N 2( ∑ i ∈ [ n ] X i -n/ 2) / N ≥ inf t ∈ [1 / c , 1] Ψ N ( t ) , we can deduce that for any x ∈ R ,

<!-- formula-not-decoded -->

As B ( t + c -1 ) d = Z + ¯ B ( t ) for another standard Brownian motion ¯ B ( t ) and Z ∼ N (0 , c -1 ) independent from ¯ B ( t ) , we can further bound the right-hand side (RHS) of the inequality in the above display as follows. Denote by Φ the cumulative distribution function (CDF) of the standard Gaussian distribution. Then we can show that

<!-- formula-not-decoded -->

Here, in step (i) above we have used the symmetry property of the Brownian motion, in step (ii) above we have used the independence between ¯ B ( t ) and Z , and in step (iii) above we have used the reflection principle.

Now choosing

<!-- formula-not-decoded -->

finishes the proof for (54). The proof for (55) is nearly the same due to the symmetric property of the Brownian motion, so will be omitted here.

## H Numerical simulations

This section summarizes the settings of all numerical experiments reported in the main text and specifies the reward distributions. Unless otherwise specified, we consider two-armed bandit instances and take arm 1 to be the optimal arm.

Common numerical settings for generating reward distributions. In all experiments, we use a Beta( α, β ) distribution to generate rewards. Given a desired mean µ and variance σ 2 , and subject to the boundedness constraint on [0 , 1] , we set

<!-- formula-not-decoded -->

Note that an implicit constraint on ( µ, σ 2 ) is σ 2 ≤ µ (1 -µ ) ; this constraint is satisfied by all reward distributions used below.

Figure 1: Distributions of n 1 ,T . In both experiments that plot histograms of arm-pull counts, we set the time horizon to T = 50 , 000 and the number of repetitions to R = 5 , 000 . The exploration hyperparameter is ρ = 2 for both UCB-V and UCB. The means and variances are set to µ 1 = µ 2 = 1 2 and σ 1 = σ 2 = 1 4 in Figure 1(a), and to σ 1 = 0 , σ 2 = 1 4 , and ∆ = σ 2 √ (log T ) /T in Figure 1(b).

Figure 2: Regret and phase transition of optimal-arm pulls. For panel (a), we set the times of repetition as 10 , exploration hyper-parameter ρ = 2 . We vary σ 1 ∈ { T -1 / 2 , T -1 / 4 , 1 } while keeping σ 2 = T -1 / 4 and µ 1 = 1 2 , ∆ 2 = T -1 / 2 , µ 2 = µ 1 +∆ 2 fixed across curves. We plot the realized regret Reg( T ) versus T and observe the trend toward O ( ( σ 2 2 /σ 1 ) √ T ) as σ 1 increases. For panel (b), we set T = 1 , 000 , 000 , and repetition time R = 30 , ρ = 2 , and fix σ 1 = 0 and σ 2 = 1 4 . We sweep Λ T = σ 2 √ ρ log T/ ( √ T ∆ 2 ) by varying ∆ 2 ; for each value we plot the median and the 30% quantile of n 1 ,T for UCB and UCB-V. The red dotted curve shows the numerical solution n ⋆ 1 ,T from (6), revealing a transition from sublinear to linear n 1 ,T around Λ T ≈ 1 .

## I Further discussion on inference with UCB-V

The asymptotic normality of the Z -statistic in (8) is established via the martingale CLT. Due to the let us consider the filtration F s generated by { X 1 , . . . , X s } . The sequence { 1 { a s = a } ( X s -µ a ) σ √ n ⋆ } s ∈ [ T ]

forms a martingale difference sequence. Moreover, the Lindeberg condition is satisfied, and thus it holds that ∑ s ∈ [ T ] E ( 1 { a s = a } ( X s -µ a ) 2 σ 2 a n ⋆ a,T ∣ ∣ F s -1 ) p - → 1 , gale CLT to establish the asymptotic normality of the Z -statistic. For a detailed derivation, see [23, Section 2.1].

adaptive nature of data collection in the bandit algorithms, traditional inference techniques based on independent and identically distributed (i.i.d.) data cannot be applied directly. However, for stable arms, a a,T which allows for the application of the martin-

To illustrate the implication of our results on the reward inference, we conduct a simulation study on the Z -statistic for UCB and UCB-V using the setting from Example 2, as shown in Figure 4. In Figure 4a, the empirical distributions of the Z -statistic for both UCB and UCB-V approximate the standard Gaussian, matching the theoretical predictions of our result in Example 2, where UCB-V is stable in this regime. In Figure 4b, the empirical distribution of the Z -statistic for UCB-V shows a noticeable bias compared to UCB. This suggests that the previously mentioned martingale CLT result no longer holds for UCB-V when Λ T = 1 . In the subsequent section, we show that the underlying reason for this deviation from the CLT is the instability of UCB-V under condition Λ T = 1 .

## J Discussion on the modification of UCB-V

In our presentation of Algorithm 1, there are several differences compared to those proposed in [4]. In [4], the UCB reward for each action a is set as

<!-- formula-not-decoded -->

Figure 4: The empirical distributions of the Z -statistic for the sub-optimal arm for UCB and UCB-V with σ 1 = 0 , σ 2 = 1 / 4 under different Λ T , both with T = 50 , 000 over 2 , 000 repetitions.

<!-- image -->

where c, ζ &gt; 0 are tunable constants. Specifically, their regret guarantee is established with c, ζ large enough. In our statement, to simplify the notation in our asymptotic analysis, we choose c and ζ such that for some ρ &gt; 0 , it holds that 2 ζ = 3 cζ = ρ , replacing log t with log T and modifying

<!-- formula-not-decoded -->

We make such modification for notational simplicity and facilitating comparison with results for the canonical UCB [21, 23]. By setting ̂ σ a,t = 1 instead of estimating it via the sample variance in the algorithm's input, our analysis shows that the asymptotic behavior of the modified algorithm becomes equivalent to that of the canonical UCB.

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1-2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

## IMPORTANT, please:

- Delete this instruction block, but keep the section heading 'NeurIPS Paper Checklist" ,
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers .

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Contributions of the paper are accurately stated in the abstract and introduction. Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Potential limitations are discussed in the Sections 3 and 4.

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

Justification: All assumptions are stated in their respective places.

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

Justification: The paper includes only simple simulations, and the settings are described in detail.

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

## Answer: [No]

Justification: The paper includes only simple simulations, and their settings are already described in detail.

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

Justification: The simulation settings are clearly described in the text.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: The goal of the simulations is to illustrate theoretical insights but not superior to some benchmark.

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

Answer: [No]

Justification: The simulations in the paper are simple and can be executed on a standard laptop.

Guidelines:

- The answer NA means that the paper does not include experiments.

- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: This theoretical work involves no sensitive data or human subjects and fully complies with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This is a theoretical study of a bandit algorithm, and we are not aware of any immediate societal impacts.

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

Justification: The paper does not release any models or datasets that pose a risk for misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: This work does not use any existing code, data, or other external assets.

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

Justification: This work does not involve crowdsourcing or research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

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