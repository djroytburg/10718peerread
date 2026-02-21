## Finite Sample Analysis of Linear Temporal Difference Learning with Arbitrary Features

## Zixuan Xie ∗

University of Virginia xie.zixuan@email.virginia.edu

## Rohan Chandra

University of Virginia rohanchandra@virginia.edu

## Abstract

Linear TD( λ ) is one of the most fundamental reinforcement learning algorithms for policy evaluation. Previously, convergence rates are typically established under the assumption of linearly independent features, which does not hold in many practical scenarios. This paper instead establishes the first L 2 convergence rates for linear TD( λ ) operating under arbitrary features, without making any algorithmic modification or additional assumptions. Our results apply to both the discounted and average-reward settings. To address the potential non-uniqueness of solutions resulting from arbitrary features, we develop a novel stochastic approximation result featuring convergence rates to the solution set instead of a single point.

## 1 Introduction

Temporal difference learning (TD, Sutton [1988]) is a fundamental algorithm in reinforcement learning (RL, Sutton and Barto [2018]), enabling efficient policy evaluation by combining dynamic programming [Bellman, 1966] with stochastic approximation (SA, Benveniste et al. [1990], Kushner and Yin [2003], Borkar [2009]). Its linear variant, linear TD( λ ) [Sutton, 1988], emerges as a practical extension, employing linear function approximation to tackle large or continuous state spaces where tabular representations become impractical. Linear TD( λ ) takes the dot product between features and weights to compute the approximated value. Establishing theoretical guarantees for linear TD( λ ), particularly convergence rates, has been a major focus of research. Most existing works (Table 1), however, require the features used in linear TD to be linearly independent. As argued in Wang and Zhang [2024], this assumption is impractical in many scenarios. For example, in continual learning with sequentially arriving data [Ring, 1994, Khetarpal et al., 2022, Abel et al., 2023], there is no way to rigorously verify whether the features are independent or not. See Wang and Zhang [2024] for more discussion on the restrictions of the feature independence assumptions. Furthermore, Dayan [1992], Tsitsiklis and Roy [1996, 1999] also outline the elimination of the linear independence assumption as a future research direction.

While efforts have been made to eliminate the linear independence assumption [Wang and Zhang, 2024], they only provide asymptotic (almost sure) convergence guarantees in the discounted setting. By contrast, this paper establishes the first L 2 convergence rates for linear TD( λ ) with arbitrary features in both discounted and average-reward settings . This success is enabled by a novel stochastic approximation result (Theorem 3) concerning the convergence rates to a solution set instead of a single point, driven by a novel Lyapunov function. This new result provides a unified approach

∗ Equal contribution

## Xinyu Liu ∗

University of Virginia xinyuliu@virginia.edu

## Shangtong Zhang

University of Virginia shangtong@virginia.edu

applicable to both discounted (Theorem 1) and average-reward (Theorem 2) settings. Notably, we do not make any algorithmic modification and do not introduce any additional assumptions. Table 1 provides a detailed comparison of existing theoretical analyses for linear TD( λ ), contextualizing our contributions within the landscape of prior work.

Table 1: Comparison of finite-sample analyses for linear TD( λ ). 'Setting' indicates the problem setting: γ &lt; 1 stands for the discounted setting and γ = 1 stands for the average reward setting. 'Features' describes assumptions on the features. 'Independent' indicates linear independence is assumed. 'Arbitrary' indicates no assumption is made on features. 'Noise Type' indicates the data generation process: Markovian samples or independent and identically distributed (i.i.d.) samples. 'Rate' is checked if a convergence rate is provided.

|                                                                                                                                                                          | Setting                                   | Features                                                                                    | Noise Type                                                                | Rate        |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|-------------|
| Tsitsiklis and Roy [1996] Bhandari et al. [2018] Lakshminarayanan and Szepesvári [2018] Srikant and Ying [2019] Wang and Zhang [2024] Chen et al. [2025a] Mitra [2025] 1 | γ < 1 γ < 1 γ < 1 γ < 1 γ < 1 γ < 1 γ < 1 | Independent Independent Independent Independent Arbitrary Independent Independent Arbitrary | Markovian Markovian i.i.d. Markovian Markovian i.i.d. Markovian Markovian | ✓ ✓ ✓ ✓ ✓ ✓ |
| Tsitsiklis and Roy [1999] Zhang et al. [2021c]                                                                                                                           | γ = 1                                     | Independent Independent                                                                     | Markovian Markovian                                                       | ✓           |
| Chen et al. [2025b]                                                                                                                                                      |                                           |                                                                                             |                                                                           |             |
| Theorem                                                                                                                                                                  | γ < 1                                     |                                                                                             |                                                                           |             |
|                                                                                                                                                                          | γ = 1                                     | Independent                                                                                 | Markovian                                                                 |             |
|                                                                                                                                                                          | γ = 1                                     |                                                                                             |                                                                           | ✓           |
| Theorem 2                                                                                                                                                                | γ = 1                                     | Arbitrary                                                                                   | Markovian                                                                 | ✓           |

## 2 Background

Notations. We use ⟨ x, y ⟩ . = x ⊤ y to denote the standard inner product in Euclidean spaces and ∥·∥ to denote the ℓ 2 norm for vectors and the associated induced operator norm (i.e., the spectral norm) for matrices, unless stated otherwise. A function f is said to be L -smooth (w.r.t. ∥·∥ ) if ∀ w,w ′ , f ( w ′ ) ≤ f ( w ) + ⟨∇ f ( w ) , w ′ -w ⟩ + L 2 ∥ w ′ -w ∥ 2 . For a matrix A , col( A ) denotes its column space, ker( A ) denotes its kernel, and A † denotes its Moore-Penrose inverse. When x is a point and U is a set, we denote d ( x, U ) . = inf y ∈ U ∥ x -y ∥ as the Euclidean distance from x to U . For sets U, V , their Minkowski sum is U + V . = { u + v | u ∈ U, v ∈ V } ; and U ⊥ denotes the orthogonal complement of U . We use 0 and 1 to denote the zero vector and the all-ones vector respectively, where the dimension is clear from context. For any square matrix A ∈ R d × d (not necessarily symmetric), we say A is negative definite (n.d.) if there exists a ξ &gt; 0 such that x ⊤ Ax ≤ -ξ ∥ x ∥ 2 ∀ x ∈ R d . For any set E ⊆ R d , we say A is n.d. on E if there exists a ξ &gt; 0 such that x ⊤ Ax ≤ -ξ ∥ x ∥ 2 ∀ x ∈ E . A is negative semidefinite (n.s.d.) if ξ = 0 in the above definition.

Markov Decision Processes. We consider an infinite horizon Markov Decision Process (MDP, Bellman [1957]) defined by a tuple ( S , A , p, r, p 0 ) , where S is a finite set of states, A is a finite set of actions, p : S × S × A → [0 , 1] is the transition probability function, r : S × A → R is the reward function, and p 0 : S → [0 , 1] denotes the initial distribution. In this paper, we focus on the policy evaluation problem, where the goal is to estimate the value function of an arbitrary policy π : A×S → [0 , 1] . At the time step 0 , an initial state S 0 is sampled from p 0 . At each subsequent time step t , the agent observes state S t ∈ S , executes an action A t ∼ π ( ·| S t ) , receives reward R t +1 . = r ( S t , A t ) , and transitions to the next state S t +1 ∼ p ( ·| S t , A t ) . We use P π to denote the state transition matrix induced by the policy π , i.e., P π [ s, s ′ ] = ∑ a ∈A π ( a | s ) p ( s ′ | s, a ) . Let d π ∈ R |S| be the stationary distribution of the Markov chain induced by the policy π . We use D π to denote the diagonal matrix whose diagonal is d π .

Linear Function Approximation. In this paper, we use linear function approximation to approximate value functions v π : S → R (to be defined shortly). We consider a feature mapping x : S → R d and a weight vector w ∈ R d . We then approximate v π ( s ) with x ( s ) ⊤ w . We use X ∈ R |S|× d to denote

the feature matrix, where the s -th row of X is x ( s ) ⊤ . The approximated state-value function across all states can then be represented as the vector Xw ∈ R |S| . The goal is thus to find a w such that Xw closely approximates v π .

Discounted Setting. In the discounted setting, we introduce a discount factor γ ∈ [0 , 1) . The (discounted) value function v π : S → R for policy π is defined as v π ( s ) . = E [∑ ∞ i =0 γ i R t + i +1 ∣ ∣ S t = s ] . We define the Bellman operator T : R |S| → R |S| as T v . = r π + γP π v , where r π ∈ R |S| is the vector of expected immediate rewards under π , with components r π ( s ) = ∑ a π ( a | s ) r ( s, a ) . With a λ ∈ [0 , 1] , the λ -weighted Bellman operator T λ is defined as T λ v . = (1 -λ ) ∑ ∞ m =0 λ m T m +1 v = r λ + γP λ v , where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This represents a weighted average of multi-step applications of T . It is well-known that v π is the unique fixed point of T λ [Bertsekas and Tsitsiklis, 1996]. Linear TD( λ ) is a family of TD learning algorithms that use eligibility traces to estimate v π ( s ) of the fixed policy π with linear function approximation. The algorithm maintains a weight vector w t ∈ R d and an eligibility trace vector e t ∈ R d , with the following update rules:

<!-- formula-not-decoded -->

Here, { α t } is the learning rate. The eligibility trace e t tracks recently visited states, assigning credit for the prediction error to multiple preceding states. Let

<!-- formula-not-decoded -->

If X has a full column rank, Tsitsiklis and Roy [1996] proves that W ∗ is a singleton and { w t } converge to -A -1 b almost surely. A key result used by Tsitsiklis and Roy [1996] is that the matrix D π ( γP λ -I ) is n.d. [Sutton, 1988]. As a result, the A matrix is also n.d. when X has a full column rank. Wang and Zhang [2024] prove, without making any assumption on X , that W ∗ is always nonempty and the { w t } converges to W ∗ almost surely. A key challenge there is that without making assumptions on X , A is only n.s.d.

Average-Reward Setting. In the average-reward setting, the overall performance of a policy π is measured by the average reward J π . = lim T →∞ 1 T E [ ∑ T -1 t =0 R t ] . The corresponding (differential) value function is defined as v π ( s ) = lim T →∞ 1 T ∑ T -1 i =0 E [( r ( S t + i , A t + i ) -J π ) | S t = s ] . We define the Bellman operator T : R |S| → R |S| as T v . = r π -J π 1 + P π v . Similarly, the λ -weighted counterpart T λ is defined as T λ v . = r λ -J π 1 -λ 1 + P λ v . Although v π is a fixed point of T λ , it is not the unique fixed point. In fact,

<!-- formula-not-decoded -->

are all the fixed points of T λ [Puterman, 2014]. Linear average-reward TD( λ ) is an algorithm for estimating both J π and v π using linear function approximation and eligibility traces. The update rules are

<!-- formula-not-decoded -->

where { α t } and { β t } are learning rates. Let

<!-- formula-not-decoded -->

If X has a full column rank and 1 / ∈ col( X ) , Tsitsiklis and Roy [1999] proves that W ∗ is a singleton and { w t } converge to -A -1 b almost surely. This is made possible by an important fact from the Perron-Frobenius theorem (see, e.g., Seneta [2006]) that

<!-- formula-not-decoded -->

Zhang et al. [2021c] further provides a convergence rate, still assuming X has a full column rank but without assuming 1 / ∈ col( X ) . When X does not have a full column rank, to our knowledge, it is even not clear whether W ∗ is always nonempty or not, much less the behavior of { w t } .

## 3 Main Results

We start with our assumptions. As promised, we do not make any assumption on X .

Assumption 3.1. The Markov chain associated with P π is irreducible and aperiodic.

Assumption LR. The learning rates are α t = α ( t + t 0 ) ξ and β t = c β α t , where ξ ∈ (0 . 5 , 1] , α &gt; 0 , t 0 &gt; 0 , and c β &gt; 0 are constants.

Discounted Setting. Wang and Zhang [2024] proves the almost sure convergence of (Discounted TD) with arbitrary features by using ∥ w -w ∗ ∥ 2 with an arbitrary and fixed w ∗ ∈ W ∗ as a Lyapunov function and analyzing the property of the ODE d w ( t ) d t = Aw ( t ) . Since A is only n.s.d., Wang and Zhang [2024] conducts their analysis in the complex number field. In this work, instead of following the ODE-based analysis originating from Tsitsiklis and Roy [1996], Borkar and Meyn [2000], we extend Srikant and Ying [2019] to obtain convergence rates by using d ( w,W ∗ ) 2 as the Lyapunov function. To our knowledge, this is the first time that such distance function to a set is used as the Lyapunov function to analyze RL algorithms, which is our key technical contribution from the methodology aspect. According to Theorem 1 of Wang and Zhang [2024], W ∗ is nonempty, and apparently convex and closed. 2 Let Γ( w ) . = arg min w ∗ ∈ W ∗ ∥ w -w ∗ ∥ be the orthogonal projection to W ∗ . We then define L ( w ) . = 1 2 d ( w,W ∗ ) 2 = 1 2 ∥ w -Γ( w ) ∥ 2 . Two important and highly non-trivial observations are

- (i) ∇ L ( w ) = w -Γ( w ) (Example 3.31 of Beck [2017]),
- (ii) L ( w ) is 1-smooth w.r.t. ∥·∥ (Example 5.5 of Beck [2017]).

Both (i) and (ii) result from the fact that W ∗ is nonempty, closed, and convex. Using L ( w ) as the Lyapunov function together with more characterization of ∇ L ( w ) (Section 5.2), we obtain

Theorem 1. Let Assumptions 3.1 and LR hold and λ ∈ [0 , 1] . Then for sufficiently large t 0 and α , there exist some constants C Thm1 and κ 1 . = αC 7 &gt; 1 such that the iterates { w t } generated by (Discounted TD) satisfy for all t

<!-- formula-not-decoded -->

The proof is in Section 5.2. Notably, Lemma 3 of Wang and Zhang [2024] states that for any w ∗ , w ∗∗ ∈ W ∗ , it holds that Xw ∗ = Xw ∗∗ . We then define

<!-- formula-not-decoded -->

for any w ∗ ∈ W ∗ . Theorem 1 then also gives the L 2 convergence rate of the value estimate, i.e., the rate at which Xw t converges to ˆ v π . The value estimate ˆ v π is the unique fixed point of a projected Bellman equation. See Wang and Zhang [2024] for more discussion on the property of ˆ v π . Additionally, by choosing a sufficiently large α , we can ensure ⌊ κ 1 ⌋ -1 ≥ 2 ξ -1 , so the rate is determined by the exponent 2 ξ -1 . For the standard choice ξ = 1 , the resulting rate becomes O (ln t/t ) , which matches existing analyses that assume linearly independent features [Bhandari et al., 2018, Srikant and Ying, 2019]. An analogous observation holds for Theorems 2 and 3 as well, since their corresponding κ is also proportional to α .

Average Reward Setting. Characterizing W ∗ is much more challenging. We first present a novel decomposition of the feature matrix X . To this end, define m . = rank( X ) ≤ min {|S| , d } . If m = 0 , all the results in this work are trivial and we thus discuss only the case m ≥ 1 .

Lemma 1. There exist matrices X 1 , X 2 such that X = X 1 + X 2 with the following properties (1) rank( X 1 ) = m -I 1 ∈ col( X ) and 1 / ∈ X 1 (2) X 2 = 1 θ ⊤ with θ ∈ R d .

The proof is in Section B.1 with I being the indicator function. Essentially, X 2 is a rank one matrix with identical rows θ (i.e., the i -th column of X 2 is θ i 1 ). To our knowledge, this is the first time that such decomposition is used to analyze average-reward RL algorithms, which is our second

2 This theorem only discusses the case of λ = 0 . The proof for a general λ ∈ [0 , 1] is exactly the same up to change of notations.

technical contribution from the methodology aspect. This decomposition is useful in three aspects. First, we have A = X ⊤ 1 D π ( P λ -I ) X 1 (Lemma 14). Second, this decomposition is the key to prove that W ∗ is nonempty (Lemma 15). Third, this decomposition is the key to characterize W ∗ in that W ∗ = { w ∗ } +ker( X 1 ) with w ∗ being any vector in W ∗ (Lemma 16). To better understand this characterization, we note that ker( X 1 ) = { w | Xw = c 1 , c ∈ R } (Lemma 16). As a result, adding any w 0 ∈ ker( X 1 ) to a weight vector w changes the resulting value function Xw only by c 1 . Two values v 1 and v 2 can be considered 'duplication' if v 1 -v 2 = c 1 (cf. (2)). So intuitively, ker( X 1 ) is the source of the 'duplication'. With the help of this novel decomposition, we obtain

Theorem 2. Let Assumptions 3.1 and LR hold and λ ∈ [0 , 1) . Then for sufficiently large α , t 0 and c β , there exist some constants C Thm2 and κ 2 . = αC 10 &gt; 1 such that the iterates { w t } generated by (Average Reward TD) satisfy for all t

<!-- formula-not-decoded -->

The proof is in Section 5.3.

Stochastic Approximation. We now present a general stochastic approximation result to prove Theorems 1 and 2. The notations in this part are independent of the rest of the paper. We consider a general iterative update rule for a weight vector w ∈ R d , driven by a time-homogeneous Markov chain { Y t } evolving in a possibly infinite space Y :

<!-- formula-not-decoded -->

where H : R d ×Y → R d defines the incremental update.

Assumption A1. There exists a constant C A1 such that sup y ∈Y ∥ H (0 , y ) ∥ &lt; ∞ ,

<!-- formula-not-decoded -->

Assumption A2. { Y t } has a unique stationary distribution d Y .

Let h ( w ) . = E y ∼ d Y [ H ( w,y )] . Assumption A1 then immediately implies that

<!-- formula-not-decoded -->

In many existing works about stochastic approximation [Borkar and Meyn, 2000, Chen et al., 2023b, Qian et al., 2024, Borkar et al., 2025], it is assumed that h ( w ) = 0 adopts a unique solution. To work with the challenges of linear TD with arbitrary features, we relax this assumption and consider a set W ∗ . Importantly, W ∗ does not need to contain all solutions to h ( w ) = 0 . Instead, we make the following assumptions on W ∗ .

Assumption A3. W ∗ is nonempty, closed, and convex.

Notably, W ∗ does not need to be bounded. Assumption A3 ensures that the orthogonal projection to W ∗ is well defined, allowing us to define Γ( w ) . = argmin w ∗ ∈ W ∗ ∥ w -w ∗ ∥ , L ( w ) . = 1 2 ∥ w -Γ( w ) ∥ 2 . As discussed before, Assumption A3 ensures that ∇ L ( w ) = w -Γ( w ) and L is 1-smooth w.r.t. ∥·∥ [Beck, 2017]. We further assume that the expected update h ( w t ) decreases L ( w t ) in the following sense, making L ( w ) a candidate Lyapunov function.

Assumption A4. There exists a constant C A4 &gt; 0 such that almost surely,

<!-- formula-not-decoded -->

Lastly, we make the most 'unnatural' assumption of W ∗ .

Assumption A5. There exists a matrix X and constants C A5 and τ ∈ [0 , 1) such that (1) ∀ w ∗ ∈ W ∗ , ∥ Xw ∗ ∥ ≤ C A5; (2) ∀ w,y , ∥ H ( w,y ) ∥ ≤ C A5 ( ∥ Xw ∥ +1) ; (3) For any n ≥ 1 :

<!-- formula-not-decoded -->

This assumption is technically motivated but trivially holds in our analyses of (Discounted TD) and (Average Reward TD). Specifically, Assumption A1 immediately leads to at-most-linear growth

∥ H ( w,y ) ∥ ≤ C A1 , 1 ( ∥ w ∥ + 1) for some constant C A1 , 1 . However, this bound is insufficient for our analysis because ∥ w ∥ ≤ ∥ w -Γ( w ) ∥ + ∥ Γ( w ) ∥ but Γ( w ) ∈ W ∗ can be unbounded. By Assumption A5, we can have ∥ Xw ∥ ≤ ∥ Xw -X Γ( w ) ∥ + ∥ X Γ( w ) ∥ ≤ ∥ X ∥∥ w -Γ( w ) ∥ + C A5. The inequality (7) is related to geometrical mixing of the chain and we additionally include Xw in the bound for the same reason. We now present our general results regarding the convergence rate of (SA) to W ∗ .

Theorem 3. Let Assumptions A1 - A5 and LR hold. Denote κ . = αC A4, then there exist some constants t 0 and C Thm3, such that the iterates { w t } generated by (SA) satisfy for all t

<!-- formula-not-decoded -->

The proof is in Section 5.1. We remark that once we have the recursion in Lemma 5, our theoretical framework can be readily extended to the constant step-size setting (akin to Chen et al. [2023b]), demonstrating its broad applicability.

## 4 Related Works

Most prior works regarding the convergence of linear TD summarized in Table 1 rely on having linearly independent features. In fact, the reliance on feature independence goes beyond linear TD and exists in almost all previous analyses of RL algorithms with linear function approximation, see, e.g., Sutton et al. [2008, 2009], Maei [2011], Hackman [2012], Bo et al. [2015], Yu [2015, 2016], Zou et al. [2019], Yang et al. [2019], Zhang et al. [2020b], Xu et al. [2020a], Zhang et al. [2020a], Xu et al. [2020b], Wu et al. [2020], Chen et al. [2021], Long et al. [2021], Qiu et al. [2021], Zhang et al. [2021a,b], Xu et al. [2021], Zhang et al. [2022], Zhang and Whiteson [2022], Zhang et al. [2023], Chen et al. [2023a], Nicolò et al. [2024], Yue et al. [2024], Swetha et al. [2024], Liu et al. [2025a], Qian and Zhang [2025], Maity and Mitra [2025], Yang et al. [2025], Chen et al. [2025b], Shaan and Siva [2025], Liu et al. [2025c]. But as argued by Dayan [1992], Tsitsiklis and Roy [1996, 1999], Wang and Zhang [2024], relaxing this assumption is an important research direction. This work can be viewed as an extension of Wang and Zhang [2024], Zhang et al. [2021c]. In terms of (Discounted TD), we extend Wang and Zhang [2024] by proving a finite sample analysis. Though we rely on the characterization of W ∗ from Wang and Zhang [2024], the techniques we use for finite sample analysis are entirely different from the techniques of Wang and Zhang [2024] for almost sure asymptotic convergence. In terms of (Average Reward TD), we extend Zhang et al. [2021c] by allowing X to be arbitrary. Essentially, key to Zhang et al. [2021c] is their proof that A is n.d. on a subspace E , assuming X has a full column rank. We extend Zhang et al. [2021c] in that we give a finer and more detailed characterization of the counterparts of their E through the novel decomposition of the features (Lemma 1) and establish the n.d. property under weaker conditions (i.e., without assuming X has a full column rank). Importantly, despite relaxing the feature-independence assumption, our convergence rate remains on par with existing finite-sample results obtained under full-rank features [Bhandari et al., 2018, Srikant and Ying, 2019]. Our improvements are made possible by the novel Lyapunov function L ( w ) and we argue that this Lyapunov function can be used to analyze many other linear RL algorithms with arbitrary features.

In terms of stochastic approximation, our Theorem 3 is novel in that it allows convergence to a possibly unbounded set. By contrast, most prior works about stochastic approximation study convergence to a point [Borkar and Meyn, 2000, Chen et al., 2020, Zhang et al., 2022, Chen et al., 2023b, Qian et al., 2024, Liu et al., 2025a, Borkar et al., 2025, Chen et al., 2025a]. In the case of convergence to a set, most prior works require the set to be bounded [Kushner and Yin, 2003, Borkar, 2009, Liu et al., 2025a,b]. Only a few prior works allow stochastic approximation to converge to an unbounded set, see, e.g., Bravo and Cominetti [2024], Chen [2025], Blaser and Zhang [2025], which apply to only tabular RL algorithms.

## 5 Proofs of the Main Results

## 5.1 Proof of Theorem 3

Proof. From the 1-smoothness of L ( w ) and (SA), we can get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We then bound the RHS one by one. ⟨ w -Γ( w ) , h ( w ) ⟩ is already bounded in Assumption A4.

Lemma 2. There exists a positive constant C 2, such that for any w ,

<!-- formula-not-decoded -->

The proof is in Section C.1. With Lemma 2 and Assumption A5, the last term in (8) can be bounded easily.

Lemma 3. There exists a constant C 3 such that ∥ H ( w t , Y t ) ∥ 2 ≤ C 3 ( ∥ w t -Γ( w t ) ∥ 2 +1) .

The proof is in Section C.2. To bound ⟨ w t -Γ( w t ) , H ( w t , Y t ) -h ( w t ) ⟩ , leveraging (7), we define

<!-- formula-not-decoded -->

as the number of steps that the Markov chain needs to mix to an accuracy α . In addition, we denote a shorthand α t 1 ,t 2 . = ∑ t 2 i = t 1 α i . Then with techniques from Srikant and Ying [2019], we obtain

Lemma 4. There exists a constant C 4 such that

<!-- formula-not-decoded -->

The proof is in Section C.3. Plugging all the bounds back to (8), we obtain

Lemma 5. There exists some D t = O ( α t α t -τ αt ,t -1 ) , such that

<!-- formula-not-decoded -->

The proof is in Section C.4. Recursively applying Lemma 5 then completes the proof of Theorem 3 (See Section C.5 for details).

In the following sections, we first map the general update (SA) to (Discounted TD) and (Average Reward TD) by defining H ( w,y ) , h ( w ) , and L ( w ) properly. Then we bound the remaining term ⟨∇ L ( w t ) , h ( w t ) ⟩ to complete the proof.

## 5.2 Proof of Theorem 1

Proof. We first rewrite (Discounted TD) in the form of (SA). To this end, we define Y t +1 . = ( S t , A t , S t +1 , e t ) , which evolves in an infinite space Y . = S × A × S × { e | ∥ e ∥ ≤ C e } with C e . = max s ∥ x ( s ) ∥ 1 -γλ being the straightforward bound of sup t ∥ e t ∥ . We define the incremental update H : R d ×Y → R d as

<!-- formula-not-decoded -->

using shorthand y = ( s, a, s ′ , e ) . We now proceed to verifying the assumptions of Theorem 3. Assumption A1 is verified by the following lemma.

Lemma 6. There exists some finite C 6 such that

<!-- formula-not-decoded -->

Moreover, sup y ∈Y ∥ H (0 , y ) ∥ &lt; ∞ .

The proof is in Section D.1.

For Assumption A2, Theorem 3.2 of Yu [2012] confirms that { Y t } has a unique stationary distribution d Y . Yu [2012] also computes that

<!-- formula-not-decoded -->

Assumption A3 trivially holds by the definition of W ∗ .

For Assumption A4, the key observation is that A Γ( w ) + b = 0 always holds because Γ( w ) ∈ W ∗ . Then we have h ( w ) = Aw + b = ( Aw + b ) -( A Γ( w ) + b ) = A ( w -Γ( w )) . Thus the term ⟨∇ L ( w ) , h ( w ) ⟩ can be written as ( w -Γ( w )) ⊤ A ( w -Γ( w )) . We now prove that for whatever X , it always holds that A is n.d. on ker( A ) ⊥ .

Lemma 7. There exists a constant C 7 &gt; 0 such that for ∀ w ∈ ker( A ) ⊥ , w ⊤ Aw ≤ -C 7 ∥ w ∥ 2 . Furthermore, for any w ∈ R d , it holds that w -Γ( w ) ∈ ker( A ) ⊥ .

The proof is in Section D.3. We then have

<!-- formula-not-decoded -->

which satisfies Assumption A4.

For Assumption A5, (6) verifies Assumption A5(1). Assumption A5(2) is verified by the following lemma.

Lemma 8. There exists a constant C 8 such that for ∀ w,y , ∥ H ( w,y ) ∥ ≤ C 8 ( ∥ Xw ∥ +1) .

The proof is in Section D.4. Assumption A5(3) is verified following a similar procedure as Lemma 6.7 in Bertsekas and Tsitsiklis [1996] (Lemma 18). Invoking Theorem 3 then completes the proof.

## 5.3 Proof of Theorem 2

Proof. We recall that in view of Lemma 1, ker( X 1 ) creates 'duplication' in value estimation. We, therefore, define the projection matrix Π ∈ R d × d that projects a vector into the orthogonal complement of ker( X 1 ) , i.e., Π w . = arg min w ′ ∈ ker( X 1 ) ⊥ ∥ w -w ′ ∥ . It can be computed that Π = X † 1 X 1 . We now examine the sequence { Π w t } with { w t } being the iterates of (Average Reward TD) and consider the combined parameter vector ˜ w t . = [ ˆ J t Π w t ] ∈ R 1+ d . The following lemma characterizes the evolution of ˜ w t . Let Y t = ( S t , A t , S t +1 , e t ) ∈ S × A × S × { e ∈ R d | ∥ e ∥ ≤ max s ∥ x ( s ) ∥ 1 -λ } , then

Lemma 9. ˜ w t +1 = ˜ w t + α t ( ˜ A ( Y t ) ˜ w t + ˜ b ( Y t )) , where we have, with y = ( s, a, s ′ , e ) ,

<!-- formula-not-decoded -->

This view is inspired by Zhang et al. [2021c] and the proof is in Section E.1. We now apply Theorem 3 to { ˜ w t } .

The verification of Assumptions A1 and A2 is identical to that in Section 5.2 and is thus omitted.

For Assumption A3, we define ˜ W ∗ . = {[ J π Π w ]∣ ∣ ∣ ∣ w ∈ W ∗ } . It is apparently nonempty, closed, and convex.

For Assumption A4, we define ˜ A . = E y ∼ d Y [ ˜ A ( y ) ] and ˜ b . = E y ∼ d Y [ ˜ b ( y ) ] and therefore realize the h in (SA) as h ( ˜ w ) = ˜ A ˜ w + ˜ b . Noticing that ˜ A Γ( ˜ w ) + ˜ b = 0 (Lemma 19), we then have h ( ˜ w ) = ˜ A ( ˜ w -Γ( ˜ w )) . The term ⟨∇ L ( ˜ w ) , h ( ˜ w ) ⟩ can thus be written as ( ˜ w -Γ( ˜ w )) ⊤ ˜ A ( ˜ w -Γ( ˜ w )) . Next, we prove that when c β is large enough, ˜ A is n.d. on R × ker( X 1 ) ⊥ .

Lemma 10. Let c β be sufficiently large. Then there exists a constant C 10 &gt; 0 such that ∀ z ∈ R × ker( X 1 ) ⊥ , z T ˜ Az ≤ -C 10 ∥ z ∥ 2 .

The proof is in Section E.3. By definition, we have ˜ w t ∈ R × ker( X 1 ) ⊥ and Γ( ˜ w ) ∈ R × ker( X 1 ) ⊥ . So ˜ w -Γ( ˜ w ) ∈ R × ker( X 1 ) ⊥ , yielding

<!-- formula-not-decoded -->

which verifies Assumption A4.

For Assumption A5, we define ˜ X = [ 1 0 ⊤ 0 X ] . Assumption A5(1) is verified below.

Lemma 11. There exists a positive constant C 11, such that for any ˜ w ∈ ˜ W ∗ , ∥ ∥ ∥ ˜ X ˜ w ∥ ∥ ∥ = C 11 .

The proof is in Section E.4. With H ( ˜ w,y ) = ˜ A ( y ) ˜ w + ˜ b ( y ) , the verification of Assumption A5(2) and (3) is similar to Lemmas 8 and 18 and is thus omitted. Invoking Theorem 3 then yields the convergence rate of E [ L ( ˜ w t )] , i.e., the convergence rate of d ( ˜ w t , ˜ W ∗ ) 2 by the definition of L . The next key observation is that d ( ˜ w t , ˜ W ∗ ) 2 = ( ˆ J t -J π ) 2 + d ( w t , W ∗ ) 2 (Lemma 20), which completes the proof.

## 6 Experiments

We now empirically examine linear TD with linearly dependent features. Following the practice of Sutton and Barto [2018], we use diminishing learning rates α t = α t + t 0 and β t = β t + t 0 which closely match our Assumption LR with ξ = 1 and t 0 = 10 7 . We use a variant of Boyan's chain [Boyan, 1999] with 15 states ( |S| = 15 ) and 5 actions ( |A| = 5 ) under a uniform policy π ( a | s ) = 1 / |A| , where the feature matrix X ∈ R 15 × 5 is designed to be of rank 3 (more details in Section F). 3 The weight convergence to a set is indeed observed. It is within expectation that different λ requires different α, β .

Figure 1: Convergence of (Discounted TD) with γ = 0 . 9 , α 0 ∈ { 0 . 005 , 0 . 01 } . Curves are averaged over 10 runs with shaded regions (too small to be visible) indicating standard errors.

<!-- image -->

Figure 2: Convergence of (Average Reward TD) with β 0 = 0 . 01 , α 0 ∈ { 0 . 01 , 0 . 02 , 0 . 1 } . Curves are averaged over 10 runs with shaded regions (too small to be visible) indicating standard errors.

<!-- image -->

## 7 Conclusion

This paper provides the first finite sample analysis of linear TD with arbitrary features in both discounted and average reward settings, fulfilling the long standing desiderata of Dayan [1992], Tsitsiklis and Roy [1996, 1999], enabled by a novel stochastic approximation result concerning the convergence rate to a set. The key methodology contributions include a novel Lyapunov function based on the distance to a set and a novel decomposition of the feature matrix for the average-reward setting. We envision the techniques developed in this work can easily transfer to the analyses of other linear RL algorithms. That being said, one limitation of the work is its focus on linear function

3 The code for this paper is available at https://github.com/WennyXie/LinearTDLambda .

approximation. Extension to neural networks with neural tangent kernels (cf. Cai et al. [2023]) is a possible future work. Another limitation is that this work considers only L 2 convergence rates but the convergence mode of random variables are versatile. Establishing almost sure convergence rates, L p convergence rates, and high probability concentration bounds (cf. Qian et al. [2024]) is also a possible future work. Finally, another promising direction is the integration of Polyak-Ruppert averaging (cf. Patil et al. [2023], Naskar et al. [2024]), which potentially leads to parameter-free convergence rates.

## Acknowledgments and Disclosure of Funding

This work is supported in part by the US National Science Foundation under the awards III-2128019, SLES-2331904, and CAREER-2442098, the Commonwealth Cyber Initiative's Central Virginia Node under the award VV-1Q26-001, and an Nvidia academic grant program award.

## References

- David Abel, André Barreto, Benjamin Van Roy, Doina Precup, Hado van Hasselt, and Satinder Singh. A definition of continual reinforcement learning. In Advances in Neural Information Processing Systems, 2023.
- Amir Beck. First-order methods in optimization. MOS-SIAM Series on Optimization, 2017.
- Richard Bellman. A markovian decision process. Journal of Mathematics and Mechanics, 1957.
- Richard Bellman. Dynamic programming. Science, 1966.
- Albert Benveniste, Michel Métivier, and Pierre Priouret. Adaptive Algorithms and Stochastic Approximations. Springer, 1990.
- Dimitri P Bertsekas and John N Tsitsiklis. Neuro-Dynamic Programming. Athena Scientific Belmont, MA, 1996.
- Jalaj Bhandari, Daniel Russo, and Raghav Singal. A finite time analysis of temporal difference learning with linear function approximation. In Proceedings of the Conference on Learning Theory, 2018.
- Ethan Blaser and Shangtong Zhang. Asymptotic and finite sample analysis of nonexpansive stochastic approximations with markovian noise. ArXiv Preprint, 2025.
- Liu Bo, Liu Ji, Ghavamzadeh Mohammad, Mahadevan Sridhar, and Petrik Marek. Finite-sample analysis of proximal gradient td algorithms. In Proceedings of the Conference on Uncertainty in Artificial Intelligence, 2015.
- Vivek Borkar, Shuhang Chen, Adithya Devraj, Ioannis Kontoyiannis, and Sean Meyn. The ode method for asymptotic statistics in stochastic approximation and reinforcement learning. The Annals of Applied Probability, 2025.
- Vivek S Borkar. Stochastic approximation: a dynamical systems viewpoint. Springer, 2009.
- Vivek S Borkar and Sean P Meyn. The ode method for convergence of stochastic approximation and reinforcement learning. SIAM Journal on Control and Optimization, 2000.
- Justin A. Boyan. Least-squares temporal difference learning. In Proceedings of the International Conference on Machine Learning, 1999.
- Mario Bravo and Roberto Cominetti. Stochastic fixed-point iterations for nonexpansive maps: Convergence and error bounds. SIAM Journal on Control and Optimization, 2024.
- Qi Cai, Zhuoran Yang, Jason D Lee, and Zhaoran Wang. Neural temporal difference and q learning provably converge to global optima. Mathematics of Operations Research, 2023.
- Xuyang Chen, Jingliang Duan, Yingbin Liang, and Lin Zhao. Global convergence of two-timescale actor-critic for solving linear quadratic regulator. Proceedings of the AAAI Conference on Artificial Intelligence, 2023a.
- Zaiwei Chen. Non-asymptotic guarantees for average-reward q-learning with adaptive stepsizes. ArXiv Preprint, 2025.

- Zaiwei Chen, Siva Theja Maguluri, Sanjay Shakkottai, and Karthikeyan Shanmugam. Finite-sample analysis of contractive stochastic approximation using smooth convex envelopes. In Advances in Neural Information Processing Systems, 2020.
- Zaiwei Chen, Siva Theja Maguluri, Sanjay Shakkottai, and Karthikeyan Shanmugam. Finite-sample analysis of off-policy td-learning via generalized bellman operators. In Advances in Neural Information Processing Systems, 2021.
- Zaiwei Chen, Siva Theja Maguluri, Sanjay Shakkottai, and Karthikeyan Shanmugam. A lyapunov theory for finite-sample guarantees of markovian stochastic approximation. Operations Research, 2023b.
- Zaiwei Chen, Siva Theja Maguluri, and Martin Zubeldia. Concentration of contractive stochastic approximation: Additive and multiplicative noise. The Annals of Applied Probability, 2025a.
- Zaiwei Chen, Sheng Zhang, Zhe Zhang, Shaan Ul Haque, and Siva Theja Maguluri. A non-asymptotic theory of seminorm lyapunov stability: From deterministic to stochastic iterative algorithms. ArXiv Preprint, 2025b.
- Peter Dayan. The convergence of td( λ ) for general λ . Machine Learning, 1992.
- Leah M Hackman. Faster gradient-td algorithms. Master's thesis, University of Alberta, 2012.
- Khimya Khetarpal, Matthew Riemer, Irina Rish, and Doina Precup. Towards continual reinforcement learning: A review and perspectives. Journal of Artificial Intelligence Research, 2022.
- Harold Kushner and G George Yin. Stochastic approximation and recursive algorithms and applications. Springer Science &amp; Business Media, 2003.
- Chandrashekar Lakshminarayanan and Csaba Szepesvári. Linear stochastic approximation: How far does constant step-size and iterate averaging go? In Proceedings of the International Conference on Artificial Intelligence and Statistics, 2018.
- Shuze Liu, Shuhang Chen, and Shangtong Zhang. The ODE method for stochastic approximation and reinforcement learning with markovian noise. Journal of Machine Learning Research, 2025a.
- Xinyu Liu, Zixuan Xie, and Shangtong Zhang. Extensions of robbins-siegmund theorem with applications in reinforcement learning. ArXiv Preprint, 2025b.
- Xinyu Liu, Zixuan Xie, and Shangtong Zhang. Linear q -learning does not diverge in l 2 : Convergence rates to a bounded set. In Proceedings of the International Conference on Machine Learning, 2025c.
- Yang Long, Zheng Gang, Zhang Yu, Zheng Qian, Li Pengfei, and Pan Gang. On convergence of gradient expected sarsa( λ ). In Proceedings of the AAAI Conference on Artificial Intelligence, 2021.
- Hamid Reza Maei. Gradient temporal-difference learning algorithms. PhD thesis, University of Alberta, 2011.
- Sreejeet Maity and Aritra Mitra. Adversarially-robust td learning with markovian data: Finite-time rates and fundamental limits. ArXiv Preprint, 2025.
- Aritra Mitra. A simple finite-time analysis of td learning with linear function approximation. IEEE Transactions on Automatic Control, 2025.
- Ankur Naskar, Gugan Thoppe, Abbasali Koochakzadeh, and Vijay Gupta. Federated td learning in heterogeneous environments with average rewards: A two-timescale approach with polyak-ruppert averaging. In IEEE Conference on Decision and Control, 2024.
- Dal Fabbro Nicolò, Adibi Arman, Mitra Aritra, and J. Pappas George. Finite-time analysis of asynchronous multi-agent td learning. ArXiv Preprint, 2024.
- Gandharv Patil, Prashanth L. A., Dheeraj Nagaraj, and Doina Precup. Finite time analysis of temporal difference learning with linear function approximation: Tail averaging and regularisation. In Proceedings of the International Conference on Artificial Intelligence and Statistics, 2023.
- Martin L Puterman. Markov decision processes: discrete stochastic dynamic programming. John Wiley &amp; Sons, 2014.
- Xiaochi Qian and Shangtong Zhang. Revisiting a design choice in gradient temporal difference learning. In Proceedings of the International Conference on Learning Representations, 2025.
- Xiaochi Qian, Zixuan Xie, Xinyu Liu, and Shangtong Zhang. Almost sure convergence rates and concentration of stochastic approximation and reinforcement learning with markovian noise. ArXiv Preprint, 2024.

- Shuang Qiu, Zhuoran Yang, Jieping Ye, and Zhaoran Wang. On finite-time convergence of actor-critic algorithm. IEEE Journal on Selected Areas in Information Theory, 2021.
- Mark Bishop Ring. Continual learning in reinforcement environments. PhD thesis, The University of Texas at Austin, 1994.
- Eugene Seneta. Non-negative matrices and Markov chains. Springer Science &amp; Business Media, 2006.
- Ul Haque Shaan and Theja Maguluri Siva. Stochastic approximation with unbounded markovian noise: A general-purpose theorem. In Proceedings of the International Conference on Artificial Intelligence and Statistics, 2025.
- Rayadurgam Srikant and Lei Ying. Finite-time error bounds for linear stochastic approximation andtd learning. In Proceedings of the Conference on Learning Theory, 2019.
- Richard S. Sutton. Learning to predict by the methods of temporal differences. Machine Learning, 1988.
- Richard S Sutton and Andrew G Barto. Reinforcement Learning: An Introduction (2nd Edition). MIT press, 2018.
- Richard S. Sutton, Csaba Szepesvári, and Hamid Reza Maei. A convergent o(n) temporal-difference algorithm for off-policy learning with linear function approximation. In Advances in Neural Information Processing Systems, 2008.
- Richard S. Sutton, Hamid Reza Maei, Doina Precup, Shalabh Bhatnagar, David Silver, Csaba Szepesvári, and Eric Wiewiora. Fast gradient-descent methods for temporal-difference learning with linear function approximation. In Proceedings of the International Conference on Machine Learning, 2009.
- Ganesh Swetha, Uddin Mondal Washim, and Aggarwal Vaneet. Order-optimal global convergence for average reward reinforcement learning via actor-critic approach. ArXiv Preprint, 2024.
- John N. Tsitsiklis and Benjamin Van Roy. Analysis of temporal-diffference learning with function approximation. In IEEE Transactions on Automatic Control, 1996.
- John N. Tsitsiklis and Benjamin Van Roy. Average cost temporal-difference learning. Automatica, 1999.
- Jiuqi Wang and Shangtong Zhang. Almost sure convergence of linear temporal difference learning with arbitrary features. ArXiv Preprint, 2024.
- Yue Wu, Weitong Zhang, Pan Xu, and Quanquan Gu. A finite-time analysis of two time-scale actor-critic methods. In Advances in Neural Information Processing Systems, 2020.
- Tengyu Xu, Zhe Wang, and Yingbin Liang. Improving sample complexity bounds for (natural) actor-critic algorithms. In Advances in Neural Information Processing Systems, 2020a.
- Tengyu Xu, Zhe Wang, and Yingbin Liang. Non-asymptotic convergence analysis of two time-scale (natural) actor-critic algorithms. ArXiv Preprint, 2020b.
- Tengyu Xu, Zhuoran Yang, Zhaoran Wang, and Yingbin Liang. Doubly robust off-policy actor-critic: Convergence and optimality. In Proceedings of the International Conference on Machine Learning, 2021.
- Peng Yang, Jin Kaicheng, Zhang Liangyu, and Zhang Zhihua. Finite sample analysis of distributional td learning with linear function approximation. ArXiv Preprint, 2025.
- Zhuoran Yang, Yongxin Chen, Mingyi Hong, and Zhaoran Wang. Provably global convergence of actor-critic: A case for linear quadratic regulator with ergodic cost. In Advances in Neural Information Processing Systems, 2019.
- Huizhen Yu. Least squares temporal difference methods: An analysis under general conditions. SIAM Journal on Control and Optimization, 2012.
- Huizhen Yu. On convergence of emphatic temporal-difference learning. In Proceedings of the Conference on Learning Theory, 2015.
- Huizhen Yu. Weak convergence properties of constrained emphatic temporal-difference learning with constant and slowly diminishing stepsize. Journal of Machine Learning Research, 2016.
- Wang Yue, Zhou Yi, and Zou Shaofeng. Finite-time error bounds for greedy-gq. Machine Learning, 2024.

- Shangtong Zhang and Shimon Whiteson. Truncated emphatic temporal difference methods for prediction and control. Journal of Machine Learning Research, 2022.
- Shangtong Zhang, Bo Liu, and Shimon Whiteson. GradientDICE: Rethinking generalized offline estimation of stationary values. In Proceedings of the International Conference on Machine Learning, 2020a.
- Shangtong Zhang, Bo Liu, Hengshuai Yao, and Shimon Whiteson. Provably convergent two-timescale offpolicy actor-critic with function approximation. In Proceedings of the International Conference on Machine Learning, 2020b.
- Shangtong Zhang, Yi Wan, Richard S. Sutton, and Shimon Whiteson. Average-reward off-policy policy evaluation with function approximation. In Proceedings of the International Conference on Machine Learning, 2021a.
- Shangtong Zhang, Hengshuai Yao, and Shimon Whiteson. Breaking the deadly triad with a target network. In Proceedings of the International Conference on Machine Learning, 2021b.
- Shangtong Zhang, Remi Tachet, and Romain Laroche. Global optimality and finite sample analysis of softmax off-policy actor critic under state distribution mismatch. Journal of Machine Learning Research, 2022.
- Shangtong Zhang, Remi Tachet Des Combes, and Romain Laroche. On the convergence of sarsa with linear function approximation. In Proceedings of the International Conference on Machine Learning, 2023.
- Sheng Zhang, Zhe Zhang, and Siva Theja Maguluri. Finite sample analysis of average-reward td learning and q -learning. In Advances in Neural Information Processing Systems, 2021c.
- Shaofeng Zou, Tengyu Xu, and Yingbin Liang. Finite-sample analysis for sarsa with linear function approximation. In Advances in Neural Information Processing Systems, 2019.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claims in the abstract and introduction accurately reflect the paper's contributions, establishing novel L 2 convergence rates for linear TD( λ ) with arbitrary features in both discounted and average-reward settings without additional assumptions.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: The Conclusion section explicitly discusses two limitations of the work:

- the focus on linear function approximation, which may not extend directly to non-linear methods like neural networks;
- the restriction to L 2 convergence rates, excluding other convergence modes such as almost sure convergence, L p convergence, or high-probability bounds.

These points address the scope of the theoretical results and potential extensions, aligning with NeurIPS guidelines for reflecting on limitations.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best

judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: All main theoretical results (Theorems 1, 2, and 3) are presented with their complete proofs in Section 5 and the Appendices B, C, D, E. The necessary assumptions for these results are explicitly stated (e.g., Assumptions 3.1, LR, A1-A5).

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

Justification: Section 6 describes the experimental setup, and Appendix F provides further details on the environment (Boyan's chain variant, state/action space, feature matrix design, reward function, and policy). The specific learning rates used are also mentioned in the figure captions (Figures 1 and 2). This information should be sufficient for others to reproduce the main experimental results.

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

Justification: The paper provides access to the code via a footnote in the Experiments section. All experimental settings, including the environment, hyperparameters, and the full feature matrix, are detailed in Sections 6 &amp; F to ensure reproducibility.

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

Justification: The paper specifies the experimental setup in Section 6 and F, including the Markov Decision Process, policy, feature matrix, and learning rate schedules.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

## Answer: [Yes]

Justification: The experimental results presented in Figures 1 and 2 show curves that are averaged over 10 independent runs. The captions explicitly state that shaded regions, indicating standard errors, are present, though they are too small to be seen. This provides information about the variability of the results across different runs.

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

Justification: The experiments involve running linear TD( λ ) under both discounted and average-reward setting for up to 1 . 5 × 10 6 steps, averaged over 10 runs. These were conducted on a server with an AMD EPYC 9534 64-Core Processor. Each individual run completed in approximately 1 minute. Memory requirements are negligible. The information is detailed in Section F.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: This research on linear TD( λ ) convergence conforms to the NeurIPS Code of Ethics, as we submit anonymized code with instructions to ensure reproducibility, aligning with data-related and artifact accessibility requirements. The theoretical nature of our work,

tested on a synthetic Boyan's chain, poses no foreseeable harms (e.g., safety, discrimination, or surveillance risks), and all assumptions (e.g., Assumptions 3.1, LR, A1-A5) are transparently stated, with no human subjects or sensitive data involved.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This paper provides a foundational theoretical analysis of the linear TD( λ ) algorithm, focusing on its L 2 convergence rates under relaxed assumptions. As a primarily theoretical contribution to an existing algorithmic framework, it does not introduce new applications or systems with societal impacts that need a specific discussion. Any societal impact would be indirect, arising from the broader application of reinforcement learning.

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

Justification: This paper presents a theoretical analysis and does not release any new data, pretrained models, or code that would pose a high risk for misuse. The experiments are based on a standard, well-defined reinforcement learning environment (Boyan's chain) and do not involve datasets or generative models.

Guidelines:

- The answer NA means that the paper poses no such risks.

- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: This paper focuses on a theoretical analysis and uses foundational algorithms linear TD( λ ) and a variant of Boyan's chain [Boyan, 1999] described in prior academic literature. These are not external datasets, code packages, or pre-trained models that would carry specific licenses. All prior work and conceptual contributions are properly credited through citations.

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

Justification: This paper does not release any new assets. The contributions are mathematical results and insights into an existing algorithm.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve any crowdsourcing experiments or research with human subjects. The work is purely theoretical and computational, focusing on the analysis of a reinforcement learning algorithm.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve any research with human subjects. Therefore, Institutional Review Board (IRB) approval or equivalent was not required.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core methodology, theoretical analysis, and experimental design in this paper do not involve the use of LLMs. Any LLM usage was limited to aiding in writing, editing, or formatting, and did not impact the scientific content or originality of the research.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Auxiliary Lemmas and Notations

Lemma 12 (Discrete Gronwall Inequality, Lemma 8 in Section 11.2 of Borkar [2009]) . For nonnegative real sequences { x n , n ≥ 0 } and { a n , n ≥ 0 } and scalar L ≥ 0 , it holds

<!-- formula-not-decoded -->

Lemma 13 (Lemma 11 of Zhang et al. [2022]) . For sufficiently large t 0 , it holds that

<!-- formula-not-decoded -->

Lemma 13 ensures that there exists some t &gt; 0 (depending on t 0 ) such that for all t ≥ t , it holds that t ≥ τ α t . Also, it ensures that for sufficiently large t 0 , we have α t -τ αt ,t -1 &lt; 1 . Throughout the appendix, we always assume t 0 is sufficiently large and t ≥ t . We will refine (i.e., increase) t along the proof when necessary.

## B Proofs in Section 3

## B.1 Proof of Lemma 1

Proof. Let x i ∈ R d denote the i -th column of X . Without loss of generity, let the first m columns be linearly independent.

̸

Case 1: When 1 ∈ col( X ) , there must exist m scalars { c i } such that ∑ m i =1 c i x i = 1 . Apparently, at least one of { c i } must be nonzero. Without loss of generity, let x m = 0 . We then have

<!-- formula-not-decoded -->

In other words, x m can be expressed as linear combination of { x 1 , . . . , x m -1 } and 1 . Since X has a column rank m , we are able to express { x m +1 , . . . , x d } by linear combination { x 1 , . . . , x m } and thus further by linear combination of { x 1 , . . . , x m -1 } and 1 . Let Z 1 . = [ x 1 , . . . , x m -1 ] be the first m -1 columns of X and Z 2 . = [ x m , . . . , x d ] be the rest. We now know that there exists some C ∈ R ( m -1) × ( d -m +1) (i.e., coefficients of the lienar combination) such that

<!-- formula-not-decoded -->

where θ m , . . . θ d are scalars (i.e., 'coordinates' along the 1 -axis), e.g., θ m = 1 c m . This means that we can express X as

<!-- formula-not-decoded -->

with θ 1 = · · · = θ m -1 = 0 . Now define

<!-- formula-not-decoded -->

We note that 1 / ∈ col( Z 1 ) . Otherwise, there would exist scalars { c ′ i } such that ∑ m -1 i =1 c ′ i x i = 1 . Then we get ∑ m -1 i =1 ( c i -c ′ i ) x i + c m x m = 0 , which is impossible because { x i } i =1 , ··· ,m are linearly independent. Since col( X 1 ) = col( Z 1 ) , we then have 1 / ∈ col( X 1 ) . Case 2: When 1 / ∈ col( X ) , we can trivially define X 1 = X and X 2 = 0 . Additionally, we can still further decompose X 1 as

<!-- formula-not-decoded -->

where Z 1 is now the first m columns of X . Apparently, we still have 1 / ∈ col( X 1 ) .

Lemma 14. Let Assumption 3.1 hold. Then A = X 1 D π ( P λ -I ) X 1 , b = X ⊤ 1 D π ( r λ -J π 1 -λ 1 ) .

Proof. Apply the decomposition shown in Lemma 1, we can get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last equality holds because ( P λ -I ) 1 = 0 and 1 ⊤ D π ( P λ -I ) = d ⊤ π P λ -d ⊤ π = 0 . Similarly, for b we can obtain

<!-- formula-not-decoded -->

Here, the fourth inequality holds because d ⊤ π ( I -λP π ) = (1 -λ ) d ⊤ π , which gives us d ⊤ π = (1 -λ ) d ⊤ π ( I -λP π ) -1 . The last inequality holds since J π = d ⊤ π r π . This completes the proof.

Lemma 15. Let Assumption 3.1 hold. Then W ∗ is nonempty.

Proof. In view of (11) and (12), we have X 1 = [ Z 1 Z 1 C ] . Notably, Z 1 has a full column rank and 1 / ∈ col( Z 1 ) . Decompose w . = [ w 1 w 2 ] accordingly and recall (4) and Lemma 14, we can rewrite Aw + b = 0 as

<!-- formula-not-decoded -->

which thus gives us the following simultaneous equations

<!-- formula-not-decoded -->

We now prove the claim by constructing a solution. Choose any w 2 ∈ ker( Z 1 C ) (e.g., w 2 = 0 ), the equations then become

<!-- formula-not-decoded -->

Since Z 1 is full rank and 1 / ∈ Z 1 , Lemma 7 of Tsitsiklis and Roy [1999] shows Z ⊤ 1 D π ( P λ -I ) Z 1 is n.d. and thus invertible. Choose w 1 = -( Z ⊤ 1 D π ( P λ -I ) Z 1 ) -1 Z ⊤ 1 D π ( r λ -J π 1 -λ 1 ) then satisfies the equations. This completes the proof.

Lemma 16. Let Assumption 3.1 hold. Then W ∗ = { w ∗ } +ker( X 1 ) and ker( X 1 ) = { w | Xw = c 1 , c ∈ R } .

Proof. For any solution w ∗ , w ∗∗ ∈ W ∗ , according to the definition of W ∗ in (4), we have Aw ∗ + b = 0 and Aw ∗∗ + b = 0 . That is A ( w ∗ -w ∗∗ ) = 0 . By multiplying ( w ∗ -w ∗∗ ) ⊤ on both side we can get

<!-- formula-not-decoded -->

According to the Perron-Frobenius theorem with Assumption 3.1, v ⊤ D π ( P λ -I ) v = 0 if and only if v = c 1 for some c ∈ R . Therefore, we must have X ( w ∗ -w ∗∗ ) = c 1 for some c ∈ R . That is, ( X 1 + X 2 )( w ∗ -w ∗∗ ) = c 1 . Recall the definition of X 2 in (11), we have X 2 ( w ∗ -w ∗∗ ) = ( θ ⊤ ( w ∗ -

w ∗∗ )) 1 . This means X 1 ( w ∗ -w ∗∗ ) = c ′ 1 with c ′ = c -θ ⊤ ( w ∗ -w ∗∗ ) . Since 1 / ∈ col( X 1 ) , we must have c ′ = 0 . That is, w -w ∈ ker( X ) . Thus, we have established that W = { w } +ker( X )

Furthermore, if w ∈ ker( X 1 ) , we have Xw = ( X 1 + X 2 ) w = ( θ w ) 1 . If Xw = c 1 , we have X 1 w = c 1 -X 2 w = ( c -θ ⊤ w ) 1 . But 1 / ∈ col( X 1 ) . So we must have c -θ ⊤ w = 0 , i.e., w ∈ ker( X 1 ) . This completes the proof of ker( X 1 ) = { w | Xw = c 1 , c ∈ R } .

∗ ∗∗ 1 ∗ ∗ 1 . ⊤

## C Proofs in Section 5.1

Lemma 17. For sufficiently large t 0 , there exists a constant C 17 such that the following statement holds. For any t ≥ t and any i ∈ [ t -τ α t , t ] , it holds that

<!-- formula-not-decoded -->

Proof. In this proof, to simplify notations, we define shorthand t 1 . = t -τ α t and C x . = max s ∥ x ( s ) ∥ . Given Lemma 13, we can select a sufficiently large t 0 such that for any t ≥ t ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

≤

C

α

(

∥

where C . = C C . We then have

A5

t

1

,i

-

1

Xw

t

1

∥

+1)exp(

C

17

,

1

α

t

1

,t

-

1

)

,

(Lemma 12)

<!-- formula-not-decoded -->

where C 17 , 2 . = 3 C A5. Thus, we have

<!-- formula-not-decoded -->

where C 17 . = 2 C 17 , 2 ( C 2 +1) . This completes the proof.

## C.1 Proof of Lemma 2

Proof.

<!-- formula-not-decoded -->

We then bound

## C.2 Proof of Lemma 3

Proof. According to the definition of H ( w t , Y t ) in (10),

<!-- formula-not-decoded -->

where C 3 . = 2 C 2 A5 (2 C 2 2 +1) . This completes the proof.

## C.3 Proof of Lemma 4

Proof. We first decompose ⟨ w t -Γ( w t ) , H ( w t , Y t ) -h ( w t ) ⟩ into three components similarly to Srikant and Ying [2019] as

<!-- formula-not-decoded -->

We leverage Lemma 2 and (9) to bound them one by one as follows. Bounding T 1 :

<!-- formula-not-decoded -->

For the first term, we have

<!-- formula-not-decoded -->

For the second term, we have

<!-- formula-not-decoded -->

where C 4 , 1 . = 2 C A5 ( C 2 +1) . Therefore, we can get

<!-- formula-not-decoded -->

Choosing C 4 ,a . = 4 C 17 C 4 , 1 then yields the bound

<!-- formula-not-decoded -->

Bounding T 2 :

<!-- formula-not-decoded -->

For the first term, we have:

<!-- formula-not-decoded -->

For the second term, we have:

<!-- formula-not-decoded -->

Combine the result in (13) and (14), we have:

<!-- formula-not-decoded -->

Choosing C 4 ,b . = 2 C 4 , 2 C 4 , 3 C 17 then yield the bound

<!-- formula-not-decoded -->

Bounding T 3 :

<!-- formula-not-decoded -->

Take expectation on both sides, we can get

<!-- formula-not-decoded -->

We have

<!-- formula-not-decoded -->

Thus, together with (13), we obtain

<!-- formula-not-decoded -->

where C 4 ,c . = C 4 , 2 C 4 , 4 . Finally, denote C 4 . = C 4 ,a + C 4 ,b + C 4 ,c then completes the proof.

## C.4 Proof of Lemma 5

Proof. We recall that

<!-- formula-not-decoded -->

Aligning Assumption A4, Lemmas 3 and 4 with (8), we get

<!-- formula-not-decoded -->

Furthermore, we aim to derive an upper bound for E [ L ( w t )] that depends on the initial expected loss E [ L ( w 0 )] and decreases over time. First, let's denote the coefficients as C t and D t :

<!-- formula-not-decoded -->

For sufficiently large t 0 and t ≥ t , we obtain 4 C 4 α t -τ αt ,t -1 + C 3 α t &lt; C A4. Thus, the recursive inequality further becomes:

<!-- formula-not-decoded -->

where D t = O ( α t α t -τ αt ,t -1 ) .

## C.5 Proof of Theorem 3

Proof. To express E [ L ( w t )] in terms of E [ L ( w 0 )] , we recursively apply the inequality:

<!-- formula-not-decoded -->

Denote E 1 . = ∏ t i = t (1 -C A4 α i ) E [ L ( w t )] , E 2 . = ∑ t j = t ( ∏ t i = j +1 (1 -C A4 α i ) ) ln( j + t 0 ) ( j + t 0 ) 2 ξ , and κ = C A4 α . Recall we have α t = α ( t + t 0 ) ξ . For E 1 , set t 0 &gt; κ = C A4 α , we have

<!-- formula-not-decoded -->

For E 2 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Starting from the update of w t +1 , we have

<!-- formula-not-decoded -->

That is, ∥ w t +1 ∥ ≤ α 0 C A1 + ∑ t i =0 ( α 0 C A1 +1) ∥ w i ∥ . Applying discrete Gronwall inequality, we obtain ∥ w t ∥ ≤ ( C A1 + ∥ w 0 ∥ ) exp ( ∑ t -1 t =0 (1 + α 0 C A1 ) ) = ( C A1 + ∥ w 0 ∥ ) exp ( t + tα 0 C A1 ) .

Denoting C Thm3,1 . = exp ( 2 t +2 tα 0 C A1 ) and C Thm3,2 . = 2max( C Thm3,4 , C Thm3,6 ) then completes the proof.

## D Proofs in Section 5.2

## D.1 Proof of Lemma 6

Proof. Let y = ( s, a, s ′ , e ) ∈ Y and C x . = max s ∥ x ( s ) ∥ . We have

<!-- formula-not-decoded -->

Furthermore,

<!-- formula-not-decoded -->

which completes the proof.

## D.2 Proof of Lemma 18

Lemma 18. There exist a constant C 18 and τ ∈ [0 , 1) such that ∀ w

<!-- formula-not-decoded -->

Proof. Given the Markov property, we only need to prove the case of t = 1 . Recall that we use y = ( s, a, s ′ , e ) . Define shorthand

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By (10), we can get

By expanding e n , we get

We then have

<!-- formula-not-decoded -->

In the proof of Lemma 6.7 of Bertsekas and Tsitsiklis [1996], it is proved that

<!-- formula-not-decoded -->

which coincides with h ( w ) . Thus the rest of the proof is dedicated to proving that f 1 ( n ) and f 2 ( n ) decay geometrically. For f 2 ( n ) , we have ∥ ∥ ¯ δ n +1 ( w ) x ( ¯ S k ) ∥ ∥ ≤ C 18 , 1 ( ∥ Xw ∥ + 1) for some C 18 , 1 (cf. (16)). We then have

<!-- formula-not-decoded -->

For f 1 ( n ) , since { S t } adopts geometric mixing, there exists some τ 1 ∈ [0 , 1) and C 18 , 2 &gt; 0 such that

<!-- formula-not-decoded -->

Then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now define a two-sided Markov chain { ¯ S t , ¯ A t } t = ..., -2 , -1 , 0 , 1 , 2 ,... such that Pr ( ¯ S t = s ) = d π ( s ) , Pr ( ¯ A t = a | ¯ S t = s ) = π ( a | s ) , i.e., the new chain always stay in the stationary distribution of the original chain. Similarly, define

<!-- formula-not-decoded -->

Noticing that E [ δ n +1 ( w ) | S k = s ] = E [ ¯ δ n +1 ( w ) | ¯ S k = s ] due to the Markov property, we obtain

<!-- formula-not-decoded -->

This means

<!-- formula-not-decoded -->

Noticing that

<!-- formula-not-decoded -->

then completes the proof.

## D.3 Proof of Lemma 7

Proof. We start with proving ∀ w ∈ ker( A ) ⊥ , w ⊤ Aw ≤ -C 7 ∥ w ∥ 2 . This is apparently true if w = 0 . Now fix any w ∈ ker( A ) ⊥ and w = 0 , which implies that Aw = 0 . Now we prove by contradiction that w ⊤ Aw = 0 . Otherwise, if w ⊤ Aw = 0 , we have w ⊤ X ⊤ D π ( γP λ -I ) Xw = 0 . Since D π ( γP λ -I ) is n.d., we then get Xw = 0 , further implying Aw = 0 , which is a contradiction. We have now proved that w ⊤ Aw = 0 . We next prove that w ⊤ Aw &lt; 0 . This is from the fact that A is n.d., i.e., for ∀ z ∈ R d , z ⊤ Az ≤ 0 . But w ⊤ Aw = 0 . So we must have w ⊤ Aw &lt; 0 . Finally, we use an extreme theorem argument to complete the proof. Define Z . = { w | w ∈ ker( A ) ⊥ , ∥ w ∥ = 1 } . Because z ∈ Z implies z ∈ ker( A ) ⊥ and z = 0 , we have ∀ z ∈ Z, z ⊤ Az &lt; 0 . Since Z is clearly compact, the extreme value theorem confirms that the function z ↦→ z ⊤ Az obtains its minimum value in Z , denoted as -C 7 &lt; 0 , i.e., we have

̸

<!-- formula-not-decoded -->

For any w ∈ ker( A ) ⊥ and w = 0 , we have w ∥ w ∥ ∈ Z , so w ⊤ Aw ≤ -C 7 ∥ w ∥ 2 , which completes the proof of the first part.

̸

We now prove that ∀ w ∈ R d , w -Γ( w ) ∈ ker( A ) ⊥ . We recall that Γ is the orthogonal projection to W ∗ = { w | Aw + b = 0 } . Since Γ is the orthogonal projection to W ∗ , we know w -Γ( w ) ∈ W ⊥ ∗ . Fix any w ∗ ∈ W ∗ and let z ∈ ker( A ) , we then have A ( w ∗ + z ) + b = 0 so w ∗ + z ∈ W ∗ . We then have

<!-- formula-not-decoded -->

confirming that w -Γ( w ) ∈ ker( A ) ⊥ , which completes the proof.

## D.4 Proof of Lemma 8

Proof. Let y = ( s, a, s ′ , e ) ∈ Y , since ∣ ∣ x ( s ) ⊤ w ∣ ∣ ≤ max s ∈ S ∣ ∣ x ( s ) ⊤ w ∣ ∣ ≤ ∥ Xw ∥ , according to (10), we have

<!-- formula-not-decoded -->

where C 8 . = C e ( C R + γ +1) . For ∥ h ( w ) ∥ , we have

<!-- formula-not-decoded -->

which completes the proof.

̸

̸

̸

̸

̸

## E Proofs in Section 5.3

## E.1 Proof of Lemma 9

Proof. The update to { ˆ J t } in (Average Reward TD) is

<!-- formula-not-decoded -->

This matches the first row of

<!-- formula-not-decoded -->

Now consider the update for w t

<!-- formula-not-decoded -->

Applying the projection matrix Π on both sides yields

<!-- formula-not-decoded -->

To see the last equality, we recall Lemma 1 and recall Π = X † 1 X 1 . We then have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we use x 1 ( s ) to denote the s -th row of X 1 . We also have

<!-- formula-not-decoded -->

which confirms the last equality and then completes the proof.

## E.2 Proof of Lemma 19

Lemma 19. ˜ A Γ( ˜ w ) + ˜ b = 0

Proof.

<!-- formula-not-decoded -->

Therefore, for the first row of ˜ A Γ( ˜ w ) + ˜ b , we get c β ( J π -J π ) = 0 . For the second row, we can get

<!-- formula-not-decoded -->

where the second equality comes with the definition of Π . This completes the proof.

This means that

## E.3 Proof of Lemma 10

Proof. If z = 0 , the lemma trivially holds. So now let Let z = [ z 1 z 2 ] ∈ R × ker( X 1 ) ⊥ , z = 0 . With (17), we have

̸

<!-- formula-not-decoded -->

For simplicity, define q . = E d Y [ e ] , B . = X ⊤ 1 D π ( P λ -I ) X 1 . We then have

<!-- formula-not-decoded -->

Recall that Π = X † 1 X 1 and it is symmetric, we can get

<!-- formula-not-decoded -->

where the last equality holds because z 2 ∈ ker( X ⊥ 1 ) . Thus,

<!-- formula-not-decoded -->

We now characterize z ⊤ 2 Bz 2 . Apparently, z ⊤ 2 Bz 2 ≤ 0 always holds because D π ( P λ -I ) is n.s.d. In view of (5), the equality holds only if X 1 z 2 = c 1 . But 1 / ∈ col( X 1 ) and z 2 ∈ ker( X 1 ) ⊥ . So the equality holds only when z 2 = 0 . Now we have proved that ∀ z 2 ∈ ker( X 1 ) ⊥ , z 2 = 0 , it holds that z ⊤ 2 Bz 2 &lt; 0 . Using the normalization trick and the extreme value theorem again (cf. (15)), we confirm that there exists some constant C 10 , 1 &gt; 0 such that ∀ z 2 ∈ ker( X 1 ) ⊥ ,

<!-- formula-not-decoded -->

Since z = 0 , we now discuss two cases.

̸

̸

Case 1: z 1 = 0 , z 2 = 0 . In this case, we have z ⊤ ˜ Az = z ⊤ 2 Bz 2 &lt; 0 .

̸

Case 2: z 1 = 0 . In this case, we have

<!-- formula-not-decoded -->

̸

By completing squares, it is easy to see that when c β is sufficiently large (depending on ∥ q ∥ and C 10 , 1 ), it holds z ⊤ ˜ Az &lt; 0 because z 1 = 0 .

̸

Combining both cases, we have proved that ∀ z ∈ R × ker( X 1 ) ⊥ , z = 0 , it holds that

<!-- formula-not-decoded -->

Using the normalization trick and the extreme value theorem again (cf. (15)) then completes the proof.

## E.4 Proof of Lemma 11

Proof. By definition, ˜ W ∗ = {[ J π Π w ]∣ ∣ ∣ ∣ w ∈ W ∗ } . In view of Lemma 16, let w ∗ be any fixed vector in W ∗ . Then any ˜ w ∗ ∈ ˜ W ∗ can be written as

<!-- formula-not-decoded -->

with some w 0 ∈ ker( X 1 ) . We then have

<!-- formula-not-decoded -->

where the last equality holds because Π is the orthogonal projection to ker( X 1 ) ⊥ . This means that ˜ X ˜ w ∗ is a constant regardless of ˜ w ∗ , which completes the proof.

̸

## E.5

## Proof of Lemma 20

<!-- formula-not-decoded -->

Proof. We recall that Π is the orthogonal projection to ker( X 1 ) ⊥ . Let Π ′ be the orthogonal projection to ker( X 1 ) . We recall from Lemma 16 that W ∗ = { w ∗ } +ker( X 1 ) with w ∗ being any fixed point in W ∗ . Thus for any w ∗ ∈ W ∗ , we can write it as w ∗ + w 0 with some w 0 ∈ ker( X 1 ) . Then for any w ∈ R d , we have

<!-- formula-not-decoded -->

where the last equality holds because we can select w 0 = Π ′ w -Π ′ w ∗ . Define Π W ∗ . = { Π w | w ∈ W ∗ } . Then we have

<!-- formula-not-decoded -->

where the last equality holds because w 0 ∈ ker( X 1 ) and Π is the projection to ker( X 1 ) ⊥ so Π w 0 = 0 . We now have ∀ w,d ( w,W ∗ ) = d (Π w, Π W ∗ ) . Then we have

<!-- formula-not-decoded -->

which completes the proof.

## F Details of Experiments

We use a variant of Boyan's chain [Boyan, 1999] with 15 states ( s 0 , s 1 , . . . , s 14 ) and 5 actions ( a 0 , . . . , a 4 ). The chain has deterministic transitions. For s 2 , . . . , s 14 , the action a 0 goes to s i -1 and the actions a 1 to a 4 go to s i -2 ; s 1 always transitions to s 0 ; s 0 transitions uniformly randomly to any state. The reward function is

<!-- formula-not-decoded -->

.

We use a uniform random policy π ( a | s ) = 0 . 5 . The feature matrix X ∈ R 15 × 5 is designed to be of rank 3.

|  0 . 07   | 0 . 11   | 0 . 18   | 0 . 14   | 0 . 61   |
|------------|----------|----------|----------|----------|
| 0 . 13     | 0 . 19   | 0 . 32   | 0 . 26   | 0 . 45   |
|   0 . 11 | 0 . 17   | 0 . 28   | 0 . 22   | 0 . 39   |
|   0 . 24 | 0 . 36   | 0 . 60   | 0 . 48   | 0 . 84   |
|   0 . 18 | 0 . 28   | 0 . 46   | 0 . 36   | 1 . 00   |
|   0 . 20 | 0 . 30   | 0 . 50   | 0 . 40   | 1 . 06   |
|   0 . 31 | 0 . 47   | 0 . 78   | 0 . 62   | 1 . 45   |
|  0 . 29   | 0 . 45   | 0 . 74   | 0 . 58   | 1 . 39   |
|   0 . 42 | 0 . 64   | 1 . 06   | 0 . 84   | 1 . 84   |
|   0 . 40 | 0 . 62   | 1 . 02   | 0 . 80   | 1 . 78   |
|   0 . 47 | 0 . 73   | 1 . 20   | 0 . 94   | 2 . 39   |
|   0 . 53 | 0 . 81   | 1 . 34   | 1 . 06   | 2 . 23   |
|  0 . 58   | 0 . 9    | 1 . 48   | 1 . 16   | 2 . 78   |
|   0 . 60 | 0 . 92   | 1 . 52   | 1 . 20   | 2 . 84   |
| 0 . 67     | 1 . 03   | 1 . 70   | 1 . 34   | 3 . 45   |

Each experiment runs for 1 . 5 × 10 6 steps, averaged over 10 runs. These experiments were conducted on a server equipped with an AMD EPYC 9534 64-Core Processor, with each run taking approximately 1 minute to complete. Memory requirements are negligible.