## Improved Regret and Contextual Linear Extension for Pandora's Box and Prophet Inequality

## Junyan Liu ∗

University of Washington junyanl1@cs.washington.edu

## Kun Wang ∗

Purdue University wang5675@purdue.edu

## Ziyun Chen ∗

University of Washington ziyuncc@cs.washington.edu

## Haipeng Luo

University of Southern California haipengl@usc.edu

## Lillian J. Ratliff

University of Washington ratliffl@uw.edu

## Abstract

We study the Pandora's Box problem in an online learning setting with semi-bandit feedback. In each round, the learner sequentially pays to open up to n boxes with unknown reward distributions, observes rewards upon opening, and decides when to stop. The utility of the learner is the maximum observed reward minus the cumulative cost of opened boxes, and the goal is to minimize regret defined as the gap between the cumulative expected utility and that of the optimal policy. We propose a new algorithm that achieves ˜ O ( √ nT ) regret after T rounds, which improves the ˜ O ( n √ T ) bound of Agarwal et al. [2024] and matches the known lower bound up to logarithmic factors. To better capture real-life applications, we then extend our results to a natural but challenging contextual linear setting, where each box's expected reward is linear in some known but time-varying d -dimensional context and the noise distribution is fixed over time. We design an algorithm that learns both the linear function and the noise distributions, achieving ˜ O ( nd √ T ) regret. Finally, we show that our techniques also apply to the online Prophet Inequality problem, where the learner must decide immediately whether or not to accept a revealed reward. In both non-contextual and contextual settings, our approach achieves similar improvements and regret bounds.

## 1 Introduction

Pandora's Box is a fundamental problem in stochastic optimization, initiated by Weitzman [1978]. In this problem, the learner is given n boxes, each associated with a known cost and a known reward distribution. The learner may open a box at a time, paying the corresponding cost to observe a realized reward. Based on the known distributions, the learner designs a policy that specifies the order of inspecting boxes and a stopping rule for when to halt the process and select a reward. Weitzman [1978] shows that the optimal policy computes a threshold (i.e., reservation value) for each box based on its cost and distribution, and opens boxes in descending order of these thresholds until a stopping condition is met. This model abstracts a wide range of sequential decision-making scenarios where

∗ Equal contribution

information is costly to acquire and decisions must be made under uncertainty. For example, in the hiring process, an employer interviews candidates one by one, where each interview incurs a cost (e.g., time), and each candidate's performance is drawn from a known distribution (e.g., candidate's profile). The employer must decide when to stop interviewing and make an offer, balancing the expected benefit of finding a better candidate in the future against the accumulated cost of continued search. Other real-world applications include online shopping and path planning [Atsidakou et al., 2024].

While the classic Pandora's Box problem and its variants have been extensively studied (see survey [Beyhaghi and Cai, 2024]), recent work [Fu and Lin, 2020, Guo et al., 2021, Gatmiry et al., 2024, Agarwal et al., 2024, Atsidakou et al., 2024] has started to explore settings where the reward distributions are unknown . This line of work is motivated by practical scenarios where distributions are unavailable upfront and must be learned through interaction. These formulations move beyond the classical setting by requiring the learner to adapt based on observed outcomes. Prior studies have considered probably approximately correct (PAC) guarantees, establishing the near-optimal sample complexity Fu and Lin [2020], Guo et al. [2021], and regret minimization over a T -round repeated game Gergatsouli and Tzamos [2022], Gatmiry et al. [2024], Agarwal et al. [2024], Atsidakou et al. [2024]. In the basic regret minimization setting, the learner repeatedly solves a Pandora's Box problem at each round and observes the rewards of the opened boxes (a.k.a. semi-bandit feedback), drawn from fixed but unknown distributions. However, even in this setting, a gap remains between the Ω( √ nT ) lower bound of Gatmiry et al. [2024] and the best known upper bound of ˜ O ( Un √ T ) by Agarwal et al. [2024], where U is the magnitude of the utility function that can be as large as n .

In addition to the fixed-distribution setting, prior work studied the online Pandora's Box problems with time-varying reward distributions, which better capture many real-world scenarios Gergatsouli and Tzamos [2022], Atsidakou et al. [2024]. For example, in the hiring process, the distribution of candidate quality may shift over time due to market trends or seasonal patterns. To model such dynamics, Gergatsouli and Tzamos [2022] consider a setting where the rewards are chosen by an oblivious adversary and show that sublinear regret is achievable when the algorithm competes against a benchmark weaker than the optimal policy, obtained by imposing constraints on the set of boxes it can open. Later, Gatmiry et al. [2024] showed that in the same setting, when the algorithm competes against the optimal policy, no algorithm can achieve sublinear regret, even with full information feedback (i.e., the learner observes the rewards of all boxes regardless of which policy is played). To make the problem tractable while still allowing distributions to change over time, Atsidakou et al. [2024] propose a contextual model where, in each round, the learner observes a context vector for each box, whose optimal threshold can be approximated by a function of the context and an unknown vector. Their approach reduces the problem to linear-quadratic online regression, which provides generality but leads to a regret bound with poor dependence on the horizon T . In particular, if the function is linear, they achieve ˜ O ( nT 5 / 6 ) regret bound with semi-bandit feedback (referred to as bandit feedback in their work).

Motivated by these limitations, we address two main challenges in this paper. First, we close the gap in the regret dependence on the number of boxes n . Second, we propose a new contextual model that allows the reward distributions to vary over time, and under this setting, we achieve improved dependence on the time horizon T . Our contributions are summarized as follows, and further related work is discussed in Appendix B.

- For the online Pandora's Box problem, we propose a new algorithm that achieves a regret bound of ˜ O ( √ nT ) , improving upon the ˜ O ( Un √ T ) bound by Agarwal et al. [2024] and matching the known Ω( √ nT ) lower bound (up to logarithmic factors) [Gatmiry et al., 2024]. Our algorithm builds on the optimism-based framework of Agarwal et al. [2024], but introduces a simple yet crucial modification. Specifically, we first construct empirical distributions for each box using the observed reward samples, and then shift probability mass to form optimistic distributions that encourage exploration. In contrast to the fixed-mass shifting scheme used in Agarwal et al. [2024], we adaptively reallocate mass at each point based on problem-dependent factors. As we demonstrate below, this adaptive construction is central to enabling a more refined regret analysis.
- We extend these results to the contextual linear setting, where each box's expected reward is linear in some known but time-varying d -dimensional context and the noise distribution is fixed over time. Our algorithm preserves the overall structure of the non-contextual case, but modifies the construction of optimistic distributions to account for context-dependent variation. Since the reward

samples are no longer identically distributed, we adjust not only the probability mass but also the observed rewards themselves in a value-optimistic manner to maintain optimism. While the algorithm of Atsidakou et al. [2024] applies to our setting with ˜ O ( nT 5 / 6 ) regret (see Appendix B), we establish ˜ O ( nd √ T ) regret. As long as d = o ( T 1 / 3 ) , our algorithm yields a better regret guarantee.

- Finally, we extend our algorithms and analytical techniques to the Prophet Inequality problem in both non-contextual and contextual settings. In the non-contextual case, our algorithm achieves ˜ O ( √ nT ) regret bound, which again improves upon the ˜ O ( n √ T ) regret bound of Agarwal et al. [2024]. For the contextual linear setup, we also establish a ˜ O ( nd √ T ) regret bound. Due to space constraints, all results and proofs for the Prophet Inequality problem are deferred to the appendix.

Technical Overview for ˜ O ( √ nT ) Improvement. Inspired by Agarwal et al. [2024], we decompose the regret at any round t by ∑ i Term t,i , where Term t,i captures the difference in expected reward when replacing the optimistic distribution of the i -th box, denoted by ˆ E t,i , with its true distribution D i , while keeping all other boxes fixed. Agarwal et al. [2024] bound Term t,i by O ( TV ( ˆ E t,i , D i )) , where TV denotes the total variation distance. However, simply bounding Term t,i by the TV distance results in ˜ O ( n √ T ) regret bound. To obtain a tighter bound, we take a different approach to handle Term t,i . Let ˜ R i ( σ t , z ) be the expected utility when using threshold vector σ t , and the i -th box has a fixed value z . We then write Term t,i = ∫ 1 0 ( F D i ( z ) -F ˆ E t,i ( z ) ) ∂ ∂z ˜ R i ( σ t ; z ) dz where F D is the cumulative density function of distribution D . Our main techniques to bound this term are as follows.

Technique 1: Bernstein-type Bound. Both our analysis and algorithm design are based on a Bernstein-type concentration bound. In particular, our algorithm adaptively adjusts the probability mass when constructing empirical distributions, guided by this bound. This introduces an extra factor √ F D i ( · ) (1 -F D i ( · )) in the upper bound on | F D i ( · ) -F E t,i ( · ) | , where E t,i is the empirical distribution constructed at round t by assigning equal probability mass to iid samples from D i .

Technique 2: Sharp Bound of ∂ ∂z ˜ R i ( σ t , z ) . We show 1 -Lipschitz and monotonically-increasing properties for ˜ R i ( σ t , z ) , which imply that 0 ≤ ∂ ∂z ˜ R i ( σ t , z ) ≤ 1 . Moreover, when z is small, we prove a sharper bound where ∂ ∂z ˜ R i ( σ t , z ) is bounded by the probability that the first i -1 boxes have values no larger than z .

Next, we apply the techniques above to obtain a sharper bound for Term t,i = ∫ 1 0 ( F D i ( z ) -F ˆ E t,i ( z ) ) ∂ ∂z ˜ R i ( σ t ; z ) dz . The high-level idea is as follows: when z is large, the Bernstein-type bound for ( F D i ( z ) -F ˆ E t,i ( z )) contributes to a factor √ F D i ( z ) (1 -F D i ( z )) , which is small since a large z implies a small 1 -F D i ( z ) ; on the other hand, when z is small, we use the sharper bound on ∂ ∂z ˜ R i ( σ t , z ) to obtain a tighter bound.

Techniques for Contextual Linear Case. We first build a virtual empirical distribution ˆ D t,i for each box i at round t , by using all available noise samples from this box plus the mean of the current round. This is virtual since we only observe non-i.i.d. reward samples, not the raw noises. To overcome this issue, we introduce a value-optimistic empirical distribution ˆ E t,i by using the linear mean model to shift the observed samples to overestimate ˆ D t,i . We then decompose Term t,i into three terms, two of which can be handled in a similar way as in the non-contextual case. The remaining piece is E z ∼E t,i [ ˜ R i ( σ ˆ E t ; z )] -E z ∼ ˆ D t,i [ ˜ R i ( σ ˆ E t ; z )] which measures the mean estimation error. This term is bounded by applying the 1 -Lipschitz property of ˜ R i ( σ ˆ E t ; z ) and contextual linear bandit techniques.

## 2 Preliminaries

In this section, we introduce online Pandora's Box problem in both non-contextual and contextual settings. Due to the limited space, the Prophet Inequality problem is deferred to Appendix A.

Online Pandora's Box. In this problem, the learner is given n boxes, each with a known cost c i ∈ [0 , 1] and an unknown fixed reward distribution D i supported on [0 , 1] . The learner plays a T -round repeated game where at each round t ∈ [ T ] , the learner decides an order to inspect boxes

```
Algorithm 1 Generic descending threshold algorithm for Pandora's Box -th largest threshold.
```

```
Input : threshold σ = ( σ 1 , . . . , σ n ) . Initialize : V max = -∞ . Let π i ∈ [ n ] be the box with the i The environment generates an unknown realization v = ( v 1 , . . . , v n ) . for i = 1 , . . . , n do Pay c π i to open box π i , observe v π i , and update V max = max { V max , v π i } . If i = n or V max ≥ σ π i +1 , then stop and take reward V max .
```

and a stopping rule. Upon opening a box i , the learner pays cost c i and receives a reward v t,i ∈ [0 , 1] independently sampled from D i . Once the stopping rule is met, she selects the highest observed reward so far. The utility for the round is the chosen reward minus the total cost of opened boxes.

Contextual Linear Pandora's Box. This setting extends the above by allowing each box's distribution to vary across rounds. Specifically, each box i ∈ [ n ] is associated with a known cost c i ∈ [0 , 1] , an unknown parameter θ i ∈ R d , and a fixed unknown noise distribution D i with zero-mean, supported on [ -1 4 , 1 4 ] . At the beginning of each round t ∈ [ T ] , the learner observes a context x t,i ∈ R d for each box i ∈ [ n ] . These contexts { x t,i } i ∈ [ n ] can be chosen arbitrarily by an adaptive adversary. Based on these contexts and past observations, the learner then decides an inspection order and a stopping rule to open boxes. Different from the non-contextual setup, upon opening box i , the learner observes the reward v t,i = η t,i + µ t,i where µ t,i = θ ⊤ i x t,i ∈ [ 1 4 , 3 4 ] is the mean and η t,i is a noise independently drawn from D i . 2 That is, the reward is drawn from the [0 , 1] -bounded distribution D t,i = µ t,i + D i which shifts D i by the context-dependent mean µ t,i . Following standard assumptions in contextual linear bandits, we assume that ∥ θ i ∥ 2 ≤ 1 and ∥ x t,i ∥ 2 ≤ 1 for all t, i .

Optimal Policy and Regret. To measure the performance, we compete our algorithm against the per-round optimal policy. For both settings, the optimal policy at round t is the Weitzman's algorithm [Weitzman, 1978] which operates with the distribution D t = ( D t, 1 , . . . , D t,n ) . In the non-contextual setting, the distribution is fixed across rounds i.e., D t = D = ( D 1 , . . . , D n ) , and thus, we will retain the notation D t in the following. More specifically, Weitzman's algorithm proceeds by computing a threshold σ ∗ t,i for each box i , which is the solution of E x ∼ D t,i [( x -σ ∗ t,i ) + ] = c i where ( x ) + = max { 0 , x } . Then, the algorithm inspects the boxes in descending order of σ ∗ t,i and stops as soon as the maximum observed reward so far exceeds the threshold of the next unopened box.

To formally define regret, we introduce the following notations. For any product distribution D = ( D 1 , . . . , D n ) , let σ D = ( σ D, 1 , . . . , σ D,n ) denote the optimal threshold vector, where each σ D,i is the solution of E x ∼ D i [( x -σ D,i ) + ] = c i . For any threshold vector σ = ( σ 1 , . . . , σ n ) ∈ [0 , 1] n and reward realization v = ( v 1 , . . . , v n ) ∈ [0 , 1] n , we use W ( σ ; v ) ⊆ [ n ] to denote boxes opened by Algorithm 1. Further, we define the utility function and expected utility function respectively as:

<!-- formula-not-decoded -->

With these definitions, the optimal expected utility is R ( σ D t ; D t ) since when v ∼ D t , the set W ( σ D t ; v ) corresponds to the boxes opened by the Weitzman's algorithm. Let σ t ∈ R n denote the threshold vector selected by the algorithm at round t . Throughout the paper, all our algorithms follow Algorithm 1 using σ t as input at round t , and thus the expected utility our algorithm at round t is R ( σ t ; D t ) . The cumulative regret in T rounds Reg T is then defined as follows: Reg T = E [ ∑ T t =1 R ( σ D t ; D t ) -R ( σ t ; D t ) ] , where D t = D for the non-contextual case.

## 3 ˜ O ( √ nT ) Regret Bound for Non-Contextual Pandora's Box

In this section, we first present Algorithm 2 for the non-contextual Pandora's Box problem and the main result, and then discuss the analysis.

2 The boundedness assumptions on the noise and mean ensure that rewards lie in [0 , 1] , which avoids rederiving standard results under a different scaling. Our results readily extends to sub-Gaussian rewards. We refer readers to Appendix C.7 for details.

## 3.1 Algorithm and Main Result

While Algorithm 2 is inspired by the principle of optimism in the face of uncertainty , its application differs from that in the classic multi-armed bandit setting, such as in the Upper Confidence Bound (UCB) algorithm [Auer et al., 2002]. In bandit problems, optimism is typically applied to the mean reward of each arm. In contrast, the Pandora's Box problem requires learning the entire distribution of each box, particularly its cumulative distribution function (CDF). We follow a similar idea to Agarwal et al. [2024], but employ a different approach in the construction of an optimistic estimate of the distribution. The threshold σ t is computed from this optimistic distribution, which implicitly ensures optimism in the algorithm's decisions. More formally, we begin by introducing the notion of stochastic dominance.

Definition 3.1.1 ( Stochastic dominance ) . Let D and E be two probability distributions with CDFs F D and F E , respectively. If P X ∼ E ( X ≥ a ) ≥ P Y ∼ D ( Y ≥ a ) for all a ∈ R , we say that E stochastically dominates D , and we denote this by E ⪰ SD D .

For any two product distributions D = ( D 1 , . . . , D n ) and E = ( E 1 , . . . , E n ) , we use E ⪰ SD D to indicate that E i ⪰ SD D i for all i ∈ [ n ] . Let m t,i be the number of samples for each box i at round t and let t i ( j ) be the round that the j -th reward sample of box i is observed by the learner. We define empirical distribution and optimistic distribution as follows.

- Empirical distribution E t,i . For each box i ∈ [ n ] , we use m t,i i.i.d. samples { v t i ( j ) ,i } m t,i j =1 to construct E t,i with respect to the underlying distribution D i by assigning each sample with probability mass 1 m t,i . Specifically, F E t,i ( x ) = 1 m t,i ∑ m t,i j =1 I { v t i ( j ) ≤ x } for all x ∈ [0 , 1] .
- Optimistic distribution ˆ E t,i . Let L = 4log(2 nT 2 /δ ) . The CDF of ˆ E t,i is constructed as follows:

<!-- formula-not-decoded -->

As verified in Lemma C.1.4, the construction in Eq. (1) ensures that F ˆ E t,i ( · ) is a valid CDF. We denote E t = ( E t, 1 , . . . , E t,n ) and ˆ E t = ( ˆ E t, 1 , . . . , ˆ E t,n ) . This type of construction has been previously used in [Guo et al., 2019, 2021], but our use differs substantially in the analysis. The following lemma shows that the optimistic product distribution stochastically dominates the underlying product distribution.

Lemma 3.1.2. With probability at least 1 -δ , for all t ∈ [ T ] , we have ˆ E t ⪰ SD D .

It is noteworthy that our approach to constructing the optimistic distribution is more adaptive than that of [Agarwal et al., 2024]. In particular, Agarwal et al. [2024] move a fixed amount of probability mass, approximately 1 / √ m t,i , to the maximal value that D i can take. In contrast, we adjust the CDF based on a data-dependent confidence interval, resulting in a more fine-grained and adaptive optimistic distribution. As stated in the following theorem, this construction results in a tighter regret bound.

Theorem 3.1.3. For Pandora's Box problem, with δ = T -1 Algorithm 2 ensures Reg T = ˜ O ( √ nT ) .

Again, our bound matches the Ω( √ nT ) lower bound of Gatmiry et al. [2024] up to logarithmic factors. It is worth noting that their lower bound is proven under the easier full-information setting and thus applicable to our setting.

Remark 3.1.4 ( High-probability regret bound ) . Under regret metric ∑ T t =1 ( R ( σ D ; D ) -R ( σ t ; D )) , our proposed algorithm also achieves a high-probability bound of ˜ O ( √ nT ) via two modifications only in our analysis. We refer readers to Appendix C.6 for details.

## 3.2 Analysis

In this subsection, we sketch the main proof ideas for Theorem 3.1.3 and summarize the new analytical techniques. We start by leveraging a monotonicity property for Pandora's Box problem, established by Guo et al. [2021]: for any distributions D,E , if E ⪰ SD D , then R ( σ E ; E ) ≥ R ( σ D ; D ) . Since

## Algorithm 2 Near-optimal algorithm for Pandora's Box

Input : confidence δ ∈ (0 , 1) , horizon T . Initialize : open all boxes once and observe rewards. Set m 1 ,i = 1 for all i ∈ [ n ] . for t = 1 , 2 , . . . , T do For each i ∈ [ n ] , construct an optimistic distribution ˆ E t,i according to Eq. (1). Compute σ t = ( σ t, 1 , . . . , σ t,n ) where σ t,i is the solution of E x ∼ ˆ E t,i [ ( x -σ t,i ) + ] = c i . Run Algorithm 1 with σ t to open a set of boxes, denoted by B t , and observe rewards { v t,i } i ∈B t . Update counters m t +1 ,i = m t,i +1 , ∀ i ∈ B t and m t +1 ,i = m t,i , ∀ i / ∈ B t .

the algorithm uses threshold σ t = σ ˆ E t at each round t and Lemma 3.1.2 shows ˆ E t ⪰ SD D , the regret at round t , denoted by Reg ( t ) , is bounded as

<!-- formula-not-decoded -->

For simplicity of the exposition, without loss of generality, we assume that σ t, 1 ≥ σ t, 2 ≥ · · · ≥ σ t,n . Following Agarwal et al. [2024], we define

<!-- formula-not-decoded -->

Therefore H t,n = D so that

<!-- formula-not-decoded -->

The decomposition in Eq. (3) breaks the reward difference into a sequence of steps, where each step replaces one coordinate of the optimistic distribution with its true counterpart, thereby allowing for isolation of the effect of each individual coordinate change.

Let E σ,i be the event that box i is opened under the threshold σ , and let E c σ,i be the corresponding complementary event. We define Q t,i as the probability that σ t opens box i given the true product distribution D . Using the same argument as in [Agarwal et al., 2024], we have that

<!-- formula-not-decoded -->

The key step is to carefully bound Term t,i , and this is where our analysis fundamentally diverges from Agarwal et al. [2024]. Before presenting our analysis, we explain below why the previous analysis fails to achieve the ˜ O ( √ nT ) regret bound.

˜ O ( Un √ T ) bound of Agarwal et al. [2024]. In the analysis of Agarwal et al. [2024], if the product distribution D is discrete, then they bound Term t,i ≤ O ( U · TV ( D i , ˆ E t,i )) (cf. [Agarwal et al., 2024, Lemma 1.3]) where U is the upper bound of the absolute value of function R and could be as large as n . Therefore,

<!-- formula-not-decoded -->

Since their algorithm moves probability mass by a fixed amount, approximately ˜ O (1 / √ m t,i ) for each distribution i , a simple calculation gives TV ( D i , ˆ E t,i ) ≤ ˜ O (1 / √ m t,i ) , which in turn implies that

<!-- formula-not-decoded -->

where the second inequality follows from the standard analysis of optimistic algorithms (e.g., [Slivkins, 2019]), and the last inequality uses the fact that ∑ n i =1 Q t,i ≤ n for all t . It is noteworthy that

∑ n i =1 Q t,i = 1 appears in multi-armed bandit problems since only one arm is played in each round. However, in the problems we consider, there is a chance to open all boxes in one round.

Refined analysis for ˜ O ( √ nT ) bound. To get the ˜ O ( √ nT ) regret bound, it suffices to show Reg ( t ) ≤ ˜ O ( √∑ i Q t,i /m t,i ) since

<!-- formula-not-decoded -->

where the second inequality follows from the Cauchy-Schwarz inequality, and the last inequality results from the bound ∑ t,i Q t,i /m t,i ≤ ˜ O ( n ) . For any threshold σ ∈ [0 , 1] n and z ∈ [0 , 1] , define

<!-- formula-not-decoded -->

Based on these definitions and the fact that E σ t ,i is independent of i -th box's reward, we write

<!-- formula-not-decoded -->

The following lemma is a key enabler for our refined analysis of both problems.

Lemma 3.2.1 ( 1 -Lipschitzness and monotonicity ) . Consider the Pandora's box problem. For all t ∈ [ T ] , i ∈ [ n ] , the function ˜ R i ( σ t ; z ) is 1 -Lipschitz and monotonically-increasing with respect to z .

Since ˜ R i ( σ t ; z ) is 1-Lipschitz with respect to z , the map ˜ R i ( σ t ; z ) is absolutely continuous on [0 , 1] , which implies that it is differentiable almost everywhere in [0 , 1] . By the fundamental theorem of calculus, for any x ∈ [0 , 1] , we have that ˜ R i ( σ t ; x ) = ˜ R i ( σ t ; 0) + ∫ x 0 ∂ ∂z ˜ R i ( σ t ; z ) dz. Then it is straightforward to deduce that

<!-- formula-not-decoded -->

where the inequality holds since ˜ R i ( σ t ; z ) is 1 -Lipschitz with respect to z and thus its derivative is bounded in [ -1 , 1] almost everywhere. Here, we decompose Term t,i into two parts A t,i and B t,i to isolate contributions from different value regions, enabling a problem-dependent analysis that yields a sharper regret bound. An advantage of this analysis is to avoid using [Agarwal et al., 2024, Lemma 1.3], which incurs a linear dependence on U . Indeed, we have

<!-- formula-not-decoded -->

where the second inequality bounds | F D i ( z ) -F E t,i ( z ) | by a Bernstein-type Dvoretzky-KieferWolfowitz inequality (cf. Lemma C.1.2 in Appendix C.1) and bounds | F E t,i ( z ) -F ˆ E t,i ( z ) | by the

construction Eq. (1); the third inequality bounds 1 -F D i ( z ) ≤ 1 -F D i ( σ t,i +1 ) , and an analogous inequality holds for 1 -F E t,i ( z ) since z ≥ σ t,i +1 ; the last inequality replaces F E t,i ( σ t,i +1 ) with F D i ( σ t,i +1 ) by paying a term of the same order (cf. Eq. (19)).

To handle the term B t,i , instead of bounding the derivative by one as before, we introduce the following lemma which more tightly bounds the derivative by the probability that the first i -1 boxes have values no larger than z , conditioning on opening box i .

Lemma 3.2.2. Suppose that σ t, 1 ≥ σ t, 2 ≥ · · · ≥ σ t,n . Pandora's Box problem satisfies

<!-- formula-not-decoded -->

For shorthand, define a z i = ∏ j ≤ i F D j ( z ) . By Lemma 3.2.2 and some calculations (deferred to Appendix C), we have that

<!-- formula-not-decoded -->

where the second inequality follows from a z i -1 ≤ a σ t,i +1 i -1 ≤ a σ t,i i -1 = Q t,i for all z &lt; σ t,i +1 , and the last inequality repeats the same argument used to bound A t,i to bound | F D i ( z ) -F E t,i ( z ) | and | F E t,i ( z ) -F ˆ E t,i ( z ) | , respectively.

Plugging the above bounds of A t,i into Eq. (3) and Eq. (4), the regret contribution of { A t,i } i at round t (omitting L/m t,i and some constant) is bounded by

<!-- formula-not-decoded -->

where the first inequality follows from the Cauchy-Schwarz inequality, and the second inequality follows from ∑ i ∈ [ n ] Q t,i (1 -F D i ( σ t,i +1 )) ≤ 1 . Indeed, since Q t,i = a σ t,i i -1 and σ t, 1 ≥ σ t, 2 . . . ≥ σ t,n , we have that Q t,i ≤ a σ t,j +1 i -1 , which in turn yields the telescoping sum ∑ n i =1 Q t,i (1 -F D i ( σ t,i +1 )) ≤ ∑ n i =1 ( a σ t,j +1 i -1 -a σ t,j +1 i ) ≤ 1 .

Similarly, the regret contribution of { B t,i } i at round t (omitting L/m t,i and some constant) is bounded by

<!-- formula-not-decoded -->

where the first inequality uses the Cauchy-Schwarz inequality and the second inequality uses the telescoping sum and a z i ≤ 1 for all i, z . Hence, the cumulative regret is bounded by ˜ O ( √∑ i Q t,i /m t,i ) . Combining all the results completes the proof of Theorem 3.1.3.

## 4 ˜ O ( nd √ T ) Regret Bound for Contextual Linear Pandora's Box

In this section, we present Algorithm 5 (in Appendix D) for the contextual linear Pandora's Box problem. While our algorithm inherits the main structure of Algorithm 2, it introduces key modifications to ensure optimism, which is critical in the contextual setting. To ensure the optimism, we need to construct an optimistic product distribution which stochastically dominates the underlying product distribution D t for each round t . In the non-contextual case, the optimistic distribution is constructed by using iid samples to first construct empirical distribution and then move the probability mass.

However, due to context-dependent shifts, this strategy no longer applies. To address this, we leverage the fact that the shift in each box's distribution is determined by a fixed but unknown vector θ i .

A natural idea here is to construct empirical distribution by using samples { z t t i ( j ) ,i } m t,i j =1 where z t t i ( j ) ,i = η t i ( j ) ,i + µ t,i , since they can be treated as m t,i iid samples drawn from D t,i . However, the learner never directly observes the noise sample η t i ( j ) ,i . To overcome this, we maintain an estimation ̂ θ t,i at each round t by using ridge regression on past observations. Specifically, for each opened box i ∈ B t , we make the following updates: ̂ θ t,i = V -1 t,i ∑ s ≤ t : i ∈B s x s,i v s,i where V t,i = I + ∑ s ≤ t : i ∈B s x s,i x ⊤ s,i . Let X denote the context space. By the analysis of linear bandits (e.g., [Abbasi-Yadkori et al., 2011]), for each box i ∈ [ n ] , there exists a known function β t i : X × (0 , 1) → R &gt; 0 such that with probability at least 1 -δ , for all x ∈ X and all t ,

<!-- formula-not-decoded -->

With a confidence radius of θ , we construct a value-optimistic empirical distribution E t,i by assigning each of the following with equal probability mass m -1 t,i .

<!-- formula-not-decoded -->

where for any context x ∈ X , any failure probability δ ∈ (0 , 1) , and any θ ∈ R d

<!-- formula-not-decoded -->

The construction of E t,i is value-optimistic because ˆ z t t i ( j ) ,i ≥ z t t i ( j ) ,i holds. If one has m t,i samples at round t for box i , then E t,i stochastically dominates the empirical distribution constructed by using samples { z t t i ( j ) ,i } m t,i j =1 . Then, for each box i , we construct the optimistic distribution ˆ E t,i by moving a small amount probability mass for the value-optimistic empirical distribution E t,i so that it dominates the true distribution (the detailed construction of ˆ E t,i and the proof are deferred to Appendix D).

Lemma 4.1. With probability at least 1 -δ , for all t ∈ [ T ] , ˆ E t ⪰ SD D t .

Once the optimism is ensured, we prove the following regret bound for Algorithm 5 in Appendix D.

Theorem 4.2. For contextual linear Pandora's Box problem, choosing δ = T -1 for Algorithm 5 ensures Reg T = ˜ O ( nd √ T ) regret bound.

While the algorithm of [Atsidakou et al., 2024] applies to our setting with ˜ O ( nT 5 / 6 ) regret, Theorem 4.2 gives a √ T -type regret bound in the contextual linear setting, which is better whenever d = o ( T 1 / 3 ) . The tightness of ˜ O ( nd √ T ) remains an open question. In the contextual linear bandit setting with θ 1 = · · · = θ n , the known lower bound is Ω( d √ T ) [Dani et al., 2008]. Our setting however requires learning a separate noise distribution for each box, and we conjecture that this makes the linear dependence on n unavoidable, even when all boxes share the same θ .

Analysis. As Lemma 4.1 ensures that ˆ E t ⪰ SD D t for all t , we follow the same argument in the non-contextual case to arrive at Eq. (4). The decomposition of the main term in Eq. (4) slightly differs from that of the non-contextual case. We introduce an empirical distribution ˆ D t,i for the decomposition, which is constructed by assigning equal probability mass m -1 t,i for each { z t t i ( j ) ,i } m t,i j =1 . Notice that conditioning on history before t , ˆ D t is a product distribution. Then, conditioning on the history before t , for any i , the main term is rewritten as

<!-- formula-not-decoded -->

Here, term (I) captures the error of optimistic reweighting, which arises from shifting probability mass in the empirical distribution E t,i to construct the optimistic distribution ˆ E t,i . Term (III) quantifies the estimation error between the empirical distribution ˆ D t,i and the true underlying distribution D i . Both can be bounded in a similar way as in the non-contextual case. It thus remains to handle (II), which captures the error incurred by the optimistic value shifts. One can show that

<!-- formula-not-decoded -->

where the inequality uses 1 -Lipschitz property (see Lemma 3.2.1). Following standard linear bandit analysis (e.g., [Abbasi-Yadkori et al., 2011]), the total regret incurred by (II) for all boxes is bounded by (omitting constant)

<!-- formula-not-decoded -->

The first summation term can be bounded by

<!-- formula-not-decoded -->

where the last inequality applies the Cauchy-Schwarz inequality followed by the elliptical potential lemma [Abbasi-Yadkori et al., 2011, Lemma 11]. The second summation term can be bounded similarly. We refer readers to Appendix D for the omitted details.

## 5 Conclusion and Future Work

In this paper, we propose new algorithms for the Pandora's Box problem in both non-contextual and contextual settings. In the non-contextual case, our algorithm achieves a regret of ˜ O ( √ nT ) which matches the known lower bound up to logarithmic factors. For the contextual case, we design a modified algorithm that attains ˜ O ( nd √ T ) regret. We further extend our methods and analysis to the Prophet Inequality problem in both settings, achieving similar improvements and regret bounds. Our results also elicit compelling open questions:

1. Minimax regret for Prophet Inequality. We establish an upper bound of ˜ O ( √ nT ) for Prophet Inequality problem, while Gatmiry et al. [2024] proved a lower bound of Ω( √ T ) , leaving a √ n -gap. In fact, Jin et al. [2024] show that the optimal sample complexity of Prophet Inequality is ˜ O ( 1 ϵ 2 ) , independent of n , which suggests that a simple explore-then-commit strategy achieves an n -independent regret upper bound of ˜ O ( T 2 / 3 ) . Whether ˜ O ( √ T ) is achievable remains open.
2. Tighter regret for contextual linear case. In the non-contextual case, our bound scales as √ n , while in the contextual linear setting, it grows linearly with n . Can this dependence be improved? It also remains unclear whether the linear dependence on the dimension d is improvable.

## Acknowledgments and Disclosure of Funding

HL is supported by NSF Award IIS-1943607. LJR and JL are supported in part by ONR YIP N000142012571, and NSF awards AF-2312775, CPS-1844729.

## References

- Y. Abbasi-Yadkori, D. Pál, and C. Szepesvári. Improved algorithms for linear stochastic bandits. In Advances in Neural Information Processing Systems , 2011.
- A. Agarwal, R. Ghuge, and V. Nagarajan. Semi-bandit learning for monotone stochastic optimization. In 2024 IEEE 65th Annual Symposium on Foundations of Computer Science (FOCS) , pages 1260-1274. IEEE, 2024.

- S. Alaei. Bayesian combinatorial auctions: Expanding single buyer mechanisms to many buyers. SIAM Journal on Computing , 43(2):930-972, 2014.
- A. Atsidakou, C. Caramanis, E. Gergatsouli, O. Papadigenopoulos, and C. Tzamos. Contextual pandora's box. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 10944-10952, 2024.
- P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning , 2002.
- H. Beyhaghi and L. Cai. Pandora's problem with nonobligatory inspection: Optimal structure and a ptas. In Proceedings of the 55th Annual ACM Symposium on Theory of Computing , pages 803-816, 2023.
- H. Beyhaghi and L. Cai. Recent developments in pandora's box problem: Variants and applications. ACM SIGecom Exchanges , 21(1):20-34, 2024.
- S. Boodaghians, F. Fusco, P. Lazos, and S. Leonardi. Pandora's box problem with order constraints. In Proceedings of the 21st ACM Conference on Economics and Computation , pages 439-458, 2020.
- S. Chawla, E. Gergatsouli, Y. Teng, C. Tzamos, and R. Zhang. Pandora's box with correlations: Learning and approximation. In 2020 IEEE 61st Annual Symposium on Foundations of Computer Science (FOCS) , pages 1214-1225. IEEE, 2020.
- V. Dani, T. P. Hayes, and S. M. Kakade. Stochastic linear optimization under bandit feedback. In Conference on Learning Theory , 2008.
- T. Ezra, M. Feldman, N. Gravin, and Z. G. Tang. Prophet matching with general arrivals. Mathematics of Operations Research , 47(2):878-898, 2022.
- M. Feldman, O. Svensson, and R. Zenklusen. Online contention resolution schemes with applications to bayesian selection problems. SIAM Journal on Computing , 50(2):255-300, 2021.
- H. Fu and T. Lin. Learning utilities and equilibria in non-truthful auctions. Advances in Neural Information Processing Systems , 33:14231-14242, 2020.
- H. Fu, J. Li, and D. Liu. Pandora box problem with nonobligatory inspection: Hardness and approximation scheme. In Proceedings of the 55th Annual ACM Symposium on Theory of Computing , pages 789-802, 2023.
- K. Gatmiry, T. Kesselheim, S. Singla, and Y. Wang. Bandit algorithms for prophet inequality and pandora's box. In Proceedings of the 2024 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA) , pages 462-500. SIAM, 2024.
- E. Gergatsouli and C. Tzamos. Online learning for min sum set cover and pandora's box. In International Conference on Machine Learning , pages 7382-7403. PMLR, 2022.
- E. Gergatsouli and C. Tzamos. Weitzman's rule for pandora's box with correlations. Advances in Neural Information Processing Systems , 36:12644-12664, 2023.
- N. Gravin and H. Wang. Prophet inequality for bipartite matching: Merits of being simple and non adaptive. In Proceedings of the 2019 ACM Conference on Economics and Computation , pages 93-109, 2019.
- C. Guo, Z. Huang, and X. Zhang. Settling the sample complexity of single-parameter revenue maximization. In Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing , pages 662-673, 2019.
- C. Guo, Z. Huang, Z. G. Tang, and X. Zhang. Generalizing complex hypotheses on product distributions: Auctions, prophet inequalities, and pandora's problem. In Conference on Learning Theory , pages 2248-2288. PMLR, 2021.

- J. Jiang, W. Ma, and J. Zhang. Tight guarantees for multi-unit prophet inequalities and online stochastic knapsack. In Proceedings of the 2022 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA) , 2022.
- B. Jin, T. Kesselheim, W. Ma, and S. Singla. Sample complexity of posted pricing for a single item. arXiv preprint arXiv:2406.00819 , 2024.
- R. Kleinberg and S. M. Weinberg. Matroid prophet inequalities. In Proceedings of the forty-fourth annual ACM symposium on Theory of computing , pages 123-136, 2012.
- R. Kleinberg and S. M. Weinberg. Matroid prophet inequalities and applications to multi-dimensional mechanism design. Games and Economic Behavior , 113:97-115, 2019.
- U. Krengel and L. Sucheston. Semiamarts and finite values. 1977.
- B. Kveton, C. Szepesvari, Z. Wen, and A. Ashkan. Cascading bandits: Learning to rank in the cascade model. In International conference on machine learning , pages 767-776. PMLR, 2015.
- T. Lattimore and C. Szepesvári. Bandit algorithms . Cambridge University Press, 2020.
- E. Samuel-Cahn. Comparison of threshold stop rules and maximum for independent nonnegative random variables. the Annals of Probability , pages 1213-1216, 1984.
- S. Singla. The price of information in combinatorial optimization. In Proceedings of the twenty-ninth annual ACM-SIAM symposium on discrete algorithms , pages 2523-2532. SIAM, 2018.

A. Slivkins. Introduction to multi-armed bandits. Found. Trends Mach. Learn. , 12(1-2):1-286, 2019.

M. Weitzman. Optimal search for the best alternative , volume 78. Department of Energy, 1978.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We provide clear motivations for the research questions in the introduction section and explicitly itemize our contributions, outlining how we address and resolve these challenges throughout the paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: In this paper, for the contextual model, we only study the linear structure of rewards for both Pandora's Box and Prophet Inequality problems.

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

Justification: All assumptions are clearly stated in Section 2 and we defer the proof to Appendix.

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

Justification: This paper is theoretically oriented. There are no experiments.

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

Justification: This paper is theoretically oriented. There are no experiments.

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

Justification: This paper is theoretically oriented. There are no experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: This paper is theoretically oriented. There are no experiments.

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

Justification: This paper is theoretically oriented. There are no experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have reviewed NeurIPS Code of Ethics and commit to it.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: In the introduction, we highlight that our research can benefit sequential decision-making problems, modeled by Pandora's Box and Prophet Inequality. We do not foresee any negative societal impacts arising from this work.

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

Justification: This paper is theoretically oriented and does not release any data or models.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: This is a theory-oriented paper with no experiments or code.

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

Justification: This is a theory-oriented paper. There are no new assets introduced.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This is a theory-oriented paper. There are no experiments or research with human objects in this paper.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This is a theory-oriented paper. There are no experiments or research with human objects in this paper.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: This work does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Appendix

| A   | Problem Setting of Prophet Inequality            | Problem Setting of Prophet Inequality                                             |   21 |
|-----|--------------------------------------------------|-----------------------------------------------------------------------------------|------|
| B   | Related Work                                     | Related Work                                                                      |   21 |
| C   | Omitted Details from the Non-Contextual Settings | Omitted Details from the Non-Contextual Settings                                  |   23 |
|     | C.1                                              | Technical Lemmas . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    |   23 |
|     | C.2                                              | Lipschitz Property, Monotonicity, and Sharp Derivative for Pandora's Box . .      |   24 |
|     | C.3                                              | Lipschitz Property, Monotonicity, and Sharp Derivative for Prophet Inequality     |   27 |
|     | C.4                                              | Proof of ˜ O ( √ nT ) Regret Bound for Pandora's Box . . . . . . . . . . . . .    |   28 |
|     | C.5                                              | ˜ O ( √ nT ) Regret Bound for Prophet Inequality . . . . . . . . . . . . . . . .  |   31 |
|     | C.6                                              | High-Probability Regret Bounds . . . . . . . . . . . . . . . . . . . . . . . .    |   32 |
|     | C.7                                              | Extension to Subgaussian Rewards . . . . . . . . . . . . . . . . . . . . . .      |   32 |
| D   | Omitted Details of Contextual Linear Setting     | Omitted Details of Contextual Linear Setting                                      |   33 |
|     | D.1                                              | Preliminaries . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |   33 |
|     | D.2                                              | Algorithms and ˜ O ( nd √ T ) Bounds for Pandora's Box and Prophet Inequality     |   34 |
|     | D.3                                              | Proof of Lemma D.2.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    |   36 |

Algorithm 3 Generic threshold algorithm for Prophet Inequality Input : threshold σ = ( σ 1 , . . . , σ n ) . The environment generates an unknown realization v = ( v 1 , . . . , v n ) . for i = 1 , . . . , n do Open box i to observe v i . If v i ≥ σ i or i = n , then stop and take reward v i .

## A Problem Setting of Prophet Inequality

Online Prophet Inequality. In this problem, the learner is given n boxes, each with an unknown fixed reward distribution D i supported on [0 , 1] . The learner plays a T -round repeated game where at each round t ∈ [ T ] , the learner inspects boxes in a fixed order. Upon opening a box i , the learner receives a reward v t,i ∈ [0 , 1] independently sampled from D i . Then, she needs to decide immediately whether to accept this reward and stop the process at round t or discard this reward and open the next box. The utility for the round is the chosen reward.

Contextual Linear Prophet Inequality. This setting extends the non-contextual one, and the assumptions on reward distributions and boundedness are the same as Section 2.

Optimal Policy and Regret. To measure the performance, we compare our algorithm against the per-round optimal policy. Let D t = ( D t, 1 , ..., D t,n ) be the product distribution of boxes. The optimal policy for Prophet Inequality in round t is also a threshold policy, in which the threshold for each box i , denoted by σ ∗ t,i is derived from a reverse/backward programming .

<!-- formula-not-decoded -->

The optimal policy at round t is to run Algorithm 3 with threshold vector σ ∗ t . In fact, the optimal threshold of box i computed from Definition A.1 is equal to the expected reward obtained by running σ ∗ t,i , . . . , σ ∗ t,n directly from box i [Gatmiry et al., 2024].

More formally, similar to those definitions in Pandora's Box setting, for any product distribution D = ( D 1 , ..., D n ) , let σ D = ( σ D, 1 , . . . , σ D,n ) denote the optimal threshold vector, where each σ D,i is the threshold calculated by reverse programming. For any threshold vector σ = ( σ 1 , ..., σ n ) ∈ [0 , 1] n and realization v = ( v 1 , ..., v n ) ∈ [0 , 1] n , we use S ( σ ; v ) ∈ [ n ] corresponds to the box where Algorithm 3 stops. Further, we define the utility function and expected utility function, respectively:

<!-- formula-not-decoded -->

With these definitions, the optimal expected utility is R ( σ D t ; D t ) since when v ∼ D t , S ( σ D t ; v ) ∈ [ n ] corresponds to the box chosen by the optimal policy. Let σ t ∈ R n denote the threshold vector selected by the algorithm at round t . Throughout the paper, all our algorithms in the Prophet Inequality setting follow the general structure outlined in Algorithm 3 using σ t as input at round t , and thus the expected utility of our algorithms at round t is R ( σ t ; D t ) . The cumulative regret in T rounds Reg T is then defined as

<!-- formula-not-decoded -->

where D t = D for the non-contextual case.

## B Related Work

The Pandora's Box problem, introduced by Weitzman [1978], has been extended in various directions, including matroid constraints [Singla, 2018], correlated distributions [Chawla et al., 2020, Gergatsouli and Tzamos, 2023], order constraints [Boodaghians et al., 2020], and settings without inspection [Fu et al., 2023, Beyhaghi and Cai, 2023]. The Prophet Inequality problem was first studied by Krengel and Sucheston [1977], Samuel-Cahn [1984], with subsequent generalizations to k -selection [Alaei, 2014, Jiang et al., 2022], matroid constraints [Kleinberg and Weinberg, 2012, 2019, Feldman et al., 2021], and online matching [Gravin and Wang, 2019, Ezra et al., 2022]. These works typically assume known distributions and focus on achieving constant-factor approximation guarantees.

A growing body of work has begun to study online variants of the Prophet Inequality and Pandora's Box problems [Guo et al., 2019, Fu and Lin, 2020, Guo et al., 2021, Gatmiry et al., 2024, Agarwal et al., 2024]. In the PAC setting, Fu and Lin [2020], Guo et al. [2021] establish a sample complexity upper bound of ˜ O ( n ϵ 2 ) for Pandora's Box, matching the known lower bound up to logarithmic factors, where ϵ denotes the desired accuracy. Guo et al. [2021] further develop a general framework for learning over product distributions, which applies to both problems and yields a sample complexity of ˜ O ( n ϵ 2 ) for Prophet Inequality as well. Later, Jin et al. [2024] provide a near-optimal sample complexity ˜ O ( 1 ϵ 2 ) for Prophet Inequality, which removes the linear dependence on n . For the regret minimization problem, Gatmiry et al. [2024] establish ˜ O ( n 3 √ T ) regret bound for Prophet Inequality and ˜ O ( n 4 . 5 √ T ) regret bound for Pandora's Box under a constrained feedback model, where the learner observes only the final reward without knowing which box produced it. Subsequently, Agarwal et al. [2024] introduce a general framework for monotone stochastic optimization under semi-bandit feedback, achieving ˜ O ( n √ T ) regret for both problems. All these works assume a fixed but unknown product distribution. Moreover, despite this progress, the best known regret bounds for both Prophet Inequality and Pandora's Box under semi-bandit feedback remain at ˜ O ( n √ T ) , which falls short of the known lower bounds [Gatmiry et al., 2024].

To better capture the dynamics in decision-making scenarios, Gergatsouli and Tzamos [2022] study the Pandora's Box problem in the adversarial setting where the reward of each box is chosen by an adversary and obtain sublinear regret bounds by comparing the algorithm against a weak benchmark. Later, Gatmiry et al. [2024] show that in the same setting, when the algorithm competes against the optimal policy, no algorithm can achieve sublinear regret, even with full information feedback. To make the problem tractable while still allowing distributions to change over time, Atsidakou et al. [2024] explore a contextual Pandora's Box model where in each round, the learner observes a context vector for each box, whose optimal threshold can be approximated by a function of the context and an unknown vector. They provide a reduction from contextual Pandora's Box to an instance of online regression. Their algorithm suffers a ˜ O ( T 5 / 6 ) regret bound if the threshold can be linearly approximated. We note that their algorithm for the linear case also applies to our setting (see the discussion below), but our algorithm enjoys ˜ O ( nd √ T ) , which is improves on the aforementioned bound whenever d = o ( T 1 / 3 ) .

Discussion of [Atsidakou et al., 2024]. Here, we show how the algorithm proposed by Atsidakou et al. [2024] can be applied to our setting. Their work makes the following realizability assumption.

Assumption 1 (Realizability of [Atsidakou et al., 2024]) . There exists w 1 , . . . , w n ∈ R d and a function h such that every time t ∈ [ T ] and every box i ∈ [ n ] , the optimal threshold for distribution D t,i is equal to h ( w i , x ′ t,i ) , i.e., σ ∗ t,i = h ( w i , x ′ t,i ) where x ′ t,i is a context vector of box i at round t and

<!-- formula-not-decoded -->

Below, we show that in our setting, the optimal threshold can be realized by a linear function h (that is, h ( a, b ) = a ⊤ b ). To see this, first note that the optimal threshold σ ∗ t,i in our setting satisfies E X ∼ D t,i [ ( X -σ ∗ t,i ) + ] = c i , which, based on the definition of D t,i , is equivalent to E X ∼ D i [ ( X -( σ ∗ t,i -µ t,i )) + ] = c i where µ t,i = θ ⊤ i x t,i . Therefore, if we let σ ∗ i be the unique solution of E X ∼ D i [( X -σ ∗ i ) + ] = c i , then we have

<!-- formula-not-decoded -->

for w i = ( θ i, 1 , . . . , θ i,d , σ 1 , . . . , σ n ) ∈ R d + n , and x ′ t,i = ([ x t,i ] 1 , . . . , [ x t,i ] d , 0 , . . . , 1 , . . . , 0) ∈ R d + n , where x ′ t,i takes value one at the d + i index. This proves that the optimal threshold is realized by a linear function of some context, making the algorithm of [Atsidakou et al., 2024] applicable to our setting.

A seemingly related strand is cascading bandits [Kveton et al., 2015], where the learner presents an ordered list of items to a user, with items corresponding to boxes in our setting. Then, the user scans the list and clicks the first attractive item if any. The learner observes binary click signals for items up to that click, which yields order dependent semi bandit feedback. Despite this shared feedback structure, Pandora's Box and Prophet Inequality differ in who controls how many items are inspected. In cascading bandits the number is user driven since observation stops at the first click. In Pandora's

Box and in Prophet Inequality the number is policy driven since the algorithm adaptively opens boxes and decides when to stop.

## C Omitted Details from the Non-Contextual Settings

## C.1 Technical Lemmas

Lemma C.1.1 (Bernstein inequality) . Given m ∈ N , let X 1 , X 2 , . . . , X m be i.i.d. random variables such that E [ X i ] = 0 , E [ X 2 i ] = σ 2 , and | X i | ≤ M for some constant M &gt; 0 . Then, for all t &gt; 0

<!-- formula-not-decoded -->

Lemma C.1.2. Let D = ( D 1 , . . . , D n ) be a [0 , 1] -bounded product distribution, and for each i , let E N i ,i be the empirical distribution constructed by assigning N i i.i.d. samples from D i with equal probability mass 1 /N i . With probability at least 1 -δ , for any i ∈ [ n ] , any N i ∈ [ T ] , and any z ∈ [0 , 1] ,

<!-- formula-not-decoded -->

Proof. This proof follows a similar idea to Guo et al. [2019, Lemma 5], but we apply an additional union bound to account for all possible N i . Fix i and N i . It is suffices to show that for all z ∈ [0 , 1] such that F D i ( z ) is multiples of 1 /N i , the following holds.

<!-- formula-not-decoded -->

This is because given the results above, the general bound on the difference of CDF of any value differs at most an extra additive factor 1 N i ≤ log(2 T 2 nδ -1 ) 3 N i .

Then fix any z such that F D i ( z ) is multiples of 1 /N i . Let Y i be the indicator function of the i -th sample no smaller than z and let X i = Y i -E [ Y i ] . By Bernstein inequality (see Lemma C.1.1), taking t be N i times the RHS of Eq. (11),

<!-- formula-not-decoded -->

Since N i ≤ T , by a union bound over all N i , with probability at least 1 -δ nT , Eq. (11) holds for all z such that F D i ( z ) is multiples of 1 /N i . Further, by union bounds over all i ∈ [ n ] and N i ∈ [ T ] , we get that Eq. (10) does not hold with probability at most δ .

As F D i ( · ) is unknown, we present the following corollary, which provides a confidence bound depending on known F E i ( · ) .

Corollary C.1.3. Under the same setting of Lemma C.1.2, suppose that the concentration bounds of Lemma C.1.2 all hold, and then for any i ∈ [ n ] , any N i ∈ [ T ] and any x ∈ [0 , 1] ,

<!-- formula-not-decoded -->

Proof. Fix any i and N i . For notional simplicity, we write E i = E N i ,i . For any x ∈ [0 , 1] ,

<!-- formula-not-decoded -->

where the last inequality uses the fact that function y -y 2 is 1 -Lipschitz for y ∈ [0 , 1] .

Let L ′ = log(2 nT 2 δ -1 ) . From Eq. (10), we can show for any x ∈ [0 , 1]

<!-- formula-not-decoded -->

where the second inequality uses Eq. (13), and the last inequality uses 2 √ ab ≤ a + b for any a, b ≥ 0 (here a = L ′ /N i and b = 1 2 | F D i ( x ) -F E i ( x ) | ). Rearranging the above inequality gives the claimed bound for fixed i, N i . Since this argument holds for all i, N i , the lemma thus follows.

Lemma C.1.4. For all t, i , F ˆ E t,i ( · ) constructed by Eq. (1) is a valid CDF.

Proof. To show F ˆ E t,i ( · ) is valid, we show that it is a right-continuous, non-decreasing function in the range of [0 , 1] . Consider any given t, i . From Eq. (1), one can easily see F ˆ E t,i ( x ) ∈ [0 , 1] for all x ∈ [0 , 1] . As F E t,i ( · ) is right-continuous and max function preserves the right-continuous property, F ˆ E t,i ( · ) is again right-continuous.

Now, we verify the monotonicity of F ˆ E t,i ( · ) . Since F E t,i ( · ) is non-decreasing, we only need to show F ˆ E t,i ( x ) is non-decreasing as a function of F E t,i ( x ) . Let k = L m t,i . For the interval of x such that F E t,i ( x ) &lt; k , F ˆ E t,i ( x ) is 0 , and thus it is non-decreasing. Then we only need to consider the interval of x such that F E t,i ( x ) ≥ k . We let y = F E t,i ( x ) , and rewrite the formulation for F ˆ E t,i ( x ) as f ( y ) = y -√ 2 ky (1 -y ) -k . In this case, f ′ ( y ) = 1 -√ k (1 -2 y ) √ 2 y (1 -y ) . When y ≥ 1 2 , this is obviously non-negative, and thus f ( y ) is non-decreasing. Then, we consider the case of y &lt; 1 2 and show that √ 2 y (1 -y ) &gt; √ k (1 -2 y ) . As y ≥ k , it is enough to show √ 2 y (1 -y ) &gt; √ y (1 -2 y ) . This is equivalent to show 4 y 2 -2 y -1 &lt; 0 when y &lt; 1 2 , which can be easily verified.

As the argument holds for all t, i , the lemma thus follows.

Lemma C.1.5 (Restatement of Lemma 3.1.2) . With probability at least 1 -δ , for all t ∈ [ T ] , ˆ E t ⪰ SD D .

Proof. For this proof, we show ˆ E t,i ⪰ SD D i for all t, i . Fix t, i . It suffices to show that for all x ∈ [0 , 1] , F ˆ E t,i ( x ) ≤ F D i ( x ) . For x = 1 , F ˆ E t,i ( x ) = F D i ( x ) = 1 . For x ∈ [0 , 1) , we have

<!-- formula-not-decoded -->

where the inequality follows from Corollary C.1.3.

Repeating the same argument for all i ∈ [ n ] completes the proof.

## C.2 Lipschitz Property, Monotonicity, and Sharp Derivative for Pandora's Box

Before proving Lipschitz property and monotonicity for ˜ R i ( σ t , z ) , we first introduce the following supporting results.

Definition C.2.1 ( ( n, D, c ) -Pandora's box instance) . We use ( n, D, c ) to characterize a Pandora's box instance where product distribution D = ( D 1 , . . . , D n ) is over n boxes, each of which is associated with a cost c i .

For any ( n, D, c ) -Pandora's box instance, let W u : [0 , 1] n × [0 , 1] n → 2 [ n ] be a general threshold rule with initial value u . Specifically, for any initial value u , threshold σ , and realization x ∼ D , W u ( σ, x ) is the set of opened boxes following rule W u . For any u, σ , if the initial value u ≥ max i σ i , then W u ( σ, · ) = ∅ , that is, the learner does not open any box and keeps the initial value u . Otherwise, the learner keeps the initial value u in hand but continues to open boxes by running Algorithm 1 with V max = u and threshold σ . Let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, R ( σ ; x ; u ; W v ) is the utility if the learner adopts rule W v to run threshold σ with initial value u over realization x . As shown by Weitzman [1978], for any initial value u , the expected utility R ( σ D ; D ; u ; W u ) enjoys the optimality.

Lemma C.2.2 ( Monotonicity and 1 -Lipschitz Property for Pandora's Box ) . For any initial values u, v , if the learner runs general threshold rules W u and W v with u ≥ v , respectively on a ( n, D, c ) -Pandora's Box instance, then we have

<!-- formula-not-decoded -->

Consequenctly, the following 1 -Lipschitz property holds: for any u, v ,

<!-- formula-not-decoded -->

Proof. Consider any u ≥ v . For any realization x = ( x 1 , . . . , x n ) drawn from product distribution D , the utility given the initial value s ( = u or v ) and rule W is

<!-- formula-not-decoded -->

Taking W = W v , and s = u and v , respectively, we have

<!-- formula-not-decoded -->

Given the initial value u and threshold σ D , using rule W u recovers the Weitzman's optimal algorithm. Thus, the corresponding expected utility should be no smaller than that of running rule W v with the same threshold σ D and initial value u , i.e., E x ∼ D [ R ( σ D ; x ; u ; W v )] ≤ E x ∼ D [ R ( σ D ; x ; u ; W u )] , which implies 0 ≤ E x ∼ D [ R ( σ D ; x ; u ; W u )] -E x ∼ D [ R ( σ D ; x ; v ; W v )] .

Taking W = W u , and s = u and v , respectively, we have

<!-- formula-not-decoded -->

where the inequality holds since for any x , f x ( v ) = max { v, x } is 1 -Lipschitz with respect to v .

As W v is optimal given initial value v and threshold σ D , we have E x ∼ D [ R ( σ D ; x ; v ; W u )] ≤ E x ∼ D [ R ( σ D ; x ; v ; W v )] , implying E x ∼ D [ R ( σ D ; x ; u ; W u )] -E x ∼ D [ R ( σ D ; x ; v ; W v )] ≤ u -v . Thus, the lemma follows.

For any x ∈ [0 , 1] n , we define x &gt;i := ( x i +1 , . . . , x n ) and x &lt;i := ( x 1 , . . . , x i -1 ) . Based on the definition of H t,i = ( D 1 , · · · , D i , ˆ E t,i +1 , · · · , ˆ E t,n ) , we define

<!-- formula-not-decoded -->

For any z ∈ [0 , 1] , any x ∈ [0 , 1] n , and any σ ∈ [0 , 1] n , we define

<!-- formula-not-decoded -->

where the expectation is taken only for x &gt;i := ( x i +1 , . . . , x n ) .

Lemma C.2.3 (Restatement of Lemma 3.2.1) . For all t ∈ [ T ] and i ∈ [ n ] , the map ˜ R i ( σ t ; z ) is 1 -Lipschitz and monotonically-increasing with respect to z in the Pandora's Box problem.

Proof. For simplicity, we assume without loss of generality that σ t, 1 ≥ σ t, 2 ≥ · · · ≥ σ t,n . For any z ∈ [0 , 1] and any x ∈ [0 , 1] n , if box i is opened at round t , then the map ˜ R i ( σ t ; x &lt;i , z ) is the expected utility of running σ t on ( x i +1 , . . . , x n ) ∼ H t,&gt;i with initial value max { x 1 , . . . , x i -1 , z } . A key fact here is that σ t = σ ˆ E t and H t,&gt;i = ( ˆ E t,i +1 , . . . , ˆ E t,n ) , which imply that the algorithm runs the optimal threshold in descending order. Thus Lemma C.2.2 implies that ˜ R i ( σ t ; x &lt;i , z ) is 1 -Lipschitz with respect to z and for any u ≥ v , and that ˜ R i ( σ t ; x &lt;i , u ) -˜ R i ( σ t ; x &lt;i , v ) ∈ [0 , u -v ] . This, in turn, implies that

<!-- formula-not-decoded -->

The 1 -Lipschitz property of ˜ R i ( σ ; z ) is an immediate corollary, and the proof is thus complete. LemmaC.2.4 (Restatement of Lemma 3.2.2) . Suppose that σ t, 1 ≥ σ t, 2 ≥ · · · ≥ σ t,n . The Pandora's Box problem satisfies the following:

<!-- formula-not-decoded -->

Proof. According to Lemma 3.2.1, the map ˜ R i ( σ t ; z ) is monotonically increasing with respect to z , which implies that 0 ≤ ∂ ∂z ˜ R i ( σ t ; z ) . Then, we only need to prove for any 0 ≤ v ≤ u &lt; σ t,i +1 ,

<!-- formula-not-decoded -->

To this end, we deduce the following:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( a ) holds since conditioning on E σ t ,i (i.e., opening box i ), the expected reward remains unchanged by replacing u (or v ) by the maximum so far, the inequality uses the 1 -Lipschitz property of ˜ R i ( σ t ; x &lt;i , z ) with respect to variable z , (b) follows from the fact that max( x &lt;i , u ) -max( x &lt;i , v ) is non-zero only when max j&lt;i x j &lt; u as u ≥ v , and (c) holds due to { max j&lt;i x j &lt; u } ⊆ E σ t ,i where this containment follows from the fact that u &lt; σ t,i +1 ≤ σ t,i and E σ t ,i = { max j&lt;i x j &lt; σ t,i } .

Thus, the stated claim holds.

## C.3 Lipschitz Property, Monotonicity, and Sharp Derivative for Prophet Inequality

In the following, we prove the similar results for Prophet Inequality.

Lemma C.3.1. Consider any two product distributions D u = ( u, D 2 , D 3 , . . . , D n ) and D v = ( v, D 2 , D 3 , . . . , D n ) , where the first box deterministically produces rewards u and v , respectively, and the remaining boxes share the same distributions. If u ≥ v , then

<!-- formula-not-decoded -->

As a corollary, we have the following 1 -Lipschitz property. For any u, v ,

<!-- formula-not-decoded -->

Proof. As σ D u is computed by backward induction from Definition A.1, we have R ( σ D u ; D u ) = σ D u , 1 = max { u, σ D u , 2 } . Moreover, since the distribution of boxes from 2 , . . . , n are the same, σ D u ,i = σ D v ,i for all i ≥ 2 . For any u ≥ v ,

<!-- formula-not-decoded -->

where the inequality holds since max function is 1 -Lipschitz and σ D u , 2 = σ D v , 2 . Thus, the lemma follows.

In the Prophet Inequality problem, the definition of ˜ R i ( σ ; z ) takes the same form as in Eq. (6), but uses the utility function R defined in Eq. (9). Similarly, we follow Eq. (15) to define ˜ R i ( σ ; x &lt;i , z ) , again using the utility function defined in Eq. (9).

Lemma C.3.2. For all t ∈ [ T ] , i ∈ [ n ] , ˜ R i ( σ ; z ) is 1 -Lipschitz and monotonically-increasing with respect to z in Prophet Inequality problem.

Proof. As the order of the boxes is fixed across time, we use Lemma C.3.1 to repeat a similar argument in Lemma 3.2.1 to complete the proof. The only difference is that conditioning on event that the learner opens box i , ˜ R i ( σ ; x &lt;i , z ) can be interpreted as the expected utility of running σ t on ( x i +1 , . . . , x n ) ∼ H t,&gt;i with x i = z since x 1 , . . . , x i -1 are discarded.

Lemma C.3.3 ( Sharp Bound of Derivative for Prophet Inequality ) . The Prophet Inequality problem satisfies

<!-- formula-not-decoded -->

Proof. For any 0 ≤ v ≤ u &lt; σ t,i , we have that

<!-- formula-not-decoded -->

where the second equality uses the fact that if the learner opens box i and the realized value is smaller than σ t,i , then they will discard it and keep opening subsequent boxes. Thus, the lemma follows.

## C.4 Proof of ˜ O ( √ nT ) Regret Bound for Pandora's Box

The analysis in this section conditions on the event that the concentration bounds of Lemma C.1.2 hold with respect to D and E t for all t . Recall from Section 3.2 that regret is bounded by

<!-- formula-not-decoded -->

Consider any given round t and suppose, without loss of generality, that

<!-- formula-not-decoded -->

Since ˜ R i ( σ t ; z ) is 1-Lipschitz with respect to z , the map ˜ R i ( σ t ; z ) is absolutely continuous on [0 , 1] , which implies that it is differentiable almost everywhere in [0 , 1] . By the fundamental theorem of calculus, for any x ∈ [0 , 1] , we have that ˜ R i ( σ t ; x ) = ˜ R i ( σ t ; 0) + ∫ x 0 ∂ ∂z ˜ R i ( σ t ; z ) dz . Therefore, we have that

<!-- formula-not-decoded -->

According to the analysis in Section 3.2, we bound

<!-- formula-not-decoded -->

Now, we complete the bounds on A t,i and B t,i .

Bounding A t,i . We have

<!-- formula-not-decoded -->

where the second inequality bounds | F D i ( z ) -F E t,i ( z ) | by Lemma C.1.2 and bounds | F E t,i ( z ) -F ˆ E t,i ( z ) | by the construction Eq. (1), and the third inequality bounds 1 -F D i ( z ) ≤ 1 -F D i ( σ t,i +1 ) (similar for 1 -F E t,i ( z ) ) as z ≥ σ t,i +1 . The last inequality follows from the following reasoning:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the third inequality uses 2 ab ≤ a + b for any a, b ≥ 0 , and the last inequality bounds ∣ ∣ F E t,i ( σ t,i +1 ) -F D i ( σ t,i +1 ) ∣ ∣ by Lemma C.1.2.

Bounding B t,i . By Lemma 3.2.2, we bound

<!-- formula-not-decoded -->

where the second inequality bounds ∏ j&lt;i F D j ( z ) ≤ ∏ j&lt;i F D j ( σ t,i +1 ) ≤ ∏ j&lt;i F D j ( σ t,i ) = Q t,i for all z &lt; σ t,i +1 .

On the one hand, we use Lemma C.1.2 to bound for any z

<!-- formula-not-decoded -->

On the other hand, we use the construction of optimistic distribution ˆ E t,i in Eq. (1) to show for any z

<!-- formula-not-decoded -->

where the last inequality repeats the same argument in Eq. (19).

Plugging the two bounds above into Eq. (20), we have that

<!-- formula-not-decoded -->

where the second inequality follows from

<!-- formula-not-decoded -->

based on the fact that z ≤ σ t,i +1 and Q t,i = ∏ j&lt;i F D j ( σ t,i ) .

Regret for round t . From the bounds of A t,i and B t,i , the regret at round t is bounded as

<!-- formula-not-decoded -->

The first term is bounded as follows:

<!-- formula-not-decoded -->

where the first inequality uses the Cauchy-Schwarz inequality, and the second inequality follows from the the fact that

<!-- formula-not-decoded -->

Here, the first inequality holds since the descending sorted assumption on σ t gives σ t,j +1 ≥ σ t,i for all j &lt; i , and the last inequality follows from the fact that the summation forms a telescoping sum.

To bound the second term, we show that

<!-- formula-not-decoded -->

where the first inequality uses the Cauchy-Schwarz inequality, and the last inequality follows from the fact that ∑ i ∈ [ n ] ∏ j&lt;i F D j ( z )(1 -F D i ( z )) forms a telescoping sum.

Combining all bounds above, we have

<!-- formula-not-decoded -->

Algorithm 4 Proposed algorithm for Prophet Inequality

Input : confidence δ ∈ (0 , 1) , horizon T

.

Initialize : open all boxes once and observe rewards. Set m 1 ,i = 1 for all i ∈ [ n ] .

<!-- formula-not-decoded -->

For each i ∈ [ n ] , construct an optimistic distribution ˆ E t,i according to Eq. (1).

Use Definition A.1 with distribution ˆ E t to compute σ t = ( σ t, 1 , . . . , σ t,n ) .

Run Algorithm 3 with σ t to open a set of boxes, denoted by B t and observe rewards { v t,i } i ∈B t . Update counters m t +1 ,i = m t,i +1 , ∀ i ∈ B t and m t +1 ,i = m t,i , ∀ i / ∈ B t .

Summing Reg ( t ) over all t . The Cauchy-Schwarz inequality gives

<!-- formula-not-decoded -->

where the last inequality uses Jensen's inequality.

Finally, it remains to bound E [ ∑ T t =1 LQ t,i /m t,i ] . Let I t,i = I { E σ t ,i } .

<!-- formula-not-decoded -->

where the second equality holds due to m t,i is deterministic given history before round t .

Combining the bound E [ ∑ T t =1 LQ t,i /m t,i ] = O ( nL log T ) with Eq. (21) completes the proof of Theorem 3.1.3.

## C.5 ˜ O ( √ nT ) Regret Bound for Prophet Inequality

The proposed algorithm for the Prophet Inequality problem is shown in Algorithm 4.

Theorem C.5.1. For the Prophet Inequality problem, choosing δ = T -1 for Algorithm 4 ensures

<!-- formula-not-decoded -->

Proof. The analysis conditions on the event that the concentration bounds of Lemma C.1.2 hold with respect to D and E t for all t . As shown by Guo et al. [2021], the Prophet Inequality satisfies the same monotonicity as Pandora's Box. Thus, the analysis follows the same approach as used in the Pandora's Box problem to bound Reg T ≤ ∑ T t =1 ∑ n i =1 E [ Q t,i · Term t,i ] where Term t,i has the same definition as in Eq. (4) with a new utility function R defined in Eq. (9).

As shown in Lemma C.3.2, the Prophet Inequality problem has both monotonicity and smoothness properties. Therefore, repeating the same analysis in Section 3.2 (splitting the integral on σ t,i instead of σ t,i +1 ) gives

<!-- formula-not-decoded -->

where the equality applies Lemma C.3.3. We use the same argument as in Eq. (18) to bound A t,i : indeed,

<!-- formula-not-decoded -->

Thus, the regret at round t is bounded by

<!-- formula-not-decoded -->

In particular, we bound

<!-- formula-not-decoded -->

where the equality holds since Q = ∏ F ( σ ) in the Prophet Inequality setting.

t,i j&lt;i D j t,j

By repeating the same argument as in Eq. (21) and Eq. (22), the proof of Theorem C.5.1 is complete.

## C.6 High-Probability Regret Bounds

The regret metric now is

<!-- formula-not-decoded -->

Following the same argument in Section 3.2, we arrive at Eq. (4) without expectation. As σ t and H t,i are deterministic given history, we have

<!-- formula-not-decoded -->

where E t [ · ] is the conditional expectation given history before round t . Then, we can follow the same analysis to arrive at √ ∑ i,t Q t,i m t,i . To handle this, let M t,i = Q t,i -I t,i m t,i , and { M t,i } t is martingale difference sequence. By Freedman's inequality, with probability at least 1 -δ , we have ∑ t M t,i ≤ √ 2 V log(1 /δ )+log(1 /δ ) where V = ∑ T t =1 E t [ M 2 t,i ] . Notice that V = Q t,i -Q 2 t,i m t,i ≤ Q t,i m t,i . Thus, we have ∑ t M t,i ≤ √ 2 log(1 /δ ) ∑ t Q t,i /m t,i +log(1 /δ ) ≤ 1 2 ∑ t Q t,i /m t,i +2log(1 /δ ) where the last step uses the AM-GM inequality. Rearranging it gives ∑ T t =1 Q t,i m t,i ≤ O (log(1 /δ ) + ∑ T t =1 I t,i m t,i ) . Therefore, by choosing δ properly and applying a union bound, with high probability, the regret is bounded by ˜ O ( √ nT ) .

## C.7 Extension to Subgaussian Rewards

For each box, if the reward distribution is sub-Gaussian, then our algorithms still work with minor modifications. Specifically, if the reward distribution of box i is K i -sub-Gaussian, one can set a range [ -K i √ 2 log(2 T 2 n ) , K i √ 2 log(2 T 2 n )] . Then, the algorithm updates parameters only if the received reward of each box i is within range of [ -K i √ 2 log(2 T 2 n ) , K i √ 2 log(2 T 2 n )] . On those rounds, the algorithm and its regret analysis coincide exactly with the bounded-reward setting. Meanwhile, by a union bound on sub-Gaussian tails, the probability that any reward ever exceeds its range is at most 1 /T . Therefore, its contribution to the expected regret is only ˜ O (1) .

## D Omitted Details of Contextual Linear Setting

## D.1 Preliminaries

Lemma D.1.1 (DKW Inequality) . Given a natural number N , let X 1 , . . . , X N be i.i.d. samples from distribution D with cumulative distribution function F D ( · ) . Let F E ( · ) be the associated empirical distribution function E such that for all x ∈ R , we define F E ( x ) := 1 N ∑ N i =1 1 { X i ≤ x } . For every ε &gt; 0 ,

<!-- formula-not-decoded -->

Lemma D.1.2. Let D = ( D 1 , . . . , D n ) be a product distribution, and for each i , let E N i ,i be the empirical distribution built from N i i.i.d. samples of D i . With probability at least 1 -δ/ 2 , for any i ∈ [ n ] , any N i ∈ [ T ] ,

<!-- formula-not-decoded -->

Proof. We first fix any i and N i and apply Lemma D.1.1. Then, the lemma follows by using a union bound over all i ∈ [ n ] and N i ∈ [ T ] .

Lemma D.1.3. With probability at least 1 -δ/ 2 , for any t &gt; 0 and any i ∈ [ n ] ,

<!-- formula-not-decoded -->

Proof. Fix any i ∈ [ n ] . Since the noise distribution of each box is 1 4 -subgaussian, and { x t,i } t is a R d -valued stochastic process such that x t,i is measurable with respect to the history, [Lattimore and Szepesvári, 2020, Theorem 20.5] directly gives the claimed result for the fixed i . Using a union bound over all i ∈ [ n ] completes the proof.

Lemma D.1.4 ( Elliptical Potential Lemma (see, e.g., Lemma 11 in [Abbasi-Yadkori et al., 2011])) . Let I ∈ R d × d be an identity matrix and a sequence of vectors a 1 , . . . , a N ∈ R d with ∥ a i ∥ 2 ≤ 1 for all i ∈ [ N ] . Let V t = I + ∑ s ≤ t a s a ⊤ s . Then, for all N ∈ N , we have

<!-- formula-not-decoded -->

Definition D.1.5 ( Definition of ˜ D t ) . Let ˜ D t = ( ˜ D t, 1 , . . . , ˜ D t,n ) where each ˜ D t,i is an empirical distribution which assigns m t,i i.i.d. noise samples from D i with equal probability mass 1 /m t,i .

Definition D.1.6 ( 'Nice" event ) . Let E be the nice event that the concentration bounds in Lemma D.1.2 for D, ˜ D t hold for all t ∈ [ T ] , and the concentration bounds in Lemma D.1.3 hold simultaneously.

Lemma D.1.7. Suppose that E holds. For any context x ∈ X and any round t ∈ [ T ] ,

<!-- formula-not-decoded -->

Proof. For any t, i, x , one can show that

<!-- formula-not-decoded -->

where the first inequality follows from the Cauchy-Schwarz inequality and the second inequality uses Lemma D.1.3.

## D.2 Algorithms and ˜ O ( nd √ T ) Bounds for Pandora's Box and Prophet Inequality

The omitted algorithm for both the Pandora's Box and Prophet Inequality problems is presented in Algorithm 5. Unlike the non-contextual case (see Eq. (1)), we use a slightly different construction for the optimistic distribution. This is because the Bernstein-type construction requires access to the CDF of an empirical distribution based on i.i.d. samples, which is unavailable in our setting. As a result, we use the following approach instead.

Optimistic distribution ˆ E t,i . Let L = 1 2 log(4 nT/δ ) . The CDF of ˆ E t,i is constructed as follows:

<!-- formula-not-decoded -->

Now, we prove Lemma 4.1 which shows that the optimistic distribution ˆ E t stochastically dominates D t for all t .

Lemma D.2.1 (Restatement of Lemma 4.1) . Suppose that E holds. For all t ∈ [ T ] , ˆ E t ⪰ SD D t .

Proof. For this proof, we show ˆ E t,i ⪰ SD D t,i for all i, t . Fix a box i ∈ [ n ] and a round t ∈ [ T ] . If c ≥ 1 , then

<!-- formula-not-decoded -->

For any c &lt; 1 , we have that

<!-- formula-not-decoded -->

The inequality holds due to the following reason:

<!-- formula-not-decoded -->

where the first inequality follows from the facts that µ t i ( j ) ,i -LCB t i ( ̂ θ t -1 ,i , x t i ( j ) ,i ) ≥ 0 , the second inequality uses the fact that UCB t i ( ̂ θ t -1 ,i , x t,i ) ≥ µ t,i , and the last inequality follows from the concentration bound given by E .

Conditioning on the 'nice" event, the argument holds for all i and t . The proof is complete.

Next, we present the following lemma, whose proof is in the next subsection.

Lemma D.2.2. Suppose that E holds. For both the Pandora's Box and Prophet Inequality problems, we have that

<!-- formula-not-decoded -->

## Algorithm 5 Proposed algorithm for contextual linear model

Input : confidence δ ∈ (0 , 1) , horizon T .

Initialize : observe context { x 0 ,i } i ∈ [ n ] for each box, open each box, observe rewards { v 0 ,i } i ∈ [ n ] , and update counters { m 1 ,i } i ∈ [ n ] and parameters { ̂ θ 0 ,i } i ∈ [ n ] where ̂ θ 0 ,i = ( I + x 0 ,i x ⊤ 0 ,i ) -1 x 0 ,i v 0 ,i . for t = 1 , 2 , . . . , T do

Observe contexts

(

x

t,

1

, . . . , x t,n

)

and compute

̂

µ

t,i

=

̂

θ

t

-

1

,i

, x t,i

for each

i

∈

[

n

]

.

For each i ∈ [ n ] , use Eq. (25) to construct an empirical distribution ˆ E t,i by using samples

<!-- formula-not-decoded -->

## if Pandora's Box then

Compute σ t = ( σ t, 1 , . . . , σ t,n ) where σ t,i is the solution of E x ∼ ˆ E t,i [( x -σ t,i ) + ] = c i . Run Algorithm 1 with σ t to open a set of boxes, denoted by B t to observe rewards { v t,i } i ∈B t .

## else if Prophet Inequality then

Use Definition A.1 with distribution ˆ E t to compute σ t = ( σ t, 1 , . . . , σ t,n ) . Run Algorithm 3 with σ t to open a set of boxes, denoted by B t to observe rewards { v t,i } i ∈B t .

Update counters m t +1 ,i = m t,i +1 , ∀ i ∈ B t and m t +1 ,i = m t,i , ∀ i / ∈ B t . Update estimate for each i ∈ B t :

<!-- formula-not-decoded -->

With Lemma D.2.2 in hand, we are now ready to prove Theorem 4.2.

Proof of Theorem 4.2. The analysis conditions on event E which is defined in Definition D.1.6 and occurs with probability at least 1 -δ . Recall that I t,i = I { E σ t ,i } . For the first term, we take the iterated expectation and use the facts that Q t,i = E t [ I t,i ] and m t,i is deterministic given the history to get

<!-- formula-not-decoded -->

Next, we bound the summation of the second term. To this end, we show that

<!-- formula-not-decoded -->

where the first equality holds due to z t t i ( j ) ,i , ˆ z t t i ( j ) ,i ∈ [0 , 1] , the first inequality uses | b -min { 1 , a }| ≤ | a -b | for b ∈ [0 , 1] , and the third inequality follows from Lemma D.1.7.

〈

〉

Using Eq. (26), we have that

<!-- formula-not-decoded -->

For the second term on the right hand side, we have that

<!-- formula-not-decoded -->

where the second inequality follows from V t -1 ,i ⪰ V t i ( j ) -1 ,i for all t and i , the third inequality uses Cauchy-Schwarz inequality, and the fourth inequality follows from Lemma D.1.4.

Analogously, we take the iterated expectation and use the fact that Q t,i = E t [ I t,i ] and ∥ x t,i ∥ V -1 t -1 ,i is deterministic given the history to conclude that

<!-- formula-not-decoded -->

Combining the bounds above completes the proof of Theorem 4.2.

## D.3 Proof of Lemma D.2.2

We follow a similar analysis of the non-contextual case to focus on the regret bound at each round and then sum them up together. Let us consider a round t . For Pandora's Box problem, we assume

W.L.O.G. that σ t, 1 ≥ σ t, 2 ≥ · · · ≥ σ t,n . Similar to the non-contextual analysis, we define for all t, i

<!-- formula-not-decoded -->

We have H t,n = D t based on this definition. As Lemma 4.1 ensures the stochastic dominance and both problems enjoy a certain monotonicity, we follow the same analysis in Section 3.2 to show that

<!-- formula-not-decoded -->

Now, we define

<!-- formula-not-decoded -->

Based on the definitions of H t,i , ˆ H t,i , ˜ H t,i , we compute the following decomposition:

<!-- formula-not-decoded -->

In the remaining analysis, we retain the same definition of ˜ R i ( σ t ; z ) as in Eq. (6). The analysis conditions on this history before round t , and thus D t is a product distribution.

Bounding (I). By using the fact that E σ t ,i is independent of i -th box's reward, we can bound

<!-- formula-not-decoded -->

where the first inequality follows from the fact that ˜ R i ( σ t ; z ) is 1 -Lipschitz for both Pandora's Box and Prophet Inequality problems (see Lemma 3.2.1 and Lemma C.3.2, respectively), and the last inequality uses the construction of ˆ E t,i in Eq. (25).

Bounding (II). One can show that

<!-- formula-not-decoded -->

where the inequality again the fact that ˜ R i ( σ t ; z ) is 1 -Lipschitz for both Pandora's Box and Prophet Inequality problems.

Bounding (III). Then, we have

<!-- formula-not-decoded -->

where the first inequality follows from the fact that ∂ ∂z ˜ R i ( σ t ; z ) ∈ [ -1 , 1] due to 1 -Lipschitzness, and the last inequality follows from the concentration bound given by E .

Putting together. Combining all bounds above, we have

<!-- formula-not-decoded -->

Summing over all t ∈ [ T ] , we complete the proof of Lemma D.2.2.