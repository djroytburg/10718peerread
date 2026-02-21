## Improved Best-of-Both-Worlds Regret for Bandits with Delayed Feedback

## Ofir Schlisselberg

Tel Aviv University ofirs4@mail.tau.ac.il

## Peter Auer

Technical University of Leoben auer@unileoben.ac.at

## Tal Lancewicki

Tel Aviv University lancewicki@mail.tau.ac.il

## Yishay Mansour

Tel Aviv University and Google Research mansour.yishay@gmail.com

## Abstract

We study the multi-armed bandit problem with adversarially chosen delays in the Best-of-Both-Worlds (BoBW) framework, which aims to achieve near-optimal performance in both stochastic and adversarial environments. While prior work has made progress toward this goal, existing algorithms suffer from significant gaps to the known lower bounds, especially in the stochastic settings. Our main contribution is a new algorithm that, up to logarithmic factors, matches the known lower bounds in each setting individually.

In the adversarial case, our algorithm achieves regret of ˜ O ( √ KT + √ D ) , which is optimal up to logarithmic terms, where T is the number of rounds, K is the number of arms, and D is the cumulative delay. In the stochastic case, we provide a regret bound which scale as ∑ i :∆ i &gt; 0 (log ( T ) / ∆ i ) + 1 K ∑ ∆ i σ max , where ∆ i is the sub-optimality gap of arm i and σ max is the maximum number of missing observations.

To the best of our knowledge, this is the first BoBW algorithm to simultaneously match the lower bounds in both stochastic and adversarial regimes in delayed environment. Moreover, even beyond the BoBW setting, our stochastic regret bound is the first to match the known lower bound under adversarial delays, improving the second term over the best known result by a factor of K .

## 1 Introduction

Delayed feedback presents a significant challenge that sequential decision-making algorithms encounter in many real-world applications. Notably, delays are often an inherent part of environments involving sequential decision-making, such as in healthcare, finance, and recommendation systems. As a central challenge in Online Learning, delays have been extensively explored in various contexts within Multi-armed Bandits (MAB), both in stochastic settings, where losses are generated i.i.d. from a fixed underlying distribution [18, 23, 38, 34, 35, 5, 12, 19, 36, 15, 29, 25] and adversarial settings, where the losses are chosen arbitrarily by an adversary [24, 4, 30, 2, 40, 16, 13, 32, 33].

Roughly speaking, under stochastic losses, delays contribute an additive regret term that does not scale with the time horizon (but with the number of missing observations), whereas under the adversarial losses, delays introduce an additive term that does scale with the horizon. More specifically, for an arbitrary sequence of delays, the best-known regret under stochastic losses is ∑ ∆ i &gt; 0 log ( T ) ∆ i + Kσ max (Joulani et al. [18]) where T is the number of rounds, K is the number of

Table 1: Comparison of regret bounds (up to constants and log ( K ) factors) to the previous state-ofthe-art regret both under stochastic and adversarial losses under adversarial delays.

| Algorithm                                                             | Regime                 | Regret                                                                                                           |
|-----------------------------------------------------------------------|------------------------|------------------------------------------------------------------------------------------------------------------|
| Joulani et al. [18]                                                   | stochastic             | ∑ ∆ i > 0 ( log ( T ) ∆ i + σ max ∆ i )                                                                          |
| Thune et al. [30] Zimmert and Seldin [40] 2 Gyorgy and Joulani [13] 2 | adversarial            | √ TK + √ D                                                                                                       |
| Masoudian et al. [22] 2                                               | stochastic adversarial | ∑ i :∆ i > 0 ( log ( T ) ∆ i + σ max ∆ i )+Φ ∗ √ TK + √ D +Φ ∗ + Kσ max Φ ∗ = min { d max K 2 / 3 , √ DK 2 / 3 } |
| Our paper 2                                                           | stochastic adversarial | ∑ i :∆ i > 0 ( log ( T ) ∆ i + σ max ∆ i K ) √ TK log ( T )+ √ D                                                 |
| Lower Bound                                                           |                        |                                                                                                                  |
| Lancewicki et al. [19] (constant delay) Masoudian et al. [21] 2       | stochastic adversarial | ∑ i :∆ i > 0 ( log ( T ) ∆ i + σ max ∆ i K ) √ TK + √ D                                                          |

arms, ∆ i is the sub-optimality gap of arm i and σ max is the maximal number of missing observations. Under adversarial losses, the optimal bound is of the order √ TK + √ D (Thune et al. [30], and later Zimmert and Seldin [40] and Gyorgy and Joulani [13] ), where D is the sum of the delays.

Remark 1 The √ D term in the optimal regret corresponds to the worst-case delay. As shown, for example, in Zimmert and Seldin [40], Masoudian et al. [21, 22], Gyorgy and Joulani [14], this can be replaced with min S ⊆ [ T ] { | S | + √ D ¯ S } , where D ¯ S is the total delay of the steps not in S . This can be significantly tighter for some delay values. For simplicity of presentation, we present all bounds in the paper in the worst-case form √ D , and note that our bound can also attain the tighter delay dependence in the adversarial case (see Corollary 5.4).

While the regret bounds of delayed Multi-armed Bandit under stochastic losses and under adversarial losses are well understood separately, the following question remains open:

Is there a single algorithm that, without knowing the nature of the losses a-priori in delayed environment, can achieve the optimal regret bounds in both regimes simultaneously?

Such an algorithm is often referred to as a best-of-both-worlds algorithm. Masoudian et al. [21, 22] have made significant progress toward answering this question. Their regret bound is O ( √ TK + √ D + Kσ max +Φ ∗ ) in the adversarial regime and O ( ∑ i :∆ i &gt; 0 ( log ( T ) ∆ i + σ max log ( K )∆ i ) +Φ ∗ ) in the stochastic regime, where Φ ∗ = min { d max K 2 / 3 , √ DK 2 / 3 } . However, these bounds are still not optimal. 1

Our contributions. In this work, we affirmatively answer the above question and present a new best-of-both-worlds algorithm for Multi-armed Bandits (MAB) with delayed feedback that simultaneously achieves the near-optimal regret bounds under both stochastic and adversarial losses. Specifically:

- In the adversarial regime our algorithm guarantees optimal ˜ O ( √ TK + √ D ) regret.

̸

- In the stochastic regime our algorithm guarantees optimal O ( ∑ i = i ⋆ ( log ( T ) ∆ i + 1 K σ max ∆ i )) regret.

In the adversarial regime, compared to Masoudian et al. [22] we have an extra logarithmic factor in the √ TK term, which is independent of the delay. However, we eliminate the additive Φ ∗ in their

1 An additional benefit of Masoudian et al. [21, 22] is that they are any time algorithms, that do no need to know the horizon T in advance. √ √

2 In these papers the D is actually min S ⊆ [ T ] { | S | + D ¯ S } , where D ¯ S is the total delay of the steps not in S . We wrote the worst-case for the simplicity of the table.

bound, which is significant when d max is very large; even a single large delay causes the regret to scale as √ DK 2 / 3 rather than our √ D delay term, which is tight to the lower bound of Masoudian et al. [21].

Even more significantly, in the stochastic regime, our bound improves the O ( ∑ i σ max ∆ i log ( K ) + Φ ∗ ) term from the bound of [22] to O ( 1 K ∑ i σ max ∆ i ) . That is, for each term in the summation, we achieve an improvement by a factor of K ∆ 2 i log ( K ) . This is a significant improvement. For example, consider the simple case of fixed delay d , which implies σ max = d , and constant number of actions. For any sub-optimality gaps our regret is at most √ T + d while there is a setting where the regret of [22] is at least √ dT . Moreover, if the maximum delay is large, Φ ∗ can be as large as √ D , offering no improvement over the additive delay term in the adversarial setting.

̸

Our bound in the stochastic regime represents an improvement even compared to state-of-the-art results for algorithms specifically designed for the stochastic case. Specifically, Joulani et al. [18] provides the best-known result for stochastic losses with adversarial delays where their bound includes an additive term of ∑ i = i ∗ σ max ∆ i , which we improve by a factor of Θ( K ) . While Lancewicki et al. [19] reduce this dependence on K , their result applies only to the case of stochastic delays. Moreover, their regret bound scales with the maximal sub-optimality gap, rather than the average. For example, in the simple case of a fixed delay d , their additive term is d max i ∆ i , whereas ours is d K ∑ i ∆ i , offering a strictly better dependence on the problem parameters in many scenarios.

## 1.1 Additional Related work

Delayed MAB with stochastic losses. The problem was first addressed by Dudik et al. [9], who analyzed the case of constant delays and established a regret bound with linear dependence on the delay. This line of work was extended by Joulani et al. [18], who allowed the delays to change through time. Subsequent work introduced several important refinements: Zhou et al. [38] distinguished between arm-dependent and arm-independent delays; Pike-Burke et al. [23] introduced an aggregated rewards model where only the sum of rewards that arrive at the same round is observed; and Lancewicki et al. [19] studied delays in the contexts of reward-dependent or reward-independent delays. More recently, Tang et al. [29], Schlisselberg et al. [25] and Zhang et al. [37] studied settings in which the delay is equal to the payoff.

Delayed MAB with adversarial losses. Delayed feedback have also been explored in adversarial settings, where both rewards and delays can be chosen adversarially. Quanrud and Khashabi [24] studied this problem in the full-information setting. The bandit setting was first addressed by CesaBianchi et al. [6], who analyzed the case of constant delay. This line of work was extended by Thune et al. [30] who considered general adversarial delays under the assumption that the delay is known at the time the arm is pulled. 3 Subsequently, Gyorgy and Joulani [13] and Zimmert and Seldin [40] removed this assumption and analyzed the case where the delay is unknown at the time of the action. Finally, Van Der Hoeven and Cesa-Bianchi [32] extended the setting to allow for arm-dependent delays.

'Best of Both Worlds' without delays. The 'Best of Both Worlds' framework in multi-armed bandits was introduced by Bubeck et al. [3], who proposed an algorithm that initially follows a stochastic-style strategy but switches to a standard adversarial algorithm upon detecting signs of adversarial losses. This adaptive approach was further developed by Auer and Chiang [1]. An alternative perspective is to start with an adversarial-style algorithm and prove that it achieves instancedependent regret bounds in stochastic settings as well. In this direction, Seldin and Slivkins [27] and Seldin and Lugosi [26] adapted the EXP3 algorithm to perform well in both regimes, while Zimmert and Seldin [39], Dann et al. [8], Ito et al. [17] extended this idea to Follow-The-Regularized-Leader (FTRL), achieving optimal performance across both adversarial and stochastic settings.

3 A similar result appeared in Bistritz et al. [2], however, there are some issues in their analysis - for further details see footnote 1 in Gyorgy and Joulani [13].

## Protocol 1 Delayed MAB

- 1: for t ∈ [ T ] do
- 2: Agent picks an action a t ∈ [ K ] .
- 3: Agent incurs loss ℓ t ( a t ) and observes feedback { ( ℓ s ( a s ) , d s ) : t = s + d s } .

## 2 Settings

Westudy the Multi-armed Bandit (MAB) problem with delayed feedback, summarized in Protocol 1. In each round t = 1 , 2 , . . . , T , an agent chooses an arm a t ∈ [ K ] and suffers loss ℓ t ( a t ) , where ℓ t ( · ) ∈ [0 , 1] K can be either stochastic or adversarial. Under the stochastic regime for each i ∈ [ K ] , { ℓ t ( i ) } T t =1 i.i.d ∼ D i where D i is some distribution with expectation µ i . Under the adversarial regime the loss sequence { ℓ t } T t =1 are chosen arbitrarily by an oblivious adversary. Unlike the standard MABsetting, the agent does not immediately observe ℓ t ( a t ) at the end of round t ; rather, only after d t rounds (namely, at the end of round t + d t ) the tuple ( t, ℓ t ( a t )) is received as feedback. The delays { d t } T t =1 are chosen by an oblivious adversary and are unknown at action time.

The performance of the agent is measured as usual by the the difference between the algorithm's cumulative expected loss and the best possible total expected reward of any fixed arm:

<!-- formula-not-decoded -->

In the stochastic case the regret can also be written as,

<!-- formula-not-decoded -->

where i ∗ denotes the optimal arm and ∆ i = µ i -µ i ∗ for all i ∈ [ K ] .

Additional notation. Wedenote the total delay by D = ∑ T t =1 d t and the maximal delay by d max = max t ∈ [ T ] d t . The amount of missing feedback at time t is defined by σ ( t ) = |{ τ | τ ≤ t, τ + d τ &gt; t }| and the maximum over σ ( t ) is denoted by σ max = max t ∈ [ T ] σ ( t ) . The rounds observed before and available at round t are denoted by B ( t ) = { s : s + d s &lt; t } . For X ∈ R , [ X ] denotes the set of all positive integers ≤ X .

Notation for the algorithms. Let S denotes a sequence of rounds that the algorithm process. S : n is the first n elements in S and S : -n is S except for the last n elements. n i ( S ) is the number of pulls of arm i in the rounds of S , ˆ µ i ( S ) = 1 n i ( S ) ∑ s ∈ S : a s = i l i ( s ) is the empirical mean over S and width i ( S ) = min { 1 , √ 2 log ( T ) n i ( S ) } is a confidence width. ucb i ( S ) = min { ˆ µ i + width i ( S ) , ucb i ( S : -1 ) } and lcb i ( S ) = max { ˆ µ i -width i ( S ) , lcb i ( S : -1 ) } are upper and lower confidence bounds with respect to the empirical average. The algorithm also maintains confidence bounds around an average importance sampling estimator. Let L i ( S ) = ∑ s ∈ S 1 [ a s = i ] ℓ i ( s ) p i ( s ) be the sum of the estimators over rounds in S , and µ i ( S ) = 1 | S | L i ( S ) be the average. We also define width ( S ) = min { 1 , √ 2 K log ( T ) | S | } , lcb i ( S ) = max { µ i ( S ) -width ( S ) , lcb i ( S : -1 ) } and ucb i ( S ) = min { µ i ( S ) + width ( S ) , ucb i ( S : -1 ) } . Finally, we define ucb ∗ ( S ) = min i { ucb i ( S ) , ucb i ( S ) } .

## 3 Algorithm

Our algorithm, sketched in Algorithm 2 and formally described in Algorithm 5, builds on the SAPO algorithm of Auer and Chiang [1]. The main idea is to integrate an external algorithm for adversarial settings, ALG . Our algorithm initially follows a stochastic-like strategy while monitoring whether the environment exhibits stochastic behavior. If this assumption is violated, it switches to ALG .

At its core, the algorithm is based on a successive elimination (SE) framework [11], maintaining a set of active arms played with equal probability. It tracks a confidence bound, width , which defines upper and lower estimates for each arm's mean. When an arm is found to be non-optimal, it is eliminated. However, unlike standard SE methods, the algorithm continues to play eliminated arms but with reduced probability. This accounts for the possibility that losses are adversarial-an arm that appears suboptimal at one point may later turn out to be optimal. To verify the stochastic nature of arms, the algorithm employs the BSC procedure to assess the nature of active arms, and a more advanced procedure EAP for assessing and determining the sampling probability of non-active arms.

```
Require: Number of arms K , number of rounds T ≥ K , Algorithm ALG . 1: Initialize active arms A = { 1 , . . . , K } , S = ⟨⟩ 2: for t = 1 , 2 , . . . , T do 3: for s ∈ B ( t ) \ S do ▷ Iterating newly received feedback 4: S = S + ⟨ s ⟩ 5: if not BSC ( S ) then ▷ Non-stochastic behavior on active arms (Procedure 3) 6: Switch to ALG . 7: A = A\{ i ∈ A : ˆ µ i ( S ) -9 width i ( S ) > ucb ∗ ( S ) } ▷ Elimination 8: for Each eliminated arm i do 9: E i = 0 , r i = 1 , C p 1 i · 2 -j i = ∅ ∀ j ∈ [log ( T )] 10: for i ∈ ([ K ] \ A ) do 11: p i ( t ) , err = EAP ( i ) ▷ Get the reduced probability for the non-active arm (Procedure 4) 12: if err then ▷ Non-stochastic behavior on nonactive arms 13: Switch to ALG . 14: ∀ i ∈ A p i ( t ) = ( 1 -∑ j ∈ ([ K ] \A ( t )) p j ( t ) ) / |A ( t ) | ▷ Equal probability for active arms 15: Sample a t ∼ p ( t ) , observe feedback and update variables
```

Algorithm 2 Sketch of Delayed SAPO Algorithm

Basic Stochastic Checks ( BSC ) Subroutine. This procedure performs two checks. The first ensures that an unbiased estimate of the mean of each arm remains within its confidence interval, expanded by an additional radius. In the stochastic regime, using standard concentration bounds we have that with high probability,

<!-- formula-not-decoded -->

Thus, in line 1 of Procedure 3, we check that the above conditions are met.

The second check in BSC constructs a lower bound on the regret and verifies that it is indeed smaller than the expected regret in the stochastic regime, which can be shown to be ˜ O ( √ TK + σ max ) under stochastic losses. To define this lower bound, we use the fact that, with high probability, µ ∗ ≤ ucb ∗ ( S ) . Thus, ∑ s ′ ∈ S ( l a s ′ ( s ′ ) -ucb ∗ ( S ) ) is a lower bound on the regret, which forms the condition in line 3 of the procedure.

```
Procedure 3 Basic Stochastic Checks (BSC) Subroutine Require: Series of processed pulls S 1: if ∃ i ∈ A : µ i ( S ) ̸∈ [ lcb i ( S ) -width ( S ) , ucb i ( S ) + width ( S )] then 2: return False 3: if ∑ s ′ ∈ S ( l a s ′ ( s ′ ) -ucb ∗ ( S ) ) > 272 √ KT log ( T ) + 10 σ max ( t ) log ( K ) then 4: return False return True
```

Eliminated Arms Processing (EAP) Subroutine. Since we do not know in advance whether we are in the stochastic or adversarial regime, we cannot completely eliminate an arm - if we did, the adversary could assign losses of 0 after elimination of an arm, and we would never detect this. Therefore, we maintain a positive sampling probability even for eliminated actions. EAP maintains these probabilities for eliminated arms and checks whether the estimated loss is significantly smaller than the empirical mean at elimination. Intuitively, if we are in the stochastic regime, we want the probability of playing an eliminated arm to decrease over time. Conversely, if we suspect the loss after elimination is significantly smaller than the empirical mean at the elimination time, we increase

## Procedure 4 Sketch of Eliminated Arms Processing (EAP) Subroutine

Note: The variables E i , r i , and C p i are initialized in Algorithm 2 and updated through multiple calls

- 1: p := p i , ˜ µ is the empirical average at the elimination time of i
- 2: Let B i be observed rounds after elimination in which i was played and the sampling probability

̸

## of this procedure. Require: Arm i r i p 10: 11: 12: 13: 14: 15:

```
was p 3: while B p i \ C p i = ∅ do 4: for s ∈ B p i \ C p i do 5: C p i = C p i ∪ { s } 6: Let S r i i be the samples processes so far in phase r i 7: if | S r i i | ˜ µ i -¯ L ( S r i i ) ≥ 1 4 ˜ ∆ i N r i i then ▷ phase error 8: E i = E i +1 , N r i +1 i = max { N 1 i , 1 2 N r i i } , p r i +1 i = min { p 1 i , 2 p r i i } , r i = r i +1 9: if E i ≥ 3 log ( T ) then return 0 , True ▷ Switch to adversarial algorithm break if | S r i i | = ⌊ N r i i ⌋ then ▷ phase ended N r i +1 i = 2 N r i i , p r i +1 i = 1 2 p r i i , r i = r i +1 break p := p r i i return p r i i , False
```

that arm's probability to monitor it more closely. If there is sufficient confidence that the arm does not behave stochastically, we switch to the adversarial algorithm.

In more detail, the probability of playing an eliminated arm i is updated in discrete phases. Let ˜ S i be the set of processed rounds at the time of elimination of arm i . We denote ˜ ∆ i = 8 width i ( ˜ S i ) , i.e the width at elimination time. As we'll later see in the analysis, ˜ ∆ i is indeed a good estimate of the sub-optimality gap of arm i in the stochastic case (see Lemma D.10). Each phase r has a maximum length N r i = Θ(1 / ( p r i ˜ ∆ 2 i )) , where p r i is the sample probability of arm i in its r th phase and p 1 i = 1 2 K + n i ( ˜ S i ) 2 T . 4 This value is always Ω(1 /K ) , but can be as high as a uniform probability over the active arms at the time of elimination. If we reach the maximum length N r i , then we have acquired additional N r i p r i = Θ(1 / ˜ ∆ 2 i ) samples from arm i . In this case, we can safely halve the sampling probability of arm i and start a new phase with a doubled maximum length (line 11). During the phase, we monitor whether the average importance sampling estimate of the loss ¯ µ ( S r i ) is smaller than ˜ µ i = ˆ µ i ( ˜ S i ) by more than Θ( ˜ ∆ i N r i / | S r i | ) , where S r i is the sequence of processed rounds in phase r . If this condition is met, referred to as a 'phase error', in means that the observed losses appear slightly non-stochastic. Thus, we terminate the phase but now double the sampling probability of arm i and halve the maximum phase length accordingly (line 6).

In the stochastic regime, phase errors occur with a constant probability, but the probability that they will happen Θ(log ( T )) times is negligible. In such cases, we transition to the adversarial algorithm.

During a phase with sampling probability p , we process only the observed rounds after elimination in which arm i was played and the sampling probability was p . If a sample is observed with a different sampling probability p ′ , it is stored in a 'probability bank' which we denote by B p ′ i and is processed only if a new phase is initiated with probability p ′ . The probability banks allow us to utilize most samples, even if they are observed after their respective phases end, and play an important role in removing a factor of the number of phases ( Θ(log ( T )) ) from the delay term in the regret.

4 We note that the initial probability assigned in the first phase differs from that in Auer and Chiang [1], and is crucial for obtaining adversarial regret bound that scales with √ KT instead of K √ T achieved in [1].

## 4 Stochastic analysis

Theorem 4.1 The regret in the stochastic settings is bounded by:

<!-- formula-not-decoded -->

The first term above is the optimal MAB regret under stochastic losses without delays. The second term is the additional regret due to delay and, in general, cannot be improved-except for the log ( K ) factor, due to the lower bound for constant delays (see Table 1). We note that with a more involved algorithm and analysis, we are able to eliminate the log ( K ) factor and match this lower bound. For simplicity of presentation, the full details are deferred to Appendix F. The dependence on σ max improves upon the BoBW result of Masoudian et al. [22] in the stochastic regime by a factor of ˜ O ( K/ ∆ 2 i ) for each i . Moreover, it is tighter by a factor of K compared to the best previous known algorithm that specifically designed for this regime (Joulani et al. [18]).

Proof sketch: The total regret can be decomposed as,

<!-- formula-not-decoded -->

where m i ( t ) is the number of pulls of arm i , up to time t and τ i is the elimination time of arm i . The first term above is the regret up to elimination and second term is the regret after elimination. (Recall that we need to keep sampling eliminated arms.)

Regret up to elimination. The regret before elimination analysis largely follows standard Stochastic Elimination (SE) with delayed feedback arguments. However, achieving dependence on ∆ avg rather than ∆ max necessitates a new algorithmic component and technical argument. We start by further decomposing the regret up to elimination:

<!-- formula-not-decoded -->

where n i ( t ) is the number of observed samples from arm i . Similar to standard non-delayed SE analysis, we can show that with high probability, each suboptimal arm is eliminated whenever Θ ( log ( T ) ∆ 2 i ) samples from arm i have been observed. Thus, n i ( τ i )∆ i = Θ ( log ( T ) ∆ i ) . For the second term above, recall that the number of missing feedback is bounded by σ max ; but only a fraction of the missing feedback is from arm i . Loosely speaking, if p max i = max t ≤ τ i p i ( t ) is the maximal probability of sampling i before elimination, then the number of missing feedback from arm i at time τ i is roughly bounded by m i ( τ i ) -n i ( τ i ) ≤ σ max p max i . Further note that if κ i is the number of active arms at the time of elimination then p max i ≤ 1 κ i . Overall, the total regret up to elimination is bounded by

<!-- formula-not-decoded -->

For the second term, each ∆ i can be trivially bounded by ∆ max , and ∑ i 1 /κ i ≤ ∑ i 1 /i ≤ 1 + log ( K ) , resulting in ∑ i σ max κ i ∆ i ≤ O ( σ max ∆ max log ( K )) . In order to have dependency with respect ∆ avg instead of ∆ max a more detailed argument is required. Unlike regular SE algorithms, an arm isn't eliminated when the ucb of some other arm is lower than its lcb. Instead, the algorithm eliminates when there are multiple widths between the two (see line 7 in algorithm 2). This stricter condition ensures that arms are roughly eliminated in decreasing order of ∆ i . Specifically, we show the following lemma:

Lemma 4.2 If arm i 1 was eliminated before i 2 then, ∆ i 2 ≤ 20∆ i 1 .

We note that this is relatively general trick that may be used in other regimes; see remark 2.

For the first half of eliminated arms where κ i ≥ K/ 2 , the additive delay term is at most order of ∑ i : κ i &gt;K/ 2 σ max κ i ∆ i ≤ σ max ∆ avg . Using the above lemma we show that for second half of

eliminated arms ∆ i = O (∆ avg ) , yielding an additive delay term of at most O ( σ max ∆ avg log ( K )) . Overall we get that the regret up to elimination is bounded by ∑ i ∈ [ K ] m i ( τ i )∆ i ≲ ∑ i log ( T ) ∆ i + σ max ∆ avg log ( K ) .

Regret after elimination. For the regret after elimination, we break the number of pulls of arm i after elimination for pulls that where processed by algorithm and pulls that where not processed by the algorithm (either because the feedback had not returned or the samples remained in the probability bank):

<!-- formula-not-decoded -->

where r i is the total number of phases of arm i , S r i are the samples processes at phase r and M p i denotes the post-elimination rounds where the probability of pulling arm i was p , but these were not processed by the algorithm (either because the feedback was not observed or the rounds remained unprocessed in the probability bank).

Recall that the maximum length of phase r is N r i = Θ(1 / ( p r i ˜ ∆ 2 i )) . Additionally, the fact that arms are only eliminated when the empirical average exceeds ucb ∗ by more than multiple widths allows us to show that ˜ ∆ i ≈ ∆ i (see Lemma D.10). Using standard concentration bounds, n i ( S r i ) ≈ N r i p r i ≈ 1 / ∆ 2 i . To bound the number of phases, note that the maximum phase length can be either doubled or halved. The number of times it is halved in the stochastic regime is at most 3 log ( T ) with high probability (see Lemmas D.5 and D.19), where in case of a failure event, we switch to the adversarial algorithm. Since the number of times it is halved is bounded by O (log ( T )) , we can also bound the number of times it is doubled before reaching the time horizon T . Formally, in Lemma C.4, we bound the total number of phases by 7 log ( T ) . Therefore the first term in Equation (2) is bounded by O ( logT ∆ 2 i ) and the regret from these rounds is O ( logT ∆ i ) .

For the second term of eq. (2), note that the size of M p 1 i 2 -j i is at most σ max , but only a small fraction of those rounds belongs to arm i . Since the probability of pulling arm i in these rounds was p 1 i 2 -j we have that n i ( M p 1 i 2 -j i ) ≈ σ max p 1 i 2 -j . Summing over this geometric series gives us ∑ log ( T ) j =0 n i ( M p 1 i 2 -j i ) = O ( σ max p 1 i ) . Recall that p 1 i = 1 2 K + n i ( τ i ) 2 T . Since the probability of pulling arm i before elimination is at most 1 /κ i , where κ i is the number of active arms at the time of elimination, n i ( τ i ) /T ≤ n i ( τ i ) /τ i ≲ 1 /κ i . That is, p 1 i ≤ O (1 /κ i ) . We get that the total regret after elimination from unprocessed pulls (multiplying the second term in eq. (2) by ∆ i and summing over i ) is of order ∑ i σ max κ i ∆ i . Again, leveraging the fact that arms are eliminated roughly in decreasing order of their sub-optimality gaps, we can bound the last sum by σ max ∆ avg log ( K ) .

Remark 2 As mentioned in the proof sketch, our algorithm adds additional width to the elimination inequality, which makes the eliminated arms to be in descending order of their sub-optimality gap. We stress that this is a general trick that can be applied in any SE-based algorithm. Specifically, for every SE-based algorithm for delayed feedback (e.g [20, 25]), this will make their additive term be dependent on ∆ avg instead of ∆ max .

## 5 Adversarial Analysis

Theorem 5.1 Assume that ALG has a regret guarantee of R ALG in terms of T , K , D and possibly d max and σ max . Then, the regret in the adversarial setting is bounded by:

<!-- formula-not-decoded -->

The log ( K ) factor can be removed with a slight algorithm modification. We deferred the details to Appendix F to reduce the complexity of the already intricate main algorithm.

Proof sketch: Fix action i ∈ [ K ] . Let ¯ T be the time the algorithm switches to ALG . Clearly, the regret after the switch is bounded by R ALG , so we focus on the regret up to time ¯ T . First, we

decompose it to the following three terms:

<!-- formula-not-decoded -->

where ¯ S is the value of S when the algorithm switches to ALG and τ i is the elimination time of arm i . Term (3) is bounded by O ( √ KT log ( T ) + log ( K ) σ max ) due to the second check of BSC . To bound term (4), note that ¯ L i is an unbiased estimator of L i . By Wald's equation, E [ L i ( ˜ S i ) ] = E [ ¯ L i ( ˜ S i ) ] = E [ | ˜ S i | µ i ( ˜ S i )] , where ˜ S i is the set of rounds in which arm i was observed before elimination. Now, since we have not switched to ALG yet, by the first check of BSC we know that for every realization S of ˜ S i ,

<!-- formula-not-decoded -->

where in the second inequality we used the fact that ucb i ( S ) -lcb i ( S ) ≤ 2 width ( S ) for any S and the last inequality is by definition of ucb ∗ . Multiplying both sides above by | ˜ S i | gives us

<!-- formula-not-decoded -->

Rearranging the terms above we get that E [∑ t ∈ ˜ S i [ ucb ∗ ( ¯ S ) -l i ( t ) ]] ≤ 3 √ 2 TK log ( T ) . Hence,

<!-- formula-not-decoded -->

where we've used the fact that ucb decreases over time and | [ τ i -1] \ ˜ S i | ≤ σ max .

The core difficulty in the adversarial analysis is bounding (5) - ensuring that an eliminated arm doesn't become much better than the active arms, before switching to ALG . Let us further decompose (5) to the phases of arm i :

<!-- formula-not-decoded -->

The main tool to upper bound the optimality of an eliminated arm is the check in Line 9 of EAP . This checks that the estimated loss (using an importance sampling estimator) isn't much higher than the loss observed when the arm was active. Using the condition in Line 9 of EAP and by bounding the difference between the loss estimator of the phase and the actual cost in terms of N r i and ˜ ∆ i we show the following lemma (the proof is deferred to the appendix - see Lemma E.2):

Lemma 5.2 For every arm i and phase r we have:

<!-- formula-not-decoded -->

where E r i is the expectation conditioned on the observed history by the beginning of the r th phase of arm i .

Note that the difference between ucb ∗ and the expected loss depends on the relationship between | S r i | and N r i . If the phase finished successfully ( | S r i | = N r i ), the expected loss exceeds ucb ∗ . If the phase was erroneous, then we may have | S r i |≪ N r i , and the expected loss can be better than ucb ∗ . Specifically:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The trick for bounding the sum of these bounds over the phases is that every successfully finished phase compensate for the erroneous phases after it, since the coefficient of ˜ ∆ i N r i under finished phases is twice as the coefficient for error phases. Since the algorithm halves N r i after an error, even O (log ( T )) erroneous phases are eventually covered by the last successful phase. The precise argument is by induction and is rather technical. For full details, see the appendix in Lemma E.3.

To conclude the proof we use the following lemma (proof is deferred to the appendix):

<!-- formula-not-decoded -->

Corollary 5.4 Using ALG as the algorithm from Zimmert and Seldin [40], we have:

<!-- formula-not-decoded -->

The first term in the regret bound is nearly optimal-up to the log ( T ) factor, it matches the worstcase regret for MAB without delays. The second term, which accounts for the delays, is also tight in general due to the lower bound of Masoudian et al. [21]. Our bound eliminates altogether the Φ ∗ term that appears in the BoBW bound of Masoudian et al. [22] for the adversarial regime (see Table 1). This is especially significant whenever d max is very large (i.e., even if a single delay is large), in which case Φ ∗ = √ DK 2 / 3 while our delay term only scales with √ D in the worst case.

## 6 Discussion

We presented a novel algorithm for the delayed-BoBW problem that achieves a near-optimal regret bound simultaneously for both stochastic and adversarial losses. Additionally, our bounds in the stochastic regime improve even compared to algorithms specifically designed for the stochastic case. As mentioned, our algorithm follows the 'adaptive approach' for BoBW-it begins with an algorithm that achieves optimal bounds in the stochastic setting, and upon identifying non-stochastic losses, it switches to an optimal algorithm for the adversarial setting. The alternative perspective Masoudian et al. [21, 22] offers a simpler algorithm but results in weaker bounds. It remains an open question whether algorithms of the latter type can achieve optimality in the delayed scenario. Apart from the simplicity of these type of algorithms, we also note that they typically have any-time guarantee, whereas ours either requires knowing T in advance or incurs an additional logarithmic factor. Additionally, while we considered worst-case adversarial delays, future research could explore delays with additional structure (such as i.i.d. or payoff-dependent delays), potentially yielding improved regret bounds.

## Acknowledgments

OS, TL and YM are supported by the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement No. 882396), by the Israel Science Foundation and the Yandex Initiative for Machine Learning at Tel Aviv University and by a grant from the Tel Aviv University Center for AI and Data Science (TAD).

## References

- [1] P. Auer and C.-K. Chiang. An algorithm with nearly optimal pseudo-regret for both stochastic and adversarial bandits, 2016. URL https://arxiv.org/abs/1605.08722 .
- [2] I. Bistritz, Z. Zhou, X. Chen, N. Bambos, and J. Blanchet. Online exp3 learning in adversarial bandits with delayed feedback. In Advances in Neural Information Processing Systems , pages 11349-11358, 2019.
- [3] S. Bubeck, N. Cesa-Bianchi, and S. M. Kakade. Towards minimax policies for online linear optimization with bandit feedback. In Conference on Learning Theory , pages 41-1. JMLR Workshop and Conference Proceedings, 2012.
- [4] N. Cesa-Bianchi, C. Gentile, Y. Mansour, and A. Minora. Delay and cooperation in nonstochastic bandits. In Conference on Learning Theory , pages 605-622, 2016.
- [5] N. Cesa-Bianchi, C. Gentile, and Y. Mansour. Nonstochastic bandits with composite anonymous feedback. In Conference On Learning Theory , pages 750-773, 2018.
- [6] N. Cesa-Bianchi, C. Gentile, and Y. Mansour. Delay and cooperation in nonstochastic bandits. The Journal of Machine Learning Research , 20(1):613-650, 2019.
- [7] C. Dann, T. Lattimore, and E. Brunskill. Unifying pac and regret: Uniform pac bounds for episodic reinforcement learning. Advances in Neural Information Processing Systems , 30, 2017.
- [8] C. Dann, C.-Y. Wei, and J. Zimmert. A blackbox approach to best of both worlds in bandits and beyond. In G. Neu and L. Rosasco, editors, Proceedings of Thirty Sixth Conference on Learning Theory , volume 195 of Proceedings of Machine Learning Research , pages 55035570. PMLR, 12-15 Jul 2023. URL https://proceedings.mlr.press/v195/dann23a. html .
- [9] M. Dudik, D. Hsu, S. Kale, N. Karampatziakis, J. Langford, L. Reyzin, and T. Zhang. Efficient optimal learning for contextual bandits. In Proceedings of the Twenty-Seventh Conference on Uncertainty in Artificial Intelligence , pages 169-178, 2011.
- [10] Y. Efroni, N. Merlis, A. Saha, and S. Mannor. Confidence-budget matching for sequential budgeted learning. In M. Meila and T. Zhang, editors, Proceedings of the 38th International Conference on Machine Learning , volume 139 of Proceedings of Machine Learning Research , pages 2937-2947. PMLR, 18-24 Jul 2021. URL https://proceedings.mlr. press/v139/efroni21a.html .
- [11] E. Even-Dar, S. Mannor, Y. Mansour, and S. Mahadevan. Action elimination and stopping conditions for the multi-armed bandit and reinforcement learning problems. Journal of machine learning research , 7(6), 2006.
- [12] M. A. Gael, C. Vernade, A. Carpentier, and M. Valko. Stochastic bandits with arm-dependent delays. In International Conference on Machine Learning , pages 3348-3356. PMLR, 2020.
- [13] A. Gyorgy and P. Joulani. Adapting to delays and data in adversarial multi-armed bandits. In International Conference on Machine Learning , pages 3988-3997. PMLR, 2021.
- [14] A. Gyorgy and P. Joulani. Adapting to delays and data in adversarial multi-armed bandits. In M. Meila and T. Zhang, editors, Proceedings of the 38th International Conference on Machine Learning , volume 139 of Proceedings of Machine Learning Research , pages 3988-3997. PMLR, 18-24 Jul 2021. URL https://proceedings.mlr.press/v139/gyorgy21a. html .
- [15] B. Howson, C. Pike-Burke, and S. Filippi. Delayed feedback in generalised linear bandits revisited. arXiv preprint arXiv:2207.10786 , 2022.
- [16] S. Ito, D. Hatano, H. Sumita, K. Takemura, T. Fukunaga, N. Kakimura, and K.-I. Kawarabayashi. Delay and cooperation in nonstochastic linear bandits. Advances in Neural Information Processing Systems , 33:4872-4883, 2020.

- [17] S. Ito, T. Tsuchiya, and J. Honda. Adaptive learning rate for follow-the-regularized-leader: Competitive analysis and best-of-both-worlds. In S. Agrawal and A. Roth, editors, Proceedings of Thirty Seventh Conference on Learning Theory , volume 247 of Proceedings of Machine Learning Research , pages 2522-2563. PMLR, 30 Jun-03 Jul 2024. URL https: //proceedings.mlr.press/v247/ito24a.html .
- [18] P. Joulani, A. Gyorgy, and C. Szepesv´ ari. Online learning under delayed feedback. In International Conference on Machine Learning , pages 1453-1461, 2013.
- [19] T. Lancewicki, S. Segal, T. Koren, and Y. Mansour. Stochastic multi-armed bandits with unrestricted delay distributions. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event , pages 5969-5978. PMLR, 2021.
- [20] T. Lancewicki, A. Rosenberg, and Y. Mansour. Learning adversarial markov decision processes with delayed feedback. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 36, pages 7281-7289, 2022.
- [21] S. Masoudian, J. Zimmert, and Y. Seldin. A best-of-both-worlds algorithm for bandits with delayed feedback. arXiv preprint arXiv:2206.14906 , 2022.
- [22] S. Masoudian, J. Zimmert, and Y. Seldin. A best-of-both-worlds algorithm for bandits with delayed feedback with robustness to excessive delays. Advances in Neural Information Processing Systems , 37:141071-141102, 2024.
- [23] C. Pike-Burke, S. Agrawal, C. Szepesvari, and S. Grunewalder. Bandits with delayed, aggregated anonymous feedback. In International Conference on Machine Learning , pages 41054113. PMLR, 2018.
- [24] K. Quanrud and D. Khashabi. Online learning with adversarial delays. Advances in neural information processing systems , 28:1270-1278, 2015.
- [25] O. Schlisselberg, I. Cohen, T. Lancewicki, and Y. Mansour. Delay as payoff in mab. Proceedings of the AAAI Conference on Artificial Intelligence , 39(19):20310-20317, Apr. 2025. doi: 10.1609/aaai.v39i19.34237. URL https://ojs.aaai.org/index.php/AAAI/article/ view/34237 .
- [26] Y. Seldin and G. Lugosi. An improved parametrization and analysis of the EXP3++ algorithm for stochastic and adversarial bandits. In S. Kale and O. Shamir, editors, Proceedings of the 2017 Conference on Learning Theory , volume 65 of Proceedings of Machine Learning Research , pages 1743-1759. PMLR, 07-10 Jul 2017. URL https://proceedings.mlr. press/v65/seldin17a.html .
- [27] Y. Seldin and A. Slivkins. One practical algorithm for both stochastic and adversarial bandits. In E. P. Xing and T. Jebara, editors, Proceedings of the 31st International Conference on Machine Learning , volume 32 of Proceedings of Machine Learning Research , pages 1287-1295, Bejing, China, 22-24 Jun 2014. PMLR. URL https://proceedings.mlr.press/v32/ seldinb14.html .
- [28] A. Slivkins. Introduction to multi-armed bandits, 2024. URL https://arxiv.org/abs/ 1904.07272 .
- [29] Y. Tang, Y. Wang, and Z. Zheng. Stochastic multi-armed bandits with strongly rewarddependent delays. In International Conference on Artificial Intelligence and Statistics , pages 3043-3051. PMLR, 2024.
- [30] T. S. Thune, N. Cesa-Bianchi, and Y. Seldin. Nonstochastic multiarmed bandits with unrestricted delays. In Advances in Neural Information Processing Systems , pages 6541-6550, 2019.
- [31] J. A. Tropp. Freedman's inequality for matrix martingales, 2011. URL https://arxiv.org/ abs/1101.3039 .

- [32] D. Van Der Hoeven and N. Cesa-Bianchi. Nonstochastic bandits and experts with armdependent delays. In International Conference on Artificial Intelligence and Statistics . PMLR, 2022.
- [33] D. van der Hoeven, L. Zierahn, T. Lancewicki, A. Rosenberg, and N. Cesa-Bianchi. A unified analysis of nonstochastic delayed feedback for combinatorial semi-bandits, linear bandits, and mdps. In The Thirty Sixth Annual Conference on Learning Theory , pages 1285-1321. PMLR, 2023.
- [34] C. Vernade, O. Capp´ e, and V. Perchet. Stochastic bandit models for delayed conversions. In Conference on Uncertainty in Artificial Intelligence , 2017.
- [35] C. Vernade, A. Carpentier, T. Lattimore, G. Zappella, B. Ermis, and M. Brueckner. Linear bandits with stochastic delayed feedback. In International Conference on Machine Learning , pages 9712-9721. PMLR, 2020.
- [36] H. Wu and S. Wager. Thompson sampling with unrestricted delays. In Proceedings of the 23rd ACM Conference on Economics and Computation , pages 937-955, 2022.
- [37] M. Zhang, Y. Wang, and H. Luo. Contextual linear bandits with delay as payoff. arXiv preprint arXiv:2502.12528 , 2025.
- [38] Z. Zhou, R. Xu, and J. Blanchet. Learning in generalized linear contextual bandits with stochastic delays. In Advances in Neural Information Processing Systems , pages 5197-5208, 2019.
- [39] J. Zimmert and Y. Seldin. An optimal algorithm for stochastic and adversarial bandits. In K. Chaudhuri and M. Sugiyama, editors, Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics , volume 89 of Proceedings of Machine Learning Research , pages 467-475. PMLR, 16-18 Apr 2019. URL https://proceedings.mlr. press/v89/zimmert19a.html .
- [40] J. Zimmert and Y. Seldin. An optimal algorithm for adversarial bandits with arbitrary delays. In S. Chiappa and R. Calandra, editors, Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics , volume 108 of Proceedings of Machine Learning Research , pages 3285-3294. PMLR, 26-28 Aug 2020. URL https: //proceedings.mlr.press/v108/zimmert20a.html .

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract reflect the paper's content and the paper contains proofs for the claims.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discussed the limitations and our assumptions throughout the paper.

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

Justification: The precise setting and assumptions are given in Section 2. All the theorems and lemmas are rigorously proved in the appendix.

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

Justification: We followed NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: We did not find any direct societal impact of the work performed.

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

Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Notations

For a series S , we denote S : k to be the first k elements of it. Additionally, S : -1 is the series without the last element.

Additionally, for every variable defined inside the algorithm, in the analysis we will add ( t ) to indicate that we refer to the value of this variable at the end of time t . For example, r i ( t ) is the value of r i at the end of time t .

| a t l i ( t ) d t µ i ¯ T n i ( S ) m i ( t ) l i ( t ) L i ( S ) L i ( S ) ˆ µ i ( S ) µ i ( S ) width i ( S ) width ( S ) lcb i ( S ) ucb i ( S ) lcb i ( S ) ucb i ( S ) ucb ∗ ( S ) max i ( t 1 , t 2 min i ( t 1 , t 2 ) p max i ( t ) p min i ( t ) B ( t ) B p i ( t ) M p i ( t )   | chosen arm at step t loss of arm i at step t delay at step t (stochastic) mean loss of arm i switch to algorithm ALG point number of pulls of arm i in S number of pulls of arm i until time t sample estimator for the loss of arm i in step t total loss of arm i in set S sample estimator for the loss of arm i w.r.t the steps in S average loss of arm i w.r.t the steps in S sample estimator average loss of arm i w.r.t the steps in S average confidence width of arm i w.r.t the steps in S estimator confidence width w.r.t the steps in S average lower confidence bound of arm i w.r.t set S average upper confidence bound of arm i w.r.t set S estimator lower confidence bound of arm i w.r.t set S estimator upper confidence bound of arm i w.r.t set S maximum pull probability of arm i in the interval minimum pull probability of arm i in the interval Set of the steps whose feedback was observed Set of observed inactive steps that were pulled with prob. p Set of inactive steps up to time t that were pulled with prob. p   | &#124; s ∈ S : a s = i &#124; &#124; s ∈ [ t ] : a s = i &#124; l i ( t ) 1 [ a t = i ] p i ( t ) ∑ s ∈ S l i ( s ) ∑ s ∈ S l i ( s ) 1 n i ( S ) ∑ s ∈ S : a s = i l i ( s ) 1 &#124; S &#124; L i ( S ) min { 1 , √ 2 log ( T ) n i ( S ) } min { 1 , √ 2 K log ( T ) &#124; S &#124; } max { ˆ µ i - width i ( S ) , lcb i ( S : - 1 ) } min { ˆ µ i + width i ( S ) , ucb i ( S : - 1 ) } max { µ i ( S ) - width ( S ) , lcb i ( S : - 1 ) min { µ i ( S )+ width ( S ) , ucb i ( S : - 1 ) min i { ucb i ( S ) , ucb i ( S ) } max t 1 ≤ t ≤ t 2 p i ( t ) min t 1 ≤ t ≤ t 2 p i ( t ) p max i (0 , t ) p min i (0 , t ) { s : s + d s < t } { s ∈ B ( t ) : p i ( s ) = p ∧ s ≥ τ i } { τ i ≤ s ≤ t : p i ( s ) = p }   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## B Algorithm

## Algorithm 5 Delayed SAPO Algorithm

```
Require: Number of arms K , number of rounds T ≥ K , Algorithm ALG . 1: Initialize active arms A = { 1 , . . . , K } , S = ⟨⟩ 2: for t = 1 , 2 , . . . , T do 3: for s ∈ B \ S do 4: S = S + ⟨ s ⟩ 5: if not BSC ( S ) (Procedure 7) then 6: Switch to ALG . 7: U ( t ) = { i ∈ A : ˆ µ i ( S ) -9 width i ( S ) > ucb ∗ ( S ) } ▷ Elimination 8: A = A\ U 9: for i ∈ U do ▷ Initialization for eliminated arms 10: Set τ i = t , p 1 i = 1 2 K + n i ( S ) 2 T , ˜ S i = S , ˜ µ i = ˆ µ i ( S ) , ˜ ∆ i = 8 width i ( S ) , N 1 i := 1280 / ( p 1 i ˜ ∆ 2 i ) , E i = 0 , r i = 1 , S 1 i = ⟨⟩ , C p 1 i · 2 -j i = ∅ ∀ j ∈ [log ( T )] 11: for i ∈ ([ K ] \ A ) do 12: p i ( t ) , err = EAP ( i ) (Procedure 6) 13: if err then 14: Switch to ALG . 15: ∀ i ∈ A p i ( t ) = ( 1 -∑ j ∈ ([ K ] \A ( t )) p j ( t ) ) / |A ( t ) | 16: Observe feedback and update variables
```

## Procedure 6 Eliminated Arms Processing (EAP) Subroutine

## Require: Arm i

```
1: p := p r i i 2: while B p i \ C p i = ∅ do 3: for s ∈ B p i \ C p i do 4: S r i i = S r i i + ⟨ s ⟩ , C p i = C p i ∪ { s } 5: if | S r i i | ˜ µ i -¯ L ( S r i i ) ≥ 1 4 ˜ ∆ i N r i i then ▷ phase error 6: E i = E i +1 , N r i +1 i = max { N 1 i , 1 2 N r i i } , p r i +1 i = min { p 1 i , 2 p r i i } , S r i +1 i = ⟨⟩ , r i = r i +1 7: if E i ≥ 3 log ( T ) then return 0 , True ▷ Switch to adversarial algorithm 8: break 9: if | S r i i | = ⌊ N r i i ⌋ then ▷ phase ended 10: N r i +1 i = 2 N r i i , p r i +1 i = 1 2 p r i i S r i +1 i = ⟨⟩ , r i = r i +1 11: break 12: p := p r i i return p r i i , False
```

̸

## Procedure 7 Basic Stochastic Checks (BSC) Subroutine

```
Require: Series of processed pulls S 1: if ∃ i ∈ A : µ i ( S ) ̸∈ [ lcb i ( S ) -width ( S ) , ucb i ( S ) + width ( S )] then 2: return False 3: if ∑ s ′ ∈ S ( l a s ′ ( s ′ ) -ucb ∗ ( S ) ) > 272 √ KT log ( T ) + 10 σ max ( t ) log ( K ) then 4: return False return True
```

## C General Lemmas

Lemma C.1 (Freedman's Inequality, Theorem 1.1 in Tropp [31]) Let { X k } k ≥ 1 be a real valued martingale difference sequence adapted to a filtration { F t } t ≥ 0 . If X k ≤ R a.s. Then, for all t ≥ 0

and σ 2 ≥ 0 ,

<!-- formula-not-decoded -->

Lemma C.2 (Lemma F.4 in Dann et al. [7]) Let { X t } T t =1 be a sequence of Bernoulli random and a filtration F 1 ⊆ F 2 ⊆ ... F T with P ( X t = 1 | F t ) = P t , P t is F t -measurable and X t is F t +1 -measurable. Then, for all t ∈ [ T ] simultaneously, with probability 1 -δ ,

<!-- formula-not-decoded -->

Lemma C.3 (Consequence of Freedman's Inequality, e.g., Lemma 27 in Efroni et al. [10]) Let { X t } t ≥ 1 be a sequence of random variables, supported in [0 , R ] , and adapted to a filtration F 1 ⊆ F 2 ⊆ ... F T . For any T , with probability 1 -δ ,

<!-- formula-not-decoded -->

Lemma C.4 For every arm i , r i ( T ) ≤ 7 log ( T )

Proof: Every phase must finish with Line 6 or Line 11. Let r 1 be the number of phases finished with Line 6. We have:

<!-- formula-not-decoded -->

From Line 7, r 1 ≤ 3 log ( T ) . Thus:

## Proof:

<!-- formula-not-decoded -->

If the 7 log ( T ) 's round is reached and finished, then the horizon has arrived. Else, the algorithm will switch.

Lemma C.5 Let i be some arm, and S be a series of steps. Denote p min = min s ∈ S p i ( s ) . Then, if | S | ≥ log ( 1 δ ) p min , w.p 1 -δ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, for every k the variance is bounded by:

<!-- formula-not-decoded -->

Using Lemma C.1 with σ 2 = | S | p min and R = 1 p min :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Which means that w.p 1 -δ :

<!-- formula-not-decoded -->

Which is exactly the first inequality in the lemma. The second has the same proof.

## D Stochastic

## D.1 Good Event

Definition D.1 Let G sto be the event that:

<!-- formula-not-decoded -->

If the delays are stochastic, also:

<!-- formula-not-decoded -->

Lemma D.2 For every state of S and arm i , w.p 1 -2 T :

<!-- formula-not-decoded -->

Proof: Directly from Equation 1.6 of Slivkins [28].

Lemma D.3 With probability 1 -2 T , For every arm i and n ≤ ∣ ∣ ∣ ˜ S i ∣ ∣ ∣ :

<!-- formula-not-decoded -->

Proof: Denote S = S : n for brevity.

If | S | ≤ 3 K log ( T ) then width ( S ) ≥ 1 and the bound is trivial.

Notice that since n ≤ ∣ ∣ ∣ ˜ S i ∣ ∣ ∣ , for s ∈ S we have p i ( s ) ≥ 1 K . If | S | ≥ 3 K log ( T ) , w.p 1 -2 T 3 from Lemma C.5:

<!-- formula-not-decoded -->

Union bound on all arms and n ≤ ∣ ∣ ∣ ˜ S i ∣ ∣ ∣ we get the desired results.

Lemma D.4 With probability 1 -2

<!-- formula-not-decoded -->

Proof: Notice that n i ( S ) is a sum of bernoulli variables. Fix S and i , from Lemmas C.2 and C.3, w.p 1 -2 T 3 :

<!-- formula-not-decoded -->

Union bound for all the options for S and i gives us the desired results.

Lemma D.5 For every arm, w.p 1 -1 T :

<!-- formula-not-decoded -->

Proof: Using Hoeffding, with probability 1 -1 T 2 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From Lemma C.4:

Union bound for all arms concludes the proof.

Lemma D.6 For every state of S , w.p 1 -1 T :

<!-- formula-not-decoded -->

Proof: From Lemma C.1, with σ 2 = T and R = 1 :

<!-- formula-not-decoded -->

Lemma D.7 If the delays are stochastic, with probability 1 -1 T :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Fix some t ≤ T . From Lemma C.3, w.p 1 -1 T 2 :

<!-- formula-not-decoded -->

Union bound for all t concludes the proof.

Corollary D.8 G sto happens w.p 1 -9 T

Proof: Union bound of Lemmas D.2 to D.7

<!-- formula-not-decoded -->

Proof: For every t ≤ T :

<!-- formula-not-decoded -->

## D.2 Regret Analysis

Lemma D.9 Assume G sto , the optimal arm i ∗ will not be evicted.

Proof: Assume by contradiction that arm i ∗ was evicted. Namely, there is S such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From the definition of ucb ∗ and G sto , there is an arm i such that:

<!-- formula-not-decoded -->

Which contradicts the fact that i ∗ is optimal.

Lemma D.10 Assume G sto , For every two arms i 1 , i 2 and series n ≤ min {∣ ∣ ∣ ˜ S i 1 ∣ ∣ ∣ , ∣ ∣ ∣ ˜ S i 2 ∣ ∣ ∣ } :

<!-- formula-not-decoded -->

Additionally, for every arm i :

<!-- formula-not-decoded -->

Proof: Denote S = S : n for brevity.

Since i 1 was not eliminated:

<!-- formula-not-decoded -->

From G sto :

From G sto :

If n i ( S ) ≥ 192 log ( T ) :

<!-- formula-not-decoded -->

Notice that since i 1 and i 2 were not evicted in the steps of S , we have E [ n i 1 ( S )] = E [ n i 2 ( S )] . Thus:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If n i ( S ) ≤ 192 log ( T ) :

In both cases:

Additionally, from G sto :

<!-- formula-not-decoded -->

Lemma D.11 Let N be a set over R with average µ . Then, at most half of N are greater than 2 µ .

Proof: Let X be a r.v sampled from N with uniform distribution. Easy to see that E [ X ] = µ . From Markov inequality:

<!-- formula-not-decoded -->

Lemma D.12 (restatement of Lemma 4.2) If arm i 1 was eliminated before i 2 then,

<!-- formula-not-decoded -->

Proof: Let i 1 and i 2 be two arms such that i 1 was evicted before i 2 . From Lemma D.10:

<!-- formula-not-decoded -->

Lemma D.13 Assume G sto , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Proof:

Let A ′ be the state of A such that |A| = K 2 (namely, the last half active arms). We show that:

<!-- formula-not-decoded -->

If ∆ i ≤ 2∆ avg it is trivial. Otherwise, from Lemma D.11 there is an arm j / ∈ A ′ such that ∆ i ≤ 2∆ avg . From Lemma D.12, ∆ i ≤ 20∆ j ≤ 40∆ avg .

Notice that if arm i was the j th evicted arm, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Which means:

<!-- formula-not-decoded -->

Lemma D.14 Assume G sto , then for any arm i and step t &lt; τ i :

<!-- formula-not-decoded -->

Proof: Let S be the set of observed pulls at time t . From Lemma D.10:

<!-- formula-not-decoded -->

At step t , there are σ ( t ) missing pulls. When those missing pulls were pull, the probability of arm i was bounded by p max i ( τ i -1) (as it is its general maximum probability). Thus, from G sto , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma D.15 Assume G sto and that K ≤ T 12 log ( T ) , then for any arm i and step τ i &lt; t ≤ ¯ T :

<!-- formula-not-decoded -->

Proof: From the algorithm defintion we have:

<!-- formula-not-decoded -->

From G sto , for every i and every phase r i :

<!-- formula-not-decoded -->

Thus:

From Lemma C.4 and G sto :

<!-- formula-not-decoded -->

Let s ≤ t be the last pull of a phase w.p p , which means that B p i ( s ) \ C p i ( s ) = ∅ , so | M p i ( t ) \ C p i ( t ) | = | M p i ( s ) \ C p i ( s ) | ≤ σ ( s ) ≤ σ max ( t ) . Then:

<!-- formula-not-decoded -->

Thus, from Equation (6):

<!-- formula-not-decoded -->

From G sto and the assumption that K ≤ T 12 log ( T ) :

<!-- formula-not-decoded -->

From Lemma D.10:

<!-- formula-not-decoded -->

Corollary D.16 Assume G sto , for every t ≤ ¯ T :

<!-- formula-not-decoded -->

Proof: If K ≤ T 12 log ( T ) , it follows directly from Lemmas D.13 to D.15. Else, we have:

<!-- formula-not-decoded -->

## D.3 With high probability the algorithm doesn't switch

Lemma D.17 Assume G sto , for every arm i and S ⊆ ˜ S i :

<!-- formula-not-decoded -->

Proof: From G sto :

<!-- formula-not-decoded -->

Lemma D.18 Assume G sto , for every state of S at time t :

<!-- formula-not-decoded -->

Proof: From Corollary D.16:

<!-- formula-not-decoded -->

From G sto , for every state of S :

<!-- formula-not-decoded -->

Lemma D.19 Assume G sto , for every arm i , E i ( ¯ T ) ≤ 3 log ( T ) .

Proof: Fix phase r . We will show that w.p 31 32 ,

<!-- formula-not-decoded -->

If | S r i | ≤ 1 4 ˜ ∆ i N r i , this is trivial. Otherwise, it means that:

<!-- formula-not-decoded -->

Now we can use Lemma C.5, w.p 31 32 , as the inequality | S r i | ≥ log (32) p r i is satisfied. We have:

<!-- formula-not-decoded -->

Additionally, from G sto :

From G sto :

<!-- formula-not-decoded -->

All of the above is true to all states throughout the phase (since Lemma C.5 is true for max k ).

This means that in every phase Line 6 happens w.p 1 32 . Thus:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Corollary D.20 Assume G sto , T = ¯ T .

Proof: Directly from Lemmas D.17 to D.19

## D.4 Conclusion

Theorem D.21 For adversarial delays:

<!-- formula-not-decoded -->

For stochastic delays we can also say:

<!-- formula-not-decoded -->

Proof: Assume G sto , from Corollaries D.16 and D.20, The above is true with probability 1 -2 T . From Corollary D.8, this is asymptotically true even without the assumption of G sto .

## E Adversarial

Lemma E.1 Let X be a random variable such that for every x ≥ 0 there is some a &gt; 0 such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof: We use the CDF representation of the expectation:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then:

Lemma E.2 (restatement of Lemma 5.2) Let H r i be the history (i.e., chosen actions) of rounds that are observed by the begining of the r phase of arm i . Denote E r i [ · ] = E [ · | H r i ] and Pr r i [ · ] = Pr[ · | H r i ] . For every arm i and phase r we have:

<!-- formula-not-decoded -->

Proof: Let S be the sequence of N r i observed rounds starting from the beginning of r phase of arm i , so that the first | S r i | rounds in S are exactly S r i . Let also X 1 , ..., X N r i i.i.d ∼ Bernoulli ( p r i ) . From Lemma C.5, if N r i ≥ m p r i , we have for every m&gt; 0 , conditioned on the history H r i , with probability of at least 1 -e -m :

<!-- formula-not-decoded -->

Now, note that for k = | S r i | , ∑ t ∈ S : k [ X t l t p r i -l i ( t ) ] distributes exactly like L i ( S r i ) -¯ L i ( S r i ) . Thus, conditioned on the history H r i , with probability of at least 1 -e -m :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Notice that if the phase was not finished it means that | S r i | ˜ µ i -¯ L r i ≤ 1 4 ˜ ∆ N r i . Given that we have:

If N r i ≤ m p r i :

From Lemma E.1:

<!-- formula-not-decoded -->

Combining the above with Equation (8) completes the proof.

Lemma E.3 For every arm i and phase r ,

<!-- formula-not-decoded -->

Proof: We will prove using reverse induction on r .

For r = r i ( T ) (namely, after all phases) we have:

<!-- formula-not-decoded -->

Assume true for r +1 , we prove for r .

If Line 11 was triggered it means that | S r i | = N r i and N r +1 i = 2 N r i . We have:

<!-- formula-not-decoded -->

̸

If Line 6 was triggered and N r i = N 1 i it means that N r +1 i = 1 2 N r i . We have:

<!-- formula-not-decoded -->

If N r i = N 1 i it means that N r +1 i = N r i . We have:

<!-- formula-not-decoded -->

Lemma E.4 For every arm i :

<!-- formula-not-decoded -->

Proof: Lemma E.3 with r = 1 gives:

<!-- formula-not-decoded -->

From Lemma C.4:

If n i ( ˜ S i ) T ≥ 1 K :

If n i ( ˜ S i ) T ≤ 1 K :

In any case:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since for every i , ˜ S i is a sub-series of ¯ S , we have ucb ∗ ( ˜ S i ) ≥ ucb ∗ ( ¯ S ) . From Equations (9) and (10):

<!-- formula-not-decoded -->

## Theorem E.5

<!-- formula-not-decoded -->

Proof: From Line 3:

<!-- formula-not-decoded -->

From Wald's equation and Line 1, for every arm i :

<!-- formula-not-decoded -->

Adding the missing pulls we get:

<!-- formula-not-decoded -->

From Equations (11) and (12) and lemma E.4, for every arm i :

<!-- formula-not-decoded -->

Since after ¯ T the algorithm switches to ALG , we have:

<!-- formula-not-decoded -->

Which concludes the proof.

Lemma E.6 (Restatement of Lemma 5.3)

<!-- formula-not-decoded -->

Proof: Let S ∗ be the set that minimizes | S | + √ D ¯ S . If | S ∗ | ≥ 1 2 σ max it concludes the proof. Continuing with the case that | S ∗ | &lt; 1 2 σ max .

Let t be the step such that σ ( t ) = σ max . Since | S ∗ | &lt; 1 2 σ max , after skipping there are at least 1 2 σ max non-skipped missing steps at time t . Let s 1 , ..., s 1 2 σ ( t ) ∈ ¯ S ∗ be the series of those 1 2 σ ( t ) missing steps, ordered in descending order of when they were pulled. Namely, s 1 is the most recent pull in the series and s 1 2 σ ( t ) is the oldest pull.

Since there are at least i -1 missing pulls that were pulled after s i , we have t -s i ≥ i . Additionally, since s i is missing, we have s i + d s i &gt; t . Combining both we have d s i &gt; i . Thus:

<!-- formula-not-decoded -->

## F Removing the log ( K ) factor

We show that a simple modification of the algorithm can eliminate the log ( K ) factor from the additive delay term in both the adversarial and stochastic settings (Theorems D.21 and E.5). To avoid adding complexity to the already intricate algorithm, we present this modification separately as an optional, opt-in feature.

## Algorithm 8 Delayed SAPO Algorithm with reduced log ( K )

```
Require: Number of arms K , number of rounds T ≥ K , Algorithm ALG . 1: Initialize active arms A = { 1 , . . . , K } , S = ⟨⟩ , h = 1 , G = ∅ 2: for t = 1 , 2 , . . . , T do 3: for s ∈ B \ S do 4: S = S + ⟨ s ⟩ 5: if not BSC ( S ) (Procedure 7) then 6: Switch to ALG . 7: U ( t ) = { i ∈ A : ˆ µ i ( S ) -9 width i ( S ) > ucb ∗ ( S ) } ▷ Ghosting 8: G = G ∪ U 9: for i ∈ U do ▷ Initialization for phases variables 10: Set p 1 i = 1 2 K + n i ( S ) 2 T , ˜ S i = S , ˜ µ i = ˆ µ i ( S ) , ˜ ∆ i = 8 width i ( S ) , N 1 i := 1280 / ( p 1 i ˜ ∆ 2 i ) , E i = 0 , r i = 1 , S 1 i = ⟨⟩ , C p 1 i · 2 -j i = ∅ ∀ j ∈ [log ( T )] 11: if min i width i ( S ) ≤ 2 -h then ▷ Elimination point 12: for i ∈ G do 13: τ i = t , S g i = S \ ˜ S i 14: A = A\ G , G = ∅ , h = h +1 15: for i ∈ ([ K ] \ A ) do 16: p i ( t ) , err = EAP ( i ) (Procedure 6) 17: if err then 18: Switch to ALG . 19: ∀ i ∈ A p i ( t ) = ( 1 -∑ j ∈ ([ K ] \A ( t )) p j ( t ) ) / |A ( t ) | 20: Observe feedback and update variables
```

Procedure 9 Basic Stochastic Checks (BSC) Subroutine with reduced log ( K )

Require: Series of processed pulls S

- 1: if ∃ i ∈ A : µ i ( S ) ̸∈ [ lcb i ( S ) -width ( S ) , ucb i ( S ) + width ( S )] then
- 2: return False

<!-- formula-not-decoded -->

- 4: return False return True

The key change involves introducing elimination points, where the h th elimination point is when the confidence width of at least one arm falls below 2 -h . When an arm is eliminated under the current algorithm, it enters a ghost period-a phase during which it remains practically active (receiving the same pull probability as active arms) and is then formally eliminated at the next elimination point. Additionally, we modify the threshold in BSC 's Line 3 by removing the log ( K ) term from its additive component.

We first show that the leading term in the stochastic regret remains asymptotically unchanged, since the number of pulls during the ghost period is asymptotically smaller than the number of pulls during the active period (Lemma F.1). We then prove a variant of Lemma D.13 without the log ( K ) factor (Lemma F.3), which is the original source of this term in the regret bound. With these changes, the updated version of BSC 's Line 3 (with an appropriate choice of the constant C ) still doesn't triggers a switch in the stochastic settings.

The log ( K ) factor is also removed from the adversarial regret, without affecting the asymptotic behavior. The main contribution to adversarial regret comes from the check in BSC 's Line 3, where we removed the log ( K ) term. We also need to verify that the relation between the losses and ucb ∗ given in Equation (12) still holds. This is indeed the case, since the check in BSC 's Line 1 continues to be valid during the ghost period (as i ∈ A still holds).

Lemma F.1 Assume G sto , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof: Assume i is eliminated in the h th elimination point and denote S h to be S at that time. Let i h be the arm whose width crossed 2 -h in the h th elimination point. From the definition of width , we still have that its width is greater then 1 2 2 -h . Thus:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma F.2 Assume G sto , for every i ∈ [ d ] that is eliminated at the h th elimination point we have:

<!-- formula-not-decoded -->

Proof: Let S h be S at the time of h th elimination point. From Lemma D.10:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Again from Lemma D.10:

Lemma F.3 Assume G sto , we have:

This concludes to:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof: In the same way as Lemma D.13, we denote A ' to be the state of A such that | A | = K 2 . In the same way we have that ∀ i ∈ A ′ ∆ i ≤ 40∆ avg and:

<!-- formula-not-decoded -->

Fix elimination point h , denote I h to be the set of arms eliminated at that point. By definition, we have for every i ∈ I h that p max i ≤ 1 | I h | . From Lemma F.2:

<!-- formula-not-decoded -->

Let h 1 be the first elimination point in which arms from A ' are eliminated. Again from Lemma F.2, we have for some i ∈ A ′ :

<!-- formula-not-decoded -->