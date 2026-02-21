## STaR-Bets: Sequential Target-Recalculating Bets for Tighter Confidence Intervals

## Václav Voráˇ cek

Second Foundation ˚

vasek.voracek@gmail.com

<!-- image -->

## Francesco Orabona

King Abdullah University of Science and Technology francesco@orabona.com

## Abstract

The construction of confidence intervals for the mean of a bounded random variable is a classical problem in statistics with numerous applications in machine learning and virtually all scientific fields. In particular, obtaining the tightest possible confidence intervals is vital every time the sampling of the random variables is expensive. The current state-of-the-art method to construct confidence intervals is by using betting algorithms. This is a very successful approach for deriving optimal confidence sequences, even matching the rate of law of iterated logarithms. However, in the fixed horizon setting, these approaches are either sub-optimal or based on heuristic solutions with strong empirical performance but without a finite-time guarantee. Hence, no betting-based algorithm guaranteeing the optimal

O p σ 2 log 1 δ n q width of the confidence intervals are known. This work bridges this gap. We propose a betting-based algorithm to compute confidence intervals that empirically outperforms the competitors. Our betting strategy uses the optimal strategy in every step (in a certain sense), whereas the standard betting methods choose a constant strategy in advance. Leveraging this fact results in strict improvements even for classical concentration inequalities, such as the ones of Hoeffding or Bernstein. Moreover, we also prove that the width of our confidence intervals is optimal up to an 1 ` o p 1 q factor diminishing with n . The code is available on github.

## 1 Introduction

Quantifying uncertainty is a cornerstone of statistical inference, and Confidence Intervals (CIs) remain one of the most widely used tools for this purpose. Typically constructed within the frequentist paradigm, a p 1 ´ δ q -CI provides a range of plausible values for an unknown parameter, derived from observed data, with the guarantee that the procedure yields intervals covering the true parameter value in 1 ´ δ proportion of hypothetical repetitions. In particular, in this paper we are interested in providing CI for the mean of a bounded random variable. So, more formally, after observing n i.i.d. samples X 1 , . . . , X n of a random variable in r 0 , 1 s with mean µ , we want to find l n and u n such that

<!-- formula-not-decoded -->

Weaim for narrow confidence intervals, i.e., small u n ´ l n . Classical methods provide well-established procedures with optimal asymptotic properties, but their performance deteriorates when n is small.

This paper explores an alternative approach to constructing confidence intervals, based on the framework of game-theoretic probability and sequential betting. In this framework, the algorithm makes sequential bets on the values of X i ´ m . If m is close to the true mean µ , then the random

˚ work done while with University of Tübingen.

variables X i ´ m is approximately zero mean. Now, it is intuitive that no strategy can hope to gain money betting on a fair random variable and we expect little money betting on a random variable with mean very close to zero. Conversely, if the algorithm makes enough money, we can infer that m is far from the true mean µ . The above reasoning can be carried out in rigorous way, using the concept of test martingales [20]. In particular, one can show that we can construct the confidence interval as the set of all values of m such that the wealth does not reach 2 δ . In this view, a better betting algorithm results in a tighter confidence interval.

These approaches give state-of-the-art theoretical and empirical results in the time-uniform case. However, they fall short in the finite-sample regime, both in theory and in practice.

Contributions. In this paper, we propose a new strategy for betting in the finite-sample regime. We explicitly take into account how many rounds we still have and that by how much we still need to multiply our current wealth to reach the threshold. This give rise to a new family of betting algorithms for the finite-sample setting, the ‹ -algorithms (STaR=Sequential Target-Recalculating). We show that this strategy can be used, for example, to immediately improve the confidence intervals calculated through Hoeffding and Bernstein, without losing anything . Moreover, we present a new algorithm, STaR-Bets, that achieve valid state-of-the-art confidence intervals on a variety of distributions. Additionally, we show that its variant, Bets, attains the optimal rates for confidence interval width-marking the first such result for betting-based confidence interval methods.

## 2 Related Work

The literature on how to construct confidence intervals for the mean of a random variable is vast. Classic results like Hoeffding's [10] and Bernstein's [3] inequalities require the knowledge of the variance of the random variable (which can always upper bounded for bounded random variables). Later, Maurer and Pontil [12] and Audibert et al. [2] proved that it is possible to prove empirical Bernstein's inequalities, that is, with optimal dependency on the variance but without its prior knowledge. However, these results tend to be loose on small samples because they sacrifice the tightness of the interval with a simple and interpretable formula.

On the other hand, it is possible to design procedures to numerically find the intervals, without having a closed formula. A classic example is the Clopper-Pearson confidence interval [5] for Bernoulli random variables, that is obtained by inverting the Cumulative Distribution Function (CDF) of the binomial distribution. This approach is general, in the sense that it can be applied to any random variable when the CDF is known. However, no optimal solution is known if we do not have prior knowledge of the CDF of the random variable.

An alternative approach is the betting one, proposed by Cover [6] for the finite-sample case and by Shafer and Vovk [19] and Shafer et al. [20] for the time-uniform case, as a general way to perform statistical testing of hypotheses. The first paper to consider an implementable strategy for testing through betting is Hendriks [9]. Later, Jun and Orabona [11] proposed to design the betting algorithms by specifically minimizing the regret w.r.t. a constant betting scheme. Waudby-Smith and Ramdas [23] suggested the use of betting heuristics, motivated by asymptotic analyses and online convex optimization approaches to betting algorithms [14, 8]. Recently, Orabona and Jun [13] have shown that regret-based betting algorithms based on universal portfolio algorithms [7] can recover the optimal law of iterated logarithm too, and achieve state-of-the-art performance for time-uniform confidence intervals. However, these approaches fail to give optimal widths in the finite-sample regime that we consider in this paper. We explicitly note that even the optimal portfolio strategy for the finite-time setting [15] does not achieve the optimal widths because it suffers from an additional log T factor. This is due to the fact that minimizing the regret with respect to a fixed strategy does not seem to be right objective to derive confidence intervals in the finite-sample case.

The problem of achieving the tightest possible confidence interval for random variables for small samples has received a lot of interest in the statistical community. It is enough to point out that even the Clopper-Pearson approach has been considered to be too wasteful, because of the discrete nature of the CDF that does not allow an exact inversion. 2 In this view, sometimes people prefer to use approximate methods, that is, methods that do not guarantee the exact level of confidence 1 ´ δ , because they are less conservative [see, e.g., the reviews in 1, 18]. Recently, Phan et al. [16] proposed an approach based on constructing an envelope that contains the CDF of the random variable with

2 See Voracek [22], for an optimal, randomized procedure.

high probability. This approach was very promising in the small sample regime, however in Section 6 we will show that our approach is consistently better.

In this paper, we build on the ideas of Cover [6] regarding the construction of optimal betting strategies for Bernoulli random variables and design a general betting scheme in that spirit.

## 3 A Coin-Betting Game

The statistical testing framework we rely on is based on betting . In particular, we will set up simple betting games based on the hypothesis we want to test. The outcome of the testing will be linked to how much money a betting algorithm makes. Hence, we will be interested in designing 'optimal' betting strategies. In the following, we first describe a simple betting game to draw some intuition about the optimal betting strategies. Then, in Section 4 we formally introduce the testing by betting framework.

We introduce a simple sequential coin betting game in which we bet on the outcomes of a coin toss. If the result is Heads, we gain the staked amount, otherwise we lose it. The game is as follows: We will bet for n rounds, and we win if we multiply our initial wealth by at least a factor of k . Hence, we are not interested in how much money we make, but only if we pass a certain threshold.

Let's now discuss some possible scenarios, to see how the optimal betting strategy changes.

Scenario 1, n ' 1 , k ' 2 : It is optimal to bet everything and if the outcome is Heads, we win. Generally, in the last round we should always bet everything we have.

Scenario 2, n ' 5 , k ' 0 . 9 : In this case, it is optimal to not bet anything, and we win.

Scenario 3, n ' 2 , k ' 4 { 3 : Consider the possible outcomes: Heads/Heads, Heads/Tails, Tails/Heads, Tails/Tails. In the last case we always lose. If we want to win in the Heads/Tails case, we would need to bet 1 { 3 , in which case we reach the desired wealth after observing Heads and we stop betting. In the case we first observe Tails and we end up with 2 { 3 money. Then, we recover Scenario 1 and bet all the money.

Before we continue with more general scenarios, we make some observations. These considerations will be useful when we will link betting strategies to the proof of standard concentration inequalities, such as Hoeffding's or Bernstein's.

- The original values of n, k do not matter. Instead, what matter are how these quantities 'change over time,' that is, how many rounds are left and how many times we still need to multiply our wealth.
- When we hit the required wealth, we should stop betting.
- For a given n , the smaller k is, the less we need to bet. Conversely, if k is large, we need to bet aggressively.
- We should always finish the game by either going bankrupt or by hitting the target.

Already in [6, Example 3], it was shown that if a betting strategy is optimal (for hypothesis testing, in the sense of Proposition 7), it either has to reach the target, or go bankrupt. The contrary was also proven, that there exists such an algorithm. More generally, all possible outcomes of the coin-betting game were described. Let's now consider more complex scenarios.

Scenario 4, n , k « 1 ` 2 ´ n : In this case, our bets would be « 2 ´ n , so that we win as soon as we observe a single heads outcome.

Scenario 5, n , k « 2 n : In this case, our bets would be « 1 in general, as we need to roughly double our wealth every round and hope for lots of heads.

a

Scenario 6, n , k : In the general case, we bet « log k { n . In this way, if we observe H heads and T tails, the wealth would be roughly

<!-- formula-not-decoded -->

as long as H ´ T Á ? n log k .

Let us examine the event H ´ T Á ? n log k : If the coin is fair, the event H ´ T Á ? n log k -in which case we successfully multiplied our wealth by a factor k -happens with probability at most « 1 k , and it is not just a coincidence. Thus, if we multiply our wealth by factor of k , we can rule out the possibility that the game is fair at a confidence level 1 ´ 1 k . This statement is the cornerstone of the testing by betting framework and we will make it formal and prove it in the next section.

## 4 Testing and Confidence Intervals by Betting

Before describing this technique in detail, we introduce the precise mathematical framework. We start with the definition of test processes: Sequences of random variables modeling fair 3 betting games. In these games, our wealth is always non-negative and stays constant (or decreases) in expectation. In the literature, test processes are also called e-processes and they are (super)-martingales.

Definition 1 (Test process) . Let W 0 , W 1 , . . . W n be a sequence of non-negative random variables. We call it a test process if E r W 0 s ' 1 and E r W i ` 1 | W 0 , . . . , W i s ď W i , for i ě 0 .

Next, Markov's inequality quantifies how unlikely it is for a test process to grow large.

Proposition 1 (Markov's inequality) . Let W 0 , W 1 . . . , W n be a test process. For any δ P p 0 , 1 q it holds that P ␣ W n ě 1 δ ( ď δ .

<!-- formula-not-decoded -->

Finally, we introduce the betting-based confidence intervals.

Theorem 2. A confidence interval obtained within the following scheme has coverage at least 1 ´ δ .

| Confidence interval by betting   | Confidence interval by betting                                                                                                                                                                                                            |
|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Objective: Procedure:            | Construct a ( 1 ´ δ )-confidence interval for E r X s from X 1 ,...,X n P r 0 , 1 s . For each m P r 0 , 1 s , form a null hypothesis H 0 p m q : E r X s ' m . Let S ' t m &#124; H 0 p m q not rejected by betting at 1 ´ δ level u . . |
| Outcome:                         | Interval I such that S Ă I , then I is a ( 1 ´ δ )-CI for E r X s                                                                                                                                                                         |
| Testing by betting               | Testing by betting                                                                                                                                                                                                                        |
| Objective: Procedure: Outcome:   | Test the null hypothesis H 0 on X 1 ,...,X n at confidence level 1 ´ δ . Construct W 1 ,...,W n such that it is a test process under H 0 . If W n ě 1 δ , reject H 0 at confidence level 1 ´ δ .                                          |

Proof. If the mean µ is not contained in the resulting confidence interval, the null hypothesis H 0 p µ q : E r X s ' µ was rejected by betting. The corresponding sequence W µ 1 , . . . , W µ n is a test process and because it was rejected, we have W µ n ě 1 δ . But this happens with probability at most 1 ´ δ by Markov's inequality.

Test process under a null hypothesis: A popular process (optimal in a certain sense [4]) for testing a null hypothesis H 0 p m q : µ ' m for a given m P r 0 , 1 s is:

<!-- formula-not-decoded -->

where ℓ i P ' 1 m ´ 1 , 1 m ı is our betting strategy. The betting strategy is, of course, independent of X ě i but can depend on X ă i and the other known quantities m,n,δ . Under the null hypothesis, we have E r W i ` 1 | W 1 , . . . W i s ' E r W i s . Additionally, by the bound on ℓ i , we ensure that W i m ě 0 , and so t W i m u n i ' 1 is a test process. We have already used this test process in the coin-betting example, if we identify Heads (resp. Tails) with 1 (resp. 0 ) and set m ' 1 2 , then ℓ i { 2 is the fraction of money we bet.

Theorem 2 provides a scheme on how to construct confidence intervals by testing all the possible values m of the expectation of the random variable. Such an approach would require us to test a continuum of hypotheses which is impossible in general. There are two standard ways to proceed:

3 We allow for the game to be unfair against us.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

1. Root finding: If the function m ÞÑ W n m has a simple structure (e.g., quasi-convex), then the set of non-rejected mean candidates form an interval and we can easily find its end points by numerical methods, or even in a closed form.
2. Discretization: Discretize the interval r 0 , 1 s into G grid points m 1 ă m 2 ă ¨ ¨ ¨ ă m G . Use the same betting strategies for all m in m i ď m ď m i ` 1 and all 1 ď i ď G ´ 1 . In this case, the function m ÞÑ W n m is monotonic in between the grid points, and so evaluating m ÞÑ W i m only at grid-points provide sufficient information for the construction of the confidence interval. We adopt this approach.

## 4.1 Hoeffding and Sequential Target-Recalculating ( ‹ ) Bets

In this section, we recover the classical confidence intervals by Hoeffding's inequality within the betting framework using Theorem 2-recovering Bernstein's intervals follows the same steps, so it is deferred to Appendix B. Our plan is to show that the Hoeffding betting strategy violates the design principles we have discussed earlier in this section. Hence, we can fix those problems and obtain tighter confidence intervals, essentially for free. More generally, we introduce ‹ -betting strategies as the ones that have the better design principles.

We will use the following lemma to derive the test processes.

Lemma 3 (Hoeffding's lemma) . Let X P r 0 , 1 s with mean µ ' E r X s , then

<!-- formula-not-decoded -->

Furthermore, the sequence W 0 ' 1 , W i ` 1 ' W i exp ´ ℓ ¨ p X i ´ µ q ´ ℓ 2 8 ¯ for i ě 0 is a test process.

We compare the testing-by-betting algorithms. The standard Hoeffding test in Algorithm 1, and our improved, ‹ -Hoeffding test in Algorithm 2. We observe that the constant betting of Algorithm 1 violates our desiderata for a good betting algorithm. Concretely, it may happen that at some point W ě 1 δ , but we keep betting and end up with W ď 1 δ . Additionally, it does not adapt to the current situation of how much do we need to increase W and in how many rounds. In particular, the terms n and log 1 δ defining the bet become irrelevant as t increases. Instead, as we said, the only important quantities are i) 1 δW : how many times we need to multiply our wealth and ii) n ´ t : the number of remaining rounds. This motivates the following definition.

We call a betting algorithm for a test process t W t u n t ' 1 a ‹ -algorithm if it uses the quantities the log 1 W t δ and n ´ t to compute the bet at time t .

As an example, we can immediately generate the ‹ -version of the Hoeffding algorithm in Algorithm 2 and prove that it is never worse than the original algorithm.

Proposition 4. Let m P r 0 , 1 s . Whenever Algorithm 1 rejects the null hypothesis H 0 p m q : m ' E r X s , then so does Algorithm 2 if they share the realizations X t , 1 ď t ď n .

Figure 1: Comparison of the Algorithms 1,2, 5, and 6 with δ ' 0 . 05 on 1000 realizations of the Bernoulli random variable with mean 0 . 9 . ( L ): We show the final value of W depending on the choice of m for the algorithms. The vanilla versions have exponential dependency on m , while the ‹ versions virtually always end up with W P t 0 , 1 δ u . Additionally, we can confirm that the ‹ versions reject the null hypothesis for more values of m . ( R ): Here we show the evolution of W throughout the runs of the algorithms for m ' 0 . 86 . We can see that the Bernstein's testing algorithm already achieved the required wealth, but later lost it, unlike ‹ -Bernstein's testing which stopped betting after reaching it. We can also see that towards the end, ‹ -Hoeffding betting started betting very aggressively in order to have a chance to reach the desired wealth.

<!-- image -->

Proof. Consider Algorithm 1. We first show that ℓ H is selected in a way that ř n i ' 1 X i required to reject the null hypothesis is smallest possible. That is:

<!-- formula-not-decoded -->

The constraint is ř n i ' 1 X i ě nm ` log 1 δ { λ ` nλ { 8 . Minimizing the RHS over λ , the solution is λ ' b 8 log 1 δ { n .

Now, consider Algorithm 2. By the same argument, ℓ H ‹ at time t is minimizing ř n i ' t X i under the constraint that the null hypothesis is rejected. Consequently, the required ř n i ' 1 X i for Algorithm 2 is initially the same as for Algorithm 1, but it decreases whenever ℓ H ‹ changes with respect to the previous iteration.

Corollary 5. Consider generating confidence intervals using Theorem 2 and a betting algorithm. The confidence intervals given by the betting Algorithm 2 are not-wider than the confidence intervals by Algorithm 1, all other things being equal. Moreover, there are cases when Algorithm 2 produces strictly smaller confidence intervals.

Remark 6 . Many popular confidence intervals including Hoeffding's, Bernstein's, and Bennet's can benefit from ‹ -betting. A standard way to derive concentration inequalities is to bound the cumulant generating function. We have seen this in Hoeffding's inequality with E r exp p ℓ ¨ p X ´ µ qqs ď exp p ℓ 2 { 8 q . In general, we get E r exp p ℓ ¨ p X ´ µ qqs ď f p ℓ q . Then, ℓ is selected to minimize ř n i ' 1 X i under the condition that ℓ ř n i ' 1 p X i ´ m q ´ log f p ℓ q ě log 1 δ , which is precisely the reason why ‹ -betting outperforms standard betting in Proposition 4. We demonstrate this in Figure 1.

We observe that ‹ -algorithms usually end with either 0 or 1 δ wealth. This is a natural consequence of the adaptiveness, as we described in Figure 1 from both sides - if ‹ algorithm reaches the target, it stops betting. On the other hand, if the target is not reached yet, the bets become more and more aggressive, often resulting in bankruptcy. This is not a bad property. If we want to design confidence intervals with exact coverage 1 ´ δ , the following proposition shows that it is actually necessary.

Proposition 7. Consider the process W i ' W i ´ 1 p 1 ` ℓ i ¨ p X i ´ µ qq with W 0 ' 1 and an algorithm for selecting bets ℓ i . If the algorithm always finishes with W n P t 0 , 1 δ u , then the algorithm falsely rejects the hypothesis H 0 p µ q : µ ' µ at confidence level 1 ´ δ with probability precisely 1 ´ δ .

Proof. By construction, E r W n s ' 1 . Let the algorithm halt with W n ' 1 δ with probability p , then E r W n s ' p δ , yielding p ' δ as required.

## 5 STaR-Bets Algorithm

In the previous section, we have described the key concept of ‹ -betting that can be implemented within the majority of common (finite time) betting schemes. It remains to choose a good betting scheme. So, here will introduce our new algorithm: STaR-Bets. Now, we will present an informal reasoning to derive a first version of our betting algorithm. Then, we prove formally that it produces intervals with the optimal width.

Let X 1 , . . . , X n P r 0 , 1 s be an i.i.d. sample of a random variable X for which we aim to find a one-sided interval for the mean parameter. In the following, we also assume that n " log 1 δ , that is, our sample is large enough. This is not a restrictive assumption. If it is violated, then the optimal confidence interval has width « 1 for a general distribution on r 0 , 1 s and we cannot hope to have a short interval anyway.

We use the processes from (2) of the form W i ` 1 ' W i p 1 ` ℓ p X i ´ m qq to refute the hypothesis m ' E r X s . So, we construct a betting strategy that aims at achieving at least 1 δ wealth after n rounds. First, we approximate the logarithm of the final wealth using a second-order Taylor expansion:

<!-- formula-not-decoded -->

For the moment, we will assume that this is a good approximation, but eventually, we will derive it formally.

Define S fi ř n i ' 1 p X i ´ m q and V fi ř n i ' 1 p X i ´ m q 2 . From the r.h.s. of (4), the constant bet ℓ ˚ maximizing the approximate wealth is

<!-- formula-not-decoded -->

A reasonable approach is to try to estimate ℓ ˚ over time, in order to achieve a wealth close to the one in (5). This is roughly the approach followed in previous work [see, e.g., 8]. Instead, here we follow the approach suggested in Waudby-Smith and Ramdas [23]: We are only interested in the case where the (approximate) log-wealth reaches log 1 δ . In other words, the outcome of the betting game is binary: We either reach the desired log-wealth for the candidate of the mean, m , and we reject the null hypothesis H 0 p m q : E r X s ' m , or we fail to do so. This means that we want

<!-- formula-not-decoded -->

Estimating V online, Shekhar and Ramdas [21] proved that this strategy will give asymptotically optimal confidence intervals almost surely. However, we are aiming for a finite-time guarantee, which requires a completely different angle of attack.

First of all, it might not be immediately apparent why we expressed (5) as function of V only, while we could also get an S term. The reason is that this is an easier quantity to estimate: Estimating V n « E rp X ´ m q 2 s P Θ p 1 q is easier than estimating S n « E r X ´ m s P Θ p 1 ? n q in the terms of relative error. Relative error is relevant because if we overestimate ℓ ˚ by a factor of 2 , then we would end up with 0 (approximated) log-wealth.

b

Also, observe that ℓ ‹ P Θ p log 1 δ { n q which justifies the Taylor approximation in (4). Indeed, we aim to approximate ℓ ‹ and | X i ´ m | ď 1 , so the error of the Taylor approximation is

<!-- formula-not-decoded -->

This error is of an order smaller than log 1 δ as long as n " log 1 δ , as we have assumed. Recalling our goal of achieving a log-wealth of log 1 δ , this means that the Taylor approximation is sufficiently good.

Let us now examine the approximate length of the confidence interval consisting of candidate means m for which we did not reach log-wealth log 1 δ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Algorithm 4 Testing with ‹ -Bets Require: i.i.d. X 1 , . . . , X n P r 0 , 1 s Require: α, δ ą 0 , m P r 0 , 1 s , n V Ð 0 , lgW Ð 0 for 1 ď t ď n do v Ð ´ V {p t ´ 1 q ` 10 log 8 α n p t ´ 1 q 2 ¯ ^ 1 ℓ Ð b 2 p log 1 δ ´ lgW q{pp n ´ t ` 1 q v q ^ 1 lgW Ð lgW ` log p 1 ` ℓ p X i ´ m qq V Ð V `p X i ´ m q 2 if lgW ě log 1 δ then Rejected

We can further approximate V « E r V s ' n V r X s ` E r S s 2 { n « n V r X s , since n V r X s P Θ p n q , while E r S s 2 { n P Θ p 1 q , then the bound from (6) matches the width from the Bernstein's inequality, which is the one of a Normal approximation and thus cannot be improved.

The only missing ingredient is how to estimate V over time. Recalling that we aim for a low relative error, after seeing t outcomes we will use the estimator V { n « ř t i ' 1 p X i ´ m q 2 t ` n log 1 α t 2 , where the second term is carefully constructed to guarantee that the estimate has a small relative error with high probability (depending on α ) uniformly over m and 1 ď t ď n .

The final algorithm is shown in Algorithm 3, where we run the above betting procedure for a possible values of m P r 0 , 1 s and reject them as mean candidates if the log-wealth is at least log 1 δ .

Theorem 8. For every random variable X P r 0 , 1 s with mean µ and variance σ 2 ą 0 , every α, δ P p 0 , 1 q , and c ą 0 , there is n 0 depending on α, δ, σ 2 , c , such that for all n ě n 0 , Algorithm 3 rejects at confidence level 1 ´ δ every m satisfying

<!-- formula-not-decoded -->

with probability at least 1 ´ α . Furthermore, we have that n 0 P O p c ´ 4 q when treating the other variables as constants.

## The proof is in Appendix A.

Corollary 9. Algorithm 3 can be used in the framework of Theorem 2 (using the discretization technique for constructing intervals) to construct confidence intervals of the width up to a 1 ` o p 1 q factor diminishing with n .

Our proposed algorithm is ‹ -Bets in Algorithm 4, the ‹ -version of Algorithm 3. We discuss the implementation details in Appendix D.

We remark that while the algorithm is tailored to confidence intervals, it naturally produces a confidence sequence as a by-product. More concretely, we do not need to construct the test process W 1 , . . . W n from Theorem 2 at once and then collect the non-rejected mean candidates. We can construct the test processes sequentially when new samples arrive and after each of which we can recompute the confidence interval. A mild modification (either replace Markov's inequality by Ville's inequality, or do some brain gymnastics) of the original argument shows that all the confidence intervals contain the mean simultaneously with probability 1 ´ δ in the frequentist sense. Straightforward modification of Theorem 8 shows that the width of the confidence interval after

b

observing t samples is (up to a constant factor) log 1 δ n { t 2 . This being said, we believe that there are generally better options when one needs confidence sequences as discussed in Section 2, such as [13].

We conclude this section by emphasizing that while we have proven an optimality result of Algorithm 3 in Theorem 8, and other optimality results of ‹ -technique in Proposition 4, when they are combined in Algorithm 4, we were not able to prove these optimality properties, and we only have empirical evidence of the superiority over the competitors. The validity of the hypothesis testing (and thus of the resulting confidence interval) still holds by Theorem 2.

Figure 2: We directly compare the widths of the confidence intervals. Note the log ´ log scale. For all the methods and every n ' 8 , 16 , . . . , 256 , we have estimated the mean 1000 ˆ of a fresh realization of the corresponding random variable and plotted the average distance to the mean. (L): When estimating the mean of beta distribution, we observe that that with increasing n , we are getting closer to the performance of T-test. (R): When estimating Bernoulli mean, the performance of ‹ -Bets is very similar to the specialized optimal methods.

<!-- image -->

## 6 Experiments

Now we provide some experiments suggesting that ‹ -Bets yields shorter confidence intervals than alternative methods. Here, we provide a 'teaser' of our experiments, while the extensive experimental evaluation is in Appendix C. We identified several methods as our direct or indirect competitors and will briefly discuss them.

- Confidence interval derived from the T-test. This is a widely used confidence interval in practice, but it does not have guaranteed coverage in general. We show that our ‹ -Bets algorithm is competitive with the T-test and has guaranteed coverage for X P r 0 , 1 s .
- Clopper-Pearson [5] is the best deterministic confidence interval for a binomial sample (sum of Bernoulli random variables). Nevertheless, ‹ -Bets often produces shorter intervals.
- Randomized Clopper-Pearson [22] is the optimal binomial confidence interval. ‹ -Bets is still competitive.
- Hedged-CI [23] is a confidence interval based on betting and currently the best known algorithm for constructing confidence intervals. It is very similar to our Bets algorithm from 3 with comparable performance. The difference lies in the fact the we estimate E rp X ´ m q 2 s , while [23] estimates E rp X ´ µ q 2 s . However, our ‹ -Bets is significantly stronger.
- Hoeffding's inequality and the empirical Bernstein bound are the standard ways to construct confidence intervals, and so we include them in the experiments. Empirical Bernstein bound is usually weak because of the additive terms. Hoeffding's inequality is generally weak, but when V r X s « 1 2 , it is competitive with some of the methods, but not with ‹ -Bets.
- The method 4 of Phan et al. [16] was state of the art at the time of introduction, aiming at short intervals for very small ( 10 ´ 50 ) sample sizes. We show that ‹ -Bets is is stronger even in this regime. Furthermore, this method does not scale well to large samples.

First, we perform several experiments with Beta and Bernoulli distribution to quickly assess the competing methods. We show in Figure 2 how the widths of the confidence intervals evolve as we increase n and conclude, that from our direct competitors, Hedged-CI of Waudby-Smith and Ramdas [23] is the strongest one, thus we use it in our further experiments. In Figure 3, we provide teaser for the extensive experimental evaluation in Appendix C. Specially, we introduce a CDF figure that shows the distribution of the lower-confidence bounds over 1000 repetitions of the experiments with the same setting, which allows for a systematic comparison of the methods.

4 It is called practical mean bounds for small samples, so we abbreviate it as PMBSS.

Figure 3: CDF figure: When a curve corresponding to a method passes through a point p x, y q , it means that the y -fraction (of 1000 repetitions) of lower confidence bounds was smaller than x . The vertical magenta line shows the mean position, and the vertical one shows the 1 ´ δ quantile. We can see that in both cases, ‹ -Bets passes through the intersection, implying that the coverage is « 1 ´ δ . (Top) Estimation of the mean of Beta distribution (L): we can see that T-test produces shorter intervals than ‹ -Bets, but that it has significantly smaller coverage than claimed. (R): Here, ‹ -Bets produces shortest intervals. (Bottom) Estimation of the mean of a Bernoulli distribution using 30 (L) and 1000 (R) samples. We observe that in the low sample regime, ‹ -Bets is mildly worse than the unbeatable randomized Clopper-Pearson, arguably better than standard Clopper-Pearson, and significantly better than Hedging of [23]. In the regime of larger samples, we can see that ‹ -Bets stays very close to the optimal intervals, while the competitor is still significantly worse.

<!-- image -->

Conclusions: We have introduced ‹ -technique in the construction of confidence intervals that directly (strictly) improves many betting algorithms and concentration inequalities, such as the ones of Hoeffding and Bernstein. Then, we have proposed a new betting algorithm for which he have proven that it can construct confidence intervals of the optimal length up to diminishing factor. While the ‹ -technique is powerful in experiments, we have only proven that it never hurts (certain class of algorithms) and that it usually helps. How much it helps remains an open theoretical question.

## References

- [1] A. Agresti and B. A. Coull. Approximate is better than 'exact' for interval estimation of binomial proportions. The American Statistician , 52(2):119-126, 1998.
- [2] J.-Y. Audibert, R. Munos, and C. Szepesvári. Variance estimates and exploration function in multi-armed bandit. In CERTIS Research Report 07-31 . 2007.
- [3] S. Bernstein. On a modification of Chebyshev's inequality and of the error formula of Laplace. Ann. Sci. Inst. Sav. Ukraine, Sect. Math , 1(4):38-49, 1924.
- [4] E. Clerico. On the optimality of coin-betting for mean estimation. arXiv preprint arXiv:2412.02640 , 2024.
- [5] C. J. Clopper and E. S. Pearson. The use of confidence or fiducial limits illustrated in the case of the binomial. Biometrika , 26(4):404-413, 1934.
- [6] T. M Cover. Universal gambling schemes and the complexity measures of Kolmogorov and Chaitin. Technical Report 12, Department of Statistics, Stanford University, 1974. URL https://purl.stanford.edu/js411qm9805 .
- [7] T. M. Cover and E. Ordentlich. Universal portfolios with side information. IEEE Transactions on Information Theory , 42(2):348-363, 1996.
- [8] A. Cutkosky and F. Orabona. Black-box reductions for parameter-free online learning in Banach spaces. In Proc. of the Conference on Learning Theory (COLT) , 2018.
- [9] H. Hendriks. Test martingales for bounded random variables. arXiv preprint arXiv:2109.08923 , 2018.
- [10] W. Hoeffding. Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association , 58(301):13-30, 1963.
- [11] K.-S. Jun and F. Orabona. Parameter-free online convex optimization with sub-exponential noise. In Proc. of the Conference on Learning Theory (COLT) , 2019.
- [12] A. Maurer and M. Pontil. Empirical Bernstein bounds and sample variance penalization. In Proc. of the Conference on Learning Theory , 2009.
- [13] F. Orabona and K.-S. Jun. Tight concentrations and confidence sequences from the regret of universal portfolio. IEEE Trans. Inf. Theory , 70(1):436-455, 2024.
- [14] F. Orabona and D. Pál. Coin betting and parameter-free online learning. In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing Systems 29 , pages 577-585. Curran Associates, Inc., 2016.
- [15] E. Ordentlich and T. M. Cover. The cost of achieving the best portfolio in hindsight. Mathematics of Operations Research , 23(4):960-982, 1998.
- [16] M. Phan, P. Thomas, and E. Learned-Miller. Towards practical mean bounds for small samples. In International Conference on Machine Learning , pages 8567-8576. PMLR, 2021.
- [17] A. Ramdas and R. Wang. Hypothesis testing with e-values. arXiv preprint arXiv:2410.23614 , 2024.
- [18] M. F. Schilling and J. A. Doi. A coverage probability approach to finding an optimal binomial confidence procedure. The American Statistician , 68(3):133-145, 2014.
- [19] G. Shafer and V. Vovk. Probability and finance: it's only a game! John Wiley &amp; Sons, 2001.
- [20] G. Shafer, A. Shen, N. Vereshchagin, and V. Vovk. Test martingales, Bayes factors and p -values. Statistical Science , 26(1):84-101, 2011.
- [21] S. Shekhar and A. Ramdas. On the near-optimality of betting confidence sets for bounded means. arXiv preprint arXiv:2310.01547 , 2023.

- [22] V. Voracek. Treatment of statistical estimation problems in randomized smoothing for adversarial robustness. Advances in Neural Information Processing Systems , 37:133464-133486, 2024.
- [23] I. Waudby-Smith and A. Ramdas. Estimating means of bounded random variables by betting. Journal of the Royal Statistical Society Series B: Statistical Methodology , 86(1):1-27, 2024.

## A Proof of Theorem 8

Before we prove the theorem, we introduce several propositions we use in the sequel. We start with a (time uniform) version of Bennett's inequality.

Proposition 10 (part of Lemma 1 of [2]) . Let X 1 , . . . , X n ď b be i.i.d. real-valued random variables and let b 1 ' b ´ E r X s . For any δ P p 0 , 1 q , simultaneously for all 1 ď t ď n , we have

<!-- formula-not-decoded -->

More generally, let X 1 , . . . X n ă b be a sequence of martingale differences (i.e., E r X i | X 1 , . . . , X i ´ 1 s ' 0 for all i ). Let nV ' ř n i ' 1 V r X i | X 1 , . . . , X i ´ 1 s . For any δ P p 0 , 1 q , simultaneously for all 1 ď t ď n , we have

<!-- formula-not-decoded -->

Remark 11 . The first part of the proposition is implied by the second one and is the standard Freedman's inequality up to a factor of 2 in the last term. This factor was removed in [2], but the result was stated for an i.i.d. sequence (the first part of the lemma) and not for martingale differences. However, the proof of [2] applies also to the second part.

Now, we use it to bound the deviation of our second moment estimator from the mean for all m and t simultaneously with high probability.

Proposition 12. Let X 1 , . . . , X n P r 0 , 1 s be i.i.d. random variables with variance V r x s ' σ 2 . For any α P p 0 , 1 q , simultaneously for all 1 ď t ď n and all m P r 0 , 1 s we have all the following inequalities with probability at least 1 ´ α :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We use Proposition 10 on random variables X, ´ X,X 2 , ´ X 2 and union bound, using the fact that V r X 2 s ď V r X s as X P r 0 , 1 s . This yields the latter two identities. Then, as X 2 ´ 2 mX ` m 2 ' p X ´ m q 2 , we have accumulated 1 ` 2 m ď 3 of the identical error terms, yielding

<!-- formula-not-decoded -->

from which we express the stated bounds, using E rp X ´ m q 2 s ě σ 2 and log 4 α { t ď n log 4 α { t 2 . The lower bound then follows from completing the square.

Further, we use the uniform version of Bennett's inequality to bound the sum of the observations in the first t rounds; again, for all m simultaneously.

Proposition 13. Let X 1 , . . . , X n P r 0 , 1 s be i.i.d. random variables with mean µ ' E r X s and variance σ 2 ' V r X s , and let 0 ď λ i ď C be random variables such that λ i is independent of X j for j ě i for some positive constant C . With probability at least 1 ´ α we have for all m P r 0 , µ s and all 1 ď t ď n simultaneously that

<!-- formula-not-decoded -->

Proof. We decompose the random variables λ i p X i ´ m q into a martingale difference sequence M i ' λ i p X ´ µ q and a drift term D i ' λ i p µ ´ m q . We bound deterministically the drift term:

<!-- formula-not-decoded -->

and so it holds for all m and t simultaneously. We bound the martingale using the Freedman's-style inequality from Proposition 10:

<!-- formula-not-decoded -->

Adding up the two bounds concludes the proof.

Finally, we introduce a quadratic lower-bound of a logarithm near zero.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We inspect the behavior of f p x q ' log p 1 ` x q ´ x ` cx 2 . First, f p 0 q ' 0 . Now we look at the derivatives f 1 p x q ' 1 {p x ` 1 q ´ 1 ` 2 cx and f 2 p x q ' ´ 1 {p x ` 1 q 2 ` 2 c . Setting f 1 to zero, we get x p 1 ´ 2 c ´ 2 cx q ' 0 , so the roots are x 1 ' 0 and x 2 ' 1 {p 2 c q ´ 1 , these points are local extremes, concretely a local minimum and local maximum respectively by the second derivative test, and so the inequality holds for x ě 1 {p 2 c q ´ 1 .

Now we are ready to restate the main theorem we are to prove.

Theorem 15 (Theorem 8 restated) . For every random variable X P r 0 , 1 s with mean µ and variance σ 2 ą 0 , every α, δ P p 0 , 1 q , and c ą 0 , there is n 0 depending on α, δ, σ 2 , c , such that for all n ě n 0 , Algorithm 3 rejects at confidence level 1 ´ δ every m satisfying

<!-- formula-not-decoded -->

with probability at least 1 ´ α .

Proof. Statement (9) holds true if the events from Proposition 12 and Proposition 13 (applied on X 1 , . . . , X c 2 n , to be specified later) are met. Namely, Proposition 12 ensures that our estimator of E rp X ´ m q 2 s is never too small and is consistent. It also ensures that empirical mean of X converges to µ . Proposition 13 provides a bound on the wealth in an early stage. We note that both of the propositions hold uniformly for all m P r 0 , µ s . Thus, we instantiate both of the bounds (with failure probabilities α { 2 ) and further assume that the events are met. Throughout the sketch, we will be

introducing new constants on the fly. For all constants, we have that we can choose a large enough n 0 to make them arbitrarily close to 0.

We fix m P r 0 , µ s whose exact expression will be decided at the end of the proof. Let Y ' X ´ m and ε ' µ ´ m . Let ℓ opt ' b 2 log 1 δ n p σ 2 ` ε 2 q . By Lemma 14, for any constant 1 { 2 ě c 1 ą 0 , we have that log p 1 ` ℓY q ě ℓY ´p 1 { 2 ` c 1 q ℓ 2 Y 2 for all ℓ ď ? 2 ℓ opt .

Analysis of the run of algorithm: We split the n steps into an arbitrarily (relatively) short 'warm up' phase of c 2 n steps, where things can go poorly, but the effect of this will be negligible. Then, there will be a 'convergent phase' of p 1 ´ c 2 q n steps, in which p 1 ´ c 3 q ℓ opt ď ℓ i ď p 1 ` c 3 q ℓ opt . We briefly comment on the constants. By Lemma 14 it holds that c 1 ď 3 ℓ opt ' O p 1 { ? n q as long as ℓ opt ă 0 . 1 . c 2 is arbitrary, so we can set it to O p n ´ 1 4 q . By Proposition 12, we have c 3 ď 1 c 2 ´b 18 log 8 α {p σ 2 n q ` 11 log 8 α {p σ 2 n q ¯ , and so given the choice of c 2 , we have c 3 P O p n ´ 1 { 4 q .

Warm up phase: In this phase, we have 0 ď ℓ i ď ? 2 ℓ opt per the second part of Proposition 12 and the fact that E rp X ´ m q 2 s ' σ 2 ` ε 2 .

First, deterministically upper bound the quadratic term:

<!-- formula-not-decoded -->

Next, by Proposition 13 applied on X 1 , . . . , X c 2 n and C ' ? 2 ℓ opt, we have

<!-- formula-not-decoded -->

where we see that both the quadratic and linear term go to zero as n increases and c 2 decreases.

Convergent phase: By Proposition 12, we have p 1 ´ c 3 q ℓ opt ď ℓ i ď p 1 ` c 3 q ℓ opt . From the definition of ℓ i we have that ℓ 2 n ř n i ' 1 Y 2 i ď 2 log 1 δ . Thus,

<!-- formula-not-decoded -->

Now, we add up the lower bounds from the warm-up phase and from the quadratic term in the convergent case:

<!-- formula-not-decoded -->

for some c 4 ą 0 . Given the choices of constants above, we can have c 4 , c 5 , c 6 P O p n ´ 1 { 4 q for the constants to come. Finally, we lower bound the log-wealth:

<!-- formula-not-decoded -->

which is greater than log 1 δ if

<!-- formula-not-decoded -->

From Proposition 13, we have assumed the events

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

We now reveal our choice of m :

<!-- formula-not-decoded -->

We can now show the value of m we have selected satisfies (10) and so it will result in a log-wealth bigger than 1 δ . Observe that

<!-- formula-not-decoded -->

Hence, we have

<!-- formula-not-decoded -->

Moreover, observe that these last inequalities apply for all m 1 ď m .

Thus, for the chosen m , we have reached log-wealth log 1 δ and so it is rejected. Finally, we have assumed the event ˇ ˇ 1 n ř n i ' 1 X i ´ µ ˇ ˇ À 1 { ? n from (8). So, recalling that ε ' µ ´ m , for our choice of m we also have ε « 1 { ? n . Hence, we can make p c 5 ` ? 2 qp 1 ` ε { σ q arbitrarily close to ? 2 and c 2 , c 6 arbitrarily close to 0, finishing the proof of the main part. Given our tracking of constants, we can see that the final leading term is off by a factor 1 ` O p n ´ 1 { 4 q , concluding the second statement.

## B Bernstein betting

We derive the algorithm for Bernstein testing referenced in Figure 1 and show that ‹ -ing it is a strict improvement. The steps are analogical to the Hoeffding's betting derivation. We note that the provided version of Bernstein's inequality is mildly weaker than the standard one in the interest of simplicity.

Lemma 16 (Bernstein simplified) . Let X P r 0 , 1 s with mean µ ' E r X s and variance σ 2 ' V r X s , then

<!-- formula-not-decoded -->

Furthermore, the sequence W 0 ' 1 , W i ` 1 ' W i exp ` ℓ ¨ p X i ´ µ q ´ σ 2 ℓ 2 ˘ for i ě 0 is a test process.

Proposition 17. Let m P r 0 , 1 s . Whenever Algorithm 5 rejects the null hypothesis H 0 p m q : m ' E r X s , then so does Algorithm 6 if they share the realizations X t , 1 ď t ď n .

Proof. Consider Algorithm 5. We first show that ℓ B is selected in a way that ř n i ' 1 X i required to reject the null hypothesis is smallest possible. That is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, consider Algorithm 6. By the same argument, ℓ B ‹ at time t is minimizing ř n i ' t X i under the constraint that the null hypothesis is rejected. Consequently, the required ř n i ' 1 X i for Algorithm 6 is initially the same as for Algorithm 5, but it decreases whenever ℓ B ‹ changes with respect to the previous iteration.

## Algorithm 5 Bernstein testing

<!-- formula-not-decoded -->

## C Experiments

We provide CDF plots as we believe they contain most information about the behavior of the confidence interval. We repeat the description from the main paper.

In short, the more the curve is to the right, the better it is.

Every algorithm provides a lower bound on the mean parameter of the corresponding distribution. We repeat the experiment 1000 times and for every algorithm plot the empirical CDF of the produced lower bounds. I.e., a curve passing through point p x, y q should be understood as y ´ fractions of the lower bounds are smaller than x . We include a vertical and a horizontal magenta line representing the mean (vertical) and 1 ´ δ (horizontal) as the desired coverage. The eCDF of the algorithm passes the vertical line at point p µ, δ 1 q , where µ is the true mean and 1 ´ δ 1 is the empirical coverage. We have zoomed in to a box centered at p µ, 1 ´ δ q to see what is the coverage; also, we have added black vertical lines corresponding to 0 . 95 ´ one-sided confidence intervals. If the eCDF meets the vertical magenta line above (resp. below) the black line, it has coverage smaller (resp. bigger) than 1 ´ δ at confidence level 0.95. All algorithms apart from T-test have guaranteed coverage at least 1 ´ δ , so if they occur under the black line, it is by a chance.

## C.1 Experimental results

Here, we provide general experimental results. We always show STaR bet from Algorithm 4 with details from D. Hedged-CI is from [23] with the default settings. T-test and (randomized) ClopperPearson intervals are standard.

Bernoulli: Here, we can see that ‹ is performing very closely to randomized Clopper-Pearson and outperforming standard Clopper-Pearson on small sample sizes. On the larder sample sizes, the performance of all three algorithms become very similar, significantly outperforming Hedge-CI.

Beta: Here, we compete with T-test based confidence interval with no formal guarantees. We can see that as long as a ą b ( a, b are parameters of the beta distributions), T-test is outperforming ‹ ; however, it clearly has larger coverage than 1 ´ δ , violating the principles of confidence intervals. When b ą a , ‹ usually provides shorter intervals than T-test. Hedged-CI provides much larger ones. With increasing n , the performance of T-test and of ‹ become essentially identical.

Coverage: In the majority of cases, the coverage of ‹ is statistically indistinguishable from 1 ´ δ . Coverage of Hedged-CI is usually 1 .

<!-- formula-not-decoded -->

## Bernoulli, delta = 0.05, averaged over 1000 runs

<!-- image -->

## Bernoulli, delta = 0.05, averaged over 1000 runs

<!-- image -->

Beta,  n = 10,  delta = 0.05, averaged over 1000 runs

<!-- image -->

Beta,  n = 50,  delta = 0.05, averaged over 1000 runs

<!-- image -->

Beta,  n = 500,  delta = 0.05, averaged over 1000 runs

<!-- image -->

## C.2 Influence of δ

The majority of the experiments are made with δ ' 0 . 05 ; here, we present the subset of the experiments with different values of δ . The results follow the patterns observed with δ ' 0 . 05 .

## Bernoulli, delta = 0.001, averaged over 1000 runs

<!-- image -->

## Bernoulli, delta = 0.01, averaged over 1000 runs

<!-- image -->

Beta,  n = 100,  delta = 0.001, averaged over 1000 runs

<!-- image -->

Beta,  n = 100,  delta = 0.001, averaged over 1000 runs

<!-- image -->

## C.3 Selection of c

As discussed in Appendix D.3, we estimate the expectation E rp X ´ m q 2 s using the empirical mean with an additive cmn {p t ´ 1 q 2 term, where t is the current round, n is the number of rounds and c is a free parameter. We provide experiments suggesting that the algorithm is not so sensitive about the choice, especially as n increases, and we choose c ' 1 as a natural choice to not overfit on our benchmark distributions.

In the following experiments, we used Algorithm 4 with the details in Appendix D sweeping over exponentially spaced grid of values of c .

<!-- image -->

<!-- image -->

<!-- image -->

## C.4 Last round bet

Here we provide experiments quantifying the effect of (an additional) last round betting described in Appendix D.2.

- STaR-Bets is Algorithm 4 with detail from D.
- Bets is STaR-Bets without the ‹ -component.
- (STaR) bets w/o last bet are the first two algorithms without the last round bet (described in Appendix D.2).
- Hedged-CI is the confidence interval of [23].

We can see that Hedged-CI performs similarly to Bets without last bet, and STaR bets without the last bet is significantly stronger. Adding last bet help both algorithms, but the effect is less significant on STaR, since by design, it tries to end up with 0 or 1 δ money. Sometimes the performance of Hedged-CI and Bets without last bet (and also of the pair STaR with/without last bet) are so similar, that the curves are indistinguishable.

<!-- image -->

<!-- image -->

<!-- image -->

## C.5 Variance of the confidence interval with data shuffling

While the majority of confidence intervals are independent of the data order, it is not the case for the betting based one. Here, we present several experiments where for every setting, the corresponding sample of random variables was obtained and then it was only shuffled for all the 1000 experiments. Note that here the coverage is meaningless, as we do not draw fresh samples.

## Bernoulli, delta = 0.05, averaged over 1000 runs

<!-- image -->

Beta,  n = 50,  delta = 0.05, averaged over 1000 runs

<!-- image -->

## D Implementation details

There are certain aspects where the algorithm we implemented deviates from the vanilla version in Algorithm 3, but the underlying process is still a Test process, so the resulting confidence intervals are valid. Here, we describe these details.

## D.1 Clipping

Instead of clipping the estimate of E rp X ´ m q 2 s to its upper bound max t m 2 , p 1 ´ m q 2 u , we clip it to m p 1 ´ m q instead. It would be the maximal value of E rp X ´ m q 2 q if m ' E r X s . The testing problem is hard when m « E r X s (and easy otherwise), so this way, we help the algorithm on the hard values of m and hurt it on the easy ones, yielding better empirical performance.

## D.2 Last round randomization

Just as the conservativeness of Clopper-Pearson is fixed by randomization, we attempt similar thing in the betting games. 5 After finishing the testing procedure, we end up with wealth W. Then, we draw a uniform random variable U supported in r 0 , 1 s ; if W ě U { δ , we set it to 1 δ . Otherwise, we set it to 0 . It is easy to see that after this manipulation, W is still a test-variable, since it is still non-negative and its expectation did not increase. The effect of this is usually small for ‹ -Bets, since it is encouraged by design to end up with 1 { δ or no wealth. The exceptions are instances with small n , small variance, or instances with E r X s ' 1 , in which cases the bets has to be very small, and so it is not easy to properly adapt the betting strategy and ensure that we bet aggressively enough if needed. See Appendix C.4 for experiments.

## D.3 Choice of second moment estimator

Algorithms 3, 4 have a hyperparameter α corresponding to the lower bound of probability of having short intervals in the proofs. It influences how conservative we should be in estimating E rp X ´ m q 2 s . We have (empirically) observed that the larger m is, the more conservative we should be. The intuition is that our win (or loss) ℓ p X ´ m q can be as low as ´ ℓm , and so with larger m , we should be more careful about the choice of ℓ . We instantiate the 10 log 8 α n {p t ´ 1 q 2 term as cmn {p t ´ 1 q 2 with c ' 1 . In Appendix C.3 we provide experiments suggesting that the exact choice of c is not that crucial, but a proper argument about how should c depend on m is not given.

5 The resulting random variable is known as all-or-nothing random variable, see [17].

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

Justification: Yes, we do provide an algorithm and prove some proeprties.

## Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [NA]

Justification: Well, we wanted to provide a confidence interval for bounded random variables and we did so.

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

Justification: Assumptions are present, the proofs are ready and will be submited with supplement.

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

Justification: it will be in suplement.

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

Justification: yes, is suplement.

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

Justification: no training in the paper.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: yes, we derive confidence intervals and we are disgusted that you explicitly say that 1-sigma error is 96% CI.

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

Justification: no resource intensive stuff

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: we are ethical people.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: come on, we derive confidence intervals.

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

Justification: no, we would like people ot use our confidence intervals.

Guidelines:

- The answer NA means that the paper poses no such risks.

- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: no assets.

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

Justification: no

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: no crows

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: we really do just confidence intervals.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: nothing to discuss.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.