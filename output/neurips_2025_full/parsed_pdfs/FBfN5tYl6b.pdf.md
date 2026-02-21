## Optimal Estimation of the Best Mean in Multi-Armed Bandits

Takayuki Osogami IBM Research - Tokyo osogami@jp.ibm.com

## Junya Honda

Kyoto University, RIKEN AIP honda@i.kyoto-u.ac.jp

## Junpei Komiyama

New York University, RIKEN AIP junpei@komiyama.info

## Abstract

We study the problem of estimating the mean reward of the best arm in a multiarmed bandit (MAB) setting. Specifically, given a target precision ε and confidence level 1 -δ , the goal is to return an ε -accurate estimate of the largest mean reward with probability at least 1 -δ , while minimizing the number of samples. We first establish an instance-dependent lower bound on the sample complexity, which requires handling the infinitely many possible candidates of the estimated best mean. This lower bound is expressed in a non-convex optimization problem, which becomes the main difficulty of this problem, preventing the direct application of standard techniques such as Track-and-Stop to provably achieve optimality. To overcome this difficulty, we introduce several new algorithmic and analytical techniques and propose an algorithm that achieves the asymptotic lower bound with matching constants in the leading term. Our method combines a confidence ellipsoid-based stopping condition with a two-phase sampling strategy tailored to manage non-convexity proposed algorithm is simple, nearly free of hyperparameters, and achieves the instance-dependent, asymptotically optimal sample complexity. Experimental results support our theoretical guarantees and demonstrate the practical effectiveness of our method.

## 1 Introduction

AI agents and foundation models are developed to be adapted for, or applied to, a wide range of tasks, users, and deployment contexts. Given this diversity of applications, it is essential to ensure that these systems perform reliably even under challenging or unfavorable conditions. For example, a foundation model should provide a minimal level of utility across various downstream tasks, and an AI agent should consistently satisfy all required safety criteria [14]. In other words, before deployment, we must ensure that the expected performance is acceptable even in worst-case scenarios.

This problem of evaluating expected performance in the worst-case scenario can be naturally formulated within the multi-armed bandit (MAB) framework as the task of estimating the mean of the best arm (i.e., best-mean estimation). This paper adopts the standard MAB convention in which rewards are to be maximized and focuses on designing an algorithm that accurately estimates the best mean.

Our primary focus in algorithm design is minimizing sample complexity, motivated by the substantial cost of evaluating AI agents and foundation models. This concern is particularly relevant given the growing trend toward inference-time reasoning [16, 7, 23], which often demands substantial test-time computation. An AI agent may rely on Best-of-N sampling with millions or billions of candidates [16, 7] or Monte Carlo tree search to select its next action [24], making each evaluation costly.

More specifically, we develop a probably approximately correct (PAC) algorithm for best-mean estimation, achieving asymptotically optimal sample complexity. The algorithm guarantees an estimation error of at most ε with probability at least 1 -δ , and the expected number of samples it uses matches the theoretical lower bound that we establish in this paper. This ensures that, as δ → 0 , the sample complexity achieved by our algorithm is asymptotically optimal.

Our lower bound proof leverages the approach of [2], which addresses the problem of selecting an arm when multiple (a finite number of) correct arms may exist. However, our setting differs in a crucial way: we must select a real-valued estimate of the best mean from infinitely many possibilities . This fundamental distinction requires modifications to the technique in [2]. We show that, for Gaussian rewards, any algorithm must use at least 2 R 2 f ( µ ) log(1 /δ ) -o (log(1 /δ )) samples in expectation, where f ( µ ) is the optimal value of an optimization problem that characterizes the optimal allocation of samples among arms. While f ( µ ) = O ( K/ε 2 ) in the worst case, it is instance-dependent and can be significantly smaller. The key challenge, therefore, is to design an algorithm that adapts to the structure of each instance and achieves the corresponding sample complexity.

Our algorithm builds on the martingale-based anytime confidence bound introduced by [1], a technique that has been widely adopted in the linear bandit literature. We adapt this technique in a novel way to the MAB setting, using it to jointly estimate the expected rewards of arms as a confidence ellipsoid. A key insight is that this ellipsoidal representation enables efficient testing of global hypotheses-such as whether at least one arm has an expected reward above a given threshold µ ′ -which would be difficult to verify using conventional, per-arm confidence intervals. Moreover, to achieve asymptotically optimal sample complexity, we conduct a detailed analysis showing that the original ( K +1) -dimensional non-convex characterizing optimization problem can be reduced to a one-dimensional non-convex optimization over a narrow interval. This enables efficient grid search within the narrow interval to optimize the allocation of samples used by our algorithm.

The primary contribution of this paper is a novel PAC algorithm for best-mean estimation, with sample complexity that is asymptotically optimal. A key challenge in our analysis lies in handling the non-convexity of the characterizing optimization problem, which makes it difficult to ensure that the sample allocation closely approximates the optimal one. To validate our theoretical analysis and highlight the practical relevance of our approach, we also present numerical experiments. These results illustrate not only the empirical advantages of the algorithm but also certain limitations that are not captured by the asymptotic theory.

## 2 Related work

Our problem can be viewed as an instance of pure exploration problems, which requires identifying a quantity that depends on unknown parameters. The most relevant literature in this context is fixed-confidence ε -best arm identification [3, 10, 4, 11, 8], where the goal is to identify one of the ε -best arms with confidence 1 -δ . However, our problem and ε -best arm identification are different. Identifying an ε -best arm does not guarantee that the best mean is estimated within ε error, since all estimated means can have greater than ε errors. Conversely, our problem of fixed-confidence ε -best-mean estimation does not guarantee that an ε -best arm is identified. However, our algorithm can be used for 2 ε -best arm identification, since it finds U such that all means are below U as well as an arm with mean at least U -2 ε . We empirically compare our algorithm with a representative ε -best arm identification algorithm, UGapEc [4], adapted for best-mean estimation in Section 7.

The utility of the Track-and-Stop algorithm [5] for pure exploration problems is shown in [2]. While the idea of Track-and-Stop could potentially be applied to best-mean estimation, the non-convexity of the characterizing optimization problem prohibits us from establishing the continuity of its optimal solution with respect to the true means, which is utilized by Track-and-Stop. On the other hand, our two-phase algorithm leverages the continuity of the optimal value to guarantee its asymptotic optimality. Deriving the optimal sample complexity of gradient-based methods, such as the one in [21], is challenging due to the non-convex nature of the underlying optimization problem, which can result in convergence to suboptimal solutions.

Although non-convexity also appears in the characterizing optimization problems of classical best arm identification, those optimization problems typically allow convex reformulation (e.g., [12, 21]). Also, Russo and Pacchiano deal with the non-convexity by convex relaxation, which is shown to lose optimality at most by a factor of 4 [15]. On the other hand, we solve our nonconvex optimization

problem by decomposing it into convex subproblems and reducing it to a one-dimensional optimization problem within a narrow interval specified later, which we can solve via grid search with an arbitrary accuracy.

Best-mean estimation has been studied by [13], where it arises as a subproblem in mechanism design-specifically, in ensuring that desired properties hold across all agent types. [13] proposes a simple two-step approach: first identifying an (2 / 3) ε -best arm with confidence level 1 -δ/ 2 , then collecting enough additional samples from it to meet the required guarantee. This method achieves a sample complexity of O (( K/ε 2 ) log(1 /δ )) , and a matching lower bound of Ω(( K/ε 2 ) log(1 /δ )) is established. In contrast, our results provide sharp, instance-dependent upper and lower bounds, with matching constants in the leading term, establishing that, for each instance, our algorithm has asymptotically optimal sample complexity.

The problem of best-mean estimation also arises frequently in both machine learning and reinforcement learning. In machine learning, it corresponds to estimating the expected performance of the optimal predictor [9], while in reinforcement learning, it corresponds to estimating the value function [19]-that is, the maximum expected return from a state, achieved by selecting the optimal action at each state. While prior work in these areas focuses on how to best estimate the best mean given a fixed set of samples [20], our work addresses a complementary question: how to efficiently sample to estimate the best mean with high confidence.

## 3 Settings

We consider a multi-armed bandit (MAB) setting with K arms, where each arm i ∈ [ K ] := { 1 , 2 , . . . , K } is associated with a reward distribution having mean µ i . We assume that the reward distributions are R -sub-Gaussian (i.e., the reward X i from each arm i satisfies Pr( | X i | ≥ x ) ≤ 2 exp( -x 2 /R 2 ) for a known constant R ), and that the means are bounded in magnitude by a known constant S (i.e., max i ∈ [ K ] | µ i | ≤ S ). Let µ ⋆ := max i ∈ [ K ] µ i denote the mean of the best arm (best mean) and i ⋆ := argmax i µ i be the index of a best arm. In the case of multiple optimal arms, we break ties arbitrarily and fix one such arm as i ⋆ . We denote the full mean vector by µ := ( µ 1 , µ 2 , . . . , µ K ) and define the suboptimality gap for arm i as ∆ i := µ ⋆ -µ i .

Our objective is to design an algorithm that adaptively samples from the K distributions and stops after a random number of samples, denoted by τ . Upon termination, the algorithm outputs an estimate ˆ µ ⋆ of the best mean such that | µ ⋆ -ˆ µ ⋆ | ≤ ε holds with probability at least 1 -δ , for given parameters ( ε, δ ) . In Section 7.3, we discuss an equivalent formulation of returning an interval of length 2 ε that contains µ ⋆ with probability at least 1 -δ . The sample complexity of the algorithm is defined as E [ τ ] , where the expectation is taken with respect to the underlying reward distributions. Throughout, we use the following notations: ( x ) + := max { 0 , x } , x ∨ y = max { x, y } , and x ∧ y = min { x, y } .

## 4 Sample Complexity Lower Bound

We start by deriving the following lower bound on the sample complexity E [ τ ] of our best-mean estimation (BME) problem:

Theorem 1. Consider any algorithm that returns an ε -accurate estimate of the best mean with probability at least 1 -δ . Then, when rewards follow Gaussian distribution with variance R 2 , the sample complexity is lower bounded as follows:

<!-- formula-not-decoded -->

where f ( µ ) is the optimal objective value of the following optimization problem:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Figure 1: Non-uniqueness of the optimal solution for (P1).

<!-- image -->

Here, we outline the full proof of the theorem provided in Appendix A.1.

In the proof, we introduce a multiple correct answer problem (MCP) that can be reduced to BME in the sense that an algorithm for BME can also solve this MCB. We then obtain a lower bound for BME from a lower bound for MCP, which is established in [2] (see Lemma 7 in Appendix A.1). We primarily consider a lower bound for Gaussian rewards, similar to [8], but an analogous, albeit less explicit, lower bound may be obtained for more general distributions from that of the M -bin MCP.

̸

Specifically, consider the following M -bin MCP problem with parameter M ∈ N , which defines ε ′ := ε/M . The problem asks to return a bin from a finite set, B := { ( -∞ , µ ⋆ -ε ) , [ µ ⋆ -ε, µ ⋆ -ε + ε ′ ) , [ µ ⋆ -ε + ε ′ , µ ⋆ -ε +2 ε ′ ) , . . . , [ µ ⋆ + ε, µ ⋆ + ε + ε ′ ) , [ µ ⋆ + ε + ε ′ , ∞ ) } , where the answer is considered correct if it is in b ⋆ ( µ ) := { b ∈ B : b ∩ [ µ ⋆ -ε, µ ⋆ + ε ] = ∅} . Notice that any δ -correct algorithm for BME can be converted into a δ -correct algorithm for the M -bin MCP by simply mapping the output of the BME algorithm to the bin that contains the output.

In Lemma 8 (Appendix A.1), we show that, for any η &gt; 0 , there exists a large M such that the sample complexity of the M -bin MCP is bounded from below by 2(1 -η ) R 2 f ( µ ) log(1 /δ ) -o (log(1 /δ )) , where f ( µ ) is the optimal objective value of (P1). Letting η → 0 , we establish that any δ -correct algorithm for BME-one that returns an ε -accurate estimate of µ ⋆ with probability at least 1 -δ for any µ -must incur a sample complexity of at least 2 R 2 f ( µ ) log(1 /δ ) -o (log(1 /δ )) .

The optimization problem (P1) plays a central role in our analysis: besides it provides a lower bound on sample complexity, it characterizes the minimal number of samples required from each arm i ∈ [ K ] to estimate the best mean to match this bound asymptotically. Specifically, 2 R 2 r i log(1 /δ ) represents the number of samples that should be taken from arm i ∈ [ K ] .

Note that Eq. (2) is a non-convex optimization over ( r , U ) . An important observation is that, when we fix the value U ∈ ( µ ⋆ , µ ⋆ +2 ε ) , the corresponding subproblem, given by

<!-- formula-not-decoded -->

is a convex optimization over r . We denote the optimal solution and value of (P2) by { r i ( U ; µ ) } i ∈ [ K ] and f ( U ; µ ) , respectively. Although f ( U ; µ ) depends on ε , we regard ε as a fixed constant and often omit its dependence throughout the paper for brevity. The optimal solution to (P1) is then given by U ( µ ) = argmin U ∈ ( µ ⋆ ,µ ⋆ +2 ε ) f ( U ; µ ) and r i ( µ ) = r i ( U ( µ ); µ ) , and we define the corresponding optimal value as f ( µ ) = min U ∈ ( µ ⋆ ,µ ⋆ +2 ε ) f ( U ; µ ) .

Figure 1 illustrates a numerical example highlighting the nonconvex nature of the optimization problem (P1). Specifically, it shows the objective value f ( U ; µ ) of (P2) as a function of U , revealing the existence of multiple disconnected local minima, which leads to discontinuities in the optimal allocation and complicates the analysis compared to settings where convex reformulations are possible. See Appendix B.8 for further details.

Since (P1) is nonconvex optimization, giving upper and lower bounds of the optimal solution U ( µ ) is beneficial for implementation and analysis. To this end, we now introduce additional notations. Let

U ( µ ) and U ( µ ) be the solutions to g ( U ; µ ) = 0 and g ( U ; µ ) = 0 , respectively, where we define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

for any U ∈ ( µ ⋆ , µ ⋆ +2 ε ) . As we will show below, U ( µ ) and U ( µ ) can be used to define a narrow interval in which U ( µ ) is contained. Here, g and g are monotonically increasing in U , and their roots can be efficiently computed using standard methods such as bisection over the interval [ µ ⋆ , µ ⋆ +2 ε ] . For convenience, we extend their domains by setting

<!-- formula-not-decoded -->

The solution to the optimization problem (P2) can then be characterized as follows:

Lemma 2. The optimal solution and objective value of (P2) satisfy the following properties:

<!-- formula-not-decoded -->

- (ii) When U ≥ U ( µ ) , there is an optimal solution

<!-- formula-not-decoded -->

̸

which satisfies r i ⋆ ( U ; µ ) ≥ 1 ( U -µ ⋆ ) 2 . In addition, the optimal value satisfies

̸

<!-- formula-not-decoded -->

̸

- (iii) The minimizer of f ( U ; µ ) satisfies U ( µ ) ∈ [ U ( µ ) , U ( µ ) ∨ U ( µ )] .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof of the lemma is postponed to Appendix A.2. Although the function (9) is generally nonconvex, we can still confine its minimizer U ( µ ) = argmin U f ( U ; µ ) within a reasonably narrow interval using the bounds, U ( µ ) and U ( µ ) ∨ U ( µ ) . This allows for efficient approximation of the minimizer via grid search over the interval [ U ( µ ) , U ( µ ) ∨ U ( µ )] ⊂ [ µ ⋆ + ε, µ ⋆ +2 ε ) . When U ( µ ) ≥ U ( µ ) , the minimizer is U ( µ ) , and there is no need for solving the non-convex optimization.

## 5 Ellipsoid-based Estimation Algorithm

Motivated by the structure of the characterizing optimization, we propose the Ellipsoid-based Estimation algorithm (EllipsoidEst) shown in Algorithm 1. In addition to the problem parameters ( K,R,S,ε,δ ) introduced in Section 3, EllipsoidEst uses a regularization parameter λ .

EllipsoidEst is characterized by its novel stopping condition based on a tight confidence ellipsoid formed by regularized estimators. It also employs a two-phase sampling strategy to minimize the need for solving non-convex optimization problems. These components are critical to derive an optimal and computationally efficient algorithm for a probably approximately correct (PAC) estimate of the best mean. In what follows, we describe each of these components in detail.

EllipsoidEst computes upper confidence bounds (UCBs; ˆ µ i ( t ) + √ β ( t, δ ) /N i ( t ) ) in Step 9, where β ( t, δ ) is defined as

<!-- formula-not-decoded -->

̸

## Algorithm 1 EllipsoidEst

Require: K,R,S,ε,δ : problem dependent parameters; λ : regularization parameter

- 1: N i (0) ← 0 , ∀ i ∈ [ K ]
- 2: ˆ µ i ( t ) ← 0 , ∀ i ∈ [ K ]
- 3: Phase ← 1
- 4: for t = 1 , 2 , . . . do

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 7: Pull arm i ( t ) and obtain reward x i ( t ) ,N i ( t ) ( t )
- 8: Update ˆ µ i ( t ) ( t ) ← 1 N i ( t ) ( t )+ λ ∑ N i ( t ) ( t ) m =1 x i ( t ) ,m

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 11: Output ˆ µ ⋆ ← U ( t ) -ε and terminate.
- 12: if Phase = 1 and g ( U ( t ); ˆ µ ( t )) ∨ g ( U ( t ); ˆ µ ( t )) ≤ 0 then

13:

Phase

←

2

- 14: ˆ i ⋆ ← argmax ˆ µ ( t )
- 15: ˆ U ← U ( ˆ µ ( t ))

16:

- i i

r

ˆ

i

⋆

←

r

ˆ

i

⋆

( ˆ

U

and

; ˆ

µ

(

t

))

▷ Determine the target confidence interval.

Minimize (9) over

▷

U

∈

[

U

( ˆ

µ

(

t

))

, U

( ˆ

µ

(

t

))

∨

U

( ˆ

µ

(

t

))]

.

<!-- formula-not-decoded -->

denotes a regularized estimate of the mean reward based on the N i ( t ) samples { x i, 1 , . . . , x i,N i ( t ) } collected up to round t .

A natural, yet naive, stopping condition would be to terminate when U ( t ) -L i ( t ) ≤ 2 ε holds for some arm i , where U ( t ) is the highest UCB and L i ( t ) := ˆ µ k ( t ) -√ β ( t, δ ) /N i ( t ) is the lower confidence bound (LCB) for i in round t . This condition guarantees that U ( t ) -ε ∈ [ µ ⋆ -ε, µ ⋆ + ε ] when the true means are simultaneously contained in the respective confidence bounds.

When multiple arms have estimated means ˆ µ i ( t ) close to the highest UCB U ( t ) , we can improve upon the naive stopping rule. Namely, EllipsoidEst employs a more refined stopping condition, as specified in Step 10. Figure 2 provides an intuitive illustration of this condition in the case of two arms. It can be shown that, with probability at least 1 -δ , the true mean vector lies within the confidence ellipsoid defined by the individual upper and lower confidence bounds. Therefore, if the region { ( µ 1 , µ 2 ) : µ 1 ∨ µ 2 ≤ U ( t ) -2 ε } lies outside this ellipsoid, we can conclude that the true best-mean must lie within the interval [ U ( t ) -2 ε, U ( t )] , even if the confidence interval for each individual arm is not contained in [ U ( t ) -2 ε, U ( t )] .

Formally, the stopping condition of EllipsoidEst is supported by the following lemma, which we prove in Appendix A.3 using the martingale-based anytime confidence bound from [1]:

Lemma 3. Let S be such that | µ i | ≤ S, ∀ i ∈ [ K ] . Then, with probability at least 1 -δ , the regularized estimator { ˆ µ i ( t ) } satisfies for all t ≥ 2 that

<!-- formula-not-decoded -->

Moreover, when t ≥ 2 and λ ≥ 2 /K , the following simpler bound holds:

<!-- formula-not-decoded -->

▷ There are two phases.

<!-- formula-not-decoded -->

otherwise.

Figure 2: Confidence ellipsoid for two arms: µ is in the ellipsoid with probability at least 1 -δ .

<!-- image -->

Remark 1. Instead, one may construct a confidence bound based on a deviation inequality for exponential families explored in [12]. Since such a bound can be constructed with β ( t, δ ) = O (log log( N i ( t ))) instead of β ( t, δ ) = O (log( N i ( t ))) as in (10) , it can lead to a tighter bound for large N i ( t ) 's. This deviation inequality has an additional advantage that it does not require the knowledge of the constant S (such that max i ∈ [ K ] | µ i | ≤ S ). However, we confirm in Appendix B.9 that the bound based on this deviation inequality can have improvement only for quite large N i ( t ) .

The leading term 2 R 2 log(1 /δ ) in Eq. (13) is asymptotically optimal as it matches the lower bound in Theorem 1. The following theorem then follows from Lemma 3:

Theorem 4 (Correctness) . The output ˆ µ ⋆ of EllipsoidEst satisfies | ˆ µ ⋆ -µ ⋆ | ≤ ε with probability at least 1 -δ .

Proof.

<!-- formula-not-decoded -->

This means that ( U ( τ ) -2 ε ) i ∈K is outside the confidence ellipsoid for the arms in K , which implies that µ ⋆ ∈ [ U ( τ ) -2 ε, U ( τ )] with probability at least 1 -δ , which in turn implies that ˆ µ ⋆ = U ( τ ) -ε ∈ [ µ ⋆ -ε, µ ⋆ + ε ] with probability at least 1 -δ .

While the stopping condition guarantees that the output ˆ µ ⋆ = U ( τ ) -ε is a PAC estimator of the true best-mean-regardless of how the samples are collected-EllipsoidEst is carefully designed to collect samples in a way that achieves asymptotically optimal sample complexity. Our lower bound analysis reveals that all non-best arms should be sampled so that they maintain the same UCB, while the (presumed) best arm sometimes needs to be sampled more frequently to tighten its LCB and accurately estimate the best mean.

Specifically, EllipsoidEst operates in two phases. In Phase 1, it repeatedly selects the arm with the highest UCB, aiming to obtain rough estimates of the arm means without oversampling any arm. Phase 2 is designed to match the lower bound when the estimated means are close to the true values, while still ensuring termination within a reasonable number of rounds even if the estimates at the time of the phase shift are inaccurate. Specifically, the algorithm transitions to Phase 2 when the condition in Step 12 is satisfied. At this point, it tentatively identifies the best arm ˆ i ⋆ and estimates its optimal sample allocation r ˆ i ⋆ . More precisely, it determines that arm ˆ i ⋆ should be sampled at least r ˆ i ⋆ β ( t, δ ) times. In Phase 2, the algorithm continues to pull the arm with the highest UCB, except when ˆ i ⋆ has not yet received its allocated number of samples (i.e., N ˆ i ⋆ ( t ) ≤ r ˆ i ⋆ β ( t, δ ) holds); in that case, ˆ i ⋆ is pulled instead.

Note that EllipsoidEst returns the midpoint ˆ µ ⋆ = U ( τ ) -ε , but what it actually finds is the interval [ U ( τ ) -2 ε, U ( τ )] that contains µ ⋆ with high probability. We will discuss this in detail in Section 7.3.

## 6 Sample complexity analysis

Here, we establish the sample complexity of EllipsoidEst that matches the lower bound, and thus proves its asymptotic optimality. We first show that the stopping time τ of EllipsoidEst is almost surely bounded by a certain quantity T max defined as follows (see Lemma 11 in Appendix A.4):

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This is then used to establish the following sample complexity of EllipsoidEst:

Lemma 5 (Sample complexity) . Consider any sufficiently small δ &gt; 0 such that log(1 /δ ) ≥ K . Then the stopping time of EllipsoidEst satisfies τ ≤ T max almost surely for the T max defined in (15) . Furthermore, for any ε c ≤ ε/ 2 , the sample complexity satisfies

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

By setting ε c = ε (log(1 /δ )) -1 / 3 , all but the first term in the right-hand side of (18) are o (log(1 /δ )) and we obtain the following theorem (see Appendix A.4):

Theorem 6. EllipsoidEst is asymptotically optimal, that is,

<!-- formula-not-decoded -->

Lemma 5 and Theorem 6 are proved in Appendix A.4. In the proof, we consider the event

<!-- formula-not-decoded -->

whose probability approaches 1 as δ → 0 (see Lemma 14). Under A ( c ) , the estimation error in ˆ µ i ( t ) after the phase shift (i.e., for t ≥ τ 1 , where τ 1 denotes the first time the condition in Step 12 is met) can be made arbitrarily small by setting a sufficiently small c (see Lemma 12). This concentration result allows us to bound the stopping time τ under A ( c ) by a quantity whose leading term is 2 R 2 f ( µ ) log(1 /δ ) , by leveraging the structure of the characterizing optimization (see Lemma 13).

̂

̂

Figure 3: (a)-(b) Sample complexity τ of EllipsoidEst and baselines (SE and UGapEc). (c) Bias in the output (midpoint) and the maximum empirical mean. The results are based on 30 random seeds.

<!-- image -->

## 7 Experiments and discussion

## 7.1 Experimental settings and implementation

We choose to set the regularization parameter as λ = ( R/S ) 2 . As we discuss in Appendix B.1, this choice minimizes the size of the confidence ellipsoid in the limit of large sample size and is also empirically shown to approximately minimize the sample complexity in a wide range of settings. Owing to this nearly optimal choice, EllipsoidEst is essentially free of hyperparameters.

We design our experiments to evaluate the empirical properties of EllipsoidEst, focusing on three aspects. First, we evaluate how the performance of EllipsoidEst depends on the confidence parameter δ and the number of arms K , comparing it against baselines. Second, we evaluate the EllipsoidEst's sensitivity to the misspecification of the parameters R and S , which reflect assumptions about the reward distributions. Finally, we evaluate the bias in the midpoint of the interval returned by EllipsoidEst and discuss its implications. Due to space constraints, we present full results and details in Appendix B and summarize the main findings in this section.

As baselines, we consider Successive Elimination (SE) [6, 3] and UGapEc [4], both originally designed for best-arm identification and adapted here for best-mean estimation following the approach in [13]. They thus satisfy the same PAC guarantee: the best mean is estimated within ε error with probability at least 1 -δ . Figure 9 in Appendix B.4 confirms that these baselines and EllipsoidEst indeed estimate best means with the required accuracy. While SE has suboptimal asymptotic sample complexity, it is known to perform well for moderate values of δ , especially when the number of arms is large. UGapEc achieves an asymptotically optimal order of sample complexity for best-arm identification and also demonstrates strong empirical performance at practical confidence levels.

We assume that each arm's reward follows a Bernoulli distribution and that the means µ are equally spaced within [0 , 1] (specifically, µ i = i/ ( K +1) , ∀ i ∈ [ K ] ). We fix ε = 0 . 1 and study the impact of varying δ (the confidence level) and K (the number of arms). In Appendix B, we explore other settings of experiments. See Appendix B.7 for details of the computational environment.

## 7.2 Results

Figure 3a-3b compares the sample complexity of EllipsoidEst against SE and UGapEc. The results show that EllipsoidEst consistently outperforms the baselines for small and moderate K . Moreover, EllipsoidEst exhibits the slowest growth in sample complexity as δ → 0 , which is consistent with the asymptotic optimality of EllipsoidEst. However, the sample complexity of EllipsoidEst increases more rapidly than the baselines as K grows. This motivates further research on best-mean estimation algorithms that scale more gracefully with K .

The experiments with other settings in Appendix B.4 suggest that UGapEc performs poorly when K is small or when the true means are tightly clustered. SE shows a faster increase in sample complexity (as δ → 0 ) across settings. Overall, although each method has its own strengths and limitations, EllipsoidEst demonstrates robust and competitive performance in configurations of practical interest.

In Appendix B.3, we examine the sensitivity of the sample complexity of EllipsoidEst to the values of R and S . As is consistent with our asymptotic analysis (i.e., E [ τ ] ∼ 2 R 2 f ( µ ) log(1 /δ ) ), the sample

complexity exhibits approximately quadratic growth with respect to R . While the influence of S is relatively weak, the sample complexity tends to increase with S , which motivates the sample-shifting technique that we discuss in Appendix B.1.

Figure 3c evaluates the output ˆ µ ⋆ returned by EllipsoidEst (red solid curve) and the maximum of the empirical means (i.e., max i ˆ µ i ( τ ) ), based on the samples collected by EllipsoidEst (blue dashed line), relative to the true best-mean µ ⋆ . The output ˆ µ ⋆ tends to overestimate the best mean, with the bias being particularly pronounced when K is large, while the maximum empirical mean max i ˆ µ i ( τ ) is notably less biased. In the following, we will discuss this bias in detail.

## 7.3 Bias in the midpoint of the interval

In Section 3, we defined best-mean estimation as the problem of returning a point estimate ˆ µ ⋆ that lies within ε of the true best-mean µ ⋆ with probability at least 1 -δ . This may raise questions about the potential bias of ˆ µ ⋆ . However, the problem can be equivalently framed as returning an interval of length (at most) 2 ε that contains µ ⋆ with probability at least 1 -δ . Under this equivalent formulation, the point estimate in the original definition is merely the midpoint of such an interval of length 2 ε .

The original formulation might be simpler and more familiar within the PAC learning framework, which is why we chose to adopt it initially. However, it has the drawback of placing unnecessary emphasis on the output ˆ µ ⋆ , requiring it to lie at the center of an interval solely due to how the problem is framed. It should be understood that the essential output of EllipsoidEst is the interval [ U ( τ ) -2 ε, U ( τ )] , which contains µ ⋆ with probability at least 1 -δ . The output ˆ µ ⋆ = U ( τ ) -ε should be viewed as merely the midpoint of this interval, returned to satisfy the requirements of the original problem formulation.

The output is typically higher than the maximum empirical mean (i.e., ˆ µ ⋆ ≥ max i ˆ µ i ( τ ) ), as is also demonstrated in Figure 3c. Notice that the maximum empirical mean is typically biased upward due to the 'winner's curse' [18], i.e., E [max i ˆ µ i ( τ )] &gt; max i µ i , although this phenomenon is not visible in Figure 3c due to the large sample size. In practice, the maximum empirical mean could be provided as supplementary information alongside the primary output-the confidence interval itself. Additional bias correction techniques such as [17, 22] could also be applied to reduce the bias in such supplementary information.

## 8 Conclusion

We have considered the problem of estimating the mean of the best arm, which can be viewed as a pure exploration problem with an infinite number of answers. Although the characterizing optimization is non-convex, it can be decomposed into a convex subproblem if we fix one parameter. By using this structure, we propose EllipsoidEst that only calculates a non-convex optimization at most once during the runtime. The sample complexity of our algorithm is asymptotically optimal, and its practical performance is verified through numerical experiments. In particular, our algorithm has outperformed the baselines for the number of arms K up to around 16. Improving the sample complexity for large K is interesting future work.

A limitation of EllipsoidEst is that it relies on a known upper bound S on the magnitude of the true means, although this assumption is standard in the linear bandit literature with sub-Gaussian noise since [1]. In some settings (e.g., Bernoulli rewards) S is trivially known, but this is not always the case. When S is unknown, the confidence bound from Theorem 2 in [1] (which we use) becomes inapplicable. However, Theorem 1 in [1] still provides a valid anytime confidence bound that does not require knowledge of S , although the resulting confidence region becomes more complicated.

## Acknowledgements

We thank the reviewer who pointed out an error in the original proof of Theorem 1. It has since been corrected using an approach that is simpler than the one proposed during the discussion. JH was supported by JSPS KAKENHI Grant Number JP25K02232.

## References

- [1] Y. Abbasi-Yadkori, D. Pál, and C. Szepesvári. Improved algorithms for linear stochastic bandits. In J. Shawe-Taylor, R. Zemel, P. Bartlett, F. Pereira, and K. Weinberger, editors, Advances in Neural Information Processing Systems , volume 24. Curran Associates, Inc., 2011.
- [2] R. Degenne and W. M. Koolen. Pure exploration with multiple correct answers. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 32. Curran Associates, Inc., 2019.
- [3] E. Even-Dar, S. Mannor, and Y. Mansour. Action elimination and stopping conditions for the multi-armed bandit and reinforcement learning problems. Journal of Machine Learning Research , 7(39):1079-1105, 2006.
- [4] V. Gabillon, M. Ghavamzadeh, and A. Lazaric. Best arm identification: A unified approach to fixed budget and fixed confidence. In F. Pereira, C. Burges, L. Bottou, and K. Weinberger, editors, Advances in Neural Information Processing Systems , volume 25. Curran Associates, Inc., 2012.
- [5] A. Garivier and E. Kaufmann. Optimal best arm identification with fixed confidence. In V. Feldman, A. Rakhlin, and O. Shamir, editors, 29th Annual Conference on Learning Theory , volume 49 of Proceedings of Machine Learning Research , pages 998-1027, Columbia University, New York, New York, USA, 23-26 Jun 2016. PMLR.
- [6] A. Hassidim, R. Kupfer, and Y. Singer. An optimal elimination algorithm for learning a best arm. In Advances in Neural Information Processing Systems , volume 33, pages 10788-10798. Curran Associates, Inc., 2020.
- [7] A. Huang, A. Block, Q. Liu, N. Jiang, A. Krishnamurthy, and D. J. Foster. Is best-of-n the best of them? coverage, scaling, and optimality in inference-time alignment, 2025.
- [8] M. Jourdan, R. Degenne, and E. Kaufmann. An ε -best-arm identification algorithm for fixedconfidence and beyond. In Thirty-seventh Conference on Neural Information Processing Systems , 2023.
- [9] H. Kajino, K. Miyaguchi, and T. Osogami. Biases in evaluation of molecular optimization methods and bias reduction strategies. In Proceedings of the 40th International Conference on Machine Learning , volume 202, pages 15567-15585, 23-29 Jul 2023.
- [10] S. Kalyanakrishnan, A. Tewari, P. Auer, and P. Stone. PAC subset selection in stochastic multiarmed bandits. In International Conference on Machine Learning , ICML'12, page 227-234. Omnipress, 2012.
- [11] Z. Karnin, T. Koren, and O. Somekh. Almost optimal exploration in multi-armed bandits. In International Conference on Machine Learning , Proceedings of Machine Learning Research, pages 1238-1246. PMLR, 17-19 Jun 2013.
- [12] E. Kaufmann and W. M. Koolen. Mixture martingales revisited with applications to sequential tests and confidence intervals. Journal of Machine Learning Research , 22(246):1-44, 2021.
- [13] T. Osogami, H. Kinoshita, and S. Wasserkrug. Mechanism design with multi-armed bandit, 2024. https://arxiv.org/abs/2412.00345v1 .
- [14] M. Rauh, N. Marchal, A. Manzini, L. A. Hendricks, R. Comanescu, C. Akbulut, T. Stepleton, J. Mateos-Garcia, S. Bergman, J. Kay, C. Griffin, B. Bariach, I. Gabriel, V . Rieser, W. Isaac, and L. Weidinger. Gaps in the safety evaluation of generative AI. Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society , 7(1):1200-1217, Oct. 2024.
- [15] A. Russo and A. Pacchiano. Adaptive exploration for multi-reward multi-policy evaluation. In Forty-second International Conference on Machine Learning , 2025.
- [16] C. Snell, J. Lee, K. Xu, and A. Kumar. Scaling LLM test-time compute optimally can be more effective than scaling model parameters, 2024. https://arxiv.org/abs/2408.03314 .

- [17] L. Sun, A. Dimitromanolakis, L. L. Faye, A. D. Paterson, D. Waggott, T. D. R. Group, and S. B. Bull. BR-squared: A practical solution to the winner's curse in genome-wide scans. Human Genetics , 129, 2011.
- [18] R. Thaler. Anomalies: The winner's curse. Journal of Economic Perspectives , 2(1):191-202, 1988.
- [19] H. van Hasselt. Double Q-learning. In J. Lafferty, C. Williams, J. Shawe-Taylor, R. Zemel, and A. Culotta, editors, Advances in Neural Information Processing Systems , volume 23. Curran Associates, Inc., 2010.
- [20] H. van Hasselt. Estimating the maximum expected value: An analysis of (nested) cross validation and the maximum sample average, 2013. https://arxiv.org/abs/1302.7175 .
- [21] P.-A. Wang, R.-C. Tzeng, and A. Proutiere. Fast pure exploration via Frank-Wolfe. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems , volume 34, pages 5810-5821. Curran Associates, Inc., 2021.
- [22] R. Xiao and M. Boehnke. Quantifying and correcting for the winner's curse in genetic association studies. Genetic Epidemiology , 33(5):453-462, 2009.
- [23] F. Xu, Q. Hao, Z. Zong, J. Wang, Y. Zhang, J. Wang, X. Lan, J. Gong, T. Ouyang, F. Meng, C. Shao, Y. Yan, Q. Yang, Y. Song, S. Ren, X. Hu, Y. Li, J. Feng, C. Gao, and Y. Li. Towards large reasoning models: A survey on scaling LLM reasoning capabilities, 2025. https: //arxiv.org/abs/2501.09686 .
- [24] X. Yu, B. Peng, V. Vajipey, H. Cheng, M. Galley, J. Gao, and Z. Yu. ExACT: Teaching AI agents to explore with reflective-MCTS and exploratory learning. In The Thirteenth International Conference on Learning Representations , 2025.

## A Proofs and technical lemmas

## A.1 Analysis of the lower bound

Consider the following multiple correct answer problem (MCP) [2]. In addition to K arms with means µ , this problem involves a finite answer set B . There is an answer function b ⋆ : R K → 2 B that determines the set of correct answers for a given model µ . Consider an adaptive algorithm that samples rewards from K arms and stops. Let τ be the stopping time. When this algorithm stops, it outputs ˆ b ⋆ ∈ B . An algorithm is δ -correct if ˆ b ⋆ ∈ b ⋆ ( µ ) with probability at least 1 -δ under any model µ .

Lemma 7. (Lower bound in the multiple correct answer problem, Theorem 1 in [2]) Any δ -correct MCP algorithm satisfies

<!-- formula-not-decoded -->

where d ( µ i , λ i ) denotes the KL divergence from the distribution with mean µ i to that with mean λ i ; hence, d ( µ i , λ i ) = ( µ i -λ i ) 2 / (2 R 2 ) for Gaussian distributions. Here, ∆ K denotes the ( K -1) -dimensional simplex.

̸

We then introduce an M -bin MCP problem with parameter M ∈ N . Let ε ′ = ε/M . This problem involves bins B := { ( -∞ , µ ⋆ -ε ) , [ µ ⋆ -ε, µ ⋆ -ε + ε ′ ) , [ µ ⋆ -ε + ε ′ , µ ⋆ -ε +2 ε ′ ) , . . . , [ µ ⋆ + ε, µ ⋆ + ε + ε ′ ) , [ µ ⋆ + ε + ε ′ , ∞ ) } . The number of bins is finite ( 2 M +2 ). We define the answer function as b ⋆ ( µ ) := { b ∈ B : b ∩ [ µ ⋆ -ε, µ ⋆ + ε ] = ∅} .

The output ˆ µ ⋆ of the original BME problem can be put into one of the bins. We can see that, if we convert a δ -correct BME algorithm into the M-bin MCP problem by assigning the estimated best mean into a bin including it, the resulting algorithm is a δ -correct M -bin MCP algorithm. By this fact, Lemma 7 applies as a lower bound of the original BME algorithm. In the following, we lower-bound the performance of the M -bin MCP algorithm.

Lemma 8. (Lower bound optimization for M -bin MCP algorithm) Let η &gt; 0 be arbitrary. Then there exists M ∈ N such that the RHS of Eq. (22) for M -bin MCP is larger than 2(1 -η ) R 2 f ( µ ) for Gaussian rewards with variance R 2 .

Proof of Lemma 8. The denominator of RHS of Eq. (22) is

<!-- formula-not-decoded -->

Weupper-bound Eq. (23) as follows. Take any bin b ∈ b ⋆ ( µ ) = { [ µ ⋆ -ε, µ ⋆ -ε + ε ′ ) , . . . , [ µ ⋆ + ε, µ ⋆ + ε + ε ′ ) } . Let L ( b ) the leftmost point in b . Then, L ( b ) ∈ [ µ ⋆ -ε, µ ⋆ + ε ] . Let b A = ( -∞ , L ( b ) -ε -ε ′ ] , b B = [ L ( b ) + ε + ε ′ , ∞ ) . Then

<!-- formula-not-decoded -->

where we denote λ ⋆ := max i ∈ [ K ] λ i . Hence, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for Gaussian rewards. Here,

<!-- formula-not-decoded -->

since L ( b ) ≤ µ ⋆ + ε and hence L ( b ) -ε -ε ′ &lt; µ ⋆ (the minimizer is λ i = L ( b ) -ε -ε ′ if µ i ≥ L ( b ) -ε -ε ′ and λ i = µ i otherwise). Also,

<!-- formula-not-decoded -->

since L ( b ) ≥ µ ⋆ -ε and hence L ( b ) + ε + ε ′ &gt; µ ⋆ (the minimizer is λ i = L ( b ) + ε + ε ′ for the minimizer i of w i ( µ i -( L ( b ) -ε -ε ′ )) 2 + and λ i = µ i for other i ). By using these, we obtain

<!-- formula-not-decoded -->

and therefore

<!-- formula-not-decoded -->

for C ⋆ = max b ∈ b ⋆ ( µ ) max w ∈ ∆ K C ( b, w ) . We write ( b ⋆ , w ⋆ ) to denote its maximizer, which always exists because this is a maximization of a continuous function C ( b, w ) over a compact set. Here note that

<!-- formula-not-decoded -->

where 1 is the all-one vector, and ∆ i := µ ⋆ -µ i for i ∈ [ K ] . Let α := ( 1 -2 √ Kε ′ ε ) -1 &gt; 1 , U ′ := L ( b ⋆ ) + ε -ε ′ , and r ′ i := αw ⋆ i /C ⋆ for i ∈ [ K ] . Then we have

<!-- formula-not-decoded -->

where the last line follows from (30), w ⋆ i ≤ 1 , and the definition of α . From this result we see that U ′ and r ′ = ( r ′ 1 , . . . , r ′ K ) are a feasible solution of the minimization problem in (P1), and therefore the optimal value f ( µ ) of (P1) satisfies

<!-- formula-not-decoded -->

Combining this fact with (29) we obtain

<!-- formula-not-decoded -->

the RHS of which approaches 2 R 2 f ( µ ) as ε ′ ↓ 0

<!-- formula-not-decoded -->

Proof of Theorem 1. Theorem 1 is an immediate consequence of Lemma 7, Lemma 8, and the fact that any δ -correct BME algorithm can be converted into an M -bin MCP, as well as the choice of η → 0 .

## A.2 Analysis of the characterizing optimization

Proof of Lemma 2. (i) The statement can be easily obtained from

̸

<!-- formula-not-decoded -->

for U ∈ ( µ ⋆ , µ ⋆ +2 ε ) .

(ii) By the Lagrange multiplier method, we immediately obtain

̸

<!-- formula-not-decoded -->

where recall that we set one optimal arm as i ⋆ , and therefore the case i = i ⋆ happens only for one i ∈ [ K ] even if there are multiple optimal arms.

Here, r i ⋆ ( U ; µ ) = 1 ( U -µ ⋆ ) 2 holds when

<!-- formula-not-decoded -->

or equivalently

<!-- formula-not-decoded -->

that is, U ≤ U ( µ ) . In this case,

̸

̸

<!-- formula-not-decoded -->

which is decreasing in U . Therefore, U &lt; U ( µ ) cannot be the optimal solution U ( µ ) minimizing f ( U ; µ ) , and we obtain U ( µ ) ≥ U ( µ ) , where

̸

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

which is not necessarily convex in U .

̸

̸

̸

̸

(iii) U ( µ ) ≥ U ( µ ) is already proved above. By letting f ′ ( U ; µ ) = ∂f ′ ( U ; µ ) ∂U , we obtain for U ≥ U ( µ ) that

̸

<!-- formula-not-decoded -->

̸

Therefore, when g ( U ; µ ) &gt; 0 , that is, when U &gt; U ( µ ) , we see that f ( U ; µ ) is increasing. Therefore, U &gt; U ( µ ) satisfying U ≥ U ( µ ) cannot minimize f ( U ; µ ) . Therefore we obtain U ( µ ) ≤ U ( µ ) ∨ U ( µ ) .

(iv) Recall that U ( µ ) is the solution to g ( U ; µ ) , and g ( U ; µ ) is increasing with U in ( µ ⋆ , µ ⋆ +2 ε ) . Now, observe that

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then the statement is immediate from U ( µ ) ≤ U ( µ ) &lt; µ ⋆ +2 ε shown in (i) and (iii).

which attains 0 at

Hence, and we get

(v) By (32), we have

## A.3 Analysis of the confidence ellipsoid

Proof of Lemma 3. Let us define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for observation Y t = X ⊤ t θ + η t with R -sub-Gaussian η t .

Then by Theorem 1 of [1], with probability at least 1 -δ it holds for all t that

<!-- formula-not-decoded -->

Here, let us consider our setting with V = λI and regularized estimator ˆ µ i ( t ) = 1 N i ( t )+ λ ∑ N i ( t ) m =1 Y i,m , where Y i,m is the m -th observation from arm i . Then

<!-- formula-not-decoded -->

where 1 S ≥ max i | µ i | . From these results, with probability at least 1 -δ it holds for all t that

<!-- formula-not-decoded -->

1 In our settings, the max norm is more natural than the L 2 norm in [1].

where

While the right-hand side of this expression is computable and sufficient for implementation, for the theoretical analysis it becomes convenient to bound the reminder term by

<!-- formula-not-decoded -->

when t ≥ 2 and λ ≥ 2 /K .

## A.4 Analysis of the sample complexity of Algorithm 1

## A.4.1 Preliminary

Lemma 9. For any t ≥ 0 and δ &gt; 0 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, at t = T max , we have

<!-- formula-not-decoded -->

When δ is sufficiently small and satisfies log(1 /δ ) ≥ K , we have

<!-- formula-not-decoded -->

Proof. Since log(1 + x/λ ) and x/ ( x + λ ) are concave in x for λ &gt; 0 , β ( t, δ ) defined in (10) is maximized when N i ( t ) = t/K, ∀ i ∈ [ K ] . Hence,

<!-- formula-not-decoded -->

By the definition of T max in (15), we have

<!-- formula-not-decoded -->

which gives (52). When log(1 /δ ) ≥ K holds, the second term in the previous expression is bounded as follows:

<!-- formula-not-decoded -->

where which gives (53).

In the following, we derive a couple of lemmas on the properties of f ( U ; µ ) and { r i ( U ; µ ) } i that are needed for the sample complexity analysis. Define

̸

<!-- formula-not-decoded -->

An intuition is that f k ( r, U ; µ ) corresponds to the sample complexity rate to ensure that the best mean is at most U given that arm k is pulled at least r times.

Lemma 10. Let ε 1 ∈ (0 , ε/ 2) , ε 2 ∈ (0 , ε ) and take µ arbitrary. Let µ ′ be such that | µ ′ i -µ i | ≤ ε 1 for all i . For k ⋆ = argmax i µ ′ i , define r ⋆ = r k ⋆ ( µ ′ ) . Then,

<!-- formula-not-decoded -->

In this lemma, µ ′ intuitively corresponds to the estimator of µ when the proportion of the optimal arm is determined. We can use this lemma to show that the sample complexity is close to the optimal one, f ( µ ) , as far as the estimator µ ′ is not far from µ .

The technical difficulty of this lemma comes from the non-convexity of the characterizing optimization (P1). While we can easily show f k ⋆ ( r ⋆ , U ; µ ) = f ( µ ) if r ⋆ = r i ⋆ ( µ ) and U = U i ⋆ ( r ; µ ) , we cannot guarantee that the optimal allocation r k ⋆ ( µ ′ ) for µ ′ becomes close to the optimal allocation r k ⋆ ( µ ) for µ due to the non-convexity. This lemma can be used to guarantee that if r and U are determined based on an estimator close to µ then the sample complexity becomes close to the optimal one despite the non-convexity of (P1).

Proof of Lemma 10. Before evaluating f k ⋆ ( r ⋆ , U ( µ ′ ) -ε 2 ; µ ′ ) we prepare several elementary relations. First we have

<!-- formula-not-decoded -->

and similarly,

<!-- formula-not-decoded -->

For any k ∈ [ K ] , we also have

̸

̸

<!-- formula-not-decoded -->

̸

We also obtain the following from | µ ′ i -µ i | ≤ ε 1 :

̸

<!-- formula-not-decoded -->

Since r ⋆ = r k ⋆ ( µ ′ ) for k ⋆ = argmax i µ ′ i ,

<!-- formula-not-decoded -->

by Lemma 2 (ii).

̸

̸

<!-- formula-not-decoded -->

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

̸

<!-- formula-not-decoded -->

̸

Here, in (64), 1 is the all-one vector and the equality comes from the shift-invariance in µ of the original problem (3).

## A.4.2 Analysis of T max

Lemma 11. When K ≤ log(1 /δ ) holds, the algorithm stops by T max almost surely.

Proof. We begin by bounding the number of times each arm is pulled before the algorithm terminates. Recall that at Step 5, the algorithm selects an arm, and that the tentative best arm ˆ i ⋆ , identified at Step 14, is treated differently from the others. Therefore, we analyze two separate cases depending on whether arm k is equal to ˆ i ⋆ or not.

̸

Case 1: k = ˆ i ⋆ . For the arm k = ˆ i ⋆ to be pulled in round t , it must be the maximizer k ⋆ at Step 9 in round t -1 . Since the algorithm has not stopped in round t -1 , we must have

̸

<!-- formula-not-decoded -->

implying that

̸

̸

<!-- formula-not-decoded -->

Since this relation holds whenever arm k is pulled in round t ≤ τ , we must have

<!-- formula-not-decoded -->

Case 2: k = ˆ i ⋆ . For the arm ˆ i ⋆ to be pulled in round t , it must be the maximizer k ⋆ at Step 9 in round t -1 or it satisfies N ˆ i ⋆ ( t -1) &lt; r ˆ i ⋆ β ( t -1 , δ ) at Step 5 in round t . However, by the same argument as above, the first situation implies

<!-- formula-not-decoded -->

The second situation can happen only when

<!-- formula-not-decoded -->

by Lemma 2(v).

From the above argument on two cases, we obtain

<!-- formula-not-decoded -->

which means

<!-- formula-not-decoded -->

where ξ is as defined in (16).

By Lemma 9, we have under log(1 /δ ) ≥ K that

<!-- formula-not-decoded -->

where the equality follows from the definition of γ in (17). This together with (71) implies

<!-- formula-not-decoded -->

Hence, the algorithm must stop within T max rounds almost surely when log(1 /δ ) ≥ K holds.

## A.4.3 Stopping time under A ( c )

We will first show, in Lemma 12, that the estimation error in ˆ µ i ( t ) after the phase shift (i.e., for t ≥ τ 1 ) can be made arbitrarily small under the event A ( c ) with a sufficiently small c . This will then be used in Lemma 13 to bound the stopping time τ under A ( c ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Let τ 1 be the time of the phase shift, that is,

<!-- formula-not-decoded -->

Observe that g ( U ( t ); ˆ µ ( t )) ∨ g ( U ( t ); ˆ µ ( t )) ≤ 0 is equivalent to U ( t ) ≤ U ( ˆ µ ( t )) ∨ U ( ˆ µ ( t )) and that U ( ˆ µ ( t )) ∨ U ( ˆ µ ( t )) &lt; ˆ µ ⋆ ( t ) + 2 ε by Lemma 2(i). These imply that

<!-- formula-not-decoded -->

where ˆ k ⋆ ( t ) := argmax i ˆ µ i ( t ) .

Consider the event

<!-- formula-not-decoded -->

̸

Then B always holds under A ( c ) because

<!-- formula-not-decoded -->

Hence, under B ⊇ A ( c ) , we have

<!-- formula-not-decoded -->

for any t ≥ τ 1 , i ∈ [ K ] . Therefore, under A ( c ) , we have

<!-- formula-not-decoded -->

To prove (74), it suffices to show that

<!-- formula-not-decoded -->

for any ε c &gt; 0 and c ∈ [ 0 , ε c ∆ max + ε ′′ +2 ε c ] . Since g ( c ) is a quadratic function with g (0) &lt; 0 , it is straightforward to verify that g ( c ) ≤ 0 for

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

Since √ 1 + x ≥ 1 + x 2(1+ x/ 4) for x &gt; 0 , the upper bound on c can be simplified to

<!-- formula-not-decoded -->

which completes the proof of the lemma.

Lemma 13. Under A ( c ) with c in the range specified in Lemma 12 for a sufficiently small ε c such that ε c &lt; ε/ 2 and 2 cε + ε c &lt; ε , the stopping time τ of Algorithm 1 satisfies

<!-- formula-not-decoded -->

Proof. We prove the lemma by showing that sufficiently large N i ( t ) 's make the stopping condition in Step 10 of the algorithm satisfied. To derive a lower bound on a term involved in the stopping condition, consider

<!-- formula-not-decoded -->

where c ∈ (0 , 1) .

This optimization problem is feasible only when U ≥ µ , and in this case, we can see that the optimal value is

<!-- formula-not-decoded -->

Indeed, when U ≥ µ +2(1 -c ) ε , the objective value can achieve 0 with x = U -2 ε and y = 2 ε . Consider the case where U &lt; µ +2(1 -c ) ε . Observe that the minimum is achieved with x = µ -cy for any y . With this x , we must have y ≤ U -µ 1 -c . Hence,

<!-- formula-not-decoded -->

which is nonnegative under U &lt; µ +2(1 -c ) ε . This lower bound is achieved with y = U -µ 1 -c and x = µ -cy .

Therefore, under A ( c ) , we have

<!-- formula-not-decoded -->

Similarly, if we additionally have N ˆ i ⋆ ( t ) ≥ r ˆ i ⋆ β ( t, δ ) , then we can show

<!-- formula-not-decoded -->

To see this, consider the optimization problem with the additional constraint of

<!-- formula-not-decoded -->

Similar to the previous argument, the minimum is achieved with x = µ -cy for any y . With this x , we must have y ≤ U -µ 1 -c ∧ √ 1 r . Hence,

<!-- formula-not-decoded -->

which is achieved with y = U -µ 1 -c ∧ √ 1 r and x = µ -cy . From these results, under A ( c ) , if N ˆ i ⋆ ( t ) ≥ r ˆ i ⋆ β ( t, δ ) holds, then

̸

<!-- formula-not-decoded -->

Therefore, if | ˆ µ k ( τ 1 ) -µ k | ≤ ε c additionally holds, then

̸

<!-- formula-not-decoded -->

From this discussion with the definition of r , if

<!-- formula-not-decoded -->

holds, then we can show that

<!-- formula-not-decoded -->

that is, the algorithm terminates. To show this, let ˆ U := U ( ˆ µ ( τ 1 )) . Then, since (94) is decreasing with U ( t ) , we have under (95) that

̸

<!-- formula-not-decoded -->

̸

̸

Therefore, it remains to evaluate the number of samples until U ( t ) decreases to the point where (95) is satisfied.

̸

If arm i = ˆ i ⋆ is pulled in round t , we must have

<!-- formula-not-decoded -->

Hence, we must have

<!-- formula-not-decoded -->

By solving the previous inequality for N i ( t -1) , we obtain

<!-- formula-not-decoded -->

Likewise, if arm ˆ i ⋆ is pulled in round t , we must have

<!-- formula-not-decoded -->

Hence, for any t ≤ τ , we have

<!-- formula-not-decoded -->

̸

̸

Hence, for ε &lt; ε/ 2 and 2 cε + ε &lt; ε , we can use Lemma 10 to establish

<!-- formula-not-decoded -->

## A.4.4 Probability of A ( c )

Lemma 14. For any c ∈ (0 , 1) , we have

<!-- formula-not-decoded -->

Proof. Lemma 3 implies

<!-- formula-not-decoded -->

Hence, A ( c ) holds with probability at least 1 -δ ′ if δ ′ satisfies

<!-- formula-not-decoded -->

By solving this for δ ′ , we obtain

<!-- formula-not-decoded -->

which is satisfied by choosing

<!-- formula-not-decoded -->

We thus obtain

<!-- formula-not-decoded -->

## A.4.5 Proof of the upper bound theorem

Proof of Lemma 5. Note that the upper bound on T max is from Lemma 11, the upper bound on β ( T max , δ ) is from Lemma 9, α ( T max ) is given in the proof of Lemma 9, and c is in the range specified in Lemma 12 and satisfies the condition, 2 cε + ε c &lt; ε , in Lemma 13:

<!-- formula-not-decoded -->

since ε c ≤ 1 2 ε . Notice also that (110) gives c &lt; ε c 3 ε .

Lemma 11 shows that τ ≤ T max almost surely under log(1 /δ ) ≥ K , and Lemma 13 shows that, under A ( c ) with the given conditions on c and ε c , we almost surely have

<!-- formula-not-decoded -->

Hence, we have

<!-- formula-not-decoded -->

which implies (18) by the bound on Pr( ¬A ( c )) from Lemma 14.

Since the sample complexity has a lower bound of 2 R 2 f ( µ ) log(1 /δ ) + o (log(1 /δ )) by Theorem 1 and an upper bound of T max = O ( KR 2 log(1 /δ ) /ε 2 ) by Lemma 11, we have f ( µ ) = O ( K/ε 2 ) . Hence, we must have ζ = O ( K/ε 2 ) .

Proof of Theorem 6. Since ∆ max ≥ 0 and α ( T max ) = O (log log(1 /δ )) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

These together with Lemma 5 imply (20).

## B Details of experiments

## B.1 Implementation

In this section, we discuss two implementation choices that can enhance the efficiency of EllipsoidEst. These techniques are broadly applicable and are recommended when appropriate.

Now, notice that

Hence, we have

Figure 4: Sample complexity τ against the regularization parameter λ , where we set K = 10 , ε = 0 . 1 ; δ is as shown in the legend. The results are over 10 random seeds.

<!-- image -->

First, while there may be room for further refinement, we recommend setting λ = ( R/S ) 2 . With this setting, EllipsoidEst becomes effectively free of tunable hyperparameters. This choice can be supported both theoretically and empirically.

Theoretically, note that our confidence ellipsoid (specifically, β ( t, δ ) in (10)) involves terms of the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and equals zero at x = nR 2 nS 2 -R 2 , which tends to ( R/S ) 2 as n →∞ . So, for large n , x = ( R/S ) 2 tends to minimize ω ( x ) , meaning that the size of the confidence ellipsoid tends to be minimized with λ = ( R/S ) 2 as more samples are collected from the arms.

The choice of λ = ( R/S ) 2 is also supported empirically, as shown in Figure 4. The sample complexity τ is minimized around λ = ( R/S ) 2 across a wide range of settings. In the figure, µ is configured in three different settings: Uniform, Clustered, and Gaussian. These configurations will be explained in detail in Appendix B.2.

The second implementation choice concerns the selection of S . Recall that S must be chosen such that | µ i | ≤ S, ∀ i ∈ [ K ] in order to guarantee the PAC property. If we know that µ i ∈ [ ℓ, u ] , ∀ i for constants ℓ and u , we can set S = | ℓ | ∨ | u | . As demonstrated in Appendix B.3, the sample complexity of EllipsoidEst tends to increase with S (even with the choice of λ = ( R/S ) 2 ). This motivates us to shift the sample by -( ℓ + u ) / 2 , which makes the true means lie within [( ℓ -u ) / 2 , ( u -ℓ ) / 2] . Consequently, we can set S = ( u -ℓ ) / 2 , which is guaranteed to be no greater than | ℓ | ∨ | u | .

## B.2 Experimental settings and baselines

We design our experiments to evaluate the empirical properties of EllipsoidEst and compare its performance against baseline methods, focusing on three aspects. First, EllipsoidEst requires the parameters R and S , which depend on assumptions about the reward distributions. In practice, these values may be misspecified, so it is important to assess the algorithm's sensitivity to such inaccuracies. Second, the midpoint of the interval returned by EllipsoidEst tends to overestimate the best mean (i.e., E [ˆ µ ⋆ ] &gt; µ ⋆ ). It is useful and interesting to quantify the extent of this bias. Third, although EllipsoidEst achieves asymptotically optimal sample complexity as δ → 0 , this does not guarantee strong performance for moderate values of δ . We therefore empirically evaluate its performance under practical confidence levels and compare it to standard baselines.

As baselines, we consider Successive Elimination (SE) [6, 3] and UGapEc [4], both originally designed for best-arm identification and adapted here for best-mean estimation following the approach in [13]. While SE has suboptimal asymptotic sample complexity, it is known to perform well for moderate values of δ , especially when the number of arms is large. UGapEc achieves an

The derivative of ω ( x ) is

Figure 5: Sensitivity of the sample complexity τ to the values of R (top row) and S (bottom row), where we set K = 10 , ε = 0 . 1 ; δ is as shown in the legend. The results are over 10 random seeds.

<!-- image -->

asymptotically optimal order of sample complexity for best-arm identification and also demonstrates strong empirical performance at practical confidence levels. Specifically, we use the SE variant proposed in [13], which is inspired by the original SE but tailored for best-mean estimation. For UGapEc, we adopt the two-step procedure from [13]: first, identify the best arm using UGapEc, and then sample that arm to estimate its mean with the desired confidence level 2 .

In our experiments, we typically assume that each arm's reward follows a Bernoulli distribution. This choice aligns with the baselines (SE and UGapEc), which are originally designed for bounded reward distributions. To assess the robustness of our findings, we also examine settings with Gaussian reward distributions in Appendix B.6. We consider three configurations for the mean vector µ : Uniform, Clustered, and Gaussian. In the Uniform configuration, the means are evenly spaced in [0 , 1] , with µ i = i/ ( K +1) , ∀ i ∈ [ K ] . In the Clustered configuration, all arms share the same mean: µ i = 1 / 2 , ∀ i ∈ [ K ] . In the Gaussian configuration, the means follow a Gaussian-shaped distribution centered at 0.5. Specifically, we compute the percent point function (inverse CDF) of a standard normal distribution at the 100( i -0 . 5) /K percentile, then linearly scale the values so that µ 1 = 0 . 1 and µ K = 0 . 9 .

We assume that the user of EllipsoidEst knows that the rewards follow a Bernoulli distribution. This allows the user to set R = S = 1 / 2 , while shifting the sample by -0 . 5 as we have discussed in Appendix B.1. In all experiments, we fix ε = 0 . 1 and vary δ and K to evaluate the performance of EllipsoidEst under different confidence and problem-size settings.

## B.3 Sensitivity to R and S

Figure 5 examines the sensitivity of the sample complexity τ of EllipsoidEst to the values of R and S . In the top row, we fix S = 1 / 2 and vary R , while in the bottom row, we fix R = 1 / 2 and vary S .

The sensitivity of sample complexity to R is consistent with our asymptotic analysis, which shows that E [ τ ] scales as 2 R 2 f ( µ ) log(1 /δ ) , implying approximately quadratic growth with respect to R . In contrast, the influence of S is less direct but still significant, as it affects the size of the confidence ellipsoid through the term β ( t, δ ) in (10). As shown in the figure, sample complexity increases with larger values of S , which supports the sample-shifting technique described in Appendix B.1. Overall, while the impact of S is weaker than that of R , it is still non-negligible, underscoring the importance of setting both parameters carefully.

2 The proof of Lemma 4 in [13] does not hold for the case when the samples from best-arm identification is reused for best-mean estimation; hence, we set M ˆ I = 0 in Algorithm 2 in [13].

̂

̂

̂

̂

̂

̂

Figure 6: Sensitivity of the estimated best-mean ˆ µ ⋆ to the values of R (top row) and S (bottom row), where we set K = 10 , ε = 0 . 1 ; δ is as shown in the legend. The results are over 10 random seeds.

<!-- image -->

When R and S are set too high, the sample complexity increases unnecessarily; when set too low, they can lead to significant errors in the estimated best-mean ˆ µ ⋆ . This trade-off is illustrated in Figure 6, particularly in the Uniform and Gaussian configurations, where ˆ µ ⋆ exhibits significant deviation from the true best-mean µ ⋆ when R or S is set too low. For the Clustered configuration, the best mean is consistently overestimated, which we discuss in detail in Section 7.3.

## B.4 Comparison against baselines

Figure 7 compares the sample complexity of EllipsoidEst against two baseline methods-Successive Elimination (SE) and UGapEc-both adapted for best-mean estimation. All three methods satisfy the same PAC guarantee: the best mean is estimated within ε error with probability at least 1 -δ . The results show that EllipsoidEst consistently outperforms the baselines when the number of arms is small to moderate (up to around K = 16 ). Moreover, as predicted by theory, EllipsoidEst exhibits the slowest growth in sample complexity as log(1 /δ ) (i.e., as δ → 0 ). UGapEc performs poorly when K is small or when the true means are tightly clustered, while SE shows a faster increase in sample complexity (as δ → 0 ) across settings. Overall, although each method has its own strengths and limitations, EllipsoidEst demonstrates robust and competitive performance in configurations of practical interest.

Figure 8 highlights a limitation of EllipsoidEst by showing how its sample complexity τ scales with the number of arms K . As the figure illustrates, the sample complexity of EllipsoidEst increases more rapidly than that of the baseline methods as K grows. This observation motivates further research on best-mean estimation algorithms that scale more gracefully with the number of arms.

Figure 9 confirms the correctness of the algorithms in returning an estimate that is within ε of the true best-mean with probability at least 1 -δ . Here, we set ε = 0 . 1 and δ = 0 . 01 in the settings of Figure 7. The box-and-whisker plots indicate that, in every case , the returned estimates lie within ε = 0 . 1 of the true best-mean across all 30 random seeds. In Appendix B.5, we discuss the bias in the estimated best-means.

## B.5 Bias in the midpoint of the interval

As discussed in Section 7.3, EllipsoidEst returns the output ˆ µ ⋆ that typically exceeds the maximum of the empirical means, max i ˆ µ i ( τ ) , which itself tends to overestimate the true best-mean µ ⋆ due to the winner's curse. Figure 10 explores this bias in the output-the midpoint of the interval that contains the true best-mean with high probability- returned by EllipsoidEst, using the same settings as Figure 8. The red curve shows the bias in the midpoint (i.e., ˆ µ ⋆ -µ ⋆ ). As expected, the midpoint

Figure 7: Sample complexity τ of EllipsoidEst compared against baselines, Successive Elimination (SE) and UGapEc, where we vary δ , setting ε = 0 . 1 and K as indicated in above the panels. The results are based on 30 random seeds.

<!-- image -->

Figure 8: Sample complexity τ of EllipsoidEst compared against baselines, Successive Elimination (SE) and UGapEc, where we vary K , while setting δ = 0 . 01 , ε = 0 . 1 . The results are based on 30 random seeds.

<!-- image -->

returned by EllipsoidEst consistently overestimates the best mean, with the bias being particularly pronounced when the number of arms K is large or when the true means are tightly clusteredscenarios where many arms are close to optimal. Note that despite the bias, the midpoint (output) ˆ µ ⋆ still satisfies the PAC guarantee with ε = 0 . 1 in this experiment: the bias remains below ε , in line with the theoretical guarantee.

The blue curve in Figure 10 illustrates the bias in the maximum of the empirical means (i.e., max i ˆ µ i ( τ ) -µ ⋆ ), based on the samples collected by EllipsoidEst. As expected, this value-already stored in the memory of EllipsoidEst-is notably less biased than the output ˆ µ ⋆ .

Figure 9: The best means estimated by UGapEC, SE, and EllipsoidEst in the settings of Figure 7 with ε = 0 . 1 and δ = 0 . 01 .

<!-- image -->

̂

̂

<!-- image -->

̂

Figure 10: Bias in the midpoint of the confidence interval returned by EllipsoidEst, where we vary K , while setting δ = 0 . 01 , ε = 0 . 1 . The results are based on 30 random seeds.

Also, recall from Section 1 that, although we have framed the problem as best-mean estimation following the multi-armed bandit literature, our original motivation was to estimate the mean performance in worst scenarios . In such settings, the overestimation can be viewed as a form of conservatism, which is typically more appropriate than optimistic estimates in safety-critical applications. In fact, an unbiased estimate may unintentionally underestimate risks and lead to overly optimistic conclusions from the perspective of confidence guarantees, which can be undesirable from the standpoint of safety assurance.

̂

̂

̂

Figure 11: [Gaussian reward distributions] Sample complexity τ of EllipsoidEst compared against baselines, Successive Elimination (SE) and UGapEc, where we vary δ , setting ε = 0 . 1 and K as indicated in above the panels. The results are based on 30 random seeds.

<!-- image -->

## B.6 Gaussian reward distributions

In this section, we assume that each arm's reward follows a Gaussian distribution. The standard deviation of the Gaussian distribution is fixed at R = 1 / 2 . Similar to the previous experiments, we consider three configurations for the mean vector µ : Uniform, Clustered, and Gaussian. Hence, the means are in [0 , 1] , which we assume to be known and set S = 1 / 2 (while shifting the sample by -0 . 5 ).

While EllipsoidEst is guaranteed to estimate the best mean within ε error with probability at least 1 -δ in this setting, the PAC guarantees of the baselines-SE and UGapEc-do not formally extend here, as they were originally designed for bounded reward distributions. Nevertheless, we apply them as if each arm's reward follows a Bernoulli distribution, in order to assess the robustness of our findings about EllipsoidEst across different reward models.

Figures 11-12 present results analogous to those in Figures 7-8, but for the case of Gaussian reward distributions. The similarity between the two sets of results suggests that our findings based on Bernoulli rewards are robust to changes in the underlying distribution.

## B.7 Computational requirements

Although running time is not the primary concern in best-mean estimation, it is still important that algorithms do not require excessive computation. Here, we compare the running time of EllipsoidEst to that of the baselines. All experiments in Appendix B, including those reported previously, were conducted on a single CPU core with 4 GB of memory and no GPU acceleration, in a cloud environment.

Figure 12: [Gaussian reward distributions] Sample complexity τ of EllipsoidEst compared against baselines, Successive Elimination (SE) and UGapEc, where we vary K , while setting δ = 0 . 01 , ε = 0 . 1 . The results are based on 30 random seeds.

<!-- image -->

Figure 13: Running time of EllipsoidEst compared against baselines in the settings of Figure 7

<!-- image -->

Figures 13-14 report the running times of EllipsoidEst and the baselines in the same settings as Figures 7-8. The reported time is measured until each algorithm returns an output, and it includes the time to generate samples. As a result, the number of samples an algorithm requires is the primary factor affecting its running time.

While the running time of each algorithm can vary significantly with implementation-and our implementations are not fully optimized-, we can already conclude from these results that all algorithms are sufficiently efficient from computational perspectives. In particular, each algorithm requires well under one millisecond per sample.

Figure 14: Running time of EllipsoidEst compared against baselines in the settings of Figure 8

<!-- image -->

## B.8 Properties of the optimal solution of (P1)

Recall that the difficulty of best mean estimation stems from the non-convexity of the optimization problem (P1). While the prior work on best arm identification has addressed such non-convexity with convex reformulation [12, 21], (P1) does not admit such a convex reformulation, and the lower bound remains inherently nonconvex. This leads to challenges such as possible discontinuities in the optimal allocation and the lack of structure typically exploited in classical settings. In this section, we provide a numerical example where the solution of (P1) has multiple disconnected local minima.

̸

Specifically, Figure 1 shows the optimal value f ( U ; µ ) of (P2) as a function of U for the setting where ε = 0 . 1 , K = 512 , µ 1 = 1 , and µ i = 0 . 971525 for i = 1 . In this case, the objective f ( U ; µ ) has two local minima at U ≈ 1 . 1633 and U ≈ 1 . 1745 . For µ i around the above values, the optimal solution discontinuously changes between these U 's.

This phenomenon simply arises from the fact that the feasible region is nonconvex. The Hessian of the left-hand side (LHS) of (3) has eigenvalues r i -√ r 2 i +4( U -µ i ) 2 ≤ 0 and r i + √ r 2 i +4( U -µ i ) 2 ≥ 0 with respect to ( r i , U ) . This means that the LHS of (3) is not negativesemidefinite and thus the feasible region is nonconvex.

Although nonconvex optimization also appears in the lower-bound analysis of classical best arm identification, an important distinction lies in the availability of a convex reformulation.

In classical best arm identification, the lower bound (e.g., Proposition 16 in [12] and (1) in [21]) takes the form:

<!-- formula-not-decoded -->

which is a nonconvex problem if we first solve the inner minimization. However, this can be equivalently rewritten as a linear program with infinitely many constraints:

<!-- formula-not-decoded -->

where the objective and the constraints are linear (and hence convex) in ( x, w ) . While the constraint set indexed by λ ∈ Alt( µ ) is nonconvex, the resulting optimization problem enjoys desirable properties such as convexity (and often uniqueness) of the optimal solution set, and continuity of the optimal allocation with respect to µ . These properties are crucial for algorithms like Track-and-Stop, which rely on the property that the allocation gradually converges to the optimal one.

In contrast, our problem does not admit such a convex reformulation, and the lower bound remains inherently nonconvex, leading to the discontinuity in the optimal allocation as shown with Figure 1.

## B.9 Comparison to a deviation inequality for exponential families

Here, we consider a stopping rule based on a deviation inequality for exponential families proposed in [12] for the case of Gaussian distributions. Specifically, Theorem 9 in [12] suggests the stopping

rule where the β ( t, δ ) in Step 10 of Algorithm 1 is replaced with the β ef ( t, δ ) defined as follows:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ζ is the zeta function. The stopping rule based on β ef guarantees the correctness of the algorithm for Gaussian arms.

Since β ( t, δ ) = O (log N i ( t )) and β ef ( t, δ ) = O (log log N i ( t )) , the algorithm terminates earlier with the stopping rule based on β ef than with the one based on β ( t, δ ) in the limit of N i ( t ) →∞ . However, here we show β ( t, δ ) &lt; β ef ( t, δ ) for moderate values of N i ( t ) in typical cases of practical interest.

We first derive a lower bound of β ef ( t, δ ) . Observe that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where η &gt; 1 . 4169 can be verified numerically (see Figure 15a). Hence, we have

<!-- formula-not-decoded -->

We also derive an upper bound of β ( t, δ ) :

<!-- formula-not-decoded -->

Now, consider the case where R = S = 1 and n = N i ( t ) , ∀ i ∈ [ K ] . We also (optimally) set λ = ( R/S ) 2 in β ( t, δ ) . Then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It can be verified numerically (see Figure 15b) that

<!-- formula-not-decoded -->

In Figure 15c, we compare the values of β ( t, δ ) and β ef ( t, δ ) . Although β ef ( t, δ ) = O (log log N i ( t )) can get smaller than β ( t, δ ) = O (log N i ( t )) for large N i ( t ) , we observe β ( t, δ ) &lt; β ef ( t, δ ) for moderate values of N i ( t ) (specifically, N i ( t ) ≤ 5 × 10 5 ) that are of practical interest.

## B.10 Ablation

In this section, we conduct the following ablation studies to assess the effectiveness of our stopping rule and sampling strategy against relatively naive choices: (i) We change the stopping condition such that the algorithm terminates when

<!-- formula-not-decoded -->

holds. (ii) We fix the sampling strategy to UCB (i.e., the algorithm never enters Phase 2).

Figure 16 shows the sample complexity τ of EllipsoidEst with our proposed stopping rule with the confidence ellipsoid (EllipsoidEst) and the one with the naive stopping rule with (131) (Naive). Observe that the choice of stopping condition can significantly affect performance, especially when µ is clustered.

On the other hand, while the two-phase sampling strategy is crucial for our theoretical guarantees, we find that its empirical impact is limited in the settings considered in our experiments. Specifically, in the settings of Figure 16, the differences in the stopping times are at most a few percent between the two sampling strategies.

Figure 15: (a)-(b) Numerical verifications of (125) and (130). (c) The value of β ( t, δ ) in our stopping rule (sub-Gaussian) and β ef ( t, δ ) for the bound based on the deviation inequality in [12] (exp-family), where K = 16 and δ = 0 . 001 .

<!-- image -->

Figure 16: Ablation study comparing the sample complexity τ of EllipsoidEst with the proposed stopping rule (EllipsoidEst) and the naive stopping rule (Naive), where we vary K and δ as indicated, while setting ε = 0 . 1 . The results are based on 30 random seeds.

<!-- image -->

## C Societal impacts

This work is expected to have both positive and potentially negative societal impacts. On the positive side, our method improves the data efficiency of estimating the mean performance in the best or worst case, which is particularly valuable in domains where data collection is costly or time-consuming. While originally motivated by the need to provide safety guaranties for AI agents, the proposed approach may have broader societal applications in areas such as healthcare, finance, and scientific research-fields where rigorous safety and performance assurances are essential. On the negative side, our method could be misused if the PAC guaranty is misunderstood. In particular, the midpoint of the interval returned by our method should not be applied as a point estimate in use cases that require unbiased estimates.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The contributions and claims are accurately summarized at the end of Section 1. Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The proposed algorithm has relatively poor performance for a large number of arms, and this is shown and discussed with Figure 3b in Section 7 and with Figure 8 in Appendix B.4.

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

Justification: All the theoretical statements are stated with full set of assumptions, and complete proofs are presented in Section 5 and Appendix A.

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

Justification: All the details of experimental settings are explained in Section 7 or Appendix B, and source code is submitted as the supplementary material.

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

Justification: We submit the source code as a supplementary material together with README.md, which includes the instructions on how to set up the environment and how to reproduce all of the experimental results reported in the paper. This paper does not use any datasets.

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

Justification: The proposed algorithm has a single hyperparameter λ , and we set its value as λ = ( R/S ) 2 based on our analysis. Our experimental settings do not involve train-test split.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: In all figures of experimental results, we plot standard deviations as error bars.

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

Justification: The computer resources are detailed in Appendix B.7. Also, while the running time of an algorithm is not our primary concern, we report the running time in Appendix B.7.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have carefully read the NeurIPS Code of Ethics and ensured that this paper follows the code.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discuss both potential positive societal impacts and negative societal impacts in Appendix C.

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

Justification: We do not release data or models.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We use numpy, scipy, matplotlib, jupyterlab, which are explicitly mentioned in pyproject.toml files. These libraries have BSD-style or MIT licenses.

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

Answer: [Yes]

Justification: README.md provides all of the necessary information to run the source code for reproducing the experimental results.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing or human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve crowdsourcing or human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

## Answer: [NA]

Justification: We have used LLMs for writing and editing and for implementing standard methods such as bisection.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.