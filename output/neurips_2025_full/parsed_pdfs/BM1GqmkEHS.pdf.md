## Simultaneous Statistical Inference for Off-Policy Evaluation in Reinforcement Learning

## Tianpai Luo ∗ Xinyuan Fan ∗ Weichi Wu †

Department of Statistics and Data Science Tsinghua University Beijing 100084, China

{ltp21, fxy22}@mails.tsinghua.edu.cn , wuweichi@tsinghua.edu.cn

## Abstract

This work presents the first theoretically justified simultaneous inference framework for off-policy evaluation (OPE). In contrast to existing methods that focus on point estimates or pointwise confidence intervals (CIs), the new framework quantifies global uncertainty across an infinite or continuous initial state space, offering valid inference over the entire state space. Our method leverages sieve-based Q-function estimation and (high-dimensional) Gaussian approximation techniques over convex regions, which further motivates a new multiplier bootstrap algorithm for constructing asymptotically correct simultaneous confidence regions (SCRs). The widths of the SCRs exceed those of the pointwise CIs by only a logarithmic factor, indicating that our procedure is nearly optimal in terms of efficiency. The effectiveness of the proposed approach is demonstrated through simulations and analysis of the OhioT1DM dataset.

## 1 Introduction

Off-policy evaluation (OPE) is a fundamental topic in reinforcement learning (RL), aiming to assess the performance of a target policy using data collected under a different behavior policy, before adopting the target policy in practice. For this purpose, much effort has been made on the statistical inference of the value of the target policy, including obtaining an accurate estimate and valid confidence intervals for quantifying the uncertainty. See Uehara et al. (2022) for a comprehensive review.

In many real-world applications, such as healthcare (Murphy et al. 2001, Matsouaka et al. 2014, Shi, Lu &amp; Song 2020), ridesharing (Xu et al. (2018)), and autonomous driving (Sallab et al. (2017)), it is often necessary to evaluate a policy across a range of initial states. For instance, in the OhioT1DM dataset (Marling &amp; Bunescu (2020)), each patient begins in a different state of continuous glucose monitoring (CGM) blood glucose levels and self-reported life events. The evaluation of a potentially effective off-policy must be conducted without direct deployment, and it requires quantification of uncertainties across multiple initial states. However, constructing pointwise confidence intervals (CIs) for each state with Bonferroni correction inflates the overall significance level, which is well-known as the multiple testing problem. The inflation becomes especially pronounced when the state space is infinite, for example, R . To address this, we consider the following question:

Is it possible to simultaneously quantify the uncertainty of off-policy value estimators over the entire state space?

∗ Equal contribution.

† Corresponding author.

In this paper, we provide an affirmative answer. Specifically, this can be achieved by constructing simultaneous confidence regions (SCRs) that cover the whole value functions at a given significance level.

## 1.1 Related work

Existing methods for statistical inference in reinforcement learning can be categorized into three categories: (i) Direct estimation: This approach constructs CIs by directly learning the system dynamics or Q-function under the target policy. The estimations include kernel-based Q-function methods (Feng et al. (2020)), batch learning (Le et al. (2019)), and sieve estimation methods (Chen (2007), Shi et al. (2021) or equivalently, called linear function approximation Sutton et al. (2008), Lagoudakis (2017)). (ii) Importance sampling: This method re-weights the observed rewards with the density ratio of the target and behavior policies. Bootstrap methods, concentration inequalities, and empirical likelihood-based methods have been applied to construct CIs for importance sampling estimators (Thomas et al. (2015), Hanna et al. (2017), Dai et al. (2020)). (iii) Double reinforcement learning (DRL): This framework combines the first two for more robust and efficient value evaluation (Jiang &amp; Li (2016),Thomas &amp; Brunskill (2016), Jiang &amp; Huang (2020)). For instance, Kallus &amp; Uehara (2022) achieves consistent DRL estimation of the value function and computes a marginalized density ratio to build a CI.

While existing methods largely focus on point estimation or pointwise intervals, approaches tailored for a large number of states remain limited. Works such as Duan et al. (2020) and Shi et al. (2021) advanced this direction by constructing confidence intervals not only pointwise (for the value at a given state) but also for integrated value functions under a known reference distribution of initial states. However, both leave important gaps. The asymptotic theory in Shi et al. (2021) establishes validity in large samples but does not provide non-asymptotic error control. In contrast, Duan et al. (2020) supports finite-sample inference, but its confidence bounds are conservative. Neither framework provides the simultaneous inference that is uniformly valid across all states. Our work addresses these gaps by developing a framework that enables distribution-free, asymptotically correct inference for the value function at any state simultaneously, while also delivering finite-sample guarantees through the non-asymptotic bound obtained from the Gaussian approximation.

## 1.2 Contributions

In this paper, we propose a novel framework for constructing asymptotically correct SCRs for the OPE. To the best of our knowledge, this is the first work to introduce a simultaneous statistical inference framework in policy evaluation of RL. Our method shares a similar spirit to Q-learning, which estimates the state-action value function (Q-function) under the target policy. The estimation of Qfunction is achieved by the linear function approximation (i.e., sieve method). Our key contributions are as follows:

1. We establish a convex Gaussian approximation result for the sieve estimation of the Qfunction. This approximation enables us to characterize the distribution of the sieve estimator over arbitrary convex sets, thereby facilitating simultaneous inference when the initial state is not fixed. Moreover, the convex Gaussian approximation theory only requires the number of trajectories or decision points to diverge, which naturally allows the infinite-horizon setting. Our theoretical results are built upon non-asymptotic results, which do not involve any convergence from extreme value theory in statistics.
2. Based on the convex Gaussian approximation, we construct an asymptotically correct SCR whose stochastic behavior is depicted by the maxima of a Gaussian random field. The width of the SCR exceeds that of the pointwise confidence intervals only by a logarithmic factor.
3. To implement our methodology, we develop a multiplier bootstrap algorithm for constructing SCRs, which avoids the need to estimate the limiting joint distribution of policy value estimators across different initial states. We further assess the performance of the proposed simultaneous inference framework through both numerical simulations and real data analysis.

The rest of the article is organized as follows. We introduce the model setup in Section 2. In Section 3, we present the construction of SCR based on sieve estimation, convex Gaussian approximation, and the bootstrap algorithm. Simulation studies and real data analysis on the OhioT1DM dataset are

conducted in Section 4. Finally, we conclude our paper in Section 5. All proofs, along with additional simulation results, are given in the supplementary material.

## 2 Preliminaries

Consider a Markov Decision Process (MDP) represented by the tuple M = ⟨S , A , R ⟩ , where S denotes the state space, A the action space, and R : S × A → R the reward function. In this paper, we assume that S is a subspace of R d with a fixed dimension d , and A is the discrete set { 0 , 1 , . . . , m -1 } with a fixed cardinality m . Let ( S 0 ,t , A 0 ,t , R 0 ,t ) denote the state-action-reward triplet collected at time t . In the MDP framework, the following Markov assumption is imposed:

<!-- formula-not-decoded -->

where P denotes the transition probability kernel, which is time-homogeneous. Additionally, we assume that the conditional mean of the reward R 0 ,t depends only on the current state and action, i.e.,

<!-- formula-not-decoded -->

where r ( · ) is a reward function r : S × A → R . We note that if the reward R 0 ,t is a deterministic function of S 0 ,t , A 0 ,t , S 0 ,t +1 , condition (2.2) follows directly from (2.1). Both (2.1) and (2.2) are standard assumptions in the reinforcement learning literature.

Let π ( ·|· ) denote a policy which satisfies π ( a | s ) ≥ 0 for all s ∈ S , a ∈ A , and ∑ a ∈A π ( a | s ) = 1 for any s ∈ S . The objective of RL can then be expressed through the following value function:

<!-- formula-not-decoded -->

where the expectation E π is taken under the rule that actions are selected according to the policy π , and γ refers to a given discount factor, 0 ≤ γ &lt; 1 .

In this paper, we consider an offline setting where data is pre-collected and can be written as

<!-- formula-not-decoded -->

where n denotes the number of trajectories, and T i is the termination time of the i -th trajectory. For the sake of brevity, we assume T i = T, i = 1 , . . . , n , and the sample size is denoted as N = nT . Our framework only requires that either T or n diverges (namely, N →∞ ).

## 3 Simultaneous inference for OPE

In this paper, we shall construct the asymptotically correct SCR for the OPE at significance level 1 -α , α ∈ (0 , 1) via finding C α (which might depend on N ) such that

<!-- formula-not-decoded -->

where ˆ V ( π ; s ) is the estimated policy values and L ( s ) is a scaling factor related to the covariance. When only a fixed s 0 ∈ S is considered (instead of ∀ s ∈ S ), (3.1) reduce to the pointwise confidence interval. Since the state space S can be continuous and infinite, to achieve asymptotic correct simultaneous coverage, we need to well control the family-wise error rate in contrast with previous pointwise CIs in RL (e.g., Luckett et al. (2020), Shi et al. (2021),Shi et al. (2024)).

Without loss of generality, we focus on stationary policies π ( · | · ) that do not vary with time t . For the justification, we refer to Lemma 1 of Shi, Wan, Song, Lu &amp; Leng (2020) and proof of Theorem 6.2.10 in Puterman (1994). To enable simultaneous confidence inference, we impose three main assumptions, which are adopted from the literature on pointwise inference (e.g., Shi et al. (2021)). The detailed assumptions and illustrations are listed as (A1)-(A3) in Section 3.2.

## 3.1 Q-learning with linear function approximation

We adopt a Q-learning approach to develop valid inference procedures for both deterministic and random policies. The Q-function under a policy π is defined as

<!-- formula-not-decoded -->

Under conditions (2.1) and (2.2), the Q-function satisfies the Bellman equation:

<!-- formula-not-decoded -->

We consider the linear function approximation for learning the Q-function. Let Φ 1 ( s ) , Φ 2 ( s ) , . . . , Φ K ( s ) be a collection of K basis functions and Φ ( s ) = (Φ 1 ( s ) , . . . , Φ K ( s )) ⊤ . We approximate Q ( π ; s, a ) based on a linear combination of the basis functions, i.e.,

<!-- formula-not-decoded -->

Related approximation results are presented in Section F.1 of the supplementary materials, assuming that Q ( π ; · , a ) belongs to a Hölder space of smoothness p for any policy π and action a ∈ A . This condition holds under standard assumptions on the transition probability P and a smooth reward function r ( s, a ) (see Section F.1 for details). The basis functions can be chosen from orthogonal splines, Legendre polynomials, or wavelets, forming a sieve basis commonly used in sieve estimation (Chen 2007, Huang 1998, Cohen et al. 1993, Timan 2014).

By (3.3) and (3.4), the mK -dimensional vector β ∗ π = ( β ∗⊤ π, 1 , . . . , β ∗⊤ π,m -1 ) ⊤ satisifies

<!-- formula-not-decoded -->

for all a ′ ∈ A . Denote ξ i,t = ξ ( S i,t , A i,t ) , U π,i,t = U π ( S i,t ) where

<!-- formula-not-decoded -->

Then (3.5) reduces to E ξ i,t ( R i,t + γU ⊤ π,i,t +1 β ∗ π -ξ ⊤ i,t β ∗ π ) = 0 , and β ∗ π can be estimated by

<!-- formula-not-decoded -->

where ˆ β π = ( ˆ β ⊤ π, 1 , . . . , ˆ β ⊤ π,m -1 ) ⊤ and

<!-- formula-not-decoded -->

Consequently, the value for policy π can be estimated by

<!-- formula-not-decoded -->

By (3.4), we have ˆ V ( π ; s ) -V ( π ; s ) -∑ a ∈A π ( a | s )Φ( s ) ⊤ ˆ θ π = O ( ϵ K ) where ˆ θ π = ˆ β π -β ∗ π and ϵ K = max a ∈A sup s ∈S | Q ( π ; s, a ) -Φ( s ) ⊤ β ∗ π,a | . By Chen (2007), there exists β ∗ π such that ϵ K = O ( K -p/d ) when the Q-function lies in a d -dimensional space with Hölder smoothness p .

## 3.2 Convex Gaussian approximation

In this section, we establish a general convex Gaussian approximation theory for learning the distribution behavior of ˆ θ π = ˆ β π -β ∗ π for all Euclidean convex sets in R mK . To allow K to diverge, we apply convex Gaussian approximation theorem (Fang (2016), Fang &amp; Koike (2024)), which supports moderately high-dimensional scenarios. We consider the state s within a compact region S ⊂ R d . For unbounded S , modifications such as introducing a weighting or mapping function are discussed in, e.g., Tjøstheim &amp; Auestad (1994),Huang &amp; Shen (2004), Chen &amp; Christensen (2015). We impose the following assumptions.

- (A1) The Markov chain { S 0 ,t } t ≥ 0 has an unique invariant distribution with some density function µ ( s ) . Denote ν 0 ( s ) as the probability density function of S 0 , 0 . The density functions µ ( s ) and v 0 ( s ) are uniformly bounded away from 0 and ∞ .

- (A2) Suppose the following (i) and (ii) hold when T → ∞ and (i) holds when T is bounded. (i) λ min [ ∑ T -1 t =0 E { ξ 0 ,t ξ ⊤ 0 ,t -γ 2 u π ( S 0 ,t , A 0 ,t ) u ⊤ π ( S 0 ,t , A 0 ,t ) } ] ≥ T ¯ c for some constant ¯ c &gt; 0 , where u π ( x, a ) = E { U π ( S 0 , 1 ) | S 0 , 0 = x, A 0 , 0 = a } and λ min ( M ) denotes the minimum eigenvalue of a matrix M . (ii) { S 0 ,t } t ≥ 0 is geometric ergodicity in dependence measure.

<!-- formula-not-decoded -->

Remark 3.1. Assumptions (A1)-(A3) are mild assumptions and serve as the minimal requirement for the goodness of the offline dataset to support feasible evaluation. The first condition in Assumption (A1) ensures that the Markov chain would not be trapped in a small subset of the entire space. Moreover, the second conditon ensures that every state is possible to be the initial state. Assumption (A2) relaxes the condition on sample size. Previous work (Jiang &amp; Li 2016) requires the number of trajectories n →∞ . (A2) additionally allows fixed n , but length T →∞ when the action variety is sufficiently large on each chain. The geometrical decay is similar with the geometrical ergodic for the Markov chain, which is a technical assumption in theoretical deduction, and is commonly assumed as a weaker requirement of i.i.d. in deriving limit theory. Assumption (A3) requires the reward signal diversity. ω π ( s, a ) ≥ c -1 0 requires that the reward random variable is nondegenerate (not always the same). P (max 0 ≤ t ≤ T -1 | R 0 ,t | ≤ c 0 ) = 1 means the rewards are bounded. The detailed definitions of the geometrical ergodicity and dependence measure are presented in Section G of the supplementary materials to save space.

The following Theorem 3.1 shows that there exists mK -dimensional Gaussian random vector Z π such that probability of ˆ θ π = ˆ β π -β ∗ π can be approximated by Z π over any convex sets.

Theorem 3.1. Denote Z π as the mK -dimensional Gaussian random vector possesses the same covariance structure of √ N ˆ θ π , i.e.,

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Under Assumptions (A1), (A2), and (A3), suppose that K = o ( N 2 / 7 (log N ) -1 ) , then we have

<!-- formula-not-decoded -->

where O is the collection of all the convex sets in R mK .

Remark 3.2. Note that the SCR based on estimation ˆ V ( π ; s ) = Φ( s ) ⊤ ∑ a ∈A π ( a | s ) ˆ β π can be written as ∩ s ∈S { √ N ˆ θ ∈ O π,s } where

<!-- formula-not-decoded -->

O π,s is a convex set since for any θ, θ ′ ∈ O π,s , λθ + (1 -λ ) θ ′ ∈ O π,s for any λ ∈ [0 , 1] . Therefore, ∩ s ∈S O π,s is a convex set and the probability P( ∩ s ∈S { √ N ˆ θ ∈ O π,s } ) can be learned by P( ∩ s ∈S { Z π ∈ O π,s } ) .

Remark 3.3. Theorem 3.1 provides a higher-order convex Gaussian approximation for ˆ θ π = ˆ β π -ˆ β ∗ π in the OPE estimation error ˆ V ( π ; s ) -V ( π ; s ) = Φ( s ) ⊤ ∑ a ∈A π ( a | s ) ˆ θ π . Existing approaches for constructing pointwise confidence intervals typically rely on the central limit theorem, deriving the limiting distribution of the inner product Φ( s ) ⊤ ∑ a ∈A π ( a | s ) ˆ θ π for each fixed s ∈ S . However, extending these results from a fixed s to arbitrary s ∈ S is nontrivial, as it requires controlling ∆ O = ∣ ∣ ∣ P ( √ N ˆ θ π ∈ O ) -P( Z π ∈ O ) ∣ ∣ ∣ for some convex set O (see Remark 3.2 for details).

Regarding the finite-sample properties, we provide the following bound on ∆ O with respect to the sample size N and the number of basis functions K , derived from the proof of Theorem 3.1:

<!-- formula-not-decoded -->

From the proof of Theorem 3.1, it follows that the above bound converges to 0 as N → ∞ when K = o ( N 2 / 7 -c ) for any given c &gt; 0 .

By Theorem 3.1, the SCR in (3.1) can be achieved by finding appropriate critical value C α,N &gt; 0 (which may depend on N ) such that

<!-- formula-not-decoded -->

The probability in (3.13) involves the supremum of functional linear combinations of the highdimensional Gaussian vector Z π . In practice, we can approximate C α,N in (3.13) by generating simulations of the Gaussian random vector Z π and computing the empirical quantile of the supremum. This approach, known as the Gaussian multiplier bootstrap, is detailed in Section 3.3. In theory, we leverage properties of Gaussian processes along with approximation techniques from Sun &amp; Loader (1994) to analyze the desired C α,N .

Proposition 3.2. For any two positive real sequences a n and b n , we write a n ≍ b n if there exists constants 0 &lt; c &lt; C &lt; ∞ such that c ≤ lim inf n →∞ a n /b n ≤ lim sup n →∞ a n /b n ≤ C . We write a n ≲ b n ( a n ≳ b n ) if there exists constant C &gt; 0 such that a n ≤ Cb n ( Ca n ≥ b n ) for all n . Denote matrix

<!-- formula-not-decoded -->

Under same conditions in Theorem 3.1, if there exists constant c 0 , c 1 , c 2 , c ≥ 0 such that

<!-- formula-not-decoded -->

then we have appropriate C α,N ≍ log 1 / 2 N such that

<!-- formula-not-decoded -->

where α is the given significance level and α ∈ (0 , 1) .

Remark 3.4. The scaling factor L ( s ) = √ U π ( s ) ⊤ Λ π U π ( s ) aligns with the pointwise CIs in Shi et al. (2021) so that we only need to compare the critical value C α,N with that in pointwise CIs. The rates N c 1 , N c 2 , N c 0 in condition (3.15) are mild assumptions which have been frequently used in the literature of sieve nonparametric estimation and inference; see Assumption 4 of Chen &amp; Christensen (2015) and Example 1-2 in Quan &amp; Lin (2024) for more details. The rate N c for ∫ S λ min ( M ⊤ M )d s can be derived in practice given basis Φ( s ) , which would be verified in Section F.2 of the supplementary materials.

Proposition 3.2 specifies the essential scale of the width of SCR. In contrast with previous pointwise confidence intervals proposed in Shi et al. (2021), C α,N ≍ √ log N shows that only an additional logarithmic rate √ log N is introduced to extend the pointwise confidence in Shi et al. (2021) to the global region S .

## 3.3 Bootstrap implementation

In the asymptotically correct SCR provided by (3.16), calculating an approximation of C α,N is rather complicated, and the convergence would be slow. We propose the Gaussian multiplier bootstrap algorithm to circumvent these problems and derive a feasible SCR in Algorithm 1.

Algorithm 1 Gaussian multiplier bootstrap for SCR

Input: Observed data { ( R i,t , A i,t , S i,t , S i,t +1 ) } 0 ≤ t ≤ T i , 1 ≤ i ≤ n .

Step 1: Calculate ˆ β π , ˆ Σ π , ˆ Ω π according to (3.6), (3.7), and (3.10). Obtain the estimator of the value function

<!-- formula-not-decoded -->

Step 2: Generate mK -dimensional Gaussian random vector Z ( b ) π ∼ N mK ( 0 , ˆ Σ -1 π ˆ Ω π ( ˆ Σ ⊤ π ) -1 ) .

Step 3: Repeat Step 2 for B times and document the outcomes Z ( b ) π , b = 1 , . . . , B .

Step 4: For a given level α ∈ (0 , 1) , denote ˆ q 1 -α as the (1 -α ) -th sample quantile of

<!-- formula-not-decoded -->

Output: (1 -α ) -th SCR ˆ V ( π ; s ) ± ˆ q 1 -α L ( s ) / √ N .

It is worth noting that by the convex Gaussian approximation results, Algorithm 1 can yield different asymptotically correct SCRs by modifying the scaling factor L ( s ) . The function L ( s ) provides flexibility to adjust the relative weighting of states s ∈ S , where larger L ( s ) prioritizes tighter confidence bounds for state s . For instance, if one is interested in the uncertainty of maximal deviation sup s ∈S | ˆ V ( π ; s ) -V ( π ; s ) | , then L ( s ) = 1 can be a convenient choice.

Remark 3.5 (Computational remarks) . Our procedure is computationally efficient and can, for instance, be executed on a personal laptop. The term ˆ β π in (3.6) is analogous to a least squares estimate and can be computed efficiently. Moreover, Steps 1 and 4 of the bootstrap procedure are linear due to the use of a linear approximation. Overall, the time complexity of our method is O ( NK 2 + K 3 + BK ) and the space complexity is O ( N + BK ) .

## 4 Experiments

## 4.1 Simulation studies

In this section, we conduct numerical studies to evaluate the performance of the proposed SCR. Both univariate ( d = 1) and multivariate ( d &gt; 1) scenarios are considered. The code is available at https://github.com/xinyuanfan01/Simultaneous-Statistical-Inference-for-Off-Policy-Evaluationin-Reinforcement-Learning.

In our settings, the state vector S 0 ,t may not have bounded support. To address this, we apply a sigmoid transformation, defined as sigmoid ( S ( j ) 0 ,t ) = 1 1+exp( -S ( j ) 0 ,t ) for 1 ≤ j ≤ d , to obtain features with bounded support. The basis functions are constructed using the tensor product of K Legendre or spline functions. The number of basis functions is determined through cross-validation (Qiu et al. 2021). We put the detailed cross-validation procedure in Section D of the supplement. Moreover, we performed sensitivity analyses and found that both the empirical coverage and the average length of the SCRs are robust to the choice of K .

We evaluate the SCRs using two metrics, each computed across 500 independent replications: (i) Empirical Coverage Probability (ECP): The proportion of times the true value function lies within the SCR across multiple simulations. (ii) Average Length (AL): The average width of the SCRs, approximated by averaging the widths at equally spaced grid points. The experiments can be readily conducted on a standard workstation, for example, an Apple M1 machine with 16 GB of RAM running macOS Sonoma.

For the method in Shi et al. (2021) (referred to as SA VE), we apply the Bonferroni correction to adjust the pointwise confidence intervals. For each setting, we compute the SCRs over equally spaced grid points. We emphasize that, in principle, pointwise confidence intervals cannot be naturally extended to SCRs, due to the continuous nature of our state space. Overall, the proposed SCR

achieves coverage close to the nominal level (we set α = 0 . 05 ), while the Bonferroni-adjusted SA VE results in a coverage rate well above 0 . 95 .

(Scenario 1 (univariate).) Let γ = 0 . 5 , n = 25 , 50 , 75 , T = 30 , 50 , 70 , and

<!-- formula-not-decoded -->

for t ≥ 0 , where U 0 ,t i.i.d. ∼ U (0 , 1) and S 0 , 0 ∼ U ( -2 , 2) . We consider a completely randomized behavior policy, i.e., A 0 ,t i.i.d. ∼ Bernoulli (0 . 5) for t ≥ 0 . The target policy is designed as π (1 | s ) = 1 -I ( s &gt; 0) . We construct SCRs for V ( π, s ) over the domain s ∈ [ -2 , 2] . The true value function is approximated from Monte Carlo simulation. We generate 10000 of independent trajectories with initial state being s i 0 = -2 + 4 i/ 999 for each i = 0 , . . . , 999 . Actions are selected according to the target policy. We approximate V ( π, s i 0 ) by taking the average over the 10000 trajectories, and use linear interpolation to approximate V ( π, s ) for s / ∈ { s i 0 , i = 0 , . . . , 999 } .

For Scenario 1, we employ the Legendre basis, and the results for ECPs and ALs are presented in Figure 1. Moreover, we perform the sensitivity analysis by taking ( n, T ) = (25 , 50) , (50 , 50) as two illustrative examples and examining the results by varying the specification of K over a relatively wide range. The corresponding results are presented in Table S.1 in the supplement. Figure 1 shows

<!-- image -->

N

Figure 1: Comparison of the methods based on empirical coverage probability (ECP, left) and average length (AL, right) for Scenario 1.

that SCR consistently achieve the nominal coverage level in various choices of ( n, T ) . In contrast, the Bonferroni-adjusted SAVE method exhibits substantial over-coverage. As N = nT increases, the empirical performance of our method converges more closely to the nominal target. Additionally, the average width of the SCRs produced by SAVE is approximately 20% greater than that of ours, which highlights the improved efficiency of our approach. Table S.1 further shows that our method is robust to the choice of the number of basis functions, enhancing its practical applicability.

(Scenario 2 (multivariate).) Let γ = 0 . 5 ,

<!-- formula-not-decoded -->

for t ≥ 0 , where z 0 ,t i.i.d. ∼ N ( 0 , 4 I 2 ) and S 0 , 0 ∼ U ([ -2 , 2] 2 ) , where the two components are independent. For behavior policy, we consider A 0 ,t ∼ Bernoulli ( p 0 ,t ) independently, where p 0 ,t = 0 . 5 ( Sigmoid ( S (1) 0 ,t ) + Sigmoid ( S (2) 0 ,t ) ) . The target policy is designed as π (1 | s ) = I ( s (1) &gt; 0 , s (2) &gt; 0) . We construct SCRs for V ( π, s ) over the domain s ∈ [ -1 , 1] 2 . Similar to that in Scenario 1, we simulate 10000 independent trajectories, each initialized at a point in the grid { s : s (1) = -1 + 2 i/ 29 , s (2) = -1 + 2 j/ 29 , for 1 ≤ i, j ≤ 30 } , to approximate the true value function V ( π, s ) , s ∈ [ -1 , 1] 2 . We construct SCRs using tensor products of Legendre and spline basis functions, respectively. The results are summarized in Table 1. Moreover, we conducted additional simulations employing SAVE with the Sidak correction (Abdi et al. 2007), and the results are summarized in Table S.2 in the supplement.

Furthermore, we modified the state transition rule to assess the performance of our method under high noise and non-Gaussian errors. Specifically, we set z 0 ,t to be an i.i.d. two-dimensional t (8) random

variable, while keeping all other components unchanged. The results are presented in Table S.3, where the coverage and length remain robust.

In addition to the comparison with SA VE, we also evaluated our method against the importance sampling approach (Jiang &amp; Li 2016, Hanna et al. 2017) based on Scenario 2. The detailed experimental settings and results are provided in Section B in the supplement.

|     | T   | Legendre      | Legendre       | ECP(AL). Spline   | ECP(AL). Spline   |
|-----|-----|---------------|----------------|-------------------|-------------------|
| n   |     | SCR           | SAVE           | SCR               | SAVE              |
| 30  | 50  | 0.926 (8.472) | 0.982 (9.445)  | 0.936 (9.452)     | 0.978 (9.793)     |
| 50  | 30  | 0.946 (9.553) | 0.970 (10.492) | 0.922 (11.101)    | 0.942 (11.131)    |
| 40  | 50  | 0.924 (7.225) | 0.976 (8.193)  | 0.938 (8.087)     | 0.976 (8.606)     |
| 50  | 40  | 0.944 (7.138) | 0.990 (8.136)  | 0.930 (7.247)     | 0.984 (7.753)     |
| 50  | 50  | 0.942 (7.299) | 0.978 (8.249)  | 0.930 (8.156)     | 0.966 (8.630)     |
| 50  | 150 | 0.952 (6.985) | 0.966 (7.402)  | 0.934 (5.771)     | 0.978 (6.080)     |
| 50  | 200 | 0.944 (5.978) | 0.978 (6.436)  | 0.950 (7.733)     | 0.960 (7.418)     |
| 50  | 250 | 0.942 (5.957) | 0.968 (6.234)  | 0.930 (5.177)     | 0.960 (5.446)     |
| 200 | 70  | 0.934 (4.924) | 0.988 (5.370)  | 0.926 (4.766)     | 0.968 (5.045)     |
| 250 | 70  | 0.936 (4.374) | 0.986 (4.800)  | 0.906 (4.245)     | 0.968 (4.521)     |
| 300 | 70  | 0.934 (4.430) | 0.974 (4.737)  | 0.936 (5.029)     | 0.978 (5.031)     |

Table 1 provides several key insights. First, it illustrates the theoretical claim that our method is primarily governed by the product N = nT . In addition, it indicates that both spline and Legendre bases lead to similar results, with the SCRs constructed using spline bases being slightly wider. This suggests some robustness of our method to the choice of basis functions, which is appealing for practical applications.

Remark 4.1. Note that the empirical coverage probability (ECP) is the mean of binary outcomes. Therefore, we can derive the confidence interval for it. Specifically, the 95% confidence interval for ECP is given by [ p -1 . 96 √ p (1 -p ) / 500 , p +1 . 96 √ p (1 -p ) / 500] , where p denotes the empirical coverage.

## 4.2 Real data application

In this section, we apply our method to the OhioT1DM dataset 3 , which contains records of continuous glucose monitoring (CGM), insulin administration, and self-reported life events for six individuals diagnosed with type 1 diabetes. The data is partitioned into consecutive three-hour intervals and has a three-dimensional state variable S i,t for each patient i at time step t . Due to the space limitation, we leave the specific construction in the supplement. The action A i,t is constructed as a binary variable. A i,t = 1 if the cumulative insulin administered during the interval exceeds one unit; otherwise A i,t = 0 . The discount factor is set as γ = 0 . 5 to weight future outcome. The reward, R i,t , is derived from the Index of Glycemic Control (IGC), a piecewise function that penalizes both hypoglycemia and hyperglycemia while assigning zero cost to glucose values within a clinically optimal range, i.e.,

<!-- formula-not-decoded -->

The downloaded dataset has been separated as training group and testing group. Our objective is to conduct the simultaneous OPE on the testing group under the target policies obtained from the training group. In specific, we evaluate two kinds of target policies on the testing group. The first is an optimal policy π opt obtained by the double fitted Q-iteration algorithm ((Härdle &amp; Song 2010)) in the training group; implementation details are provided in Section E of the supplementary material. The second is the behavior policy b obtained by the random forest from the training data. We then estimate value functions ˆ V ( π opt , S 0 ) and ˆ V ( b, S 0 ) on the testing set by (3.8). SCRs for ˆ V ( π opt , S 0 ) and ˆ V ( b, S 0 ) are constructed for all states in the test set by Algorithm 1.

The results show that ˆ V ( π opt , S 0 ) exceeds ˆ V ( b, S 0 ) by an average of 2.61, and improvements are observed in 87.1% of the initial states. To characterize uncertainty, we examine the proportion of states under which the SCRs do not cover 0 (i.e., the average CGM blood glucose level is not within

3 https://www.kaggle.com/datasets/ryanmouton/ohiot1dm

the normal range) for both target policies. Owing to the uniform property of the SCR, the proportion of states for which the SCRs do not cover zero reflects the fraction of patients who remain in a significantly poor condition under the target policy. The results show that, at the 5% significance level, for policy b , the value function ˆ V ( b, S 0 ) is significantly less than 0 in 90.7% of the states, whereas for policy ˆ V ( π opt , S 0 ) , this proportion is 23.3%. We visualize the SCRs where the upper bound of 95% SCR is below than 0, sorted by the value estimates, in Figure 2. In terms of the average length, for ˆ V ( π opt , S 0 ) , our method yields an averaged length of 27.0, while SAVE with Bonferroni correction produces an AL of 32.4, which is 20% larger than ours. Moreover, for ˆ V ( b, S 0 ) , our method yields an average length of 7.02, compared to 7.58 for SAVE (approximately 8% longer). These findings suggest that, in the medical context, applying reinforcement learning algorithms alongside simultaneous inference could improve health outcomes for patients.

Figure 2: Left: Visualization of values where ˆ V ( b, S 0 ) is sufficiently negative (the upper bound of 95% SCR is below 0). Right: Visualization of values where ˆ V ( π opt , S 0 ) is sufficiently negative (the upper bound of 95% SCR is below 0).

<!-- image -->

## 5 Conclusion and future work

In this work, we present a novel simultaneous statistical inference framework for off-policy evaluation, proving that our SCRs are asymptotically correct via convex Gaussian approximation. The SCRs have widths exceeding pointwise confidence intervals by only a logarithmic factor. This establishes near-optimal efficiency while achieving uniform coverage. The method's validity and efficiency are demonstrated both theoretically and empirically.

The current results are limited to offline settings. Extending this framework to online RL represents a natural next research direction. Additionally, the simultaneous inference framework shows potential for extension to more general Q-learning estimation in RL, including robust value estimation (e.g., Panaganti et al. (2022), Cayci &amp; Eryilmaz (2023)).

## Acknowledgments and disclosure of funding

This work was supported by the High Performance Computing Center, Tsinghua University. Weichi Wu, the corresponding author, is supported by the NSFC No.12271287.

## References

Abdi, H. et al. (2007), 'Bonferroni and šidák corrections for multiple comparisons', Encyclopedia of measurement and statistics 3 (01), 2007.

Cayci, S. &amp; Eryilmaz, A. (2023), 'Provably robust temporal difference learning for heavy-tailed rewards', Advances in Neural Information Processing Systems 36 , 25693-25711.

Chen, X. (2007), 'Large sample sieve estimation of semi-nonparametric models', Handbook of Econometrics 6 , 5549-5632.

- Chen, X. &amp; Christensen, T. M. (2015), 'Optimal uniform convergence rates and asymptotic normality for series estimators under weak dependence and weak conditions', Journal of Econometrics 188 (2), 447-465.
- Cohen, A., Daubechies, I. &amp; Vial, P. (1993), 'Wavelets on the interval and fast wavelet transforms', Applied and computational harmonic analysis .
- Dai, B., Nachum, O., Chow, Y., Li, L., Szepesvári, C. &amp; Schuurmans, D. (2020), 'Coindice: Off-policy confidence interval estimation', Advances in neural information processing systems 33 , 9398-9411.
- Duan, Y., Jia, Z. &amp; Wang, M. (2020), Minimax-optimal off-policy evaluation with linear function approximation, in 'International Conference on Machine Learning', PMLR, pp. 2701-2709.
- Fang, X. (2016), 'A Multivariate CLT for Bounded Decomposable Random Vectors with the Best Known Rate', Journal of Theoretical Probability 29 (4), 1510-1523.
- Fang, X. &amp; Koike, Y. (2024), 'Large-dimensional central limit theorem with fourth-moment error bounds on convex sets and balls', The Annals of Applied Probability 34 (2), 2065 - 2106.
- Feng, Y., Ren, T., Tang, Z. &amp; Liu, Q. (2020), Accountable off-policy evaluation with kernel bellman statistics, in 'International Conference on Machine Learning', PMLR, pp. 3102-3111.
- Hanna, J., Stone, P. &amp; Niekum, S. (2017), Bootstrapping with models: Confidence intervals for off-policy evaluation, in 'Proceedings of the AAAI Conference on Artificial Intelligence', Vol. 31.
- Härdle, W. K. &amp; Song, S. (2010), 'Confidence bands in quantile regression', Econometric Theory 26 (4), 1180-1200.
- Huang, J. Z. (1998), 'Projection estimation in multiple regression with application to functional anova models', The annals of statistics 26 (1), 242-272.
- Huang, J. Z. &amp; Shen, H. (2004), 'Functional coefficient regression models for non-linear time series: A polynomial spline approach', Scandinavian Journal of Statistics 31 (4), 515-534.
- Jiang, N. &amp; Huang, J. (2020), 'Minimax value interval for off-policy evaluation and policy optimization', Advances in Neural Information Processing Systems 33 , 2747-2758.
- Jiang, N. &amp; Li, L. (2016), Doubly robust off-policy value evaluation for reinforcement learning, in 'International conference on machine learning', PMLR, pp. 652-661.
- Kallus, N. &amp; Uehara, M. (2022), 'Efficiently breaking the curse of horizon in off-policy evaluation with double reinforcement learning', Operations Research 70 (6), 3282-3302.
- Lagoudakis, M. G. (2017), Least-Squares Reinforcement Learning Methods , Springer US, Boston, MA, pp. 738-744.
- Le, H., Voloshin, C. &amp; Yue, Y. (2019), Batch policy learning under constraints, in 'International Conference on Machine Learning', PMLR, pp. 3703-3712.
- Luckett, D. J., Laber, E. B., Kahkoska, A. R., Maahs, D. M., Mayer-Davis, E. &amp; Kosorok, M. R. (2020), 'Estimating dynamic treatment regimes in mobile health using v-learning', Journal of the American Statistical Association .
- Marling, C. &amp; Bunescu, R. (2020), 'The ohiot1dm dataset for blood glucose level prediction: Update 2020', CEUR workshop proceedings 2675 , 71-74.
- Matsouaka, R. A., Li, J. &amp; Cai, T. (2014), 'Evaluating marker-guided treatment selection strategies', Biometrics 70 (3), 489-499.
- Murphy, S. A., van der Laan, M. J., Robins, J. M. &amp; Group, C. P. P. R. (2001), 'Marginal mean models for dynamic regimes', Journal of the American Statistical Association 96 (456), 1410-1423.
- Panaganti, K., Xu, Z., Kalathil, D. &amp; Ghavamzadeh, M. (2022), 'Robust reinforcement learning using offline data', Advances in neural information processing systems 35 , 32211-32224.

Puterman, M. L. (1994), 'Markov decision processes', Wiley Series in Probability and Statistics .

- Qiu, H., Luedtke, A. &amp; Carone, M. (2021), 'Universal sieve-based strategies for efficient estimation using machine learning tools', Bernoulli: official journal of the Bernoulli Society for Mathematical Statistics and Probability 27 (4), 2300.
- Quan, M. &amp; Lin, Z. (2024), 'Optimal one-pass nonparametric estimation under memory constraint', Journal of the American Statistical Association 119 (545), 285-296.
- Sallab, A. E., Abdou, M., Perot, E. &amp; Yogamani, S. (2017), 'Deep reinforcement learning framework for autonomous driving', Electronic Imaging 29 (19), 70-76.
- Shi, C., Lu, W. &amp; Song, R. (2020), 'Breaking the curse of nonregularity with subagging-inference of the mean outcome under optimal treatment regimes', Journal of Machine Learning Research 21 (176), 1-67.
- Shi, C., Wan, R., Song, R., Lu, W. &amp; Leng, L. (2020), Does the markov decision process fit the data: Testing for the markov property in sequential decision making, in 'International Conference on Machine Learning', PMLR, pp. 8807-8817.
- Shi, C., Zhang, S., Lu, W. &amp; Song, R. (2021), 'Statistical inference of the value function for reinforcement learning in infinite-horizon settings', Journal of the Royal Statistical Society Series B: Statistical Methodology 84 (3), 765-793.
- Shi, C., Zhu, J., Shen, Y ., Luo, S., Zhu, H. &amp; Song, R. (2024), 'Off-policy confidence interval estimation with confounded markov decision process', Journal of the American Statistical Association 119 (545), 273-284.
- Sun, J. &amp; Loader, C. R. (1994), 'Simultaneous Confidence Bands for Linear Regression and Smoothing', The Annals of Statistics 22 (3), 1328 - 1345.
- Sutton, R. S., Szepesvári, C. &amp; Maei, H. R. (2008), 'A convergent o (n) algorithm for off-policy temporal-difference learning with linear function approximation', Advances in neural information processing systems 21 (21), 1609-1616.
- Thomas, P. &amp; Brunskill, E. (2016), Data-efficient off-policy policy evaluation for reinforcement learning, in 'International conference on machine learning', PMLR, pp. 2139-2148.
- Thomas, P., Theocharous, G. &amp; Ghavamzadeh, M. (2015), High-confidence off-policy evaluation, in 'Proceedings of the AAAI Conference on Artificial Intelligence', Vol. 29.
- Timan, A. F. (2014), Theory of approximation of functions of a real variable , Elsevier.
- Tjøstheim, D. &amp; Auestad, B. H. (1994), 'Nonparametric identification of nonlinear time series: projections', Journal of the American Statistical Association 89 (428), 1398-1409.
- Uehara, M., Shi, C. &amp; Kallus, N. (2022), 'A review of off-policy evaluation in reinforcement learning', arXiv preprint arXiv:2212.06355 .
- Xu, Z., Li, Z., Guan, Q., Zhang, D., Li, Q., Nan, J., Liu, C., Bian, W. &amp; Ye, J. (2018), Largescale order dispatch in on-demand ride-hailing platforms: A learning and planning approach, in 'Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery &amp; data mining', pp. 905-913.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly state the contributions and scope introducing a theoretically justified simultaneous inference framework for off-policy evaluation. These claims are well supported by the theoretical analyses and empirical evaluations presented in the paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The paper discusses several limitations in the final section of the main article. Guidelines:

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

Justification: The paper clearly states the assumptions required for its theoretical results. All theorems are formally stated and proofs are provided in full in the supplementary material.

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

Justification: The experimental setup, including data generation processes, parameter settings, algorithms, and evaluation metrics, is described in detail.

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

Justification: The code is available at https://github.com/xinyuanfan01/SimultaneousStatistical-Inference-for-Off-Policy-Evaluation-in-Reinforcement-Learning. The datasets used were obtained from Kaggle.

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

Justification: This paper provides complete details of the experimental setup, including data generation schemes, model parameters, and the rationale behind their selection (e.g., cross-validation).

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The paper provides a way to measure the randomness of the simulation results in Remark 4.1.

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

Justification: The paper specifies that experiments were conducted on a standard personal computer. The computational requirements are minimal, and all experiments can be run efficiently on CPUs without specialized hardware.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research adheres to the NeurIPS Code of Ethics. The dataset used is publicly available on Kaggle, and the algorithms proposed do not pose known ethical or societal risks.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: There is no societal impact of the work performed.

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

Justification: The paper does not release any high-risk models or datasets and poses no identifiable risk of misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The dataset used is publicly available on Kaggle.

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

Justification: The paper does not introduce any new datasets or models. It proposes a new theoretical framework.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve any human subjects or crowdsourced data collection.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This study does not involve human subjects and does not require IRB approval.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: No large language models (LLMs) were used as part of the method development, experimentation, or analysis.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Supplementary materials for 'Simultaneous Statistical Inference for Off-Policy Evaluation in Reinforcement Learning'

## Tianpai Luo ∗ Xinyuan Fan ∗ Weichi Wu †

Department of Statistics and Data Science

Tsinghua University

Beijing 100084, China

{ltp21, fxy22}@mails.tsinghua.edu.cn , wuweichi@tsinghua.edu.cn

The supplementary materials are organized as follows. In Section A, we discuss the key challenges in policy evaluation for offline reinforcement learning and describe methods for assessing dataset quality. In Section B, we report the additional simulation results described in the main article. In Section C, we detail the construction of the state space in the real data example. In Section D, we present the cross-validation method for selecting the number of basis functions. In Section E, we provide the double fitted Q-learning algorithm. In Section F, we introduce commonly used basis functions and verify the related conditions stated in the main paper. Section G contains the proofs of the theorems, along with the relevant lemmas.

Notations in this supplement are summarized as follows. For a vector v =: ( v 1 , v 2 , . . . , v p ) ∈ R p , let | v | = (∑ p i =1 v 2 i ) 1 / 2 . For a random vector V and probability measure P , denote ∥ V ∥ P ,q =: [ E P ( | V | q )] 1 /q , q &gt; 0 where E P ( · ) is the expectation with respect to probability P . For simplicity, we shall use E ( · ) , ∥ · ∥ q , ∥ · ∥ instead of E P ( · ) , ∥ · ∥ P ,q , ∥ · ∥ P , 2 , respectively if no confusion arises. For a matrix A , the determinant of a matrix A is denoted as det( A ) . If the matrix A is real and symmetric, we use λ min ( A ) ( λ max ( A ) ) to denote the smallest (largest) eigenvalue of A . For any two positive real sequences a n and b n , write a n ≍ b n if there exists 0 &lt; c &lt; C &lt; ∞ such that c ≤ lim inf n →∞ a n /b n ≤ lim sup n →∞ a n /b n ≤ C . We write a n ≲ b n ( a n ≳ b n ) to mean that there exists a universal constant C &gt; 0 such that a n ≤ Cb n ( Ca n ≥ b n ) for all n .

## A Offline Reinforcement Learning: Evaluation Challenges and Dataset Quality

Unlike online environments (e.g., Go or MiniGrid), collecting data through online interaction in many real-world applications, such as healthcare or autonomous driving, can be costly or even hazardous. This limitation hinders the widespread adoption of traditional online RL methods. As an alternative, offline RL leverages large historical datasets, and it is particularly suited to situations where online interaction is infeasible but existing data is available for learning.

Nevertheless, offline RL introduces unique challenges compared to online RL. From the learning perspective, key difficulties include distributional shift, where the behavior policy used to collect the dataset may differ from the target policy being learned, potentially leading to poor performance or overfitting; and sample inefficiency, since learning relies entirely on a static dataset, preventing online exploration. From the evaluation perspective, estimating policy performance without online deployment necessitates off-policy evaluation (OPE), which is the focus of our work. A central challenge in OPE is the gap between estimated off-policy performance and real-world outcomes. Recent methods, such as pointwise confidence intervals, aim to quantify this gap probabilistically.

∗ Equal contribution.

† Corresponding author.

Our work extends these tools from pointwise to global inference, providing more reliable guarantees for decision-making.

The quality of an offline dataset critically affects learning and evaluation. Important factors include state-space coverage, reward signal diversity, action variety, and compatibility between the data distribution and the RL algorithm. As noted in Levine et al. (2020), formalizing a non-trivial sufficiency condition for dataset quality remains an open problem. In our framework, these considerations are captured through Assumptions (A1)-(A3), under which the dataset's quality can be roughly quantified by its sample size N . Our empirical results demonstrate reliable performance of our large-sample theory with dataset sizes around N = 2000 , outperforming modified classical methods such as SA VE when sample sizes are larger. This scale is modest compared to typical offline RL applications: for instance, the HiRID Rodemund et al. (2023) and MIMIC-III Johnson et al. (2016) datasets contain extensive ICU records collected over multiple years, and datasets like D4RL (Datasets for Deep Data-Driven Reinforcement Learning) provide large-scale data for computer science applications. These examples highlight that our approach can be reliably applied across a wide range of real-world scenarios.

## B Additional simulation results

## B.1 Sensitivity analysis for scenario 1

We perform the sensitivity analysis for scenario 1 by taking ( n, T ) = (25 , 50) , (50 , 50) as two illustrative examples and examining the results by varying the specification of K over a relatively wide range. The results are presented in Table S.1. It can be seen that our method is not sensitive to the choice of the number of basis functions K .

Table S.1: Sensitivity analysis under Scenario 1 for different values of K .

| n   | T   | K        | ECP            | AL                |
|-----|-----|----------|----------------|-------------------|
| 25  | 50  | 10 11 12 | 0.92 0.93 0.94 | 0.121 0.127 0.133 |
| 50  | 50  | 20 22    | 0.93 0.94      | 0.120 0.127 0.135 |
|     |     | 24 26    | 0.96 0.95      | 0.143             |

## B.2 SAVE with the Sidak correction for Secnario 2

The results for SAVE with the Sidak correction in Scenario 2 are summarized in Table S.2. Results with high noise and non-Gaussian errors are reported in Table S.3.

Table S.2: Results for Scenario 2 using SA VE with Sidak correction. Format: ECP(AL).

|   n |   T | Legendre SAVE (Sidak)   | Spline SAVE (Sidak)   |
|-----|-----|-------------------------|-----------------------|
|  30 |  50 | 0.980 (9.431)           | 0.976 (9.778)         |
|  50 |  30 | 0.968 (10.476)          | 0.942 (11.115)        |
|  40 |  50 | 0.976 (8.181)           | 0.974 (8.593)         |
|  50 |  40 | 0.990 (8.124)           | 0.984 (7.741)         |
|  50 |  50 | 0.978 (8.237)           | 0.966 (8.617)         |
|  50 | 150 | 0.974 (7.393)           | 0.972 (6.075)         |
|  50 | 200 | 0.974 (6.435)           | 0.966 (7.385)         |
|  50 | 250 | 0.952 (6.223)           | 0.980 (5.431)         |
| 200 |  70 | 0.972 (5.360)           | 0.978 (5.050)         |
| 250 |  70 | 0.972 (4.804)           | 0.958 (4.503)         |
| 300 |  70 | 0.986 (4.726)           | 0.972 (5.020)         |

Table S.3: Results for Scenario 2 with t (8) noises. Format: ECP(AL).

|   n |   T | Legendre SCR   | Legendre SAVE (Bonferroni)   | Legendre SAVE (Sidak)   | Spline SCR    | Spline SAVE (Bonferroni)   | Spline SAVE (Sidak)   |
|-----|-----|----------------|------------------------------|-------------------------|---------------|----------------------------|-----------------------|
|  30 |  50 | 0.940 (4.706)  | 0.934 (4.431)                | 0.980 (4.424)           | 0.936 (3.716) | 0.934 (3.967)              | 0.986 (3.961)         |
|  50 |  30 | 0.944 (2.721)  | 0.932 (3.316)                | 0.990 (3.311)           | 0.934 (3.694) | 0.932 (3.959)              | 0.988 (3.953)         |
|  40 |  50 | 0.946 (3.349)  | 0.934 (3.658)                | 0.992 (3.652)           | 0.952 (3.632) | 0.950 (3.755)              | 0.990 (3.749)         |
|  50 |  40 | 0.932 (2.876)  | 0.928 (3.279)                | 0.990 (3.274)           | 0.944 (3.104) | 0.942 (3.393)              | 0.988 (3.388)         |
|  50 |  50 | 0.934 (3.488)  | 0.926 (3.448)                | 0.986 (3.443)           | 0.934 (2.776) | 0.932 (3.048)              | 0.994 (3.044)         |
|  50 | 150 | 0.948 (2.333)  | 0.944 (2.498)                | 0.988 (2.494)           | 0.948 (2.415) | 0.944 (2.524)              | 0.988 (2.520)         |
|  50 | 200 | 0.942 (2.012)  | 0.946 (2.168)                | 0.984 (2.165)           | 0.936 (2.078) | 0.934 (2.184)              | 0.988 (2.181)         |
|  50 | 250 | 0.932 (1.962)  | 0.934 (2.111)                | 0.994 (2.108)           | 0.950 (2.330) | 0.944 (2.332)              | 0.986 (2.329)         |
| 200 |  70 | 0.946 (1.970)  | 0.996 (2.024)                | 0.996 (2.021)           | 0.944 (2.176) | 0.944 (2.196)              | 0.980 (2.192)         |
| 250 |  70 | 0.948 (1.644)  | 0.980 (1.784)                | 0.980 (1.780)           | 0.944 (1.943) | 0.944 (1.968)              | 0.982 (1.965)         |
| 300 |  70 | 0.930 (1.495)  | 0.986 (1.625)                | 0.986 (1.623)           | 0.948 (1.770) | 0.944 (1.800)              | 0.980 (1.797)         |

## B.3 Comparison with the importance sampling method for Scenario 2

We evaluated our method against the importance sampling (IS) approach (Jiang &amp; Li 2016, Hanna et al. 2017) under Scenario 2. Specifically, we set S 0 = ( -2+0 . 4 i, -2+0 . 4 j ) ⊤ where 0 ≤ i, j ≤ 10 . For each combination of i and j , we generated n 0 = 100 trajectories, each of length 10, while keeping all other settings unchanged. For bootstrapping IS, we employed the algorithm from Hanna et al. (2017), which provides confidence intervals for each V ( π, S 0 ) . The Bonferroni correction was then applied to obtain the simultaneous confidence bands (SCB).

It is worth noting that IS approach estimates the value function by directly reweighting trajectories, whereas in Scenario 2, the target policy π (1 | s ) = I ( s (1) &gt; 0 , s (2) &gt; 0) is discontinuous in s . Moreover, the choice of target policy frequently results in weights of zero, reducing the effective sample size and causing the IS estimates to be dominated by a small subset of samples. This leads to biased estimates and, consequently, degraded performance. We mention that Hanna et al. (2018) also highlighted the same issue in their Section 7. Our method is not impacted by this problem, further showcasing its practical applicability.

From the results, under this setting, the IS method achieves an empirical coverage probability (ECP) of 0.638, with an average length (AL) of 5.300. This is below the nominal level of 0.95. Increasing the sample size can improve the coverage of the IS method, however, the associated bootstrap procedure becomes computationally expensive. In contrast, our method performs well even with a smaller sample size ( n 0 = 10 ), achieving an empirical coverage probability (ECP) of 0.954 and an average length (AL) of 4.575.

## C Specific construction for states in the real data example

In the real data example in the main article, we construct a three-dimensional state variable S i,t for each patient i at time step t . Specifically, S (1) i,t represents the average CGM blood glucose level over the preceding three-hour interval. S (2) i,t is a decayed sum of carbohydrate intake within the same period, where each meal's carbohydrate estimate is discounted according to its temporal distance from the current interval. Specifically, if meals are recorded at times t 1 , t 2 , . . . , t N ∈ [ t -1 , t ) with corresponding carbohydrate estimates CE 1 , CE 2 , . . . , CE N , then S (2) i,t = ∑ N j =1 CE j · 0 . 5 36( t j -t +1) .

S (3) i,t denotes the average basal rate over the same three-hour window, capturing the background level.

## D Cross-validation for choosing the number of basis functions

The method of cross-validation is widely used in machine learning and sieve methods (see, for example, Van Der Laan &amp; Dudoit (2003), Hansen (2014), Bates et al. (2024)). Based on the key equation (3.5) in the main article, we adopt the following 5-fold cross-validation approach, as described in Algorithm S.1.

Algorithm S.1 5-Fold Cross-Validation

- 1: Input: Observed data D = { ( R i,t , A i,t , S i,t , S i,t +1 ) } 0 ≤ t ≤ T i , 1 ≤ i ≤ n ; candidate set of choices K can = { k 1 , . . . , k l } .
- 2: Randomly partition D into 5 approximately equal-sized folds: D 1 , D 2 , D 3 , D 4 , D 5 .
- 3: for j = 1 to 5 do
- 4: Set the j -th fold as validation set: D val ←D j .

̸

- 5: Set the remaining 4 folds as training set: D train ← ⋃ i = j D i .
- 6: for k = 1 to l do
- 7: Obtain ˆ β ( j,k ) π based on D train using equation (3.6) with k k basis functions.
- 8: Let

<!-- formula-not-decoded -->

- 9: end for

- 10: end for

- 11: Let k ∗ = arg min k =1 , ··· ,l ∑ 5 j =1 CV ( j, k ) .

- 12: Output: Select k k ∗ as the number of basis functions.

## E Double-fitted Q-iteration algorithm

The double-fitted Q-iteration algorithm (Hasselt 2010) is presented in Algorithm S.2. The Q-function Q ( · , · ; θ ) can be specified using any model indexed by θ , and we use a linear model with basis functions to approximate Q .

## Algorithm S.2 Double Fitted Q-Iteration Algorithm

- 1: Input: Observed data { ( R , A , S , S ) } ; initialize parameters θ , θ
- 2: repeat
- 3: Step 1: For all i, t , compute target values:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 4: Step 2: Update parameters by minimizing squared errors:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 5: until convergence
- 6: Output: Learned parameter ˆ θ A .

## F Sieve method

In this section, we introduce commonly used sieve basis and verify the related conditions in the main paper. We list several commonly used sieve basis as follows, which can be used in our simultaneous inference framework.

Example F.1 (Legendre) . Define Legendre polynomials

<!-- formula-not-decoded -->

Then continuous function f ( x ) on [ -1 , 1] can be written as

<!-- formula-not-decoded -->

- i,t i,t i,t i,t +1 0 ≤ t ≤ T i , 1 ≤ i ≤ n ˆ A ˆ B .

Example F.2 (Fourier) . Consider real-valued function f ( x ) ∈ L 2 [ -1 , 1] i.e. ∫ 1 -1 f ( x )d x &lt; ∞ . By Fourier transformation, f ( x ) can be written as

<!-- formula-not-decoded -->

where { ϕ j ( x ) } ∞ j = -∞ = { (cos( jπx ) + i sin( jπx )) / √ 2 } ∞ j = -∞ forms an orthonormal basis for L 2 [ -1 , 1] .

Example F.3 (Harr wavelet) . The Haar sequence was proposed in 1909 by Haar (1910). Haar used these functions to give an example of an orthonormal system for the space of square-integrable functions. For every pair n, k of integers in Z , the Haar function h n,k is defined on the real line R by the formula

<!-- formula-not-decoded -->

where h ( t ) is the Harr wavelet's mother wavelet function

<!-- formula-not-decoded -->

The Haar system on the real line is the set of functions

<!-- formula-not-decoded -->

which is an orthonormal basis.

Example F.4 (Daubechies wavelet) . For N ∈ N , a Daubechies mother wavelet of class DaubechiesN is a function ϕ ∈ L 2 ( R ) defined by

<!-- formula-not-decoded -->

where h 0 , h 1 , · · · , h 2 N -1 ∈ R are constant and satisfy ∑ N -1 k =0 h 2 k = 1 √ 2 = ∑ N -1 k =0 h 2 k +1 , as well as, for l = 0 , 1 , · · · , N -1 ,

̸

<!-- formula-not-decoded -->

The φ ( x ) is the scaling wavelet function supported on [0 , 2 N -1) and satisfies the recursion equation φ ( x ) = √ 2 ∑ 2 N -1 k =0 h k φ (2 x -k ) , as well as the normalization ∫ R φ ( x ) dx = 1 , ∫ R φ (2 x -k ) φ (2 x -l ) dx = 0 , k = l . As listed in Daubechies (1992), the filter coefficients h 0 , . . . , h 2 N -1 can be efficiently computed. The order N decides the support [0 , 2 N -1) and provides the regularity condition

<!-- formula-not-decoded -->

The Harr wavelet as introduced above can be regarded as a special Daubechies wavelet with N = 1 . In our simulations and data analysis, we employ Daubechies wavelet with a sufficiently high order N to construct a sequence of orthogonal sieve basis as proposed in Daubechies (1988). For a given J n and J 0 , we consider the following periodized wavelets on [0 , 1]

<!-- formula-not-decoded -->

or equivalently, by Yves (1989),

̸

<!-- formula-not-decoded -->

The 2 J n equals to our basis number K . Additionally, we refer to Chen (2007) for a more general example of orthogonal wavelets.

## F.1 Sieve approximation

For the approximation (3.4), we show that the sieve method can approximate any function in the Hölder space with smoothness p . Given d -tuple α = ( α 1 , . . . , α d ) of nonnegative integers and [ α ] = α 1 + · · · + α d , the Hölder space with smoothness p , Λ p C ( S ) , is defined as

̸

<!-- formula-not-decoded -->

where C &gt; 0 is a constant, p = m + γ, γ ∈ (0 , 1] , C m ( S ) is the class of m -times continuously differentiable real-valued functions on S , and the differential operator

<!-- formula-not-decoded -->

For function Q ( π ; · , a ) ∈ Λ p C ( S ) , sup s ∈S ,a ∈A | Q ( π ; s, a ) -Φ( s ) ⊤ β ∗ π,a | = O ( K -p/d ) if Φ( s ) is the tensor product of sieve bases such as B-splines, Legendre polynomials, orthogonal wavelets, or Fourier series if it is periodic; see Section 2.3.1 in Chen (2007) or Timan (2014),Yves (1989),Chen (2007). As discussed in Shi, Wan, Chernozhukov &amp; Song (2021), there exists some transition density function q such that P (d s ′ , a ) = q ( s ′ | s, a )d s if the transition kernel P ( ·| s, a ) is absolutely continuous with respect to the Lebesgue measure. The following Lemma shows that Q ( π ; · , a ) ∈ Λ p C ( S ) if q ( s ′ |· , a ) and reward r ( s, a ) follow certain mild conditions.

Lemma F.1 (Lemma 1 in Shi, Zhang, Lu &amp; Song (2021)) . If there exist some p, C &gt; 0 such that r ( · , a ) , q ( s ′ |· , a ) ∈ Λ p C ( S ) for any a ∈ A , s ′ ∈ S , then there exists constant C ′ &gt; 0 such that Q ( π ; ˙ ,a ) ∈ Λ p C ′ ( S ) for any policy π and a ∈ A .

## F.2 Geometric properties of sieve space

In this section, we verify the condition (3.15) in Proposition 3.2. Condition (3.15) are simplified requirement on the sieve basis which will yield a polynomial rate N c ( c ≥ 0 ) for the geometric quantities, including volume, curvature, and boundary of the manifold { Φ( s ) / | Φ( s ) | : s ∈ S} . For simplicity, we only verify ∫ S λ d/ 2 min ( M ⊤ M )d s ≳ N c in condition (3.15). We refer to Assumption 4 of Chen &amp; Christensen (2015) and Example 1-2 in Quan &amp; Lin (2024) for the rest polynomial rate conditions in (3.15). Define ξ K,N =: sup s ∈S | Φ( s ) | and ∆ K,N =: sup s ∈S |∇ Φ( s ) | . Then there exists ¯ ω, ω 0 , ω 1 , ω ′ 1 ≥ 0 s.t. ξ K,N ≲ N ω 1 , and ∆ K,N ≲ N ω ′ 1 and N ¯ ω ≪ K ≪ N ω 0 .

Lemma F.2 (Lemma E.1 in Shi, Zhang, Lu &amp; Song (2021)) . There exists some constant c ∗ ≥ 1 such that

<!-- formula-not-decoded -->

We verify condition (3.15) using trigonometric basis functions as a representative example. A similar procedure applies to other types of basis functions.

Suppose that d = 1 , S = [ -π, π ] , the number of basis functions is K = 2 ˜ K + 1 , ˜ K ≥ 1 , and Φ( s ) = (1 , sin( s ) , cos( s ) , · · · , sin( ˜ Ks ) , cos( ˜ Ks )) ⊤ . Then | Φ( s ) | = √ ˜ K +1 , M ( s ) = (0 , cos( s ) , -sin( s ) , · · · , ˜ K cos( ˜ Ks ) , -˜ K sin( ˜ Ks )) ⊤ / √ ˜ K +1 , and M ⊤ M = ∑ ˜ K i =1 i 2 / ( ˜ K +1) ≳ ˜ K 2 . As a result, we have ∫ S λ min ( M ⊤ M )d s ≳ K ≳ N ¯ ω .

Now consider the case where d = 2 and S = [ -π, π ] 2 . Suppose that the number of basis functions is K = (2 ˜ K + 1) 2 . Then Φ( s ) = ϕ ( s 1 ) ⊗ ϕ ( s 2 ) where ϕ ( s ) = (1 , sin( s ) , cos( s ) , · · · , sin( ˜ Ks ) , cos( ˜ Ks )) ⊤ . M 1 ( s ) = ψ ( s 1 ) ⊗ ϕ ( s 2 ) / ( ˜ K +1) , M 2 ( s ) = ϕ ( s 1 ) ⊗ ψ ( s 2 ) / ( ˜ K +1) where ψ ( s ) = (0 , cos( s ) , -sin( s ) , · · · , ˜ K cos( ˜ Ks ) , -˜ K sin( ˜ Ks )) ⊤ .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then we have ∫ S λ min ( M ⊤ M )d s ≳ K ≳ N ¯ ω .

## G Technical proofs

## G.1 Dependence measure and geometric ergodicity

To measure the dependence in the Markov chain, we introduce the concept of physical dependence measure in Wu (2005). For a pair of jointly distributed random variables ( X,Y ) , let F XY ( x, y ) = P ( X ≤ x, Y ≤ y ) , x, y ∈ R , be the joint distribution function and F Y | X ( y | x ) = P ( Y ≤ y | X = x ) the conditional distribution function of Y given X = x . For u ∈ (0 , 1) , define the conditional quantile function G ( x, u ) = inf { y ∈ R : F Y | X ( y | x ) ≥ u } . Let U be a uniform (0 , 1) distributed random variable and assume that U and X are independent. Then we can view Y as the outcome of the bivariate function Y = d G ( X,U ) such that

<!-- formula-not-decoded -->

For many standard constructions of stochastic processes (see e.g. Deák (1990), chapter 5), a stochastic process { X t } can be represented as

<!-- formula-not-decoded -->

where H 1 , . . . , H n are measurable functions. The above representation can characterize many Markov sequences (see e.g. Rüschendorf &amp; de Valk (1993)). If { X t } is a Markov chain, then the conditional quantile G t ( X t -1 , U t ) can be viewed as a function of X t -1 . Wiener (1958) first considered this representation problem for stationary and ergodic processes. For a Markov chain { X t } with the form X i = G i ( X i -1 , U i ) , Wu &amp; Mielniczuk (2010) asserts that, there exists a copy ˜ X i of X i such that ( ˜ X i ) i ∈ Z = d ( X i ) i ∈ Z and ˜ X i is expressed as H i ( . . . , U i -1 , U i ) , a function of iid random variables.

Lemma G.1 (Theorem 4.1 in Wu &amp; Mielniczuk (2010)) . Assume that { X i } satisfies the recursion

<!-- formula-not-decoded -->

where U i are i.i.d. standard uniform random variables. Here F i are independent random maps F i ( x ) = G i ( x, U i ) . Assume that for some α &gt; 0 we have

̸

<!-- formula-not-decoded -->

and for some x 0 ,

<!-- formula-not-decoded -->

Then the backward iteration F i ◦ F i -1 ◦ F i -2 . . . converges almost surely and the limit forms a non-stationary Markov chain which is a solution to (S.3) .

Let ε i , i ∈ Z , be i.i.d. random variables and let H i = ( . . . , ζ i -1 , ζ i ) . Based on Lemma G.1, we can consider the irreducible and aperiodic Markov chain { X i } as

<!-- formula-not-decoded -->

where H i are measurable functions. We view ξ i as the input and X i as the output of the system. If H i does not depend on i , i.e., H i ≡ H , then the process ( X i ) is stationary. Then we introduce the following definition to measure the dependence of { X i } in (S.4): Let { ζ ′ i } be an i.i.d. copy of { ζ i } , denote H i,k = ( H i -k -1 , ζ ′ i -k , ζ i -k +1 , . . . , ζ i ) .

Definition G.1 (physical dependence measure) . Assume that sup i ∥ H i ( H i ) ∥ q &lt; ∞ for q &gt; 0 , then we can define the physical dependence measure of { X i } as

<!-- formula-not-decoded -->

where H i,k = ( H i -k -1 , ζ ′ i -k , ζ i -k +1 , . . . , ζ i ) .

Note that the Lemma G.1 and Definition G.1 allow a nonstationary Markov chain; our theorem can actually be generalized to nonstationary cases, which can be a promising future work.

For our stationary state observation { S 0 ,t } , by Lemma G.1, our geometric ergodicity assumes { S 0 ,t } is an irreducible and aperiodic Markov chain where there exists { ˜ S 0 ,t } = S ( H t ) in the form of (S.4) such that { ˜ S 0 ,t } = d { S 0 ,t } and the physical dependence measure is geometrically decaying, i.e. δ S ( k, 1) = O ( χ k ) for some constant χ ∈ (0 , 1) . In fact, for contracting Markov chains (e.g., autoregressive models), this assumption generally holds. Notice that

<!-- formula-not-decoded -->

δ S ( k, q ) = O ( χ k ) holds for any given q ∈ N + since state space S is bounded. Furthermore, denote Φ( S 0 ,t ) = G ( H t ) = Φ ◦ S ( H t ) , then the physical dependence measure

<!-- formula-not-decoded -->

Note that | Φ( S 0 ,t ) | ≤ sup s | Φ( s ) | = ξ K,N . Using the fact min { x, 1 } ≤ x α , x ≥ 0 for any given α ∈ (0 , 1) , we can have δ Φ ( k, q ) = O ( ξ K,N ∆ α K,N χ αk ) for any given α ∈ (0 , 1) .

## G.2 Proof of Theorem 3.1

We introduce the following lemmas before proving Theorem 3.1. In the proof of Theorem 3.1 and the following Lemmas, we will omit the subscript π in U π ( · ) , u π ( · ) , Σ π , ˆ Σ π , ˆ β π , β ∗ π , ω π , etc, for brevity. For simplicity, we deduce our proof under the condition (3.4). In other words, we consider the Q-function Q ∗ ( π ; s, a ) = Φ( s ) ⊤ β ∗ π,a instead of Q ( π ; s, a ) . This can be achieved when the Q -function is smoothing enough as discussed in Section F.1. We denote the dimension of ˆ β as p =: mK where p ≍ K since m is fixed. For the convex Gaussian approximation, we introduce a smoothed function

<!-- formula-not-decoded -->

where ω ∈ R p , convex set A ⊂ R p , and

<!-- formula-not-decoded -->

To show the convex Gaussian approximations, we introduce the following Lemmas.

Lemma G.2 (Lemma 5.3 in Fang &amp; Koike (2024)) . For any p -dimensional random vector W ,

<!-- formula-not-decoded -->

where Z is a p -dimensional Gaussian random vector with invertible covariance matrix and O is the collection of all the convex sets in R p .

Lemma G.3 (Theorem 2.1 in Fang (2016)) . Let W = ∑ n i =1 X i be a sum of p -dimensional random vectors such that E ( X i ) = 0 and Cov( W ) = Σ . Suppose W can be decomposed as follows:

1. ∀ i ∈ [ n ] , ∃ i ∈ N i ⊂ [ n ] , such that W -X N i is independent of X i , where [ n ] = { 1 , · · · , n } . 2. ∀ i ∈ [ n ] , j ∈ N i , ∃ N i ⊂ N ij ⊂ [ n ] , such that W -X N ij is independent of { X i , X j } . 3. ∀ i ∈ [ n ] , j ∈ N i , k ∈ N ij , ∃ N ij ⊂ N ijk ⊂ [ n ] such that W -X N ijk is independent of { X i , X j , X k } .

Suppose further that for each i ∈ [ n ] , j ∈ N i , k ∈ N ij ,

<!-- formula-not-decoded -->

where | · | is the Euclidean norm of a vector. Then there exists a universal constant C such that

<!-- formula-not-decoded -->

where Z is a p -dimensional Gaussian random vector preserving the covariance structure of W and where O denotes the collection of all the convex sets in R p .

Lemma G.4 (Lemma E.2 in Shi, Zhang, Lu &amp; Song (2021)) . Suppose the conditions in Theorem 3.1 hold. We have as N → ∞ that ∥ ∥ Σ -1 ∥ ∥ F ≤ 3¯ c -1 , ∥ Σ ∥ F = O (1) , ∥ Σ -Σ ∥ F = O p { K 1 / 2 ( nT ) -1 / 2 log N } , ∥ ∥ ∥ ̂ Σ -1 -Σ -1 ∥ ∥ ∥ F = O p { K 1 / 2 N -1 / 2 log N } and ∥ ∥ ∥ ̂ Σ -1 ∥ ∥ ∥ F ≤ 6¯ c -1 with probability approaching 1.

Lemma G.5 (Lemma E.3 in Shi, Zhang, Lu &amp; Song (2021)) . Suppose the conditions in Theorem 3.1 hold. As N → ∞ , we have λ max ( T -1 ∑ T -1 t =0 E ξ 0 ,t ξ ⊤ 0 ,t ) = O (1) , λ max { N -1 ∑ n i =1 ∑ T -1 t =0 ξ i,t ξ ⊤ i,t } = O p (1) , λ min ( T -1 ∑ T -1 t =0 E ξ 0 ,t ξ ⊤ 0 ,t ) ≥ ¯ c/ 2 and λ min { N -1 ∑ n i =1 ∑ T -1 t =0 ξ i,t ξ ⊤ i,t } ≥ ¯ c/ 3 with probability approaching 1.

Proof of Theorem 3.1. By definition and the arguments in Section F.1, we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Denote G i,t as the sub-dataset { S i,t , A i,t } ∪ { ( R i,j , A i,j , S i,j ) } 1 ≤ j&lt;t . By the Bellman equation and conditions (2.1) and (2.2), we have

<!-- formula-not-decoded -->

Recall the definition ξ i,t = ξ ( S i,t , A i,t ) in (3.5), we have for any 0 ≤ t 1 &lt; t 2 ≤ T -1

<!-- formula-not-decoded -->

Therefore, for any 0 ≤ t 1 &lt; t 2 ≤ T -1 and 1 ≤ i 1 &lt; i 2 ≤ n we have E ε i 1 ,t 1 ε i 2 ,t 2 ξ ⊤ i 1 ,t 1 ξ i 2 ,t 2 = 0 and

<!-- formula-not-decoded -->

By Assumption (A3) and Lemma F.2, we have

<!-- formula-not-decoded -->

By Markov's inequality, N -1 ∑ n i =1 ∑ T -1 t =0 ξ i,t ε i,t = O p ( √ K/N ) , and together with Lemma G.4, we have

<!-- formula-not-decoded -->

In the following, we show the convex Gaussian approximation on Σ -1 ( 1 √ N ∑ n i =1 ∑ T -1 t =0 ξ i,t ε i,t ) . Denote Z N = 1 √ N ∑ n i =1 ∑ T -1 t =0 z i,t and z i,t = ξ i,t ε i,t , define the truncated z i,t as

<!-- formula-not-decoded -->

Denote ¯ Z N = ∑ n i =1 ∑ T -1 t =0 ¯ z i,t / √ N and ¯ Z ∗ N =: ¯ Z N -E ¯ Z N . Suppose T/m = k ∈ N w.l.o.g., and define ¯ z ( m ) i,t =: E (¯ z i,t |F m ( t )) , (S.13)

where F m ( t ) = σ ( ζ t -m +1 , . . . , ζ t ) . Then ¯ z ( m ) i,k , ¯ z ( m ) i,j are independent as long as | k -j | &gt; m . Further let ¯ Z ( m ) N =: ∑ n i =1 ∑ T -1 t =0 ¯ z ( m ) i,t / √ N and ˜ Z ( m ) N =: ¯ Z ( m ) N -E ¯ Z ( m ) N , then ¯ Z ∗ N -˜ Z ( m ) N = ¯ Z N -¯ Z ( m ) N . ( m ) ¯ ( m )

Denote the covariance matrices Ω and Ω of Z N and Z N respectively. Introduce a p -dimensional standard Gaussian random vector G and denote

<!-- formula-not-decoded -->

so that G N and ˜ G ( m ) N preserve the covariance structure Ω , Ω ( m ) , respectively. We then introduce the convex Kolmogorov distance to measure the convex distribution probability difference between p -dimensional random vectors X and Y ,

<!-- formula-not-decoded -->

where O is the collection of all the convex sets in R p . Combining (S.9) and (S.11), it suffices to show K ( Z N , G N ) = o (1) . By Lemma G.2 and |∇ h A,ϵ | ≤ 2 ϵ -1 , we can decompose the K ( Z N , G N ) as

<!-- formula-not-decoded -->

Based on decomposition (S.15), for q &gt; 4 and appropriate m ≍ log n , we shall prove the following assertions as follows:

- (1) Truncation error
- (2) M-decomposition error
- (3) Gaussian comparison
- (4) Gaussian approximation

<!-- formula-not-decoded -->

Truncation error In view of E Z N = 0 and z i,t -¯ z i,t = z i,t 1 | z i,t | &gt;π N , we have for q &gt; 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which yields (S.16) using the fact ∥ z i,t ∥ q = O (sup s | Φ( s ) | ) = O ( ξ K,N ) for any given q ∈ N + . Moreover, we can also have for q &gt; 1 ,

<!-- formula-not-decoded -->

where T N,q =: √ nπ 1 -q N ∥ z i,t ∥ q 2 q .

M-decomposition error Denote operator P ( k ) ¯ z i,t =: E (¯ z i,t |F k ( t )) -E (¯ z i,t |F k -1 ( t )) using the fact ¯ z i,t = lim j →∞ E (¯ z i,t |F t + j ( t )) , we have on a richer space,

<!-- formula-not-decoded -->

where R i,T,j =: ∑ T i =( m -j +1) ∨ 1 P ( t + j ) ¯ z i,t . By Jensen's inequality, for q &gt; 1 ,

<!-- formula-not-decoded -->

Note that for given i , process { R i,T,j , j ≥ m -n + 1 } is martingale difference with respect to filtration σ ( ζ -j +1 , ζ -j +2 , . . . ) . If q ≥ 2 , by Burkholder's inequality, there exists constant C q &gt; 0 such that

<!-- formula-not-decoded -->

Using the fact δ Φ ( k, q ) = O (∆ α K,N χ αk ξ K,N ) for any given α ∈ (0 , 1) , (S.23) yields

<!-- formula-not-decoded -->

Combining (S.22), (S.23), (S.24), and (S.25) elementary calculation yields

<!-- formula-not-decoded -->

Setting appropriate m-decomposition m ≍ log N (e.g., m = ( q -4) | ω 1 + ω 0 / 12 -1 / 6 | log N α log(1 /χ ) ), we have for q &gt; 4 ,

<!-- formula-not-decoded -->

with α &lt; min { 1 , q -4 2 ω ′ 1 | ω 1 + ω 0 / 12 -1 / 6 |} .

Gaussian comparison For a matrix A , denote ∥ A ∥ F as the Frobenius norm of A i.e. ∥ A ∥ F = ( tr( A ⊤ A ) ) 1 / 2 . Recall Ω = E ( ∑ n i =1 ∑ T -1 t =0 z i,t )( ∑ n i =1 ∑ T -1 t =0 z ⊤ i,t ) /N and

<!-- formula-not-decoded -->

Consider the difference of covariance matrix between ¯ Z ( m ) N and Z N based on Frobenius norm, using the fact E Z N = 0 and E ¯ Z ( m ) N = E ¯ Z N ,

<!-- formula-not-decoded -->

By (S.21) and (S.26),

<!-- formula-not-decoded -->

and similarly, we also have

<!-- formula-not-decoded -->

Besides, by (S.20),

<!-- formula-not-decoded -->

Combining (S.29), (S.30), (S.31),

<!-- formula-not-decoded -->

By Assumption (A3) and Lemma G.5, we have inf s,a ω ( s, a ) ≥ c -1 0 and

<!-- formula-not-decoded -->

which yields

<!-- formula-not-decoded -->

Combing (S.32), (S.34) and sub-multiplicativity of Frobenius norm, we have

<!-- formula-not-decoded -->

By similar arguments in (S.27), ξ K,n ∆ α K,n χ αm = o ( T n,q ) by appropriate m and α , which yields (S.18).

Gaussian approximation Plug n 1 = m , n 2 = 2 m , n 3 = 3 m , κ = N -1 / 2 π N into Lemma G.3 with (S.34), we have

<!-- formula-not-decoded -->

Moreover, in the proof of Lemma G.3, equation (4.23) in Fang (2016) yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let π N = √ κ 1 κ 2 , where

<!-- formula-not-decoded -->

Combining (S.16), (S.17), (S.18), (S.19), and (S.15),

<!-- formula-not-decoded -->

Therefore, with appropriate ϵ 1 and q &gt; 4 , we have when K = o ( n 2 / 7 -c ) for any given c &gt; 0 ,

<!-- formula-not-decoded -->

Furthermore, combining Lemma G.4 and using the fact K ≪ √ N/ log N , by similar arguments in (E.29) of Shi, Wan, Chernozhukov &amp; Song (2021), one can show ∥ ˆ Σ -1 ˆ Ω( ˆ Σ ⊤ ) -1 -Σ -1 ΩΣ -1 ∥ F = o p (1) , which yields the validation of the Bootstrap algorithm from the Slutsky's theorem.

## G.3 Proof of Proposition 3.2

Proof. It suffices to find C α,N such that as N →∞ ,

<!-- formula-not-decoded -->

where T ( s ) = l ( s ) / | l ( s ) | = U ( s ) ⊤ Λ 1 / 2 / √ U ( s ) ⊤ Λ U ( s ) and G is standard p -dimensional random vector. Denote manifold

<!-- formula-not-decoded -->

and let κ 0 be the volume of the manifold M , and ζ 0 be the area of the boundary ∂ M ; Let κ 2 and ζ 1 be measures of the curvature of M and ∂ M respectively, and m 0 measures the rotation angles in the regions ∂ 2 M .

By Proposition 3 in Sun &amp; Loader (1994), for the α in (S.40), we have

<!-- formula-not-decoded -->

where χ 2 d is the chi-square random variable with the degree of freedom d .

̸

where

To bound the positive geometric quantities κ 0 , ζ 0 , κ 2 , ζ 1 , m 0 appearing in (S.42), we give the following formulations for numerical computation. For simplicity, we suppose S = [0 , 1] d and the boundary ∂ S consists of those points s with exactly one component 0 or 1. The regions where two faces of ∂ S meet are denoted ∂ 2 S . Denote matrix A = ( T 1 ( s ) , . . . , T d ( s )) where T j ( s ) = ∂ T ( s ) /∂x j with s = ( x 1 , . . . , x d ) ⊤ and indicator vector e j = ( e j, 1 , . . . , e j,d ) ⊤ such that e j,j = 1 and e j,k = 0 if k = j . By (3.2) and (3.3) in Sun &amp; Loader (1994), the κ 0 and κ 2 can be computed as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

For ζ 1 , ζ 0 measuring the boundary ∂ M , by (3.4) in Sun &amp; Loader (1994) and the second and third equation on Page 1335 of Sun &amp; Loader (1994),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

where indicator vector e ∗ j = ( e j, 1 , . . . , e j,d -1 ) ⊤ such that e j,j = 1 and e j,k = 0 if k = j .

<!-- formula-not-decoded -->

on the face s ∈ ∂ S at which s d is maximized, with similar definitions for ζ 1 ( s ) , U j ( s ) , A ∗ on other faces where s j is maximized. Moreover, by the fifth and seventh equations on Page 1335 of Sun &amp; Loader (1994),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

at a point s at the meeting of the faces s d -1 = 1 and s d = 1 , with similar definitions for m 0 ( s ) on other meetings of the two faces of ∂ S .

Denote ˜ Φ( s ) = Λ 1 / 2 U ( s ) and ∂ s j ˜ Φ( s ) = ∂ ˜ Φ( s ) /∂s j , then basic calculation yields

<!-- formula-not-decoded -->

Using the fact det 1 /d ( A ⊤ A ) ≤ tr( A ⊤ A ) /d and tr( A ⊤ A ) = ∑ d j =1 | T j ( s ) | 2 , by (S.43), (S.45), (S.48). Note that | ˜ Φ( s ) | ≥ √ λ min (Λ) | Φ( s ) | ≳ n c 0 by condition (3.15), Assumptions (A2) and (A3), there exists constant c 1 &gt; 0 such that

<!-- formula-not-decoded -->

and similarly, ζ 0 = O ( N c 1 ) , m 0 = O ( N c 1 ) since tr( A ⊤ ∗∗ A ∗∗ ) ≤ tr( A ⊤ ∗ A ∗ ) ≤ tr( A ⊤ A ) . For κ 2 , note that matrix A ( A ⊤ A ) -1 A ⊤ is idempotent, thus

<!-- formula-not-decoded -->

By condition (3.15), for some constant c 2 &gt; 0 ,

<!-- formula-not-decoded -->

thus κ 2 = O ( N c 3 ) and similarly, we have ζ 1 = O ( N c 3 ) for some constant c 3 &gt; 0 .

To sum up, there exists constant ¯ c &gt; 0 such that

<!-- formula-not-decoded -->

By Theorem 6 in Zhang &amp; Zhou (2020), for constants ˜ c, ˜ C, ¯ C &gt; 0 , the tail bounds of χ 2 d

<!-- formula-not-decoded -->

Combining (S.50), (S.42), (S.51) and the fact α is a fixed value, we have n ¯ c exp( -˜ CC 2 α,N ) ≳ 1 , which implies that C α,N = O ( √ log N ) . On the other hand, by condition (3.15), there exists constant c &gt; 0 , such that

<!-- formula-not-decoded -->

Combining (S.42), (S.51) and the fact α is a fixed value, we have N c exp( -˜ CC 2 α,N ) ≲ 1 , which shows C α,N ≳ √ log N . Combining (S.40),(3.11), and (3.4), we have appropriate C α,N ≍ log 1 / 2 N such that (3.16) holds.

## References

- Bates, S., Hastie, T. &amp; Tibshirani, R. (2024), 'Cross-validation: what does it estimate and how well does it do it?', Journal of the American Statistical Association 119 (546), 1434-1445.
- Chen, X. (2007), 'Large sample sieve estimation of semi-nonparametric models', Handbook of Econometrics 6 , 5549-5632.
- Chen, X. &amp; Christensen, T. M. (2015), 'Optimal uniform convergence rates and asymptotic normality for series estimators under weak dependence and weak conditions', Journal of Econometrics 188 (2), 447-465.
- Daubechies, I. (1988), 'Orthonormal bases of compactly supported wavelets', Communications on pure and applied mathematics 41 (7), 909-996.
- Daubechies, I. (1992), Ten Lectures on Wavelets , Society for Industrial and Applied Mathematics.
- Deák, I. (1990), 'Random number generators and simulation', Mathematical methods of operation research .
- Fang, X. (2016), 'A Multivariate CLT for Bounded Decomposable Random Vectors with the Best Known Rate', Journal of Theoretical Probability 29 (4), 1510-1523.
- Fang, X. &amp; Koike, Y. (2024), 'Large-dimensional central limit theorem with fourth-moment error bounds on convex sets and balls', The Annals of Applied Probability 34 (2), 2065 - 2106.
- Haar, A. (1910), 'Zur theorie der orthogonalen funktionensysteme', Mathematische Annalen 69 , 331371.
- Hanna, J. P., Stone, P. &amp; Niekum, S. (2018), 'Bootstrapping with models: Confidence intervals for off-policy evaluation'.
- Hanna, J., Stone, P. &amp; Niekum, S. (2017), Bootstrapping with models: Confidence intervals for off-policy evaluation, in 'Proceedings of the AAAI Conference on Artificial Intelligence', Vol. 31.
- Hansen, B. E. (2014), 'Nonparametric sieve regression: Least squares, averaging least squares, and cross-validation'.
- Hasselt, H. (2010), 'Double q-learning', Advances in neural information processing systems 23 .
- Jiang, N. &amp; Li, L. (2016), Doubly robust off-policy value evaluation for reinforcement learning, in 'International conference on machine learning', PMLR, pp. 652-661.
- Johnson, A., Pollard, T. &amp; Mark III, R. (2016), 'Mimic-iii clinical database (version 1.4). physionet. 2016', Available from:[DOI] .
- Levine, S., Kumar, A., Tucker, G. &amp; Fu, J. (2020), 'Offline reinforcement learning: Tutorial, review, and perspectives on open problems', arXiv preprint arXiv:2005.01643 .

- Quan, M. &amp; Lin, Z. (2024), 'Optimal one-pass nonparametric estimation under memory constraint', Journal of the American Statistical Association 119 (545), 285-296.
- Rodemund, N., Kokoefer, A., Wernly, B. &amp; Cozowicz, C. (2023), 'Salzburg intensive care database (sicdb), a freely accessible intensive care database', PhysioNet https://doi. org/10.13026/ezs8-6v88 .
- Rüschendorf, L. &amp; de Valk, V. (1993), 'On regression representations of stochastic processes', Stochastic Processes and their Applications 46 (2), 183-198.
- Shi, C., Wan, R., Chernozhukov, V. &amp; Song, R. (2021), Deeply-debiased off-policy interval estimation, in 'International conference on machine learning', PMLR, pp. 9580-9591.
- Shi, C., Zhang, S., Lu, W. &amp; Song, R. (2021), 'Statistical inference of the value function for reinforcement learning in infinite-horizon settings', Journal of the Royal Statistical Society Series B: Statistical Methodology 84 (3), 765-793.
- Sun, J. &amp; Loader, C. R. (1994), 'Simultaneous Confidence Bands for Linear Regression and Smoothing', The Annals of Statistics 22 (3), 1328 - 1345.
- Timan, A. F. (2014), Theory of approximation of functions of a real variable , Elsevier.
- Van Der Laan, M. J. &amp; Dudoit, S. (2003), 'Unified cross-validation methodology for selection among estimators and a general cross-validated adaptive epsilon-net estimator: Finite sample oracle inequalities and examples'.
- Wiener, N. (1958), Nonlinear Problems in Random Theory , MIT Press, Cambridge, MA.
- Wu, W. B. (2005), 'Nonlinear system theory: Another look at dependence', Proceedings of the National Academy of Sciences 102 (40), 14150-14154.
- Wu, W. B. &amp; Mielniczuk, J. (2010), A new look at measuring dependence, in 'Dependence in probability and statistics', Springer, pp. 123-142.
- Yves, M. (1989), Ondelettes et opérateurs . I, Ondelettes / Yves Meyer , Actualités mathématiques, Hermann, Paris.
- Zhang, A. R. &amp; Zhou, Y. (2020), 'On the non-asymptotic and sharp lower tail bounds of random variables', Stat 9 (1), e314.