## Stable Matching with Ties: Approximation Ratios and Learning

## Shiyun Lin ∗

Center for Statistical Science School of Mathematical Sciences, Peking University shiyunlin@stu.pku.edu.cn

## Nadav Merlis

Technion - Israel Institute of Technology nmerlis@technion.ac.il

## Simon Mauras

INRIA, FairPlay Joint Team simon.mauras@inria.fr

## Vianney Perchet

CREST, ENSAE, IP Paris Criteo AI Lab, FairPlay Joint Team Vianney.perchet@normalesup.org

## Abstract

We study matching markets with ties, where workers on one side of the market may have tied preferences over jobs, determined by their matching utilities. Unlike classical two-sided markets with strict preferences, no single stable matching exists that is utility-maximizing for all workers. To address this challenge, we introduce the Optimal Stable Share (OSS)-ratio, which measures the ratio of a worker's maximum achievable utility in any stable matching to their utility in a given matching. We prove that distributions over only stable matchings can incur linear utility losses, i.e., an Ω( N ) OSS-ratio, where N is the number of workers. To overcome this, we design an algorithm that efficiently computes a distribution over (possibly non-stable) matchings, achieving an asymptotically tight O (log N ) OSS-ratio. When exact utilities are unknown, our second algorithm guarantees workers a logarithmic approximation of their optimal utility under bounded instability. Finally, we extend our offline approximation results to a bandit learning setting where utilities are only observed for matched pairs. In this setting, we consider worker-optimal stable regret, design an adaptive algorithm that smoothly interpolates between markets with strict preferences and those with statistical ties, and establish a lower bound revealing the fundamental trade-off between strict and tied preference regimes.

## 1 Introduction

Two-sided matching markets are prevalent in various contexts, such as matching students to schools [2, 3], doctors to hospitals [50], or workers to jobs [5]. In this paper, we model the market as a company that assigns jobs to workers . Each participant has a preference ordering over the other side of the market. For example, jobs rank workers by ability, while workers rank jobs by personal preference. Stability ensures a fair equilibrium where workers receive sufficiently desirable jobs while respecting the preferences and priorities of all parties. When preferences are strict, the deferred acceptance algorithm [23] efficiently computes a worker-optimal stable matching - no worker can get a better job without violating stability.

In online marketplaces, for example, the online crowd-sourcing platform Amazon Mechanical Turk, workers are usually uncertain of their preferences over jobs at the beginning, since they do not have hands-on experience. However, there are numerous similar tasks to be delegated

∗ This work was performed when Shiyun Lin was a visiting student at CREST, ENSAE, IP Paris.

on the platform and, fortunately, the uncertain preferences can thus be learnt during the iterative matchings. Recent research has explored this scenario within the framework of multi-player multiarmed bandits [42, 43, 8, 37]. Under the strict preferences assumption, these works combine bandit learning algorithms with the deferred acceptance procedure to guide the market toward the workeroptimal stable matching.

However, in real-life scenarios, workers could be indifferent between some jobs due to inherent uncertainty or coarse evaluations. For instance, conference management systems like the Toronto Paper Matching System (TPMS): while the system generates continuous scores to evaluate the suitability of each reviewer for a paper, which theoretically avoids ties, the bidding process introduces unavoidable indifference through discrete categorical ratings (e.g., 'Eager', 'Willing', 'In a pinch', 'Not willing'), creating natural ties in preferences. The challenge becomes even more pronounced in learning-based matching markets, where statistically indistinguishable utility estimates produce effective ties between options. This presents a fundamental limitation for bandit learning approaches, as standard algorithms typically fail to provide meaningful regret guarantees when facing such indifference structures in the preference landscape. In particular, when utility differences become small (statistically indistinguishable), existing regret bounds break down completely, and handling this regime was previously considered impossible [42].

With indifferent preferences, a stable matching can be obtained by arbitrarily breaking ties and applying the deferred-acceptance algorithm. However, the resulting matching is no longer worker-optimal, as different tie-breaking rules may lead to different stable matchings preferred by different workers - potentially creating dramatic utility disparities across outcomes. This challenge is particularly acute in bandit learning settings, where statistically indistinguishable utilities for one worker may lead to arbitrarily large regret for others due to the cascading effects of tie-breaking decisions. In fair resource allocation, fractional matching is a standard technique for balancing competing interests when a single integral matching is infeasible [33, 27, 9]. The Birkhoff-von Neumann (BvN) theorem [10, 58] establishes that such a fractional matching is equivalent to a probability distribution over integral matchings.

These observations motivate our core research question: For markets with tied preferences, can we approximate a stable solution by considering distributions over matchings, while guaranteeing all workers a fair, minimum level of satisfaction ?

To answer this question, we define a worker's optimal-stable-share (OSS) as her maximum achievable utility across all stable matchings. We then introduce the OSS-ratio as a fairness metric, which measures the fraction of the OSS that each worker is guaranteed to receive under any allocation.

We begin by analyzing the offline setting with known preferences, establishing tight OSS-ratio bounds across different matching classes. These results naturally extend to settings with preference uncertainty. Building on these offline results, we further formulate the problem within a multi-player multiarmed bandit framework for online learning scenarios, and show how our approximation guarantees provide the crucial foundation for achieving sublinear regret in matching markets with indifference.

## 1.1 Main Contributions

Offline Approximation Oracle and Matched Upper and Lower Bounds. We first demonstrate that restricting to stable matchings yields only a trivial (and tight) lower bound on the OSS-ratio (Theorem 1), motivating our study for broader matching classes. We then establish a logarithmic lower bound for general matchings (Theorem 2) and construct an approximation oracle (Algorithm 1) achieving this bound while maintaining internal stability (Theorem 3).

Robustness to Approximated Preferences. We prove our positive results are robust to utility uncertainty: when exact utilities are unknown but lie within a given uncertainty set, we maintain the same guarantees with only an additive error bounded by the maximum uncertainty (Theorem 6). This holds especially for rectangular uncertainty sets, which model utility matrices estimated from data.

Bandit Learning in Matching Markets with Indifference. Building on our offline approximation results, we introduce α -approximation stable regret Reg α i ( T ) , using an α -fraction of the optimalstable-share as a tractable benchmark for markets with (statistical) ties. Our adaptive algorithm ETCO (Algorithm 3) seamlessly handles both strict and tied preferences. Theorem 7 establishes its regret bounds, which match the lower bound [52] in markets with large preference gaps. Theorem 8 further reveals a fundamental trade-off: no algorithm can simultaneously achieve optimal regret in both large-gap (standard regret) and small/no-gap (approximation regret) regimes.

## 1.2 Techniques Involved and Developed

The upper bound on the approximation ratio is the first key technical contribution of our paper. We establish this result via three main steps: 1) Introducing a novel component - the duplication index - into the algorithm design; 2) Constructing a directed forest where edges encode conflicts between workers competing for the same job copies across different matchings; 3) Leveraging the tree structure and stability constraints to derive the upper bound inductively.

In the bandit learning setting, the primary technical challenge and key contribution lie in the lower bound proof. To establish this result, we carefully construct two instances with 4 workers and 4 jobs, where the utility matrices differ in only one critical entry that determines whether meaningful ties exist. This construction reveals how ties in one worker's preferences propagate to affect other workers' regret. Furthermore, we employ an information-theoretic argument to demonstrate that the algorithm must sample this critical entry sufficiently often to avoid incurring linear regret. To our knowledge, we are the first to provably show a tradeoff between standard regret and approximation regret in bandit settings.

## 1.3 Related Work

Stable Matching with Ties. A natural extension of Gale and Shapley's work [23] considers settings with tied or incomplete preferences. Irving [29] introduced three stability notions - weak, strong, and super-stability - with weak stability being the most studies [25, 26, 35, 46], as it always guarantees existence, unlike strong or super-stability. However, weakly stable matchings may vary in size, and finding a maximum one is NP-hard [30], while verifying weak stability is NP-complete [46]. Unlike prior work focused on maximizing matching size, we instead study fair job allocations, ensuring each worker receives a utility within a guaranteed fraction of their optimal stable matching, and we characterize the approximation ratio of such allocations.

Fairness in Two-sided Matching. Recent work has increasingly addressed fairness in two-sided markets. In fair division, Freeman et al. [21] introduces double envy-freeness up to one match (DEF1) and double maximin share guarantee (DMMS) for many-to-many matching, while Igarashi et al. [28] studies many-to-one matching, enforcing EF1 for one side while preserving stability. In machine learning, Karni et al. [32] incorporates preference-informed individual fairness (PIIF) [34], requiring allocations to satisfy individual fairness [18] while respecting preferences. Our work diverges by focusing on one-to-one markets, where standard notions like EF1 and MMS are inapplicable. We propose a novel share-based fairness concept (OSS-ratio) to measure workers' gains relative to their optimal-stable-share. Our algorithm returns a random matching that is ex-ante stable (no justified envy) and ex-post internally stable, achieving a best-of-both-worlds guarantee.

Bandit Learning in Matching Markets. Das and Kamenica [16] first formalized bandit problems in matching markets, with subsequent work [42, 43, 8, 52, 37] exploring this model. In this setting, players (with unknown utilities) and arms (with known preferences) form a two-sided market. Playeroptimal stable regret [42] measures the utility difference between a player's outcome and their optimal stable match. Yet, existing results are limited to markets with strict preferences, as stable regret becomes linear and ill-defined when ties exist. Kong et al. [38] recently studied indifference cases, but their player-pessimistic regret benchmark cannot recover optimal stable matches in tie-free settings. Our work bridges this gap by: (1) establishing a tight logarithmic OSS-ratio for offline matching with ties, (2) introducing approximation regret as a tractable objective for tied markets, and (3) developing an adaptive algorithm that achieves optimal regret bounds in both tied and tie-free settings.

## 2 Preliminaries

We model the matching market as a company that assigns jobs to workers. There are N workers, W = { w 1 , w 2 , · · · , w N } and K jobs, A = { a 1 , a 2 , · · · , a K } . The company assigns jobs to workers such that each job is assigned to at most one worker and each worker performs at most one job. The assignment is therefore a matching µ . We shall use µ ( w ) to represent the allocated job to worker w , and µ ( a ) to denote the worker with job a . If a worker w or a job a remains unmatched, we will use the notation µ ( w ) = ⊥ or µ ( a ) = ⊥ .

For every job, the company has a strict rating over the workers based on their expertise and ability on this job. Specifically, if w ≻ a w ′ , worker w performs job a strictly better than w ′ . On the other hand,

workers also have preferences over the jobs, and it is possible that a worker is indifferent among several jobs. The preferences of workers on jobs are represented through a utility matrix U , where U ( w,a ) ∈ [0 , 1] denotes the preference of worker w on job a . If U ( w,a ) &gt; U ( w,a ′ ) , worker w prefers job a over a ′ , and U ( w,a ) = U ( w,a ′ ) implies that w is indifferent between jobs a and a ′ . For simplicity, we will assume that a worker w will refuse to be matched with job a if it has utility U ( w,a ) = 0 ; stated otherwise, either U ( w, ⊥ ) is positive but infinitely small or U ( w, ⊥ ) = 0 and ties are broken in favor of ⊥ . As a consequence, a problem instance ( U , P a ) is defined by a utility matrix U and a preference profile P a representing the preferences of jobs over workers.

Stability is a key concept in two-sided matching markets, which ensures there is no justified envy in the market, i.e., the only jobs a worker prefers over her own job are the ones that she is less suitable to face than the currently assigned worker. When preferences include ties, multiple stability notions arise, and we focus on weak stability [29]. A matching µ is weakly stable if no worker-job pair exists where both strictly prefer each other over their allocated partners:

Definition 1 (Weak Stability) . A matching µ is weakly stable if there is no blocking pair ( w,a ) such that w ≻ a µ ( a ) and U ( w,a ) &gt; U ( w,µ ( w )) .

If a matching is weakly stable, there exists a tie-breaking mechanism such that this matching is stable in the resulting instance with strict preferences. Conversely, any stable matching that is generated using a tie-breaking mechanism is also weakly stable in the original instance. Without causing ambiguity, we will refer to weak stable as stable for brevity. Furthermore, internally stable matching [44] refers to a matching where there are no blocking pairs when only considering the matched workers and jobs.

Definition 2 (Internal Stability) . A matching µ is internally stable if there is no internally blocking pair ( w,a ) such that 1) both w and a are matched in µ , and 2) w ≻ a µ ( a ) and U ( w,a ) &gt; U ( w,µ ( w )) .

Given a problem instance, we define the following classes of matchings: M := { µ : µ is a matching } , S := { µ : µ is a stable matching } , and I := { µ : µ is an internally stable matching } .

In a matching market with ties, stable matchings are not unique, given different tie-breaking mechanisms. A job a is a valid stable match of worker w if there exists a stable matching that matches w with a . We say a is the optimal stable match of worker w if it is the most preferred valid stable match, i.e., there exists a matching µ ∗ ∈ S such that µ ∗ ( w ) = a and U ( w,µ ∗ ( w )) = max µ ∈S U ( w,µ ( w )) . We call U ( w,µ ∗ ( w )) the optimal stable share (OSS) for worker w , denoted as U ∗ ( w ) .

The canonical results in two-sided matching markets are the Gale-Shapley theorem and algorithm (GS) [23], which guarantee both the existence of stable matchings and an efficient O ( n 2 ) computation. The GS algorithm operates through an iterative proposal process. First, workers sequentially propose to their most preferred available jobs. Each job tentatively accepts its most preferred proposal and rejects others. After that, rejected workers continue proposing to their next preferences. The process terminates when no rejections occur, yielding a stable matching. In markets with strict preferences, GS produces a matching that is optimal for all proposers. However, when preferences contain ties, this optimality no longer holds uniformly.

Example 1 (Stable matching with indifference) . Let W = { w 1 , w 2 , w 3 } be workers and A = { a 1 , a 2 } be jobs with w 1 ≻ w 2 ≻ w 3 for all jobs. The utility matrix that encodes the preference of workers over jobs is given by: U =   1 1 1 0 0 1  

There are 2 stable matchings in this instance: µ 1 = { ( w 1 , a 1 ) , ( w 3 , a 2 ) } , µ 2 = { ( w 1 , a 2 ) , ( w 2 , a 1 ) } . There are 4 extra non-empty internally stable matchings, where exactly one worker is assigned a job of utility 1 , and unmatched workers/jobs cannot be involved in blocking pairs.All workers have an OSS of 1 . More precisely, w 1 receives utility U ∗ ( w 1 ) = U ( w 1 , a 1 ) = U ( w 1 , a 2 ) = 1 in both stable matchings, w 2 receives utility U ∗ ( w 2 ) = U ( w 2 , a 1 ) = 1 in µ 2 , and w 3 receives utility U ∗ ( w 3 ) = U ( w 3 , a 2 ) = 1 in µ 1 .

Example 1 demonstrates that different workers may achieve their optimal outcomes in different stable matchings. However, it is impossible to simultaneously guarantee all workers their OSS with a single matching (even non-stable). Based on this impossibility result, a natural question arises as to whether an allocation exists such that every worker is at least satisfied at a certain level . Formally, given a problem instance and a class of matchings C , we are interested in the following optimal stable share-ratio (OSS-ratio):

<!-- formula-not-decoded -->

where ∆( C ) is the set of distributions over C and U D ( w ) is worker w 's expected utility given a distribution D , i.e., U D ( w ) = E µ ∼ D [ U ( w,µ ( w ))] . When we are constrained to the set of matchings, stable matchings and internally stable matchings, R M , R S and R I are defined accordingly.

The OSS-ratio adopts a worst-case perspective by taking the maximum over workers , ensuring every worker receives a fair share of their optimal stable utility. Formally, if max U R M ≤ α , then every worker w i is guaranteed at least 1 α U ∗ ( w i ) in expectation, regardless of the market's preference structure. The minimum over distributions reflects a central planner's optimization: the distribution represents a rotating schedule (e.g., matchings in the support correspond to daily assignments), and restricted support encodes practical constraints. For instance, limiting support to internally stable matchings ensures no justified envy arises between co-present workers in any schedule realization.

## 3 Approximation Ratios for Stable Matching with Ties

In this section, we aim to characterize the scale of the OSS-ratio R C from the worker's perspective, which allows for ties, while additional findings related to the job side are provided in Appendix J. As a first observation, S ⊂ I ⊂ M implies R M ≤ R I ≤ R S , and R S ≤ N , since uniformly selecting a worker and their favored stable matching achieves this bound.

## 3.1 Lower Bound

We first prove that the trivial upper bound on R S is asymptotically tight.

Theorem 1. There exists an instance, such that for any distribution over stable matchings, one worker only receives a 2 /N fraction of their optimal stable share, i.e., R S ≥ N 2 = Ω( N ) .

To prove Theorem 1, we construct an instance with N/ 2 highly-skilled workers and N/ 2 regular workers, such that every stable matching can satisfy at most one regular worker at a time, proving that R S ≥ N/ 2 . The formal proof is deferred to Appendix B.

However, a closer look at our instance reveals that all regular workers can be satisfied in a single (non-stable) matching (See Remark 4 in Appendix B). Thus, we turn our attention to distribution over (possibly non-stable) matchings, and the ratio R M . Theorem 2 shows that if we extend the support of D to include all matchings, i.e., D ∈ ∆( M ) , the ratio R M is still lower bounded by log N .

Theorem 2. There exists an instance s.t. for any distribution over (possibly non-stable) matchings, one worker only receives a 1 / Ω(log N ) fraction of their optimal stable share, i.e., R M = Ω(log N ) .

To prove Theorem 2, we recursively construct instances with global ranking of jobs over workers, and each worker could be assigned to a job they like, but such that the number of workers grows logarithmically faster than the number of valuable jobs, proving that each worker can only receive a logarithmic fraction of their optimal stable share. The full proof could be found in Appendix B.

## 3.2 Upper Bound

We show that the logarithmic ratio obtained in Theorem 2 is asymptotically tight, even if we consider distributions over internally stable matchings.

Theorem 3. For any problem instance, there exists a distribution D over internally stable matchings s.t. all workers only receive a 1 / O (log N ) fraction of their optimal stable share, i.e., R I = O (log N ) .

We prove Theorem 3 by constructing an offline approximation oracle (Algorithm 1), which generates a uniform distribution over m internally stable matchings ˜ µ 1 , . . . , ˜ µ m . Each worker w is matched in exactly one matching ˜ µ i , the key technical insight is that setting m &gt; log 2 N + 1 ensures U D ( w ) = U ( w, ˜ µ i ( w )) /m ≥ U ∗ ( w ) /m . To prove this, we construct a directed forest where nodes represent workers who prefer a stable matching over the algorithm's output, and edges capture conflicts where workers compete for the same job copies under different matchings. By exploiting the tree structure and stability constraint, the proof shows that if any worker were worse off, the graph would imply an exponential growth in the number of workers. For more details, please refer to Appendix C.

Remark 1. The distribution computed by Algorithm 1 is not only 'ex-post' internally stable, but also 'ex-ante' (externally) stable, in the sense that no worker has justified envy towards any other worker's (randomized) allocation.

Remark 2. In Algorithm 1, each worker is assigned a job with a probability of 1 / m . Under such an allocation, some matchings in the support only assign a subset of jobs. In practice, if some job a is not allocated in a matching ˜ µ j , but is allocated to worker w in ˜ µ i , we can give a to w in ˜ µ j without breaking internal stability of ˜ µ j . This post-processing is a Pareto improvement of our solution.

## Algorithm 1 Internally Stable Matchings for Matching Market with Indifference

Input: N workers, K jobs, Utility matrix U that encodes the preference of workers over jobs, strict preference list P a of jobs over workers, a positive number m .

- 1: For each job a ∈ A , duplicate it m times and denote the i -th copy as a ( i ) .
- 2: Each replica a ( i ) shares the same preference P a as the original job a .
- 3: For each worker w , define an ordering P w , by sorting jobs a ( i ) k by decreasing utility U ( w,a ) , breaking ties in favour of lower duplication index i , then in favour of lower index k . That is,

<!-- formula-not-decoded -->

- 4: Run Gale-Shapley algorithm on P w and P a to compute a worker-optimal stable matching ˜ µ .
- 5: For each i ∈ [ m ] , build a matching ˜ µ i , which matches each job a with ˜ µ i ( a ) := ˜ µ ( a ( i ) ) . Output: The distribution D which selects each matching ˜ µ i with probability 1 /m .

Finally, we show that Algorithm 1 cannot be manipulated by a worker who mis-reports her preferences to obtain a distribution that gives them a higher utility, whereas the proof is deferred to Appendix C.3.

Theorem 4. Algorithm 1 is dominant strategy incentive compatible: for every utility matrices U and U ′ that differ only on the row of worker w , let D and D ′ be the distributions computed by Algorithm 1, then U D ( w ) ≥ U D ′ ( w ) .

## 4 Robustness and ϵ -Stability

In Section 3, we present an asymptotically tight algorithm for approximating the optimal stable share in markets with ties under stability. However, exact stability often proves too rigid for real-world applications where preferences may fluctuate slightly. We therefore introduce ϵ -stability, which tolerates blocking pairs with utility gains below a threshold ϵ . This relaxation yields robust matching resilient to preference perturbations while maintaining theoretical guarantees.

Definition 3 ( ϵ -Stability) . Given ϵ ≥ 0 , a matching µ is ϵ -stable if there is no ϵ -blocking pair ( w,a ) such that w ≻ a µ ( a ) and U ( w,a ) &gt; U ( w,µ ( w )) + ϵ .

The notion of ϵ -stability is a relaxation of weak stability, where setting ϵ = 0 makes it equivalent to weak stability (Definition 1). In general, ϵ -stable matching is not unique, and there is not a single ϵ -stable matching that simultaneously maximizes the utilities for all workers. Therefore, similar to matching markets with ties, we define S ϵ := { µ : µ is an ϵ -stable matching } , and we call a a valid ϵ -stable match of worker w if there exists an ϵ -stable matching matches w with a , and it is the optimal ϵ -stable match of worker w if it is the most preferred valid ϵ -stable match, i.e., there exists a matching µ ∗ ϵ ∈ S ϵ such that µ ∗ ϵ ( w ) = a and U ( w,µ ∗ ϵ ( w )) = max µ ∈S ϵ U ( w,µ ( w )) . And we say U ( w,µ ∗ ϵ ( w )) is the optimal ϵ -stable share for worker w , denoted as U ∗ ϵ ( w ) .

Algorithm 2 (see Appendix D) generalizes Algorithm 1 with a different workers' preference profiles generation. It outputs a randomized matching that achieves an expected utility within a log N factor of the optimal ϵ -stable share, plus an ϵ -additive error.

Theorem 5. Given any utility matrix U , parameter m = ⌊ log 2 N +2 ⌋ , and the instability tolerance ϵ ≥ 0 , Algorithm 2 computes a distribution D ∈ ∆( I ) , such that U D ( w ) ≥ U ∗ ϵ ( w ) m -ϵ , ∀ w ∈ W .

The proof of Theorem 5 is deferred to Appendix D.2. Interestingly, the distribution D randomizes over internally stable matchings, which do not depend on ϵ .

In labor markets, worker preferences are typically estimated with uncertainty via i.i.d. observations, we construct utility uncertainty sets using concentration inequalities. Theorem 6 shows that for any utility matrix in such a set U , Algorithm 2 produces a random matching guaranteeing each

worker a logarithmic approximation to their optimal share within U , where the proof is deferred to Appendix D.4.

Theorem 6. Given an uncertainty set U , the optimal stable share within U is

<!-- formula-not-decoded -->

We define the center ˆ U of the set U as ˆ U ( w,a ) = inf U ∈U U ( w,a )+sup U ∈U U ( w,a ) 2 , and the uncertainty parameter as ϵ = 2 · sup U 1 , U 2 ∈U || U 1 -U 2 || max . Algorithm 2 with input ˆ U , m = ⌊ log 2 N +2 ⌋ , and ϵ outputs a distribution D ∈ ∆( I ) such that U D ( w ) ≥ U ∗ ( w ) m -ϵ, ∀ w ∈ W .

Example 2 illustrates an application of Theorem 6 to batch learning problems.

Example 2 (Batch learning) . Suppose that we have a dataset of size T , where each data point U is a noisy observation of the ground-truth utility matrix ˜ U , i.e., each U ( i, j ) is sampled from a 1-sub-Gaussian distribution with mean ˜ U ( i, j ) . Given a parameter δ , set ϵ = 2 √ ln ( 1 δ ) / T , and define the uncertainty set for each entry ( w,a ) as U w,a = { U ( w,a ) : | U ( w,a ) -ˆ U ( w,a ) | ≤ ϵ / 2 } , and U = ⊗ ( w,a ) ∈W×A U w,a , where ˆ U is the empirical mean utility matrix computed from the dataset. The OSS within the uncertainty set U ∗ ( w ) could be defined as in Eq.(2). By Lemma 2, we know that with probability 1 -δ , the ground-truth utility matrix ˜ U ∈ U , and hence ˜ U ∗ ( w ) ≤ U ∗ ( w ) . Therefore, by running Algorithm 2 with the empirical mean utility matrix as input, and set ϵ = 2 √ ln ( 1 δ ) / T , m = ⌊ log 2 N +2 ⌋ , we have w.p. 1 -δ that the corresponding output distribution D over matchings satisfies U D ( w ) ≥ ˜ U ∗ ( w ) ⌊ log 2 N +2 ⌋ -2 √ ln( 1 δ ) / T for all w ∈ W .

## 5 Bandit Learning in Matching Markets

Example 2 demonstrates the application of our offline oracle to learning problems. We now transition to an online learning setting, framing the matching market as a multi-player bandit problem to show how the offline results naturally connect learning scenarios both with and without statistical ties.

In online marketplaces, companies can evaluate workers through interviews, but typically lack prior knowledge of worker preferences over jobs. Still, by leveraging repeated matching opportunities, these preferences can be learned through ex-post evaluations. Recent work models this as a multi-armed bandit (MAB) problem [42, 43, 8, 37], where workers ('players') and jobs ('arms') interact over T rounds. Each round, the company outputs a matching µ t assigning jobs to workers and observes 1 -subgaussian rewards X i ( t ) for matched pairs ( w i , µ t ( w i )) with mean U ( w i , µ t ( w i )) ∈ [0 , 1] . Following bandit matching literature, we assume N ≤ K (more jobs than workers) to ensure matching feasibility. If N &gt; K , we can extend the problem by adding zero-utility jobs or randomly assigning unmatched workers.

The company seeks to learn the worker-optimal stable matching µ ∗ ( w i ) through interactions. Specifically, it aims to minimize the worker-optimal stable regret for each w i ∈ W , defined as the cumulative reward difference between being matched with µ ∗ i and that w i receives over T rounds:

<!-- formula-not-decoded -->

The expectation is taken over the randomness of the received reward and the allocation strategy.

Prior work on minimizing worker-optimal stable regret focuses exclusively on tie-free markets [42, 8, 37], rendering their results inapplicable when preferences contain ties. Crucially, existing regret bounds scale as 1 / ∆ 2 , where ∆ is the minimum utility gap across all workers w and jobs a , i.e., ∆ = min w min a,a ′ | U ( w,a ) -U ( w,a ′ ) | 2 . As shown in Example 2 in [42], this dependence is fundamental - achieving sublinear regret requires ∆ = ω (1 / √ T ) .

When the benchmark is unachievable (computationally or statistically), prior work adopts α -approximation regret to ensure sublinear regret relative to an α -fraction of the benchmark [31, 54, 14].

2 While definitions of ∆ vary slightly across works, this strongest version generalizes to other formulations.

In our setting, since ties prevent all workers from simultaneously achieving their optimal stable share, we assume access to an offline oracle that, given utility matrix U , outputs a randomized matching guaranteeing each worker at least an 1 /α of U ∗ ( w ) in expectation, with additional error ϵ . Formally,

Definition 4 (( α , ϵ )-Approximation Oracle) . An ( α , ϵ )-approximation oracle takes a rectangular uncertainty set U with width ϵ as input and returns a (randomized) matching ˜ µ satisfying: E [ U ˜ µ ( w )] ≥ α U ( w ) · U ∗ ( w ) -ϵ for every worker w , where α U ∈ (0 , 1] N is a worker-specific approximation ratio vector (often simplified to α ). If α U ( w ) = α is uniform across workers and independent of U , we call it an ( α, ϵ ) -approximation oracle,

For example, Algorithm 2 guarantees that for any input utility matrix U , α U ( w ) ≥ 1 / ⌊ log 2 N +2 ⌋ . With ties, our regret metric should not compare against the OSS each time, but against an α -fraction of the optimal stable share, since the offline oracle can only guarantee this fraction in expectation:

<!-- formula-not-decoded -->

where α ∈ (0 , 1] is the approximation ratio given by the offline oracle. When we want to emphasize that the observations X ( t ) come from a distribution ν , we write Reg i ( T ; ν ) and Reg α i ( T ; ν ) .

For markets without ties, [37] achieves a stable regret of O ( K ln T/ ∆ 2 ) , matching the Ω( N ln T/ ∆ 2 ) lower bound [52] in T and ∆ . We seek a best-of-both-worlds guarantee, i.e., an algorithm that attains Reg i ( T ) = O (ln T/ ∆ 2 ) when ∆ = ω (1 / √ T ) , and Reg α i ( T ) = o ( T ) when ∆ = O (1 / √ T ) .

## 5.1 Algorithm: Explore-then-Choose-Oracle

We present our algorithm, Explore-then-Choose-Oracle (ETCO, Algorithm 3 in Appendix E), and summarize it here. The algorithm consists of two phases. In each round of the exploration phase , the company allocates a job to every worker in a round-robin way to estimate their utilities accurately. In the second phase, the company checks for plausible ties in utilities. If none exists, it computes a matching using GS algorithm; otherwise, it uses the approximation oracle. In subsequent rounds, jobs are allocated based on the chosen oracle's output.

In the exploration phase, the company allocates jobs to workers in a round-robin way, according to the index of the workers. In this way, every K rounds, each worker is matched to every job exactly once. The maximal number of exploration rounds is bounded by a parameter T 0 . After each allocation, based on the observation, we update the estimated utility ˆ U ( i, µ t ( i )) = ˆ U ( i,µ t ( i )) · T i,µ t ( i ) + X i,µ t ( i ) ( t ) T i,µ t ( i ) +1 , and the observation count of worker w i and job µ t ( i ) as T i,µ t ( i ) = T i,µ t ( i ) +1 . The company also builds a confidence set for each utility estimate, ensuring the true expected utility is included with high probability. Particularly, the confidence interval (CI) for worker w i 's preference utility over job a j is [ LCB i,j , UCB i,j ] , with the upper and lower confidence bounds defined as

<!-- formula-not-decoded -->

When confidence sets for jobs a j and a j ′ are disjoint ( LCB i,j &gt; UCB i,j ′ or vice versa), we can determine worker w i 's strict preference between them. If all topN job CIs for w i become disjoint, we recover the true preference with high probability. If this occurs for all workers before the exploration phase T 0 ends, we switch to the Gale-Shapley oracle for exploitation, as no topN ties exist w.h.p. Otherwise, remaining CI overlaps indicate potential ties, triggering our approximation oracle instead.

## 5.2 Theoretical Analysis

Before stating the regret guarantee for ETCO algorithm, we first give a formal definition of the minimum preference gap, which measures the hardness of the learning problem.

̸

Definition 5 (Minimum Preference Gap) . For each worker w i and job a j = a j ′ , let ∆ i,j,j ′ = | U ( i, j ) -U ( i, j ′ ) | be the preference gap for w i between a j and a j ′ . Let r i be the preference ranking of worker w i and r i,k be the k -th preferred job in w i 's ranking for k ∈ [ K ] . Define ∆ min = min i ∈ [ N ]; k ∈ [ N ] ∆ i,r i,k ,r i,k +1 as the minimum preference gap among all workers and their first ( N +1) -ranked jobs.

Next, we present upper bounds for the worker-optimal stable regret for each worker when using ETCO. Theorem 7 (Upper Bound) . Following the ETCO algorithm with exploration phase of length T 0 and an ( α, 2 √ 6 K ln T T 0 ) -approximation oracle, for w i ∈ W , we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

See Appendix G for the proof. Our bound exhibits two regimes: (1) large-∆ regime : when ∆ min is large, the exploration phase learns the top-( N +1 ) job preferences w.h.p. before T 0 , enabling exact worker-optimal stability via Gale-Shapley in exploitation. This reduces to ETGS [37] under centralization; (2) small-∆ / tied regime : for small ∆ min or exact ties, worker-optimal stability is unattainable; instead, implementing an approximation oracle guarantees an α -approximation regret sublinear in T .

When ∆ min is sufficiently large, our upper bound matches the Ω( N ln T/ ∆ 2 min ) lower bound [52] for serial dictatorship markets (where all jobs share identical preferences). This tightness, however, comes at a fundamental trade-off: Theorem 8 shows that extending sublinear regret guarantees to wider ranges of ∆ min unavoidably worsens approximation regret in small- or no-gap regimes.

Prior to presenting our trade-off lower bound, we formally define two key concepts. The Paretooptimal stable matching set S U opt , comprising stable matchings where no worker's utility can be strictly improved without harming another worker, and the relevant preference utility gap ∆ rel , representing the maximum utility perturbation that preserves S U opt .

̸

Definition 6 (Pareto-optimal Stable Matching Set) . Given a utility matrix U , the worker-optimal Pareto-optimal stable matching set S U opt is the set of all matchings µ such that: 1) µ is stable; 2) If there exists a stable matching µ ′ and a worker w such that U ( w,µ ′ ( w )) &gt; U ( w,µ ( w )) , then for some w ′ = w , it holds that U ( w ′ , µ ′ ( w ′ )) &lt; U ( w ′ , µ ( w ′ )) .

Definition 7 (Relevant Utility Gap) . Given a utility matrix U , the relevant preference gap ∆ rel is

̸

<!-- formula-not-decoded -->

By definition, ∆ rel ≥ 0 . When S U opt contains multiple matchings, ∆ rel = 0 since any perturbation acts as a tie-breaker, eliminating at least one matching from S U opt (by the uniqueness of worker-optimal stable matching in tie-free markets). Furthermore, ∆ rel ≥ ∆ min because perturbations smaller than ∆ min cannot alter any worker's topN preferences or the worker-optimal matching 3 .

Theorem 8 (Trade-off between Regret and Approximation Reget) . Let δ ∈ (0 , 1 2 ) and fix N = K = 4 . Consider the class of instances with a large relevant utility gap, denoted as E ℓ ( T ) , i.e., for any instance ν ∈ E ℓ ( T ) , we have ∆ ν rel ≥ cT -1 / 2+ δ for some absolute c &gt; 0 . Assume that an algorithm π guarantees sublinear regret for all workers, for all ν ∈ E ℓ ( T ) . Then there exists an instance such that this algorithm suffers Ω( T 1 -2 δ ) approximation regret for some worker when ∆ rel = 0 w.r.t the best approximation ratio α ∗ for this instance, i.e.,

<!-- formula-not-decoded -->

where α ∗ ( w i ) = max { α ( w i ) : α ( w ) ≥ 1 /R U M , ∀ w ∈ W } , for any w i ∈ W , and R U M is the OSS-ratio on matchings with a given utility matrix U .

The proof appears in Appendix H. We construct two serial dictatorship instances with 4 workers and 4 jobs each, whose utility matrices differ in only one entry (for the highest-priority worker), yielding ∆ rel = 0 . The first instance evaluates α ∗ -approximation regret, while the second analyzes standard stable regret. Crucially, this single entry difference completely alters the benchmark utilities for the other three workers. Thus, one of the two cases happens: (1) under-sampling : without

3 Actually, if we assume an oracle that can determine whether there is a unique worker-optimal stable matching within the uncertainty set, we can prove a similar upper bound as in Theorem 7 with ∆ min replaced by ∆ rel .

enough samples of the differing entry, at least one worker incurs linear approximation regret; (2) over-sampling : After T 0 samples of the differing entry, at least one of the remaining workers suffers Ω( T 0 ) approximation regret.

Theorem 8 establishes an inherent trade-off between large-gap and small / no-gap regimes: as δ → 0 , sublinear regret in the former necessitates linear approximation regret in the latter. Consequently, the exploration length T 0 of ETCO algorithm critically determines regime-specific performance. We provide two T 0 choices and their corresponding regret bounds.

Corollary 1. Following ETCO algorithm with T 0 = T 2 / 3 ( K ln T ) 1 / 3 , for w i ∈ W , we have

<!-- formula-not-decoded -->

Choosing T 0 = T 2 / 3 ( K ln T ) 1 / 3 yields the optimal approximation regret upper bound in the small gap regime when implementing explore-then-commit type algorithms. However, this choice is not satisfiable when ∆ min ∈ [ ˜ Ω ( T -1 / 2 ) , ˜ O ( T -1 / 3 ) ] , since setting T 0 as such cannot guarantee detection of instances when ∆ min falls in this intermediate regime. For these cases, we must resort to the approximation oracle during exploitation. Cruicially, since the oracle's solution differs by a constant factor from the Gale-Shapley optimal, each exploitation round incurs constant regret when measured against Eq.(3), resulting in an overall linear regret.

Corollary 2. Following ETCO algorithm with T 0 = T 2 ln T , for w i ∈ W , we have

<!-- formula-not-decoded -->

Choosing T 0 = T/ (2 ln T ) yields the optimal regret that matches the lower bound for any ∆ min = ˜ Ω ( T -1 / 2 ) . However, for ∆ min = ˜ O ( T -1 / 2 ) , we can only guarantee sublinear approximation regret with an approximation ratio of α -1 / ln T even when using an offline α -approximation oracle.

Remark 3. The approximation regret lower bound in Theorem 8 is both non-trivial and potentially of independent interest for bandit theory. While combinatorial bandits typically use approximation regret to circumvent computational limits (with statistical lower bounds focusing on 1-regret [15, 39, 48]), our result reveals a fundamental distinction: in matching markets, this approximation factor persists even given unlimited computational resources.

## 6 Conclusion

In this paper, we study stable matching with one-sided indifference, modeled as a company assigning workers to jobs. Using a utility matrix to encode workers' potentially tied preferences over jobs, we define the optimal stable share (OSS) for each worker as the maximum utility achievable in any stable matching. To address fairness, we introduce the OSS-ratio, quantifying the fraction of the OSS a worker obtains under random matchings. We first analyze distributions over stable matchings, showing that a linear approximation to the OSS is trivial and asymptotically tight. For general matchings, we prove that no better than logarithmic approximation is possible. To achieve this bound, we propose a polynomial-time algorithm computing a distribution over internally stable matchings , which is asymptotically optimal in OSS ratio and dominant strategy incentive-compatible. Next, we extend our framework to settings where the utility matrix is uncertain but lies within a given uncertainty set. By incorporating ϵ -stable matchings and relating them to perturbations of the utility matrix, we derive a logarithmic approximation with an additive ϵ error, matching the deterministic case. Finally, we explore online learning, where existing stable regret frameworks fail to handle tied preferences. Leveraging the OSS-ratio, we define α -approximation stable regret and provide an algorithm whose upper bound matches the lower bound in the no-tied case. We further derive approximation regret bounds for small or no utility gaps and establish a fundamental trade-off between regret types, highlighting the need for careful exploration stopping time decisions.

Our work establishes the first instance-independent worker-optimal stable regret bound in bandit learning for matching markets, achieved through centralized job allocation. However, real-world marketplaces typically operate in decentralized settings where workers cannot directly coordinate. While the Gale-Shapley algorithm naturally decentralizes, extending our approximation guarantees to decentralized bandit learning remains an open challenge, which is an important direction for future research. Furthermore, exploring the application of our proposed algorithms to real-world datasets would be a valuable next step, as it would help address the practical challenge of stable matching when ties exist in preference rankings.

## Acknowledgements

This project has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Sk l odowska-Curie grant agreement No 101034255. Shiyun Lin acknowledges the financial support from the China Scholarship Council (Grant No.202306010152). Vianney Perchet's research was supported in part by the French National Research Agency (ANR) in the framework of the PEPR IA FOUNDRY project (ANR-23-PEIA-0003) and through the grant DOOM ANR-23-CE23-0002. It was also funded by the European Union (ERC, Ocean, 101071601). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## References

- [1] Atila Abdulkadiro˘ glu and Tayfun Sönmez. School choice: A mechanism design approach. American economic review , 93(3):729-747, 2003.
- [2] Atila Abdulkadiro˘ glu, Parag A Pathak, and Alvin E Roth. The new york city high school match. American Economic Review , 95(2):364-367, 2005.
- [3] Atila Abdulkadiro˘ glu, Parag A Pathak, Alvin E Roth, and Tayfun Sönmez. The boston public school match. American Economic Review , 95(2):368-371, 2005.
- [4] Elliot Anshelevich, Sanmay Das, and Yonatan Naamad. Anarchy, stability, and utopia: creating better matchings. Autonomous Agents and Multi-Agent Systems , 26(1):120-140, 2013.
- [5] Esteban Arcaute and Sergei Vassilvitskii. Social networks and stable matchings in the job market. In International Workshop on Internet and Network Economics , pages 220-231. Springer, 2009.
- [6] Haris Aziz, Rupert Freeman, Nisarg Shah, and Rohit Vaish. Best of both worlds: Ex ante and ex post fairness in resource allocation. Operations Research , 2023.
- [7] Moshe Babaioff, Tomer Ezra, and Uriel Feige. On best-of-both-worlds fair-share allocations. In International Conference on Web and Internet Economics , pages 237-255. Springer, 2022.
- [8] Soumya Basu, Karthik Abinav Sankararaman, and Abishek Sankararaman. Beyond log 2 ( t ) regret for decentralized bandits in matching markets. In International Conference on Machine Learning , pages 705-715. PMLR, 2021.
- [9] Gerdus Benade, Aleksandr M Kazachkov, Ariel D Procaccia, Alexandros Psomas, and David Zeng. Fair and efficient online allocations. Operations Research , 72(4):1438-1452, 2024.
- [10] Garrett Birkhoff. Three observations on linear algebra. Univ. Nac. Tacuman, Rev. Ser. A , 5: 147-151, 1946.
- [11] Eric Budish. The combinatorial assignment problem: Approximate competitive equilibrium from equal incomes. Journal of Political Economy , 119(6):1061-1103, 2011.
- [12] Ioannis Caragiannis, Aris Filos-Ratsikas, Panagiotis Kanellopoulos, and Rohit Vaish. Stable fractional matchings. In Proceedings of the 2019 ACM Conference on Economics and Computation , pages 21-39, 2019.
- [13] Ioannis Caragiannis, David Kurokawa, Hervé Moulin, Ariel D Procaccia, Nisarg Shah, and Junxing Wang. The unreasonable fairness of maximum nash welfare. ACM Transactions on Economics and Computation (TEAC) , 7(3):1-32, 2019.
- [14] Wei Chen, Yajun Wang, Yang Yuan, and Qinshi Wang. Combinatorial multi-armed bandit and its extension to probabilistically triggered arms. Journal of Machine Learning Research , 17 (50):1-33, 2016.
- [15] Richard Combes, Mohammad Sadegh Talebi Mazraeh Shahi, Alexandre Proutiere, et al. Combinatorial bandits revisited. Advances in neural information processing systems , 28, 2015.

- [16] Sanmay Das and Emir Kamenica. Two-sided bandits and the dating market. In Proceedings of the 19th international joint conference on Artificial intelligence , pages 947-952, 2005.
- [17] Lester E Dubins and David A Freedman. Machiavelli and the gale-shapley algorithm. The American Mathematical Monthly , 88(7):485-494, 1981.
- [18] Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel. Fairness through awareness. In Proceedings of the 3rd innovations in theoretical computer science conference , pages 214-226, 2012.
- [19] Michal Feldman, Simon Mauras, Vishnu V Narayan, and Tomasz Ponitka. Breaking the envy cycle: Best-of-both-worlds guarantees for subadditive valuations. arXiv preprint arXiv:2304.03706 , 2023.
- [20] Duncan Karl Foley. Resource allocation and the public sector . Yale University, 1966.
- [21] Rupert Freeman, Evi Micha, and Nisarg Shah. Two-sided matching meets fair division. In International Joint Conference on Artificial Intelligence , 2021.
- [22] Yi Gai, Bhaskar Krishnamachari, and Rahul Jain. Combinatorial network optimization with unknown variables: Multi-armed bandits with linear rewards and individual observations. IEEE/ACM Transactions on Networking , 20(5):1466-1478, 2012.
- [23] David Gale and Lloyd S Shapley. College admissions and the stability of marriage. The American Mathematical Monthly , 69(1):9-15, 1962.
- [24] Aurélien Garivier, Pierre Ménard, and Gilles Stoltz. Explore first, exploit next: The true shape of regret in bandit problems. Mathematics of Operations Research , 44(2):377-399, 2019.
- [25] Magnús M Halldórsson, Robert W Irving, Kazuo Iwama, David F Manlove, Shuichi Miyazaki, Yasufumi Morita, and Sandy Scott. Approximability results for stable marriage problems with ties. Theoretical Computer Science , 306(1-3):431-447, 2003.
- [26] Magnús M Halldórsson, Kazuo Iwama, Shuichi Miyazaki, and Hiroki Yanagisawa. Randomized approximation of the stable marriage problem. Theoretical Computer Science , 325(3):439-465, 2004.
- [27] Daniel Halpern and Nisarg Shah. Fair and efficient resource allocation with partial information. arXiv preprint arXiv:2105.10064 , 2021.
- [28] Ayumi Igarashi, Yasushi Kawase, Warut Suksompong, and Hanna Sumita. Fair division with two-sided preferences. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence , pages 2756-2764, 2023.
- [29] Robert W Irving. Stable marriage and indifference. Discrete Applied Mathematics , 48(3): 261-272, 1994.
- [30] Kazuo Iwama and Shuichi Miyazaki. Stable marriage with ties and incomplete lists. Encyclopedia of algorithms , pages 883-885, 2008.
- [31] Sham M Kakade, Adam Tauman Kalai, and Katrina Ligett. Playing games with approximation algorithms. In Proceedings of the thirty-ninth annual ACM symposium on Theory of computing , pages 546-555, 2007.
- [32] Gili Karni, Guy N Rothblum, and Gal Yona. On fairness and stability in two-sided matchings. In 13th Innovations in Theoretical Computer Science Conference (ITCS 2022) . Schloss-DagstuhlLeibniz Zentrum für Informatik, 2022.
- [33] Richard M Karp, Umesh V Vazirani, and Vijay V Vazirani. An optimal algorithm for on-line bipartite matching. In Proceedings of the twenty-second annual ACM symposium on Theory of computing , pages 352-358, 1990.
- [34] Michael P Kim, Aleksandra Korolova, Guy N Rothblum, and Gal Yona. Preference-informed fairness. In 11th Innovations in Theoretical Computer Science Conference, ITCS 2020 , pages 16-1. Schloss Dagstuhl-Leibniz-Zentrum fur Informatik GmbH, Dagstuhl Publishing, 2020.

- [35] Zoltán Király. Better and simpler approximation algorithms for the stable marriage problem. Algorithmica , 60(1):3-20, 2011.
- [36] Donald E Knuth. Mariages stables et leurs relations avec d'autres problèmes combinatoires (stable marriage and its relation to other combinatorial problems). In CRM Proceedings and Lecture Notes , volume 10. Les Presses de l'Université de Montréal, 1976.
- [37] Fang Kong and Shuai Li. Player-optimal stable regret for bandit learning in matching markets. In Proceedings of the 2023 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA) , pages 1512-1522. SIAM, 2023.
- [38] Fang Kong, Jingqi Tang, Mingzhu Li, Pinyan Lu, John C.S. Lui, and Shuai Li. Bandit learning in matching markets with indifference. In The Thirteenth International Conference on Learning Representations , 2025. URL https://openreview.net/forum?id=7ENakslm9J .
- [39] Branislav Kveton, Zheng Wen, Azin Ashkan, and Csaba Szepesvari. Tight regret bounds for stochastic combinatorial semi-bandits. In Artificial Intelligence and Statistics , pages 535-543. PMLR, 2015.
- [40] Tor Lattimore and Csaba Szepesvári. Bandit algorithms . Cambridge University Press, 2020.
- [41] Richard J Lipton, Evangelos Markakis, Elchanan Mossel, and Amin Saberi. On approximately fair allocations of indivisible goods. In Proceedings of the 5th ACM Conference on Electronic Commerce , pages 125-131, 2004.
- [42] Lydia T Liu, Horia Mania, and Michael Jordan. Competing bandits in matching markets. In International Conference on Artificial Intelligence and Statistics , pages 1618-1628. PMLR, 2020.
- [43] Lydia T Liu, Feng Ruan, Horia Mania, and Michael I Jordan. Bandit learning in decentralized matching markets. The Journal of Machine Learning Research , 22(1):9612-9645, 2021.
- [44] Yicheng Liu, Pingzhong Tang, and Wenyi Fang. Internally stable matchings and exchanges. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 28, 2014.
- [45] David F Manlove. The structure of stable marriage with indifference. Discrete Applied Mathematics , 122(1-3):167-181, 2002.
- [46] David F Manlove, Robert W Irving, Kazuo Iwama, Shuichi Miyazaki, and Yasufumi Morita. Hard variants of stable marriage. Theoretical Computer Science , 276(1-2):261-279, 2002.
- [47] Nadav Merlis and Shie Mannor. Batch-size independent regret bounds for the combinatorial multi-armed bandit problem. In Conference on Learning Theory , pages 2465-2489. PMLR, 2019.
- [48] Nadav Merlis and Shie Mannor. Tight lower bounds for combinatorial multi-armed bandits. In Conference on Learning Theory , pages 2830-2857. PMLR, 2020.
- [49] Pierre Perrault. When combinatorial thompson sampling meets approximation regret. Advances in Neural Information Processing Systems , 35:17639-17651, 2022.
- [50] Alvin E Roth and Elliott Peranson. The redesign of the matching market for american physicians: Some engineering aspects of economic design. American economic review , 89(4):748-780, 1999.
- [51] Alvin E Roth, Uriel G Rothblum, and John H Vande Vate. Stable matchings, optimal assignments, and linear programming. Mathematics of operations research , 18(4):803-828, 1993.
- [52] Abishek Sankararaman, Soumya Basu, and Karthik Abinav Sankararaman. Dominate or delete: Decentralized competing bandits in serial dictatorship. In International Conference on Artificial Intelligence and Statistics , pages 1252-1260. PMLR, 2021.
- [53] Hugo Steinhaus. Sur la division pragmatique. Econometrica: Journal of the Econometric Society , pages 315-319, 1949.

- [54] Matthew Streeter and Daniel Golovin. An online algorithm for maximizing submodular functions. Advances in Neural Information Processing Systems , 21, 2008.
- [55] Chung-Piaw Teo and Jay Sethuraman. The geometry of fractional stable matchings and its applications. Mathematics of Operations Research , 23(4):874-891, 1998.
- [56] Hal R Varian. Equity, envy, and efficiency. Journal of Economic Theory , 9(1):63-91, 1974.
- [57] John H Vande Vate. Linear programming brings marital bliss. Operations Research Letters , 8 (3):147-153, 1989.
- [58] John Von Neumann. A certain zero-sum two-person game equivalent to the optimal assignment problem. Contributions to the Theory of Games , 2(0):5-12, 1953.
- [59] Dietrich Weller. Fair division of a measurable space. Journal of Mathematical Economics , 14 (1):5-17, 1985.
- [60] YiRui Zhang and Zhixuan Fang. Decentralized two-sided bandit learning in matching market. In The 40th Conference on Uncertainty in Artificial Intelligence , 2024.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: In the abstract and introduction, we accurately present the setting and its motivation, as well as a summary of our contributions, all of which are proved in the appendix.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: We discussed the limitations and open challenges of our work in the conclusion section.

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

Justification: Proofs of all the stated results are provided in the appendix.

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

Justification: The paper is purely theoretical and studies a fundamental game theory model; as such, it does not have any direct ethical implications.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Due to the theoretical nature of the paper, there is no societal impact of the work performed.

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

Justification: No data or models are released with this paper.

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

## A Further Related Work

Fractional Matchings Aside from integral matchings, fractional matchings have also attracted research interests due to their practical implications. For instance, in our running example, consider a time-sharing scenario [51], where each worker could spend five days a week at work. An integral matching requires every worker to work full-time on a single job, while fractional matchings allow them to switch among different jobs, making it natural in such situations. By the well-known Birkhoff-von-Neumann (BvN) theorem [10, 58], a fractional matching could be written as a convex combination of several integral matchings.

In the context of stable matching, fractional matching has also been studied. Considering purely ordinal preferences, several notions of stability have been proposed, such as strong stability [51], ex-post stability [51], and fractional stability [57]. In these works, the stable matching problem is formulated as a linear program, Teo and Sethuraman [55] showed that any fractional solution in the stable matching polytope is a convex combination of integral stable matchings. On the other hand, concerning purely cardinal preferences, Anshelevich et al. [4] proposes the notions of stability and ε -stability, while Caragiannis et al. [12] shows that the set of stable fractional matchings that satisfies the notion can be non-convex.

In this paper, we consider a two-sided market where the worker side has cardinal preferences while the job side has ordinal preferences. We do not concern the notions of fractional stable matchings, instead, we focus on finding a distribution over integral matchings such that it is fair in the sense that every worker could receive a certain fraction of its optimal stable share in expectation.

Fair Division Fair division is the problem of dividing a set of items among several people in a fair manner. Steinhaus [53] pioneers this line of research and defines a share-based notion, i.e., proportionality , where each player gets a 1 /N fraction of all items. Foley [20] and Varian [56] define envy-freeness , where no player prefers the bundle allocated to another player, and this notion is later generalized by Weller [59]. In two-sided matching markets, a stable matching eliminates justified envy [1]. Regarding the problem of sharing indivisible goods, share-based guarantees such as MMS [11] and envy-based guarantees such as EF1 [41, 11] or EFX [13] are proposed. Recent works have studied best-of-both-world fairness [6, 7, 19], providing random allocation with fairness guarantees both in expectation and for every realization.

Approximation Regret in Bandit Learning In combinatorial bandit problems, approximation regret is often considered instead of the standard regret [31, 54, 22, 14, 47, 49]. The reason mainly lies in the complex reward structure and the computational intractability of the problem, i.e., rewards are often dependent on the combination of actions, leading to an exponentially large action space, which makes it computationally prohibitive to find the exact solution.

Besides computational intractability, there is another more fundamental reason for using the approximation regret framework in this paper. The stable regret is defined for each worker, which means the company aims to solve a multi-objective optimization problem while it couldn't satisfy everyone simultaneously. Consequently, the approximation regret serves as a compromise between fairness and efficiency.

## B Lower bounds on OSS-ratio

## B.1 Lower Bound for Distributions over Weakly Stable Matchings

The proof idea of Theorem 1 could be illustrated through Figure 1.

Proof of Theorem 1. Assume that N is even, let W = { w 1 , w 2 , · · · , w N } and A = { a 1 , a 2 , · · · , a K } with K = N 2 + 1 and w 1 ≻ w 2 ≻ ·· · ≻ w N for all the jobs. The utility matrix that encodes the preference of workers over jobs is as follows:

Figure 1: Lower bound on R S . All jobs have the same ordering over workers, from top to bottom. Any stable matching can be obtained by letting the first worker pick a job, then the second, etc. Hence, each stable matching contains at most one blue edge.

<!-- image -->

<!-- formula-not-decoded -->

̸

In any stable matching, every worker w i in { w 1 , w 2 , · · · , w N/ 2 } must be assigned to a i or a N 2 +1 , leading to a utility of 1 for them. Without loss of generality, let µ i be the matching such that µ ( w i ) = a N 2 +1 . Then in µ i , only worker w N 2 + i would receive a utility of 1, by matching it to job a i , while all workers in { w N/ 2+1 , · · · , w N/ 2+ i -1 , w N/ 2+ i +1 , · · · , w N } would be unmatched and receive a utility of 0. Indeed, for any j = i , the unique optimal match of worker w N/ 2+ j is already taken by worker w j .

For every stable matching, at most one of the workers in { w N/ 2+1 , w N/ 2+2 , · · · , w N } could be assigned to their optimal match. Since there are N/ 2 such workers, then for any distribution D , there must be at least one of the workers for which the probability to be optimally matched is smaller than 2 /N , for this worker, it holds that U ∗ ( w ) U D ( w ) ≥ N 2 , which implies R S ≥ N/ 2 .

Remark 4. Theorem 1 shows that if we only consider random allocations of stable matchings, then in the worst case, workers could only expect O (1 /N ) profit share compared to their benchmark. On the other hand, define

<!-- formula-not-decoded -->

Here, µ 1 is a stable matching while µ 2 is non-stable. We construct a distribution D as follows:

<!-- formula-not-decoded -->

then all workers have U ∗ ( w ) U D ( w ) = 2 . This result implies that if we consider possibly non-stable matchings for the support of D , there is space for improvement on the OSS-ratio.

## B.2 Lower Bound for Distributions over Matchings

The proof idea of Theorem 2 could be illustrated through Figure 2.

Figure 2: Lower bound on R M . In each example, left nodes represent workers while right nodes represent jobs. If there is an edge connecting a left node w and a right node a , we have U ( w,a ) = 1 , and U ( w,a ) = 0 otherwise. All the right nodes without edges connecting to them are hidden from the graph.

<!-- image -->

Proof of Theorem 2. Consider the following sequence of problem instances, as depicted in Figure 2. In each bipartite graph, left nodes represent the set of workers while the right nodes represent the set of jobs. Jobs share a global preference over the workers, with the topmost node being the most preferred and the preference decreasing from top to bottom. From the worker side, the utility matrix is binary. Specifically, for a worker-job pair ( w,a ) , U ( w,a ) = 1 if ( w,a ) is connected, while U ( w,a ) = 0 otherwise. For example, I 1 is the graph representation of the problem instance of Example 1.

Instance I n is constructed recursively. Given I n -1 , we first duplicate this instance, denoted the replications as upper class and lower class , respectively. Take K n -1 as the number of right nodes and N n -1 as the number of left nodes in I n -1 , and denote these nodes as { a u 1 , a u 2 , · · · , a u K n -1 } , { w u 1 , w u 2 , · · · w u N n -1 } and { a ℓ 1 , a ℓ 2 , · · · , a ℓ K n -1 } , { w ℓ 1 , w ℓ 2 , · · · w ℓ N n -1 } for the upper and lower classes, respectively. Then, we introduce K n -1 prioritized workers in I n , who are uniformly more preferred by the jobs than the workers in the upper and lower classes. In particular, denote the set of prioritized workers as { w 1 , w 2 , · · · , w K n -1 } , we have

<!-- formula-not-decoded -->

And for each w i , we have U ( w i , a u i ) = U ( w i , a ℓ i ) = 1 , and U ( w i , a ) = 0 otherwise. We first prove by induction that the optimal-stable-share is U ∗ ( w ) = 1 for any worker w . For I 0 , the unique matching is stable. Suppose that for any worker w in I n -1 , there exists at least one stable matching µ such that U ( w,µ ( w )) = 1 . Then in I n , all the prioritized workers could be matched in any stable matching. Furthermore, as long as they simultaneously choose to be matched to the jobs in the same class, the other class is free, and hence by induction assumption, every worker in that class gets a chance to be matched in at least one stable matching. Therefore, by breaking ties for the upper (lower) class, all workers in the lower (upper) class can be matched.

On the other hand, given that U ∗ ( w ) = 1 for any w , the ratio R M is equal to min D max w 1 U D ( w ) for these instances, Then, proving a lower bound on this quantity is equivalent to establishing an upper bound on max D min w U D ( w ) . In particular, we have max D min w U D ( w ) ≤ max D ∑ w U D ( w ) N , where N is the number of left nodes. Now notice that ∑ w U D ( w ) is the expected size of the matching under distribution D , since the utility is 1 when a worker is matched and 0 otherwise. We have ∑ w U D ( w ) ≤ K from the fact that the size of any matching is bounded by K . Combining the above derivation, we have

<!-- formula-not-decoded -->

Finally, in instance I n , by the recursive construction, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Solving the recursive equation with the initial condition K 0 = N 0 = 1 , we know that there are 2 n right nodes and N = ( n +2)2 n -1 left nodes in I n , which implies that R M ≥ n/ 2 + 1 . We rewrite N = ( n +2)2 n -1 to obtain 2 n = 2 N/ ( n +2) and we deduce that

<!-- formula-not-decoded -->

Taking a logarithm in the inequalities, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, n = Ω(log N ) and hence R M is Ω(log N ) .

## C Upper Bound on OSS-ratio

## C.1 Procedure Illustration of Algorithm 1

We use Example 3 to illustrate the procedure stated in Algorithm 1.

Example 3. Let W = { w 1 , w 2 , w 3 } , A = { a 1 , a 2 , a 3 } . We consider the following preference list P a of jobs over workers, and utility matrix U that encodes the preference of workers over jobs:

<!-- formula-not-decoded -->

If m = 2 , the preference profile P w generated from the algorithm is

<!-- formula-not-decoded -->

Running Gale-Shapley algorithm on P w and P a , the worker-optimal stable matching would be ˜ µ = { ( w 1 , a (1) 2 ) , ( w 2 , a (1) 1 ) , ( w 3 , a (2) 2 ) } , and we can recover two internally stable matchings from ˜ µ , i.e., ˜ µ 1 = { ( w 1 , a 2 ) , ( w 2 , a 1 ) } and ˜ µ 2 = { ( w 3 , a 2 ) } .

## C.2 Proof of Theorem 3

In Algorithm 1, each worker w is matched in exactly one matching ˜ µ i , where we call i the index of w , denoted index ( w ) . In other words, the index of a worker is the index of the job she receives, that is, index ( w ) = i if worker w receives a ( i ) j for some j .

Definition 8. Given a problem instance ( U , P a ) , run Algorithm 1 with duplication number m to generate the output distribution D . Then, for any stable matching µ with respect to ( U , P a ) , we define a graph G µ = ( V µ , E µ ) where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Informally, G µ is the graph of workers who (weakly) prefer µ to their match in distribution D , where an edge ( w,w ′ ) means that w ′ received a job that w would have liked. Next, we show properties on the graph G µ , which we illustrate in Figure 3.

Proposition 1. For any stable matching µ , the following holds

- G µ is a directed forest (there is no cycle and each vertex has at most one incoming edge),
- For every worker w ∈ V µ with i = index( w ) , and for every 1 ≤ j &lt; i , there is a worker w ′ ∈ V µ with j = index( w ′ ) such that ( w,w ′ ) ∈ E µ .

And thus

Figure 3: The graph G µ = ( V µ , E µ ) is a directed forest. The matching ˜ µ , computed in Algorithm 1 matches each worker to a single (copy of) job in ˜ µ . In the stable matching µ , each worker w is connected to all copies of µ ( w ) which have lower index.

<!-- image -->

Proof. The graph G µ is a directed forest by construction. Indeed, it has no cycle because edges connect workers to lower index workers. And every node has at most one incoming edge because ˜ µ matches each worker to at most one job, and µ matches each job to at most one worker.

Fix a worker w ∈ V µ with i = index( w ) , let 1 ≤ j &lt; i , and let a = µ ( w ) . By definition of V µ , worker w weakly prefers µ to D , that is U ( w,a ) ≥ m · U D ( w ) = U ( w, ˜ µ i ( w )) . By definition of w 's preference list P w in Algorithm 1, the lexicographic ordering gives that

<!-- formula-not-decoded -->

Because ˜ µ is a stable matching, it should not be blocked by the pair ( w,a ( j ) ) . Thus, there exists a worker w ′ ∈ W such that ˜ µ ( w ′ ) = a ( j ) and

<!-- formula-not-decoded -->

Finally, because µ is a stable matching, it should not be blocked by the pair ( w ′ , a ) , thus

<!-- formula-not-decoded -->

proving that w ′ ∈ V µ . Hence, there is an edge ( w,w ′ ) ∈ E µ , which concludes the proof.

Proposition 2. In the graph G µ , each node of index i ≥ 1 can reach 2 i -1 nodes (including itself).

Proof. We show that the property holds by induction on i . The property trivially holds for i = 1 . Let i &gt; 1 such that there is a worker w ∈ V µ with index( w ) = i . Using Proposition 1, there is an edge ( w,w j ) ∈ E µ with index( w j ) = j for every 1 ≤ j &lt; i . Because the graph is a directed forest, the set of nodes reachable from each w j are disjoint. Thus, the number of nodes reachable from w (including itself) is 1 + ∑ i -1 j =1 2 j -1 = 2 i -1 .

Finally, we conclude with the proof that Algorithm 1 computes a distribution over internally stable matching which guarantees each worker a logarithmic fraction of their optimal stable share.

Proof of Theorem 3. Algorithm 1 first computes a stable matching ˜ µ for the instance with duplicated jobs, then build m matchings ˜ µ 1 , . . . , ˜ µ m . If there were a pair ( w,a ) with index( w ) = i which blocks matching ˜ µ i , that is U ( w,a ) &gt; U ( w, ˜ µ i ( w )) and w ≻ a ˜ µ i ( a ) , then ( w,a ( i ) ) would block ˜ µ , which is a contradiction. Thus, each matching ˜ µ i is internally stable.

Now, let us assume, that there is a stable matching µ in which a worker w with index( w ) = i receives a = µ ( w ) having utility U ( w,a ) &gt; U ( w, ˜ µ i ( w )) . In the matching ˜ µ , job a ( m ) must be matched to some worker w ′ such that w ′ ≻ a w , otherwise ( w,a ) would block ˜ µ . Moreover, we must have U ( w ′ , µ ( w ′ )) ≥ m · U D ( w ′ ) otherwise ( w ′ , a ) would be blocking µ . Thus, there is a node w ′ ∈ V µ of index m , which proves that there exists at least 2 m -1 nodes, and thus that N ≥ 2 m -1 . By contrapositive, if we set m&gt; 1 + log 2 N , then we have m · U D ( w ) ≥ U ∗ ( w ) for every worker w , which concludes the proof. 4

4 Interestingly, we can show that m&gt; log 2 N suffices because each µ ( w ) ( i ) is matched in ˜ µ to a different worker w i ∈ V µ , who can reach 2 i -1 distinct workers in G µ , none of them being w (as this would contradict ˜ µ being worker-optimal), which gives at least 2 m workers in total. However, for the sake of simplicity, we do not present this improved bound.

## C.3 Dominant Strategy Incentive Compatibility of Algorithm 1

Proof of Theorem 4. We will use the fact that when workers have strict preferences, Gale and Shapley's worker-proposing deferred acceptance procedure is dominant strategy incentive compatible, i.e., it is always optimal for workers to report their true preferences [17].

First, notice that for each worker w , the ranking P w used in Algorithm 1 aligns with her utility, ensuring that all copies of a higher-utility job are ranked above copies of lower-utility ones.

To see that it is optimal for a worker w to report her true vector of utility, we will give her more strategic power, and we will let her choose her ranking P ′ w over all the duplicated jobs. By the incentive compatibility property of the deferred acceptance procedure with strict preferences, she cannot obtain any job ranked above ˜ µ ( w ) in P w . And because P w is consistent with w 's utility, it is optimal to report P ′ w = P w .

## D ϵ -Oracle for Approximated Worker Optimal Stable Matching

## D.1 ϵ -Oracle

Algorithm 2 ϵ -Oracle for Approximated Worker Optimal Stable Matching

Input: N workers, K jobs, Utility matrix U that encodes the preference of workers over jobs, strict preference profile P a of jobs, an integer m ≥ 1 , and the instability tolerance ϵ ≥ 0 .

- 1: For each job a ∈ A , duplicate it m times and denote the i -th copy as a ( i ) .
- 2: Each replica a ( i ) shares the same preference P a as the original job a .
- 3: For every worker w and job a ( i ) , define the utility

<!-- formula-not-decoded -->

and use it to generate the workers' preference profile P w (breaking ties in favor of lower indices).

- 4: Run Gale-Shapley algorithm on P w and P a to compute a worker-optimal stable matching ˜ µ .
- 5: For each i ∈ [ m ] , build a matching ˜ µ i , which matches each job a with ˜ µ i ( a ) := ˜ µ ( a ( i ) ) .

Output: The distribution D which selects each matching ˜ µ i with probability 1 /m .

## D.2 Proof of Theorem 5

Similarly to the proof of Theorem 3, we run Algorithm 2 and define the index of a worker as the index of the job she receives in ˜ µ , that is, index ( w ) = i if worker w receives a ( i ) j for some j .

Definition 9. Given a problem instance ( U , P a ) , run Algorithm 2 with duplication number m and instability tolerance ϵ to generate the output distribution D . Then, for any ϵ -stable matching µ with respect to ( U , P a ) , we define a graph G µ = ( V µ , E µ ) where

<!-- formula-not-decoded -->

Once again, G µ is the graph of workers who prefer µ to their match in distribution D , where an edge ( w,w ′ ) means that w ′ received a job that w would have liked. Next, we show properties on the graph G µ .

Proposition 3. For any stable matching µ , we have that

- G µ is a directed forest (there is no cycle and each vertex has at most one incoming edge),
- For every worker w ∈ V µ with i = index( w ) , and for every 1 ≤ j &lt; i , there a worker w ′ ∈ V µ with j = index( w ′ ) such that ( w,w ′ ) ∈ E µ .

Proof. The proof is almost identical to that of Proposition 1. Fix a worker w ∈ V µ with i = index( w ) , let 1 ≤ j &lt; i , and let a = µ ( w ) . By definition of V µ , worker w prefers µ to D , that is U ( w,a ) ≥

m · U D ( w ) -ϵ = U ( w, ˜ µ i ( w )) -ϵ . By definition of w 's preference list P w in Algorithm 2, the lexicographic ordering gives that

<!-- formula-not-decoded -->

Because ˜ µ is a ϵ -stable matching, it should not be blocked by the pair ( w,a ( j ) ) . Thus, there exists a worker w ′ ∈ W such that ˜ µ ( w ′ ) = a ( j ) and

<!-- formula-not-decoded -->

Finally, because µ is a stable matching, it should not be blocked by the pair ( w ′ , a ) , thus

<!-- formula-not-decoded -->

proving that w ′ ∈ V µ . Hence, there is an edge ( w,w ′ ) ∈ E µ , which concludes the proof.

We will once again use Proposition 2 to give a lower on the number of nodes in the graph G µ . Finally, we conclude with the proof that Algorithm 2 computes a distribution over internally ϵ -stable matching which guarantees each worker a logarithmic fraction of their optimal stable share.

Proof of Theorem 5. Algorithm 2 first computes a stable matching ˜ µ for the instance with duplicated jobs, then build m matchings ˜ µ 1 , . . . , ˜ µ m . If there were a pair ( w,a ) with index( w ) = i which ϵ -blocks matching ˜ µ i , that is U ( w,a ) &gt; U ( w, ˜ µ i ( w )) + ϵ and w ≻ a ˜ µ i ( a ) , then ( w,a ( i ) ) would block ˜ µ , which is a contradiction. Thus, each matching ˜ µ i is internally stable.

Now, let us assume, that there is a stable matching µ in which a worker w with index( w ) = i receives a = µ ( w ) having utility U ( w,a ) &gt; U ( w, ˜ µ i ( w )) + mϵ . In the matching ˜ µ , job a ( m ) must be matched to some worker w ′ such that w ′ ≻ a w , otherwise ( w,a ) would block ˜ µ . Moreover, we must have U ( w ′ , µ ( w ′ )) ≥ m · U D ( w ′ ) -ϵ otherwise ( w ′ , a ) would be blocking µ . Using Proposition 2, there is a node w ′ ∈ V µ of index m , which proves that there exists at least 2 m -1 nodes, and thus that N ≥ 2 m -1 . By contrapositive, if we set m&gt; 1 + log 2 N , then we have U D ( w ) ≥ U ∗ ( w ) /m -ϵ for every worker w , which concludes the proof.

## D.3 Robustness of ϵ -stable matching

Lemma 1. Fix the preferences of jobs over workers. Given two utility matrices U 1 and U 2 such that 5 ∥ U 1 -U 2 ∥ max &lt; ϵ 2 , then any stable matching µ for U 1 , is also ϵ -stable with respect to U 2 .

Proof. If a matching µ is stable with respect to U 1 , then for any ( w,a ) pair such that w ≻ a µ ( a ) , we must have

<!-- formula-not-decoded -->

from the definition of stable matching (Definition 1).

Since ∥ U 1 -U 2 ∥ max ≤ ϵ 2 , we have that for any ( w,a ) pair, | U 1 ( w,a ) -U 2 ( w,a ) | ≤ ϵ 2 . Therefore,

<!-- formula-not-decoded -->

where the first and the last inequality come from ∥ U 1 -U 2 ∥ max ≤ ϵ 2 , while the second inequality holds according to Eq.(10). Therefore, combining w ≻ a µ ( a ) and Eq.(11), we can conclude that matching µ is ϵ -stable with respect to U 2 .

## D.4 Proof of Theorem 6

For convenience, we denote S U (resp. S U ϵ ) the stable matchings ( ϵ -stable matchings) with respect to U .

Proof. By running Algorithm 2 with U = ˆ U , ϵ = ϵ, m = ⌊ log 2 N +2 ⌋ , from Theorem 5, we get

<!-- formula-not-decoded -->

5 We recall that the max norm of a matrix A = ( A i,j ) , is defined by ∥ A ∥ max = max i,j | A i,j | .

By construction, any utility matrix U ∈ U satisfies ∥ U -ˆ U ∥ max ≤ ϵ 2 . From Lemma 1, we know that for any matching µ ∈ S U , ∀ U ∈ U , we have µ ∈ S ˆ U ϵ , that is ⋃ U ∈U S U ⊆ S ˆ U ϵ . Therefore, for the optimal stable share, we have

<!-- formula-not-decoded -->

Combining Eq.(12) and (13), the conclusion holds.

## E Explore-then-Choose-Oracle Algorithm

Algorithm 3 is the full version of the Explore-then-Choose-Oracle algorithm.

## Algorithm 3 Explore-then-Choose-Oracle (Full version)

Input: N workers, K jobs, horizon T , exploration length T 0 &lt; T , preference profile P a for all jobs a ∈ K , approximation stable-matching oracle O .

- 1: Initialize: ˆ U ( i, j ) = 0 , T i,j = 0 , ∀ i ∈ [ N ] , j ∈ [ K ] .
- 2: Initialize: F i ← False. ▷ Whether the CIs of the first ( N +1) -ranked jobs are disjoint.

3:

Set

t

= 1

, T

0

←

K

⌊

T

0

/K

⌋

, t

m

= 0

- 4: while t ≤ T 0 and ∃ i ∈ [ N ] s.t. F i == False do

To have full rounds of round-robin.

- ▷ Phase 1, round-robin exploration.
- 5: Match µ t ( i ) ← a (( t + i -1) mod K )+1 , ∀ i ∈ [ N ] .
- 6: Observe X i,µ t ( i ) ( t ) and update ˆ U ( i, µ t ( i )) , T i,µ t ( i ) as follows:

<!-- formula-not-decoded -->

```
7: t ← t +1 8: if t mod K == 0 then ▷ Completed a full round of round-robin 9: t m ← t m +1 . 10: Compute UCB i,j and LCB i,j for all i ∈ [ N ] , j ∈ [ K ] . ▷ See Equation (5) 11: for i = 1 , 2 , · · · , N do 12: ˆ U sort ( i, · ) ← Sort ( ˆ U ( i, · ) , decreasing ) 13: ∆ i, min ← min { ˆ U sort ( i, j ) -ˆ U sort ( i, j +1) , j ∈ [ N ] } 14: if ∆ i, min > 2 √ 6 ln T t m then 15: F i ← True. 16: end if 17: end for 18: if F i == True, ∀ i ∈ [ N ] , then ▷ No ties - standard oracle 19: Compute preference list P w for all w ∈ W according to ˆ U 20: ˆ µ ∗ ← worker-optimal stable matching w.r.t P w and P a (using GS algorithm) 21: else if t = T 0 then ▷ Potential ties - approximation oracle 22: ˆ µ ∗ ← O ( ¯ U ) for ¯ U s.t. ¯ U ( i, j ) = UCB i,j for all i ∈ [ N ] , j ∈ [ K ] 23: end if 24: end if 25: end while 26: while t ≤ T do ▷ Phase 2, exploitation with the chosen oracle. 27: Match µ t ( i ) ← ˆ µ ∗ ( i ) , ∀ i . 28: t ← t +1 29: end while
```

## F Technical Lemmas

Lemma 2 (Corollary 5.5 in Lattimore and Szepesvári [40]) . Assume that X 1 , X 2 , · · · , X n are independent, σ -subgaussian random variables centered around µ . Then for any ε &gt; 0 ,

<!-- formula-not-decoded -->

▷

Lemma 3 (Divergence Decomposition, Lemma 15.1 in Lattimore and Szepesvári [40]) . For two bandit instances ν = { ν ij : i ∈ [ N ] , j ∈ [ K ] } , and ν ′ = { ν ′ ij : i ∈ [ N ] , j ∈ [ K ] } , fix some policy π and let P ν,π and P ν ′ ,π be the probability measures induced by the T -round interconnection of π and ν (respectively, π and ν ′ ), the following divergence decomposition holds,

<!-- formula-not-decoded -->

Lemma 4 (Data-processing Inequality, Lemma 1 in Garivier et al. [24]) . Consider a measurable space (Ω , F ) equipped with two distributions P 1 and P 2 , and any F -measurable random variable Z : Ω → [0 , 1] . We denote respectively by E 1 and E 2 the expectations under P 1 and P 2 . Then,

<!-- formula-not-decoded -->

where kl denotes the KL divergence for Bernoulli distributions, i.e., ∀ p, q ∈ [0 , 1] 2 , kl ( p, q ) = p ln p q +(1 -p ) ln 1 -p 1 -q .

## G Proof of Therorem 7

For convenience, let ˆ U ( t ) ( i, j ) , T ( t ) i,j , UCB ( t ) i,j , LCB ( t ) i,j be the value of ˆ U ( i, j ) , T i,j , UCB i,j , LCB i,j at the end of round t . Define F = { ∃ t ∈ [ T ] , i ∈ [ N ] , j ∈ [ K ] : | ˆ U ( t ) ( i, j ) -U ( i, j ) | &gt; √ 6 ln T T ( t ) i,j } as the bad event that some preference is not estimated well during the horizon.

## Lemma 5.

Proof.

<!-- formula-not-decoded -->

where the second last inequality results from Lemma 2.

Lemma 6. Conditional on ⌝ F , UCB ( t ) i,j &lt; LCB ( t ) i,j ′ implies U ( i, j ) &lt; U ( i, j ′ ) .

Proof. According to the definition of LCB and UCB, we have that conditional on ⌝ F ,

<!-- formula-not-decoded -->

Therefore, if UCB ( t ) i,j &lt; LCB ( t ) i,j ′ , we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 7. In round t , let T ( t ) i = min j ∈ [ K ] T ( t ) i,j . Conditional on ⌝ F , if T ( t ) i &gt; 96 ln T/ ∆ 2 min , we have LCB ( t ) i,ρ i,k &gt; UCB ( t ) i,ρ i,k +1 for any k ∈ [ N ] , and LCB ( t ) i,ρ i,N &gt; UCB ( t ) i,ρi,k for any N +1 ≤ k ≤ K .

Proof. We prove it by contradiction, suppose that there exists k ∈ [ N ] such that LCB ( t ) i,ρ i,k ≤ UCB ( t ) i,ρ i,k +1 or there exists N +1 ≤ k ≤ K such that LCB ( t ) i,ρ i,N ≤ UCB ( t ) i,ρi,k . Without loss of generality, denote j as the arm on the LHS and j ′ as the arm on the RHS.

Conditional on ⌝ F and by the definition of LCB and UCB , we have that

<!-- formula-not-decoded -->

Therefore, ∆ i,j,j ′ = U ( i, j ) -U i,j ′ ≤ 4 √ 6 ln T T ( t ) i , which implies that T ( t ) i ≤ 96 ln T ∆ 2 i,j,j ′ ≤ 96 ln T ∆ 2 min , which is a contradiction.

Lemma 8. Conditional on ⌝ F , if ∆ min &gt; √ 96 K ln T T 0 , Algorithm 3 would enter the exploitation phase and choose the Gale-Shapley oracle at some t ≤ T 0 .

Proof. If ∆ min &gt; √ 96 K ln T T 0 , we have T 0 &gt; 96 K ln T ∆ 2 min . Since for every worker, Algorithm 3 allocates jobs in a round-robin fashion, we have that T ( t ) i &gt; 96 ln T ∆ 2 min .

By Lemma 7, we know that for any worker w i , LCB ( t ) i,ρ i,k &gt; UCB ( t ) i,ρ i,k +1 for any k ∈ [ N ] , and LCB ( t ) i,ρ i,N &gt; UCB ( t ) i,ρi,k for any N +1 ≤ k ≤ K , i.e., the preference utility for the first N -ranked jobs for every worker has been estimated well enough with the confidence intervals disjoint. The flag F i would be set as True as in Line 15 in Algorithm 3 and we would enter Phase 2 at some time t ≤ T 0 .

Lemma 9. Given a utility matrix U N × K without ties, the worker-optimal stable matching job of each worker must be its first N -ranked.

Proof. We implement the Gale-Shapley algorithm with the workers as the proposing side. Once a job is proposed, it has a temporary worker. By contradiction, once N jobs have been proposed, we have N workers occupied. Therefore, each worker would be allocated with a job and the Gale-Shapley algorithm would stop. Since in the deferred-acceptance procedure, workers propose to jobs one by one according to their preference list, then the worker-optimal stable matching job of each worker must be its first N -ranked.

Proof of Theorem 7. We consider the two cases separately.

<!-- formula-not-decoded -->

Let ∆ i, max = max j ∈ [ K ] [ U ∗ ( w i ) -U ( i, j )] be the maximum worker-optimal stable regret that may be suffered by w i in all rounds, we have ∆ i, max ≤ 1 . The worker-optimal stable regret for each

worker w i by following Algorithm 3 satisfies

̸

<!-- formula-not-decoded -->

̸

where Eq.(15) comes from the fact that in a matching market without ties, there is a unique workeroptimal stable matching and hence a unique optimal stable match µ ∗ ( i ) for worker i , Eq.(16) holds based on Lemma 5, Eq.(17) holds according to Lemma 8 and 9 and the fact that Gale-Shapley algorithm could always output the worker-optimal stable matching with respect to the given utility matrix by treating worker as the proposing side.

<!-- formula-not-decoded -->

The objective function is the approximation regret Reg α i ( T ) . Denote F ( t ) d as the event that LCB ( t ) i,ρ i,k &gt; UCB ( t ) i,ρ i,k +1 for all k ∈ [ N ] , and LCB ( t ) i,ρ i,N &gt; UCB ( t ) i,ρi,k for all N +1 ≤ k ≤ K . We have

<!-- formula-not-decoded -->

where Eq.(18) comes from the fact that U ∗ ( w i ) ≤ 1 and X i ( t ) ≥ 0 . Eq.(19) holds according to Lemma 5. Eq.(20) comes from the fact that when the good event ⌝ F that all utilities are well estimated and the top ( N +1) -ranked CIs are disjoint before T 0 , the Gale-Shapley algorithm would give us the OSS in the exploitation phase, and since α ∈ (0 , 1] , the approximation regret would be no larger than αT 0 +( α -1) · ( T -T 0 ) ≤ αT 0 .

̸

Moreover, conditional on ⌝ F , we have the ground-truth utility matrix U lies in the uncertainty set constructed by the empirical mean utility matrix ˆ U ( T 0 ) and the UCB ( T 0 ) and LCB ( T 0 ) , i.e., for any ( i, j ) ∈ [ N ] × [ K ] , | ˆ U ( T 0 ) ( i, j ) -U ( i, j ) | ≤ √ 6 K ln T T 0 . If we implement an ( α , ϵ ) -oracle, with ϵ being 2 √ 6 K ln T T 0 , follow a similar proof as that for Theorem 6, in each round t in the exploitation phase, we have that

<!-- formula-not-decoded -->

Since there are in total T -T 0 rounds of exploitation, we have that

<!-- formula-not-decoded -->

Therefore, combining Eq.(20) and (22), we have

<!-- formula-not-decoded -->

## H Proof of Theorem 8

Proof. Let W = { w 1 , w 2 , w 3 , w 4 } and A = { a 1 , a 2 , a 3 , a 4 } and w 1 ≻ w 2 ≻ w 3 ≻ w 4 for all the jobs. Throughout the proof, we assume that all observations are Gaussian of unit variance, that is, when matching w i to a j at round t , we observe X i ( t ) ∼ N ( U ( i, j ) , 1) . Consider two instances ν and ν ′ with the following mean utility matrices U and U ′ , respectively.

<!-- formula-not-decoded -->

where γ &lt; 1 4 .

Lemma 10 (Properties of Instances ν and ν ′ ) . Based on the utility matrices U and U ′ , we have the following properties of ν and ν ′ :

1. Under ν , the optimal stable shares are U ∗ ( w ) = 1 2 , ∀ w ∈ W ; Under ν ′ , the optimal stable shares are ( U ′ ) ∗ ( w 1 ) = 1 2 + γ , ( U ′ ) ∗ ( w 2 ) = 1 2 , ( U ′ ) ∗ ( w 3 ) = 1 4 , and ( U ′ ) ∗ ( w 4 ) = 0 .
2. The relevant utility gaps for the two instances are ∆ ν rel = 0 , and ∆ ν ′ rel = γ .
3. Given an offline oracle that could compute the best approximation ratio, the benchmark utilities for the four workers (after multiplying the approximation ratio) are ( 1 2 , 3 8 , 3 8 , 3 8 ) under ν and ( 1 2 + γ, 1 2 , 1 4 , 0) under ν ′ .

We provide the proof of Lemma 10 in Appendix I.

For a worker w i , i ∈ [ N ] , job a j , j ∈ [ K ] and time slot t ∈ [ T ] , denote N ij ( t ) ∈ N ∪{ 0 } as the number of times worker w i is matched to job a j , up to and including time t , and denote the past information as I t := ( µ 1 , X (1) , µ 2 , X (2) , · · · , µ t -1 , X ( t -1)) , where X ( t ) = ( X 1 ( t ) , X 2 ( t ) , · · · , X N ( t )) is the realized reward vector for all N workers in round t . Finally, let P ν ,π be the joint probability measure over the history and E ν ,π be the expectation induced by instance ν and policy π , and P ν ′ ,π , E ν ′ ,π be defined similarly. By divergence decomposition theorem [40, restated in Lemma 3], we have that

<!-- formula-not-decoded -->

where ν ij is the distribution of utilities obtained when worker w i is matched to job a j in the environment ν .

Since the only change in utility distribution happens in ( w 1 , a 1 ) pair, we have that

<!-- formula-not-decoded -->

where the second equality comes from the fact that for two Gaussian distributions with means 1 2 and 1 2 + γ and variance 1 , the KL divergence is γ 2 2 .

By data-processing inequality [40, restated in Lemma 4], we know that for all σ ( I T ) -measurable random variable Z ∈ [0 , 1] , we have that

<!-- formula-not-decoded -->

where kl denotes the KL divergence between two Bernoulli distributions, i.e., ∀ p, q ∈ [0 , 1] 2 , kl( p, q ) = p ln p q +(1 -p ) ln 1 -p 1 -q .

Let Z = N 21 ( T )+ N 23 ( T ) T , then Z ∈ [0 , 1] , by Pinsker's inequality, we have that

<!-- formula-not-decoded -->

Combining Eq.(23), (24), (25), we have that

<!-- formula-not-decoded -->

We now divide into two cases, depending on the asymptotic number of matches E ν ,π [ N 11 ( T )] .

Case I: lim inf T →∞ E ν ,π [ N 11 ( T )] T 1 -2 δ = 0 . We assume that both Reg i ( T ; ν ′ ) and Reg α ∗ ( w i ) i ( T ; ν ) are sublinear for all workers and show that we have a contradiction.

Since γ = cT -1 2 + δ , by Eq. (26), we have

<!-- formula-not-decoded -->

In ν ′ , if the ground-truth utility matrix U ′ is known and we would like to achieve the benchmark utility for w 2 , we need N 21 ( T ) + N 23 ( T ) = T , since worker w 2 could only get positive utilities from jobs a 1 and a 3 and her benchmark utility is 1 2 . In particular, the regret for this worker is

<!-- formula-not-decoded -->

and to guarantee sublinear regret for w 2 for any large enough T , we must have

<!-- formula-not-decoded -->

Therefore, to satisfy Eq.(27), we must also have

<!-- formula-not-decoded -->

On the other hand, since w 4 only gets positive utilities in U (4 , 3) , to achieve the benchmark utility in ν , we need N 43 ( T ) = 3 4 T . Therefore, to guarantee sublinear approximation regret for w 4 , we must have

<!-- formula-not-decoded -->

which implies lim sup T →∞ E ν ,π [ N 23 ( T )] T ≤ 1 4 , since the total number of times that jobs a 3 being allocated cannot be more than the horizon T . Therefore, lim inf T →∞ E ν ,π [ N 21 ( T )] T ≥ 3 4 according to Eq.(28). Using again the fact that a job can be allocated no more than T times, we get

<!-- formula-not-decoded -->

Finally, we write the regret of w 3 as

<!-- formula-not-decoded -->

where the inequality is since w 3 is matched at most T times, namely, E ν ,π [ N 34 ( T )]+ E ν ,π [ N 31 ( T )] ≤ T . Combining with Eq. (29), we have lim inf T →∞ Reg α 3 ( T ; ν ,π ) T ≥ 1 8 -1 4 · 1 4 = 1 16 , which implies worker w 3 suffers linear approximation regret in ν .

Thus, to summarize, assuming that all workers in both problem exhibit sublinear regret for this case leads to a contradiction.

Case II: lim inf T →∞ E ν ,π [ N 11 ( T )] T 1 -2 δ &gt; 0 . Our goal is to prove a lower bound on the regret in ν for some worker.

For a fixed T , denote for brevity N = E ν ,π [ N 11 ( T )] , and assume with contradiction that all workers w 2 , w 3 , w 4 suffer a regret smaller than N/ 32 . Denote the cumulative allocation given by the algorithm by D ∈ [0 , T ] 4 × 4 , namely D ( i, j ) = ∑ T t =1 1 { ( w i , a j ) ∈ µ t } . In particular, we know that D (1 , 1) = N and that for all i ∈ { 2 , 3 , 4 } , it holds that

<!-- formula-not-decoded -->

We now state a set of assumptions on the matching that the policy outputs at each round. Each of these assumptions never decreases the worker utility. Thus, and since we want to prove a contradiction in the utility lower bound of Eq. (30), they could be assumed without loss of generality.

1. When w 1 is not matched to a 1 , it is always matched to a 2 ( D (1 , 1) + D (1 , 2) = T ) - so that it suffers zero regret.
2. a 3 is always matched to either w 2 or w 4 ( D (2 , 3) + D (4 , 3) = T ).
3. a 1 is always assigned to one of the first three workers ( D (1 , 1) + D (2 , 1) + D (3 , 1) = T ).
4. If a 1 is not assigned to w 3 , then a 4 is assigned to w 3 ( D (3 , 1) + D (3 , 4) = T ).

Notice that changing each individual allocation to follow this condition can only require unmatching a worker from a job that yields her no utility and matching all conditions is feasible (e.g., by µ = { ( w 1 , a 1 ) , ( w 3 , a 4 ) , ( w 4 , a 3 ) } ).

We now modify the matching allocation D to allocation ¯ D while maintaining the above properties as follows:

- We initialize ¯ D = D .
- If D (4 , 3) ≤ 3 T 4 + N 16 , we set ¯ D (4 , 3) = 3 T 4 + N 16 and ¯ D (2 , 3) = T 4 -N 16 ; otherwise, we leave ¯ D (4 , 3) = D (4 , 3) . By Eq. (30), we know that D (4 , 3) ≥ 3 T 4 -N 16 , and combined with Assumption 2, this change can only decrease the allocation to worker w 2 by

<!-- formula-not-decoded -->

This decreases the utility of worker w 2 by 1 2 ( D (2 , 3) -¯ D (2 , 3) ) ≤ N 16 , and after this modification, the cumulative utility of worker w 4 is 3 T 8 + N 32 .

- By Assumption 1, we know that D (1 , 1) = N and D (1 , 2) = T -N . We also know by assumption 4 that in all rounds where a 1 was assigned to w 1 , a 4 was assigned to w 3 , and therefore, D (3 , 4) ≥ N . In ¯ D , we move all the N assignments of ( w 1 , a 1 ) to ( w 1 , a 2 ) , so that ¯ D (1 , 1) = 0 and ¯ D (1 , 2) = T ; in particular, w 1 still gets its OSS. We split the allocation of a 1 evenly between w 2 and w 3 by letting:

1. ¯ D (2 , 1) = min { T -¯ D (2 , 3) , D (2 , 1) + N/ 2 } , thus making sure that the utility of w 2 is at least either T/ 2 ≥ 3 T/ 8 + N/ 16 or

<!-- formula-not-decoded -->

where in the last inequality we again used the assumption on the regret of w 2 in Eq.(30).

2. We move the matches from ( w 1 , a 1 ) that were not allocated to D (2 , 1) as follows:

<!-- formula-not-decoded -->

Both allocations are valid since D (3 , 4) ≥ N due to Assumption 4. In particular, this shift from a 4 to a 1 increases the utility of w 3 by at least ( 1 2 -1 4 ) N 2 = N 8 , ensuring it a total utility of at least 3 T/ 8 + 3 N/ 32 .

Notice that all changes either kept ¯ D a doubly-stochastic matrix or decreased the sum of a row - we can w.l.o.g increase another element of ¯ D or use partial matchings.

Importantly, at the end of this process, the utility of w 1 under the matching distribution ¯ D remained T/ 2 , while the utility of all other workers increased by at least N/ 16 - contradicting the fact that no matching distribution can collect more than 3 T/ 8 to all workers w 2 , w 3 , w 4 . Thus, Eq. (30) cannot hold and at least one worker must suffer a regret of at least N/ 32 . Finally, since lim inf T →∞ E ν ,π [ N 11 ( T )] T 1 -2 δ &gt; 0 , we get the same for the regret of one of the workers, concluding the proof.

## I Proof of Lemma 10

Proof. We proof the three properties as follows.

## 1. Optimal Stable Share

Under ν , consider the following stable matchings: µ 1 = { ( w 1 , a 2 ) , ( w 2 , a 1 ) , ( w 3 , a 4 ) , ( w 4 , a 3 ) } and µ 2 = { ( w 1 , a 2 ) , ( w 2 , a 3 ) , ( w 3 , a 1 ) , ( w 4 , a 4 ) } . The optimal stable share is U ∗ ( w ) = 1 2 , ∀ w ∈ W with w 1 and w 2 receives it in both matchings, w 3 receives it in matching µ 2 and w 4 receives it in matching µ 1 . Under ν ′ , the optimal stable shares are ( u ′ ) ∗ ( w 1 ) = 1 2 + γ , ( u ′ ) ∗ ( w 2 ) = 1 2 , ( u ′ ) ∗ ( w 3 ) = 1 4 , and ( u ′ ) ∗ ( w 4 ) = 0 , achieved through the stable matching µ ′ = { ( w 1 , a 1 ) , ( w 2 , a 3 ) , ( w 3 , a 4 ) , ( w 4 , a 2 ) } .

## 2. Relevant Utility Gap

Besides, the relevant utility gap for ν is ∆ ν rel = 0 since both µ 1 and µ 2 belongs to the Pareto-optimal stable matching set S U opt . On the other hand, since the jobs have global preference rankings over the workers, i.e., serial dictatorship, µ ′ is the unique stable matching with respect to ν ′ , i.e., S U ′ opt = { µ ′ } . A perturbation of -γ in U (1 , 1) or γ in U (1 , 2) brings ties for worker w 1 , and hence would change S U ′ opt tp S U opt . On the other hand, a perturbation of 1 4 in U (3 , 2) or -1 4 in U (3 , 4) would make S U ′ opt include both µ ′ and µ ′ 2 = { ( w 1 , a 1 ) , ( w 2 , a 3 ) , ( w 3 , a 2 ) , ( w 4 , a 4 ) } . All other entries need a perturbation of scale larger than 1 4 to change the Pareto-optimal stable matching set. Since γ &lt; 1 4 , we have that ∆ ν ′ rel = γ .

## 3. Benchmark Utility

Let γ = cT -1 / 2+ δ for some δ ∈ (0 , 1 2 ) . Then, under ν , we aim to minimize the approximation regret Reg α i ( T ) for every worker w i , while under ν ′ , the objective is to minimize regret Reg i ( T ) for each worker w i . Given an offline oracle that could compute the best possible approximation ratio, the benchmark utilities for the four workers (after multiplying the approximation ratio) are ( 1 2 , 3 8 , 3 8 , 3 8 ) under ν and ( 1 2 + γ, 1 2 , 1 4 , 0 ) under ν ′ . For ν ′ , by serial dictatorship, the allocation scheme is equivalent to letting the workers choose their favorite jobs one by one. Doing so leads to the matching µ = { ( w 1 , a 2 ) , ( w 2 , a 3 ) , ( w 3 , a 1 ) , ( w 4 , a 4 ) } , which is unique since for all workers, when they choose their jobs, only a single unallocated job maximizes their utility. Hence, the benchmark utilities are immediately determined by the utility under this matching.

For ν , since all workers but w 1 have no utility from job a 2 , and U ∗ ( w 1 ) = U (1 , 2) , it is always optimal to assign a 2 to w 1 deterministically and the benchmark utility for w 1 is 1 / 2 . For the other players, if we match w 2 to a 1 , then for w 3 and w 4 , there are two possible matchings, i.e., { ( w 3 , a 4 ) , ( w 4 , a 3 ) } and { ( w 3 , a 3 ) , ( w 4 , a 4 ) } , but the first one is always better since it gives both players higher utilities. Similarly, if we match w 2 to a 3 , it is always better to select the matching { ( w 3 , a 1 ) , ( w 4 , a 4 ) } rather than { ( w 3 , a 4 ) , ( w 4 , a 3 ) } , and if w 2 is matched to a 4 , then we choosing { ( w 3 , a 1 ) , ( w 4 , a 3 ) } yields higher utilities than choosing { ( w 3 , a 3 ) , ( w 4 , a 1 ) } . Since all three three players have an OSS of 1/2, to maximize the OSS ratio, we need to compute a distribution D over the three matchings µ a = { ( w 2 , a 1 ) , ( w 3 , a 4 ) , ( w 4 , a 3 ) } , µ b = { ( w 2 , a 3 ) , ( w 3 , a 1 ) , ( w 4 , a 4 ) } and µ c = { ( w 2 , a 4 ) , ( w 3 , a 1 ) , ( w 4 , a 3 ) } , such that min { u D ( w 2 ) , u D ( w 3 ) , u D ( w 4 ) } is maximized. Noticing that each of the three matchings yields the OSS to two players and a lower utility for the third one, we can conclude that the optimal balance would be u D ( w 2 ) = u D ( w 3 ) = u D ( w 4 ) -otherwise, we could increase the OSS-ratio by moving utility from the highest rewarded player to the lowest rewarded one. This condition is satisfied iff

<!-- formula-not-decoded -->

and the benchmark utility for any of these three players is 3 8 .

## J Discussion Regarding Stable Matching with One-sided Ties and Job Utilities

In this paper, we consider a matching market where one side has possibly tied cardinal preferences and the other side has strict ordinal preferences. It directly generalizes to the setting that both sides have cardinal preferences but only one side admits ties, by recovering an ordinal preference list from the utility, and we can define the OSS-ratio for the jobs in a similar fashion, denoted as R a M ; in the following, we rename the OSS-ratio for the workers as R w M for distinguishment. We claim that the setting with one-sided ties is not only practical in reality, but also important for our theoretical results, if we want to consider the OSS-ratio for both sides of the market.

Stable Matching Without Ties The distributive lattice structure is a striking feature for a matching market without ties [36], which reveals that all workers could be optimally matched simultaneously, by simply running the deferred-acceptance algorithm with worker proposing, denoted as µ w . Conversely, job-proposing deferred-acceptance algorithm, denoted as µ a , gives every job its corresponding optimal stable match. Therefore, construct the distribution D as follows:

<!-- formula-not-decoded -->

Since µ w and µ a are both stable matchings, with distribution D , we know that R w M ≤ R w S ≤ 1 2 , and the same result holds for R a M . This implies that both R w M and R a M are Θ(1) , and Ω(1) is a trivial lower bound for the two ratios.

Stable Matching With One-sided Ties The lattice structure is absent when ties exist in the preference profiles [45]. When only one side of the market admits ties, we have proved that R w M = Θ(log N ) . On the other hand, for R a M , we have the following result.

Theorem 9. For a matching market where the workers have ties while the jobs have strict preferences, there exists an instance such that R a M = Ω( N ) .

Proof. Consider the following utility matrix for a market with N workers and N jobs. The matrix encodes the preferences of the workers and the jobs simultaneously.

<!-- formula-not-decoded -->

where ε 1 &gt; ε 2 &gt; · · · &gt; 0 . u i,j ≥ u i,j ′ implies a j ≿ w i a j ′ and u i,j ≥ u i ′ ,j implies w i ≿ a j w i ′ . In this example, every worker is indifferent among all the jobs, while every job has the preference profile that w 1 ≻ w 2 ≻ · · · ≻ w N . The optimal stable share for each job is 1, and it is achieved by an appropriate tie-breaking of w 1 . However, in any matching, exactly one job would receive a utility of 1. When ε 1 , ε 2 , · · · approach 0, we have that max a U ∗ ( a ) U D ( a ) ≥ N , with the equality achieved when we consider the distribution D that assigns probability 1 /N on matching µ j , where µ j refers to a matching in which job a j gets a utility of 1. Therefore, lim ε → 0 R a M = Ω( N ) .

Stable Matching with Two-sided Ties When both sides of the market have ties, by symmetry, the OSS-ratio for the worker side and the job side would be of the same order.

Theorem 10. For a matching market where both sides admit ties, there exists an instance, such that R w M = R a M = Ω( N ) .

Proof. Consider the following N × N utility matrix which simultaneously encodes the preferences of the workers and the jobs.

<!-- formula-not-decoded -->

where the entries of this utility matrix share the same correspondence to the preference profile as those in the example in Theorem 9. In this example, worker w 1 is indifferent among all the jobs, while all the other workers share the same preference over the jobs, that is, a 1 ≻ a 2 ∼ a 3 ∼ · · · ∼ a N . The preference of jobs over workers is symmetrically derived. Every matching that involves all the workers and jobs is a stable matching, which gives a utility of 1 to worker w 1 and job a 1 , while for the remaining workers and jobs, at most one worker and one job would receive a utility of 1, and all the other workers get 0. For every worker and every job, there exists a tie-breaking mechanism such that it gets a utility of 1. Therefore, the optimal stable share is U ∗ ( w ) = U ∗ ( a ) = 1 for any w and a . And any distribution D over matchings gives U ∗ ( w 1 ) U D ( w 1 ) = U ∗ ( a 1 ) U D ( a 1 ) = 1 , max w ∈{ w 2 ,w 3 , ··· ,w N } U ∗ ( w ) U D ( w ) ≥ N -1 , and max a ∈{ a 2 ,a 3 , ··· ,a N } U ∗ ( a ) U D ( a ) ≥ N -1 , with the equality achieved when we adopt the random allocation that assigns probability 1 / ( N -1) on matching µ i , in which workers w 1 and w i , jobs a 1 and a i receive a utility of 1, while all the other workers and jobs get 0.

## K Discusssion Regarding Two-sided Bandit Learning in Matching Markets

In this paper, we consider bandit learning for one side of the market, where the preferences of jobs over workers are assumed to be known. A possible future direction would be to consider two-sided bandit learning for matching markets.

For a matching market without ties, a fundamental result indicates that the worker-optimal stable matching is necessarily job-pessimal, and no stable matching simultaneously maximizes utility for both sides. Therefore, a reasonable benchmark would be to consider the optimal stable matching for workers and the pessimal stable matching for jobs, leading to the following regret definition:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ¯ U ( w i ) represents the utility from the worker-optimal stable matching and U ( a j ) denotes the utility from the job-pessimal stable matching.

Previous work [60] studied regret minimization for both sides of the market in a setting without ties, adopting the regret definitions above. Zhang and Fang [60] establishes an O ( K log T/ ∆ 2 ) regret bound for every agent, measured against their respective benchmark (worker-optimal and

job-pessimal). Our algorithm directly generalizes to the same setting and matches the theoretical guarantees of Zhang and Fang [60] when there are no ties. This is because the initial exploration phase will find the strict preference list for both sides simultaneously w.h.p. and thus, after committing, the resulting Gale-Shapley output will be the worker-optimal / job-pessimal stable matching.

In contrast, introducing ties to the market significantly complicates the analysis. The set of stable matchings expands substantially, and no single matching simultaneously satisfies the benchmarks defined for both sides. To extend our results to markets with ties, we propose using the Optimal Stable Share (OSS) as the benchmark for workers as defined in our paper. It is thus natural to introduce the equivalent Pessimal Stable Share (PSS) - the minimum utility a job can receive across all stable matchings. However, whether efficient algorithms can achieve sublinear regret for all agents in markets with ties remains open and would be interesting to explore.