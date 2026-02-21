## Efficient Last-Iterate Convergence in Solving Extensive-Form Games

Linjian Meng 1 , Tianpei Yang 1 † , Youzhi Zhang 2 † , Zhenxing Ge 1 , Shangdong Yang 3 , Tianyu Ding 4 , Wenbin Li 1 , Bo An 5 , Yang Gao 1

1

National Key Laboratory for Novel Software Technology, Nanjing University 2 Centre for Artificial Intelligence and Robotics, Hong Kong Institute of Science &amp; Innovation, CAS 3 Jiangsu Key Laboratory of Big Data Security and Intelligent Processing, Nanjing University of Posts and Telecommunications

4 Microsoft Corporation

5 School of Computer Science and Engineering, Nanyang Technological University menglinjian@smail.nju.edu.cn, tianpei.yang@nju.edu.cn, youzhi.zhang@cair-cas.org.hk, zhenxingge@smail.nju.edu.cn, sdyang@njupt.edu.cn, tianyuding@microsoft.com, liwenbin@nju.edu.cn, boan@ntu.edu.sg, gaoy@nju.edu.cn

## Abstract

To establish last-iterate convergence for Counterfactual Regret Minimization (CFR) algorithms in learning a Nash equilibrium (NE) of extensive-form games (EFGs), recent studies reformulate learning an NE of the original EFG as learning the NEs of a sequence of (perturbed) regularized EFGs. Hence, proving last-iterate convergence in solving the original EFG reduces to proving last-iterate convergence in solving (perturbed) regularized EFGs. However, these studies only establish last-iterate convergence for Online Mirror Descent (OMD)-based CFR algorithms instead of Regret Matching (RM)-based CFR algorithms in solving perturbed regularized EFGs, resulting in a poor empirical convergence rate, as RM-based CFR algorithms typically outperform OMD-based CFR algorithms. In addition, as solving multiple perturbed regularized EFGs is required, fine-tuning across multiple perturbed regularized EFGs is infeasible, making parameter-free algorithms highly desirable. This paper show that CFR + , a classical parameter-free RM-based CFR algorithm, achieves last-iterate convergence in learning an NE of perturbed regularized EFGs. This is the first parameter-free last-iterate convergence for RM-based CFR algorithms in perturbed regularized EFGs. Leveraging CFR + to solve perturbed regularized EFGs, we get Reward Transformation CFR + (RTCFR + ). Importantly, we extend prior work on the parameter-free property of CFR + , enhancing its stability, which is vital for the empirical convergence of RTCFR + . Experiments show that RTCFR + exhibits a significantly faster empirical convergence rate than existing algorithms that achieve theoretical last-iterate convergence. Interestingly, RTCFR + show performance no worse than average-iterate convergence CFR algorithms. It is the first last-iterate convergence algorithm to achieve such performance. Our code is available at https://github.com/menglinjian/NeurIPS-2025-RTCFR .

## 1 Introduction

Extensive-form games (EFGs) are a foundational model for capturing interactions among multiple agents and sequential events, which are widely applied in simulating real-world scenarios, such as medical treatment [Sandholm, 2015], security games [Lis` y et al., 2016], and recreational games [Brown and Sandholm, 2019b]. A common goal to address EFGs is to learn a Nash equilibrium (NE), where no player can unilaterally improve their payoff by deviating from the equilibrium.

† Corresponding authors.

Recent research commonly employs regret minimization algorithms [Zhang et al., 2022b] to learn an NE in EFGs. Among them, Counterfactual Regret Minimization (CFR) algorithms are the most widely used ones for learning an NE in real-world EFGs [Bowling et al., 2015, Moravˇ cík et al., 2017, Brown and Sandholm, 2018, 2019b, Pérolat et al., 2022]. They usually use Regret Matching (RM) algorithms [Hart and Mas-Colell, 2000, Gordon, 2006, Lanctot et al., 2009, Lanctot, 2013, Tammelin, 2014, Brown and Sandholm, 2019a, Farina et al., 2021, 2023, Xu et al., 2024b] as the local regularizer, since RM algorithms usually exhibit a faster empirical convergence rate than other local regret minimizers, such as Online Mirror Descent (OMD) [Nemirovskij and Yudin, 1983]. For convenience, we refer to the CFR algorithms that employ RM algorithms and OMD algorithms as local regularizers as RM-based CFR algorithms and OMD-based CFR algorithms, respectively.

However, most regret minimization algorithms, including CFR algorithms, typically only achieve average-iterate convergence and their strategy profile may diverge or cycle, even in normal-form games (NFGs) [Bailey and Piliouras, 2018, Mertikopoulos et al., 2018]. Average-iterate convergence implies that the averaging of strategies is necessary, which increases computational and memory overhead. Additionally, when strategies are parameterized via function approximation, a new approximation function must be trained to represent the average strategy, resulting in further approximation errors. Consequently, algorithms with last-iterate convergence to NE, which ensures that the sequence of strategy profiles converges to the set of NEs, are preferable.

To establish last-iterate convergence for CFR algorithms, recent studies [Pérolat et al., 2021, 2022, Liu et al., 2023] employ the Reward Transformation (RT) framework, which (i) transforms the task of learning an NE of the original EFG into learning the NEs of a sequence of (perturbed) regularized EFGs and (ii) ensures the sequence of the NEs of these (perturbed) regularized EFGs converges to the set of NEs of the original EFG. Therefore, to ensure last-iterate convergence in learning an NE of the original EFG, it is sufficient to establish last-iterate convergence in learning an NE of (perturbed) regularized EFGs. Unfortunately, these studies only establish last-iterate convergence in learning an NE of (perturbed) regularized EFGs for OMD-based CFR algorithms, incurring a poor empirical convergence rate to the set of NEs of the original EFG, as illustrated in our experiments.

To improve the empirical convergence rate, we propose Reward Transformation CFR + (RTCFR + ), utilizing CFR + [Tammelin, 2014], a classical parameter-free RM-based CFR algorithm, to solve perturbed regularized EFGs. RTCFR + is inspired by two observations: (i) RM-based CFR algorithms (the CFR algorithms that employ RM algorithms as the local regret minimizer) usually outperform OMD-based CFR algorithms, and (ii) parameter-free algorithms, implying no parameters need to be tuned [Grand-Clément and Kroer, 2021], are desirable to solve multiple perturbed regularized EFGs because fine-tuning across all perturbed regularized EFGs is infeasible. Notably, the parameter in CFR algorithms typically refers to the step sizes. Based on the RT framework, if CFR + has lastiterate convergence in learning an NE of perturbed regularized EFGs, then RTCFR + has last-iterate convergence in learning an NE of the original EFG. Unfortunately, it remains unknown whether CFR + achieves the parameter-free (i.e., holds for any step sizes) last-iterate convergence in learning an NE of perturbed regularized EFGs. It motivates a key question:

Does CFR + have parameter-free last-iterate convergence in learning an NE of perturbed regularized EFGs?

To answer this question, we first provide the non-parameter-free (w.r.t. the step sizes) last-iterate convergence of CFR + , i.e., for any initial accumulated counterfactual regrets, CFR + achieves lastiterate convergence in learning an NE of perturbed regularized EFGs when the step size exceeds a positive constant. We then extend this non-parameter-free result to establish the parameter-free result, i.e., CFR + achieves last-iterate convergence for any initial accumulated counterfactual regrets and step sizes. Note that our parameter-free result holds for any initial accumulated counterfactual regrets-not just the zero initialization in previous works [Farina et al., 2021] 1 -enhancing the stability of CFR + [Farina et al., 2023], which is critical for the empirical convergence of RTCFR + in solving the original EFG. Without our parameter-free result, RTCFR + fails to empirically converge to the set of NEs of the original EFG! To the best of our knowledge, this is the first parameter-free last-iterate convergence guarantee for RM-based CFR algorithms in learning an NE of perturbed regularized EFGs. As a consequence, based on the convergences of the RT framework and CFR + , RTCFR + achieves last-iterate convergence in learning an NE of the original EFG.

1 While Tammelin et al. [2015] establish parameter-free average-iterate convergence of CFR + under any initialization, we show both last- and average-iterate convergence. Their proof techniques differ from ours and the recent RM-based CFR works, which are all based on Farina et al. [2021]. See details in Appendix B.

Specifically, we propose novel techniques to overcome the challenges in the above two steps of the proof. First, the primary challenge in proving the non-parameter-free result is that the smoothness of the instantaneous counterfactual regrets-the key property used in prior works [Liu et al., 2023] to establish the last-iterate convergence of CFR algorithms-cannot be leveraged, since RM algorithms update within the cone of the strategy space while the final output lies in the strategy space itself. To address this, we exploit the fact that an NE represents a best response to others at each infoset in perturbed EFGs. More specifically, this fact allows a term-related to the accumulated counterfactual regrets and the utility obtained by deviating from an NE of perturbed EFGs-can be added. It enables the smoothness of the instantaneous counterfactual regrets to be leveraged, ensuring that the cumulative squared distance between the iterated strategy profiles and the NE of perturbed regularized EFGs remains bounded by a constant across all iterations, thereby guaranteeing lastiterate convergence. Second, the main challenge of proving our parameter-free result is that the property used in prior proofs of the parameter-free property of CFR + -the strategy sequence produced by CFR + remains invariant across different step sizes-holds only when the initial accumulated counterfactual regrets are zero [Farina et al., 2021]. We address this by leveraging the linearity of the projection alongside our non-parameter-free convergence result that holds for any initial accumulated counterfactual regrets. In particular, we use the linearity of projection to show that for any given initial accumulated counterfactual regrets and step sizes, there exists an alternative choice of these parameters that yields an identical strategy profile sequence. By then applying our non-parameter-free result to this alternative setting, we establish that the resulting strategy profile sequence converges to the set of NEs of perturbed regularized EFGs, thus proving the parameter-free last-iterate convergence. Notably, We only provide parameter-free last-iterate convergence results for CFR + . In other words, RTCFR + is not a parameter-free algorithm.

Experimental results across nine instances from five standard EFG benchmarks-Kuhn Poker, Leduc Poker, Goofspiel, Liar's Dice, and Battleship, as well as two heads-up no-limit Texas Hold'em (HUNL) Subgames-demonstrate that RTCFR + achieves a significantly faster empirical convergence rate compared to existing algorithms with theoretical last-iterate convergence guarantees. Interestingly, RTCFR + even performs no worse well as average-iterate convergence CFR algorithms. Notably, it is the first last-iterate convergence algorithm to accomplish this level of performance.

## 2 Preliminaries

Extensive-form games (EFGs). EFG is a commonly used model for modeling tree-form sequential decision-making problems. An EFG can be formulated as G = {N , H , P, A, I , { u i }} . Here, N is the set of players. H is the set of all possible histories. The set of leaf nodes is denoted by Z . For each history h ∈ H , the function P ( h ) represents the player acting at node h , and A ( h ) denotes the actions available at node h . To account for private information, the nodes for each player i are partitioned into a collection I i , referred to as information sets (infosets). For any infoset I ∈ I i , histories h, h ′ ∈ I are indistinguishable to player i . Thus, P ( I ) = P ( h ) , A ( I ) = A ( h ) , ∀ h ∈ I . The notation I denotes I = {I i | i ∈ N} . We also use C i ( I, a ) to denote the set of infosets that belongs to i and will counter after executing a ∈ A ( I ) at infoset I ∈ I i . The notations A max and C max denote max I ∈I | A ( I ) | and max i ∈N ,I ∈I i ,a ∈ A ( I ) C i ( I, a ) , respectively. For each leaf node z , there is a pair ( u 0 ( z ) , u 1 ( z )) ∈ [ -1 , 1] which denotes the payoffs for the min player (player 0) and the max player (player 1), respectively. We define H as the maximum number of actions taken by all players along any path from the root to a leaf node. In two-player zero-sum EFGs, u 0 ( z ) = -u 1 ( z ) , ∀ z ∈ Z . To illustrate the components of an EFG, we provide an example in Appendix A.

Sequence-form strategy. A sequence is an infoset-action pair ( I, a ) , where I ∈ I is an infoset and a is an action belonging to A ( I ) . Each sequence identifies a path from the root node to the infoset I , selecting the action a along this path. The set of sequences for player i is denoted by Σ i . The last sequence encountered on the path from the root node r to I is denoted by ρ I ( ρ I ∈ Σ i ). In other words, ∀ i ∈ N , I ∈ I i , I ∈ C i ( ρ I ) . A sequence-form strategy for player i is a non-negative vector x i indexed over the set of sequences Σ i . For each sequence q = ( I, a ) ∈ Σ i , x i ( q ) is the probability that player i reaches the sequence q when following the strategy x i . We formulate the sequence-form strategy space as a treeplex [Hoda et al., 2010]. Let X i denote the set of sequence-form strategies for player i . We use x i ( I ) = [ x i ( I, a ) | a ∈ A ( I )] to denote the slice of a given strategy x i corresponding to sequences belonging to infoset I , where x i ( I, a ) is value of x i at the sequence ( I, a ) . For each EFG, there always exists a D such that ∀ i ∈ N and x i ∈ X i , ∥ x i ∥ 1 ≤ D .

Nash equilibrium (NE). NE describes a rational behavior where no player can benefit by unilaterally deviating from the equilibrium. For any player, her strategy is the best response to the strategies of others. From the sequence-form strategy framework, learning an NE of EFGs is represented by

<!-- formula-not-decoded -->

where A is the payoff matrix. We use X and X ∗ to denote × i ∈N X i and the set of NE, respectively.

Behavioral strategy. This strategy σ i is defined on each infoset. For any infoset I ∈ I i , the probability for the action a ∈ A ( I ) is denoted by σ i ( I, a ) . We use σ i ( I ) = [ σ i ( I, a ) | a ∈ A ( I )] ∈ ∆ | A ( I ) | to denote the strategy at infoset I , where ∆ | A ( I ) | is a ( | A ( I ) | -1) -dimension simplex. If all players follow the strategy profile σ = { σ 0 , σ 1 } and reaches infoset I , the reaching probability is denoted by π σ ( I ) . The probability contribution from player i is represented by π σ i ( I ) , while the contribution from the other players is represented by π σ -i ( I ) , where -i refers to all players except player i . Notably, ∀ i ∈ N , I ∈ I i , a ∈ A ( I ) , x i ∈ X i , x i ( I, a ) = π σ i ( I ) σ i ( I, a ) , where σ i is the corresponding behavioral strategy of x i .

Perturbed extensive-form games (Perturbed EFGs). This game is a variant of the original EFG. Specifically, the strategy space of each infoset I ∈ I in a γ -perturbed EFG is a γ -perturbed simplex ∆ | A ( I ) | γ , a subset of ∆ | A ( I ) | , rather than the standard simplex ∆ | A ( I ) | used in the original EFG, where γ &gt; 0 is a constant. Formally, for any ˆ σ i ( I ) ∈ ∆ | A ( I ) | γ and a ∈ A ( I ) , the constraint γ ≤ ˆ σ i ( I, a ) ≤ 1 holds, where i = P ( I ) . For convenience, we denote the set of sequence-form strategies for player i in the γ -perturbed EFGs as X γ i . In γ -perturbed EFGs with γ &gt; 0 , any behavioral strategy ˆ σ i , with ˆ σ i ( I ) ∈ ∆ | A ( I ) | γ for all i ∈ N and I ∈ I i , can be uniquely mapped to a sequence-form strategy ˆ x i ∈ X γ i , and vice versa. Specifically, ∀ i ∈ N , I ∈ I i , ˆ σ i ( I ) = ˆ x i ( I ) / ˆ x i ( ρ I ) ≥ γ . Notably, ∀ i ∈ N , X γ i is a subset of X i . Similarly, we use the notation X γ and X ∗ ,γ to denote the joint strategy space × i ∈N X γ i and the set of NEs of γ -perturbed EFGs, respectively.

Learning an NE via regret minimization algorithms. For any sequence of strategies x 1 i , · · · , x T i of of player i , player i 's regret is R T i = max x i ∈ X i ∑ T t =1 ⟨ ℓ t i , x t i -x i ⟩ , where ℓ t i is the loss for player i at iteration t . Regret minimization algorithms are algorithms ensuring R T i grows sublinearly. To learn an NE of EFGs via regret minimization algorithms, we set ℓ t i = ℓ x t i with ℓ x 0 = Ax 1 and ℓ x 1 = -A T x 0 . If all players follow regret minimization algorithms, then the average strategy converges to the set of NEs in two-player zero-sum EFGs. In EFGs, there always exists L and P such that, ∀ x , x ′ ∈ X , ∥ ℓ x -ℓ x ′ ∥ 1 ≤ L ∥ x -x ′ ∥ 1 and ∥ ℓ x ∥ 1 ≤ P , where ℓ x = [ ℓ x i | i ∈ N ] , as well as L &gt; 0 and P &gt; 0 are game-dependent constants.

Counterfactual regret minimization (CFR) framework. This framework [Zinkevich et al., 2007, Farina et al., 2019] is designed to solve EFGs by decomposing the global regret R T i into local regrets at each infoset, allowing for independent minimization within each infoset, rather than directly minimizing global regret. This approach has led to the development of several superhuman Game AIs [Bowling et al., 2015, Moravˇ cík et al., 2017, Brown and Sandholm, 2018, 2019b, Pérolat et al., 2022]. Formally, for player i , given the observed loss when all players follow x ∈ X is ℓ x i , the CFR framework computes the counterfactual values at each infoset I ∈ I i according to

<!-- formula-not-decoded -->

where ℓ x i ( I, a ) is the value of ℓ x i at the sequence ( I, a ) , v x i ( I ′ ) = [ v x i ( I ′ , a ′ ) | a ′ ∈ A ( I ′ )] , and σ i represents the behavioral strategy of player i corresponds to x i . Farina et al. [2019] demonstrate that

<!-- formula-not-decoded -->

where v t i ( I ) = v x t i ( I ) = [ v x t i ( I, a ) | a ∈ A ( I )] and σ t i is the behavioral strategy of player i corresponds to x t i . It indicates that minimizing the local regret max σ i ( I ) ∑ T t =1 ⟨ v t i ( I ) , σ i ( I ) -σ t i ( I ) ⟩ at I ∈ I i contributes to minimizing the global regret R T i .

Blackwell approachability framework. RMalgorithms are come from this framework whose core insight lies in reframing the problem of regret minimization within the orignial strategy space Z as

regret minimization within cone ( Z ) = { λ z | z ∈ Z , λ ≥ 0 } [Blackwell, 1956, Abernethy et al., 2011, Farina et al., 2021]. Specifically, a regret minimization algorithm is instantiated in cone ( Z ) , where its output at iteration t is θ t . This corresponds to the strategy z t = θ t / ⟨ θ t , 1 ⟩ within Z . Given the loss ℓ t at iteration t , the algorithm observes the transformed loss -m t = -⟨ ℓ t , z t ⟩ 1 + ℓ t and subsequently generates θ t +1 . The main advantage of this framework is its capacity to develop parameter-free algorithms. More details are provided below.

Regret Matching + (RM + ). To minimize local regret within each infoset, CFR algorithms commonly employ local regret minimizers based on RM [Hart and Mas-Colell, 2000, Gordon, 2006, Bowling et al., 2015, Farina et al., 2021, 2023, Xu et al., 2022, 2024b, Cai et al., 2025], which show strong empirical convergence rate and are typically parameter-free. In this paper, we focus on RM + [Tammelin, 2014], a variant of RM that typically exhibits a faster empirical convergence rate than vanilla RM. RM + is a traditional algorithm grounded in Blackwell approachability framework. It corresponds to an OMD instantiated in the cone of the simplex [Farina et al., 2021]. Formally, at each iteration t and infoset I ∈ I i ,RM + updates the strategy via

<!-- formula-not-decoded -->

where i = P ( I ) , η &gt; 0 is the step size, m t i ( I ) = -⟨ v t i ( I ) , σ t i ( I ) ⟩ 1 + v t i ( I ) represents the instantaneous counterfactual regret, and D ψ ( u , v ) = ψ ( u ) -ψ ( v ) -⟨∇ ψ ( v ) , u -v ⟩ is the Bregman divergence associated with the quadratic regularizer ψ ( · ) = ∥ · ∥ 2 2 / 2 . If θ 1 I = 0 , for all the step size η &gt; 0 , the output sequence { σ 1 i ( I ) , σ 2 i ( I ) , . . . , σ t i ( I ) , . . . } remains unchanged [Farina et al., 2021]. Combining RM + with the CFR framework yields CFR + [Tammelin, 2014], which is a parameter-free CFR algorithm and has been used to build superhuman poker AI [Bowling et al., 2015].

## 3 Problem Statement

To demonstrate the last-iterate convergence of CFR algorithms, Pérolat et al. [2021, 2022], Liu et al. [2023] employ the RT framework. This framework reformulates the objective of learning an NE for the original EFG into finding NEs for a series of (perturbed) regularized EFGs, and ensures that the sequence of NEs of the regularized EFGs converges to the set of NEs of the original EFG. Therefore, establishing last-iterate convergence in learning an NE of the original EFG reduces to establishing last-iterate convergence in learning an NE of (perturbed) regularized EFGs. Inspired by Pérolat et al. [2021], Liu et al. [2023], Abe et al. [2024], we consider the following perturbed regularized EFG:

<!-- formula-not-decoded -->

where γ &gt; 0 and µ &gt; 0 are constants, ψ ( · ) is the quadratic regularizer, and r = [ r 0 ; r 1 ] ∈ X is the reference strategy profile. The NE of this perturbed regularized EFG is unique and denoted by ˆ x ∗ ,γ,µ, r or ˆ σ ∗ ,γ,µ, r . To ensure the sequence of the NEs of the perturbed regularized EFGs converges to the set of NEs of the original EFG, a valid approach is to continuously decreasing the value of γ and updating r to ˆ x ∗ ,γ,µ, r , according to the studies in Abe et al. [2024], Bernasconi et al. [2024]. Another approach involves simultaneously reducing the values of γ and µ [Liu et al., 2023, Bernasconi et al., 2024]. Notably, in the approach where simultaneously reducing the values of γ and µ , updating r to ˆ x ∗ ,γ,µ, r is optional. Consequently, achieving the last-iterate convergence for solving Eq. (2) implies achieving the last-iterate convergence for solving Eq. (1). This paper refrains from investigating the RT framework and its convergence as these have been thoroughly investigated in other studies [Pérolat et al., 2021, Liu et al., 2023, Abe et al., 2024, Bernasconi et al., 2024, Wang et al., 2025].

The introduction of perturbation and regularization ensures the smoothness of counterfactual values and the strong monotonicity, respectively. The smoothness is ∥ v ˆ σ i ( I ) -v ˆ σ ′ i ( I ) ∥ 1 ≤ O ( ∥ ˆ x -ˆ x ′ ∥ 1 ) , ∀ ˆ x , ˆ x ′ ∈ X γ , where ˆ σ and ˆ σ ′ are the behavioral strategy profiles associated with ˆ x and ˆ x ′ , respectively. The strong monotonicity indicates that O ( ⟨ ℓ ˆ x -ℓ ˆ x ′ , ˆ x -ˆ x ′ ⟩ ) ≥ ∥ ˆ x -ˆ x ′ ∥ 2 2 , ∀ ˆ x , ˆ x ′ ∈ X γ .

Although some works have investigated the last-iterate convergence of CFR algorithms for solving perturbed regularized EFGs [Liu et al., 2023], their algorithms do not use RM-based algorithms as the local regret minimizer. The absence of RM-based algorithms leads to significantly weaker empirical last-iterate convergence performance than traditional RM-based average-iterate convergence CFR algorithms, as shown in our experiments. In addition, as solving multiple perturbed regularized EFGs

is required, fine-tuning across all perturbed regularized EFGs is infeasible. Consequently, parameterfree algorithms, implying no parameters need to be tuned [Grand-Clément and Kroer, 2021], are desirable. Based on these observations, we propose Reward Transformation CFR + (RTCFR + ), utilizing CFR+ [Tammelin, 2014], a classical parameter-free RM-based CFR algorithm, to solve perturbed regularized EFGs defined in Eq. (2) (details of RTCFR + are in Section 4). Unfortunately, it remains unknown whether CFR + achieves the parameter-free (i.e., holds for any step sizes) last-iterate convergence in solving Eq. (2). Thus, our objective is to establish the parameter-free last-iterate convergence for CFR + in solving Eq. (2). More discussions about the related works are in Appendix B.

## 4 Last-Iterate Convergence of CFR + in Solving Perturbed Regularized EFGs

Now, we show that CFR + exhibits last-iterate convergence for solving the perturbed regularized EFGs defined in Eq. (2). Before introducing the last-iterate convergence of CFR + , we first extend CFR + to perturbed EFGs as the original CFR + algorithm is only designed for the case where γ = 0 . Specifically, we (i) first update the accumulated counterfactual regrets within the original simplex's cone while ensuring strategy outputs lie within the perturbed simplex by mixing the non-perturbed strategy formed by the accumulated counterfactual regrets with the uniform vector, then (ii) compute the instantaneous counterfactual regrets using the non-perturbed strategy and the counterfactual values observed through following the output perturbed strategy. This enables the use of the strong monotonicity to establish last-iterate convergence in learning an NE of the perturbed regularized EFGs in Eq. (2), as shown in Eq. (6). Formally, the update rule of CFR + for learning an NE of the perturbed regularized EFGs in Eq. (2) at iteration t and infoset I ∈ I i is

<!-- formula-not-decoded -->

where η &gt; 0 is the step size and ˆ x t i ( I ) = π ˆ σ t i ( I )ˆ σ t i ( I ) . The second line in Eq. (3) mixes the non-perturbed strategy σ with the uniform vector 1 , while the third line constructs the instantaneous counterfactual regrets ˆ m t I using the non-perturbed strategy σ t i derived from accumulated counterfactual regrets θ t I and counterfactual values ˆ v t i obtained from the perturbed strategy ˆ σ t i .

Theorem 4.1 (Proof is in Appendix D) . Assuming all players follow the update rule of CFR + with any θ 1 I ∈ R | A ( I ) | ≥ 0 and η &gt; 0 , the strategy profile ˆ x t converges to the set of NEs of the perturbed regularized EFGs defined in Eq. (2) with any γ &gt; 0 and µ &gt; 0 .

Proof sketch of Theorem 4.1. Our proof consists of two steps. Firstly, we establish the nonparameter-free last-iterate convergence; that is, for all θ 1 I ∈ R | A ( I ) | ≥ 0 , the last-iterate convergence of CFR + in solving Eq. (2) holds when η exceeds a certain constant. The principal challenge is that the smoothness of the instantaneous counterfactual regrets cannot be used since RM algorithms update within the cone of the strategy space, cone (∆ A ( I ) ) , whereas the final output lies in the strategy space, ∆ A ( I ) . We address this challenge by leveraging the fact that an NE is a best response to other strategies at each infoset in perturbed EFGs, as shown in the text around Eq. (5) and (6), as well as Lemma 4.4. Secondly, we derive the parameter-free convergence result, namely, that the last-iterate convergence of CFR + holds for all θ 1 I ∈ R | A ( I ) | ≥ 0 and η &gt; 0 . The main challenge here is that the property used in previous proofs of the parameter-free property-that the strategy sequence produced by CFR + is invariant w.r.t. different step sizes η &gt; 0 -holds only when θ 1 I = 0 . We overcome this by exploiting the linearity of the projection in CFR + and the fact that our non-parameter-free last-iterate convergence of CFR + holds for all θ 1 I ∈ R | A ( I ) | ≥ 0 , as presented in the second paragraph following Lemma 4.4. The details of our proof sketch is shown in the following.

Lemma 4.2 (Adapted from the proof of Lemma 4 in Farina et al. [2021]) . Assuming all players follow the update rule of CFR + , then for any θ I ∈ R | A ( I ) | ≥ 0 , we have

<!-- formula-not-decoded -->

I I I By applying Lemma 4.2 with θ I = σ ∗ ,µ,γ, r i ( I ) = (ˆ σ ∗ ,µ,γ, r i ( I ) -γ 1 ) / (1 -α I ) ∈ ∆ | A ( I ) | , we get η ⟨ ˆ m t i ( I ) ,σ ∗ ,µ,γ, r i ( I ) -θ t +1 I ⟩≤ D ψ ( σ ∗ ,µ,γ, r i ( I ) , θ t I ) -D ψ ( σ ∗ ,µ,γ, r i ( I ) , θ t +1 I ) -D ψ ( θ t +1 I , θ t I ) . (4) Also, we define ˆ m ∗ ,µ,γ, r i ( I )=ˆ v ∗ ,µ,γ, r i ( I ) -⟨ ˆ v ∗ ,µ,γ, r i ( I ) ,σ ∗ ,µ,γ, r i ( I ) ⟩ 1 , ˆ v ∗ ,µ,γ, r i ( I )= -ˆ ℓ ∗ ,µ,γ, r i ( I,a )+ ∑ I ′ ∈ C i ( I,a ) ⟨ ˆ v ∗ ,µ,γ, r i ( I ′ ) , ˆ σ ∗ ,µ,γ, r i ( I ′ ) ⟩ , ˆ ℓ ∗ ,µ,γ, r 0 = A ˆ x ∗ ,µ,γ, r 1 + µ ∇ ψ (ˆ x ∗ ,µ,γ, r 0 ) -µ ∇ ψ ( r 0 ) , ˆ ℓ ∗ ,µ,γ, r 1 = -A T ˆ x ∗ ,µ,γ, r 0 + µ ∇ ψ (ˆ x ∗ ,µ,γ, r 1 ) -µ ∇ ψ ( r 1 ) . Then, adding η ⟨-ˆ m ∗ ,µ,γ, r i ( I ) , θ t +1 I -θ t I ⟩ to each hand side of Eq. (4), we can get

<!-- formula-not-decoded -->

In OMD algorithms [Sokota et al., 2023], the addition of the term η ⟨-ˆ m ∗ ,µ,γ, r i ( I ) , θ t +1 I -θ t I ⟩ is not required to exploit the smoothness of the instantaneous counterfactual regrets. However, this term is necessary to prove the last-iterate convergence of CFR + . This step is crucial in our proof, and to the best of our knowledge, no prior work has proposed a similar approach.

Lemma 4.3 (Proof is in Appendix E.1) . For any x , x ′ ∈ X , ℓ ∈ R |X| , i ∈ N , µ ≥ 0 , and γ ≥ 0 ,

<!-- formula-not-decoded -->

where v σ i ( I ) = [ v σ i ( I, a ) | a ∈ A ( I )] with v σ i ( I, a ) = -ℓ i ( I, a ) + ∑ I ′ ∈ C i ( I,a ) ⟨ v σ i ( I ′ ) , σ i ( I ′ ) ⟩ , as well as σ and σ ′ are the behavioral strategy profiles associated with x and x ′ , respectively.

Combining Eq. (5) with Lemma 4.3, and setting ζ I = (1 -α I ) β I with β I = π ˆ σ ∗ ,µ,γ, r i ( I ) , we have

<!-- formula-not-decoded -->

By using the strong monotonicity ( O ( ∑ T t =1 ∑ i ∈N ⟨ ˆ ℓ t i , ˆ x t i -ˆ x ∗ ,µ,γ, r i ⟩ ) ≥ ∥ ˆ x t -ˆ x ∗ ,µ,γ, r ∥ 2 2 , as shown in Lemma D.1) and the smoothness of instantaneous counterfactual regrets ( ∥ ˆ m t i ( I ) -ˆ m ∗ ,µ,γ, r i ( I ) ∥ 2 2 ≤ O ( ∥ ˆ x t -ˆ x ∗ ,µ,γ, r ∥ 2 2 ) ) (see details in Appendix D), we get

<!-- formula-not-decoded -->

where C 0 = |I| A 2 max ( 6( L + µ ) 2 +8( P +2 µD ) 2 ( A max C max +1) 2 /γ 2 H ) . Note that the form of smoothness we adopt differs from that commonly used in OMD algorithms [Sokota et al., 2023], where smoothness typically takes the form ∥ ˆ m t i ( I ) -ˆ m t +1 i ( I ) ∥ 2 2 ≤ O ( ∥ ˆ x t -ˆ x t +1 ∥ ) rather than ∥ ˆ m t i ( I ) -ˆ m ∗ ,µ,γ, r i ( I ) ∥ 2 2 ≤ O ( ∥ ˆ x t -ˆ x ∗ ,µ,γ, r ∥ 2 2 ) . This difference also highlights that our proof approach diverges from the approach used by OMD algorithms. Then, if 0 &lt; η ≤ µ/ (2 C 0 ) , we get

<!-- formula-not-decoded -->

Lemma 4.4 (Proof is in Appendix E.2) . ∀ i ∈ N , I ∈ I i , and θ I ∈ R | A ( I ) | ≥ 0 , ⟨-ˆ m ∗ ,µ,γ, r i ( I ) , θ I ⟩ ≥ 0 . Lemma 4.4 is from that an NE is a best response to others at each infoset in perturbed EFGs, i.e., ∀ σ i , ⟨ ˆ v ∗ ,µ,γ, r i ( I ) , ˆ σ ∗ ,µ,γ, r i ( I ) -ˆ σ i ( I ) ⟩≥ 0 , where ˆ σ i ( I ) = (1 -α I ) σ i ( I )+ γ 1 (details are in Appendix E.2). By using Lemma 4.4, we get ∀ T ≥ 1 , ∑ T t =1 ∥ ˆ x t -ˆ x ∗ ,µ,γ, r ∥ 2 2 ≤ O (1) , implying that ˆ x t converges to ˆ x ∗ ,µ,γ, r with 0 &lt;η ≤ µ/ (2 C 0 ) .

Farina et al. [2021] show that when θ 1 I = 0 , for any η &gt; 0 , the sequence { ˆ x 1 , ˆ x 2 , · · · , ˆ x t , · · · } remains the same. This implies that ˆ x t converges to ˆ x ∗ ,µ,γ, r for any η &gt; 0 , showing the parameterfree property. In this paper, we further show that for any initial θ 1 I ∈ R | A ( I ) | ≥ 0 and η &gt; 0 , ˆ x t converges to ˆ x ∗ ,µ,γ, r (see advantages in discussions). This proof is simple yet novel, with the key insights being the linearity of the projection in CFR + and that ∑ T t =1 ∥ ˆ x t -ˆ x ∗ ,µ,γ, r ∥ 2 2 ≤ O (1) holds independently of the value of θ 1 I . Specifically, from the linearity of the projection in CFR + , for any accumulated counterfactual regret sequence { θ 1 I , θ 2 I , . . . , θ t I , . . . } generated by any θ 1 I ∈ R | A ( I ) | ≥ 0 and η &gt; 0 , there exists a corresponding accumulated counterfactual regret sequence { θ 1 I ′ , θ 2 I ′ , . . . , θ t I ′ , . . . } generated by θ 1 I ′ and η ′ = µ/ (2 C 0 ) , such that the resulting strategy profile sequence { ˆ x 1 , ˆ x 2 , . . . , ˆ x t , . . . } are identical. Additionally, as the condition ∑ T t =1 ∥ ˆ x t -ˆ x ∗ ,µ,γ, r ∥ 2 2 ≤ O (1) holds independently of the value of θ 1 I ( θ 1 I ′ ). Based on this analysis, we conclude that for any accumulated counterfactual regret sequence { θ 1 I , θ 2 I , . . . , θ t I , . . . } generated by any θ 1 I and η &gt; 0 , the corresponding strategy profile sequence { ˆ x 1 , ˆ x 2 , . . . , ˆ x t , . . . } converges to ˆ x ∗ ,µ,γ, r , which indicates the parameter-free property.

Reward Transformation CFR + (RTCFR + ). RTCFR + is the RT algorithm that applies CFR + to solve perturbed regularized EFGs, whose pseudocode is in Algorithm 1. As analyzed by Abe et al. [2024], Bernasconi et al. [2024], continuously decreasing γ and updating r to ˆ x ∗ ,γ,µ, r allows the sequence of the NEs of the perturbed regularized EFGs to converge to the set of NEs of the original EFG. Specifically, as shown in Algorithm 1, after T u iterations, RTCFR + updates γ and r , with N ∗ T u representing the total number of iterations. The implementation of RTCFR + is in Appendix H.

For RTCFR + , we do not examine the convergence of the sequence of the NEs of the perturbed regularized EFGs to the set of NEs of the original EFG when the exact ˆ x ∗ ,γ,µ, r is not learned but only an approximate ˆ x ∗ ,γ,µ, r is obtained, as this problem can be solved by simultaneously decreasing the values of µ and γ , as mentioned in Section 3. Formally, line 8 of Algorithm 1 can be modified as: µ ← µ × (1 -ς ) ,γ ← γ × 0 . 5 , and r ← ˆ x T u +1 , where 0 &lt;ς&lt; 1 . When ς is close to 0 , e.g., 1e -16 , its effect on the empirical convergence rate of RTCFR + is minimal (Figure 3). Nonetheless, it ensures that the sequence of NEs for the perturbed regularized EFGs converges to the set of NEs of the original EFG, even the exact ˆ x ∗ ,γ,µ, r is not learned.

Discussions. Firstly, to the best of our knowledge, we provide the first parameter-free last-iterate convergence for RM-based

## Algorithm 1 RTCFR +

- 1: Input: N , T u , µ , γ , r
- 2: θ 1 I ← 0 , η ← 1 , ∀ I ∈ I
- 3: for each n ∈ [1 , 2 , · · · , N ] do
- 4: Build the perturbed regularized EFGs in Eq. (2) via µ , γ , and r
- 5: for each t ∈ [1 , 2 , · · · , T u ] do
- 6: Obtain ˆ x t +1 and θ t +1 I via the update rule in Eq. (3)
- 7: end for
- 8: γ ← γ ∗ 0 . 5 , r ← ˆ x T u +1
- 9: θ 1 I ← θ T u +1 I , ∀ I ∈ I
- 10: end for
- 11: Return ˆ x T u +1

CFR algorithms in learning an NE of perturbed regularized EFGs. When considering NFGs, the last-iterate convergence result of CFR + (RM + ) holds even when γ = 0 , due to that the smoothness of counterfactual values and Lemma 4.4 hold in NFGs with any γ ≥ 0 . Secondly, we extend the parameter-free results of CFR + from Farina et al. [2021], demonstrating that CFR + converges with the parameter-free property for any θ 1 I ∈ R | A ( I ) | ≥ 0 , not just when θ 1 I = 0 in Farina et al. [2021]. This new parameter-free result is significant. Specifically, it indicates that after updating γ and r (line 8 of Algorithm 1), there is no need to reset θ 1 I to 0 to get the parameter-free property (line 9 of Algorithm 1). This improves the stability of CFR + , i.e., rapid fluctuations in the strategy profiles across iterations, since such stability improves as the lower bound of the 1-norm of θ t I increases [Farina et al., 2023] (for CFR + , from the proof of Lemma C.2 of Liu et al. [2022], we get that ∥ θ t I ∥ 2 ≤ ∥ θ t +1 I ∥ 2 , and the 1-norm lower bound is related to the 2-norm lower bound). Notably, as shown in Appendix G, resetting θ 1 I to 0 after updating γ and r (line 9 of Algorithm 1 becomes θ 1 I ← 0 , ∀ I ∈ I ) causes RTCFR + to never converge (Figure 3)! Lastly, our proof approach for the parameter-free property can be used to show that CFR + 's average-iterate convergence holds for all θ 1 I ∈ R | A ( I ) | ≥ 0 and η &gt; 0 . As our primary focus is on last-iterate convergence, we discuss the parameter-free average-iterate convergence in Appendix F rather than the main text.

Figure 1: Last-iterate convergence rates of different algorithms. In all plots, the x-axis is the number of iteration, and the y-axis is exploitability, displayed on a logarithmic scale. Liar's Dice ( x ) represents that every player is given a die with x sides. Goofspiel ( x ) denotes that each player is dealt x cards. Battleship ( x ) implies the size of grids is x . The size of the tested games is in Appendix G (Table 2).

<!-- image -->

## 5 Experiments

Configurations. We now evaluate the empirical convergence rate of RTCFR + on five standard EFG benchmarks: Kuhn Poker, Leduc Poker, Goofspiel, Liar's Dice, and Battleship, all implemented using OpenSpiel [Lanctot et al., 2019]. We compare RTCFR + with classical CFR algorithms, such as CFR + , PCFR + [Farina et al., 2021], and DCFR [Brown and Sandholm, 2019a], and those with theoretical guarantees for last-iterate convergence, including R-NaD [Pérolat et al., 2021, 2022] and Reg-CFR [Liu et al., 2023]. Additionally, we evaluate traditional last-iterate convergence algorithms, such as OMWU and OGDA [Wei et al., 2021, Lee et al., 2021]. The algorithm implementations are based on the open-source LiteEFG code [Liu et al., 2024], which offers a significant speedup-approximately 100 times faster than OpenSpiel's default implementation for the same number of iterations. For RTCFR + , we set the initial values of η , γ , and µ to 1 , 1e -10 , and 1e -3 , respectively. The number of iterations T u required to update γ and r , is set to 100 . For Reg-CFR, we use the parameters from the original paper. For R-NaD, we initialize µ = 1e -5 (R-NaD does not include the parameter γ ), set T u = 1000 , and use a learning rate of η = 0 . 1 . For OMWU and OGDA, we set η to 0 . 5 and 0 . 1 , respectively. All algorithms employ alternating updates to enhance empirical convergence rates. Each algorithm is run for 20,000 ( N = 20000 /T u ) iterations to analyze long-term

Table 1: Hyperparameters used in RTCFR + (fine-tuned).

|     | Kuhn Poker      | Leduc Poker   | Battleship (3)   | Liar's Dice (4)   | Liar's Dice (5)   |
|-----|-----------------|---------------|------------------|-------------------|-------------------|
| µ   | 0.1             | 0.001         | 0.1              | 0.01              | 0.0005            |
| T u | 10              | 100           | 50               | 10                | 10                |
|     | Liar's Dice (6) | Goofspiel (4) | Goofspiel (5)    | Goofspiel (6)     |                   |
| µ   | 0.0001          | 0.1           | 0.05             | 0.005             |                   |
| T u | 500             | 10            | 100              | 50                |                   |

behavior. The experiments are conducted on a machine equipped with a Xeon(R) Gold 6444Y CPU and 256 GB of memory. More experimental results including (i) performance of RTCFR + under simultaneous decrease of µ and γ , (ii) performance of RTCFR + under reset accumulated regrets as 0 , (iii) comparison with average-iterate convergence CFR algorithms, (iv) performance of RTCFR + in HUNL Subgames, and (v) performance of RTCFR + under different hyperparameters, are in Appendix G.

Results. The experimental results are presented in Figure 1. RTCFR + demonstrates superior performance compared to all other tested algorithms except PCFR + . Specifically, RTCFR + exhibits the fastest convergence rate across all games when compared to CFR + . In comparison to existing theoretical last-iterate convergence CFR algorithms, such as Reg-CFR and R-NaD, RTCFR + is only surpassed by Reg-CFR during the initial stages in small-scale games like Kuhn Poker and Goofspiel (4). Similarly, when compared to traditional last-iterate convergence algorithms, RTCFR + is only outperformed by OGDA in small-scale games such as Kuhn Poker and Goofspiel (4). Inspired by our RTCFR + and the performance of PCFR + , we propose RTPCFR + , which employs PCFR + to solve the perturbed regularized EFG defined in Eq. (2) instead of CFR + . For RTPCFR + , we use the same parameters as RTCFR + . Among RTCFR + , RTPCFR + , and PCFR + , no single algorithm consistently outperforms the others across all EFGs, as their performance varies depending on the specific EFG. This variability may be attributed to the fact that RTCFR + and RTPCFR + have not been fine-tuned for individual EFGs. Therefore, we also include a comparison with the fine-tuned RTCFR + , which is denoted as RTCFR + (fine-tuned) in Figure 1. Our findings demonstrate that fine-tuning enables RTCFR + to outperform all tested algorithms. The parameters used for the fine-tuned RTCFR + are presented in Table 1. However, the automatic adjustment of γ , µ , and T u remains an open problem. One of our future research directions is to investigate the automotive adjustment of these parameters.

## 6 Conclusions

We explore the last-iterate convergence of parameter-free RM-based CFR algorithms. We establish that a classical parameter-free RM-based CFR algorithm, CFR + , achieves last-iterate convergence in learning an NE of perturbed regularized EFGs. To our knowledge, this is the first parameter-free lastiterate convergence of RM-based CFR algorithms in perturbed regularized EFGs. Experimental results show that our proposed algorithm, RTCFR + , exhibits a significantly faster empirical convergence rate than existing algorithms that achieve theoretical last-iterate convergence.

Limitations. The main limitation of RTCFR + is its dependency on parameter tuning. Specifically, RTCFR + requires careful fine-tuning of parameters µ , γ , and T u , which prevents it from being a parameter-free algorithm. Interestingly, when both µ and γ are simultaneously reduced, RTCFR + achieves last-iterate convergence in learning an NE of the original EFGs, irrespective of the values of µ , γ , and T u . These parameters only impact the empirical convergence rate. Therefore, advancing automated methods to learn optimal values for µ , γ , and T u represents a promising direction for future research.

## Acknowledgements

This work is supported in part by the National Natural Science Foundation of China under Grants 62192783 and 62506157, the Jiangsu Science and Technology Major Project BG2024031, the Fundamental Research Funds for the Central Universities (14380128), the Collaborative Innovation Center of Novel Software Technology and Industrialization, and the InnoHK funding.

## References

- Kenshi Abe, Kaito Ariu, Mitsuki Sakamoto, and Atsushi Iwasaki. Adaptively perturbed mirror descent for learning in games. In Proceedings of the 41st International Conference on Machine Learning , 2024.
- Jacob Abernethy, Peter L Bartlett, and Elad Hazan. Blackwell approachability and no-regret learning are equivalent. In Proceedings of the 24th Conference on Learning Theory , pages 27-46. JMLR Workshop and Conference Proceedings, 2011.
- Ioannis Anagnostides, Ioannis Panageas, Gabriele Farina, and Tuomas Sandholm. On the convergence of no-regret learning dynamics in time-varying games. In Proceedings of the 34th International Conference on Neural Information Processing Systems , 2024.
- James P. Bailey and Georgios Piliouras. Multiplicative weights update in zero-sum games. In Proceedings of the 19th ACM Conference on Economics and Computation , pages 321-338, 2018.
- Martino Bernasconi, Alberto Marchesi, and Francesco Trovò. Learning extensive-form perfect equilibria in two-player zero-sum sequential games. In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics , pages 2152-2160. PMLR, 2024.
- David Blackwell. An analog of the minimax theorem for vector payoffs. 1956.
- Michael Bowling, Neil Burch, Michael Johanson, and Oskari Tammelin. Heads-up limit hold'em poker is solved. Science , 347(6218):145-149, 2015.
- Noam Brown and Tuomas Sandholm. Strategy-based warm starting for regret minimization in games. In Proceedings of the 30th AAAI Conference on Artificial Intelligence , 2016.
- Noam Brown and Tuomas Sandholm. Superhuman AI for heads-up no-limit poker: Libratus beats top professionals. Science , 359(6374):418-424, 2018.
- Noam Brown and Tuomas Sandholm. Solving imperfect-information games via discounted regret minimization. In Proceedings of the 33rd AAAI Conference on Artificial Intelligence , pages 1829-1836, 2019a.
- Noam Brown and Tuomas Sandholm. Superhuman AI for multiplayer poker. Science , 365(6456): 885-890, 2019b.
- Noam Brown, Tuomas Sandholm, and Brandon Amos. Depth-limited solving for imperfectinformation games. In Proceedings of the 28th International Conference on Neural Information Processing Systems , 2018.
- Noam Brown, Anton Bakhtin, Adam Lerer, and Qucheng Gong. Combining deep reinforcement learning and search for imperfect-information games. In Proceedings of the 30th International Conference on Neural Information Processing Systems , 2020.
- Yang Cai, Gabriele Farina, Julien Grand-Clément, Christian Kroer, Chung-Wei Lee, Haipeng Luo, and Weiqiang Zheng. Last-iterate convergence properties of regret-matching algorithms in games. In Proceedings of the 14th International Conference on Learning Representation , 2025.
- Darshan Chakrabarti, Julien Grand-Clément, and Christian Kroer. Extensive-form game solving via Blackwell approachability on treeplexes. In Proceedings of the 35th International Conference on Neural Information Processing Systems , 2024.
- Gabriele Farina and Tuomas Sandholm. Fast payoff matrix sparsification techniques for structured extensive-form games. In Proceedings of the 36th AAAI Conference on Artificial Intelligence , 2022.
- Gabriele Farina, Christian Kroer, and Tuomas Sandholm. Online convex optimization for sequential decision processes and extensive-form games. In Proceedings of the 33rd AAAI Conference on Artificial Intelligence , pages 1917-1925, 2019.

- Gabriele Farina, Christian Kroer, and Tuomas Sandholm. Faster game solving via predictive Blackwell approachability: Connecting regret matching and mirror descent. In Proceedings of the 35th AAAI Conference on Artificial Intelligence , pages 5363-5371, 2021.
- Gabriele Farina, Julien Grand-Clément, Christian Kroer, Chung-Wei Lee, and Haipeng Luo. Regret matching+: (in)stability and fast convergence in games. In Proceedings of the 37th Conference on Neural Information Processing Systems , volume 36, pages 61546-61572, 2023.
- Sam Ganzfried and Tuomas Sandholm. Potential-aware imperfect-recall abstraction with earth mover's distance in imperfect-information games. In Proceedings of the 28th AAAI Conference on Artificial Intelligence , 2014.
- Geoffrey J. Gordon. No-regret algorithms for online convex programs. In Proceedings of the 19th International Conference on Neural Information Processing Systems , pages 489-496, 2006.
- Julien Grand-Clément and Christian Kroer. Conic Blackwell algorithm: Parameter-free convexconcave saddle-point solving. In Proceedings of the 35th International Conference on Neural Information Processing Systems , volume 34, 2021.
- Sergiu Hart and Andreu Mas-Colell. A simple adaptive procedure leading to correlated equilibrium. Econometrica , 68(5):1127-1150, 2000.
- Samid Hoda, Andrew Gilpin, Javier Pena, and Tuomas Sandholm. Smoothing techniques for computing nash equilibria of sequential games. Mathematics of Operations Research , 35(2): 494-512, 2010.
- Michael Johanson, Nolan Bard, Marc Lanctot, Richard Gibson, and Michael Bowling. Efficient nash equilibrium approximation through monte carlo counterfactual regret minimization. In Proceedings of the 11th International Conference on Autonomous Agents and Multiagent Systems-Volume 2 , 2012.
- Marc Lanctot. Monte Carlo Sampling and Regret Minimization for Equilibrium Computation and Decision-Making in Large Extensive Form Games . University of Alberta (Canada), 2013.
- Marc Lanctot, Kevin Waugh, Martin Zinkevich, and Michael Bowling. Monte carlo sampling for regret minimization in extensive games. In Proceedings of the 22nd International Conference on Neural Information Processing Systems , pages 1078-1086, 2009.
- Marc Lanctot, Edward Lockhart, Jean-Baptiste Lespiau, Vinicius Zambaldi, Satyaki Upadhyay, Julien Pérolat, Sriram Srinivasan, Finbarr Timbers, Karl Tuyls, Shayegan Omidshafiei, et al. Openspiel: A framework for reinforcement learning in games, 2019.
- Chung-Wei Lee, Christian Kroer, and Haipeng Luo. Last-iterate convergence in extensive-form games. In Proceedings of the 35th International Conference on Neural Information Processing , pages 14293-14305, 2021.
- Boning Li and Longbo Huang. Efficient online pruning and abstraction for imperfect information extensive-form games. In Proceedings of the 13th International Conference on Learning Representations , 2025.
- Boning Li, Zhixuan Fang, and Longbo Huang. Rl-cfr: improving action abstraction for imperfect information extensive-form games with reinforcement learning. In Proceedings of the 41st International Conference on Machine Learning , 2024.
- Viliam Lis` y, Trevor Davis, and Michael Bowling. Counterfactual regret minimization in sequential security games. In Proceedings of the 30th AAAI Conference on Artificial Intelligence , pages 544-550, 2016.
- Mingyang Liu, Asuman E. Ozdaglar, Tiancheng Yu, and Kaiqing Zhang. The power of regularization in solving extensive-form games. In Proceedings of the 12th International Conference on Learning Representations , 2023.
- Mingyang Liu, Gabriele Farina, and Asuman Ozdaglar. LiteEFG: An efficient python library for solving extensive-form games. arXiv preprint arXiv:2407.20351 , 2024.

- Weiming Liu, Huacong Jiang, Bin Li, and Houqiang Li. Equivalence analysis between counterfactual regret minimization and online mirror descent. In Proceedings of the 37th International Conference on Machine Learning , pages 13717-13745, 2022.
- Linjian Meng, Youzhi Zhang, Zhenxing Ge, Tianyu Ding, Shangdong Yang, Zheng Xu, Wenbin Li, and Yang Gao. Last-iterate convergence of smooth Regret Matching+ variants in learning Nash equilibria, 2025.
- Panayotis Mertikopoulos, Christos H. Papadimitriou, and Georgios Piliouras. Cycles in adversarial regularized learning. In Proceedings of the 29th Annual ACM-SIAM Symposium on Discrete Algorithms , pages 2703-2717, 2018.
- Matej Moravˇ cík, Martin Schmid, Neil Burch, Viliam Lis` y, Dustin Morrill, Nolan Bard, Trevor Davis, Kevin Waugh, Michael Johanson, and Michael Bowling. Deepstack: Expert-level artificial intelligence in heads-up no-limit poker. Science , 356(6337):508-513, 2017.
- Arkadij Semenoviˇ c Nemirovskij and David Borisovich Yudin. Problem complexity and method efficiency in optimization. 1983.
- Julien Pérolat, Rémi Munos, Jean-Baptiste Lespiau, Shayegan Omidshafiei, Mark Rowland, Pedro A. Ortega, Neil Burch, Thomas W. Anthony, David Balduzzi, Bart De Vylder, Georgios Piliouras, Marc Lanctot, and Karl Tuyls. From Poincaré recurrence to convergence in imperfect information games: Finding equilibrium via regularization. In Proceedings of the 38th International Conference on Machine Learning , pages 8525-8535, 2021.
- Julien Pérolat, Bart De Vylder, Daniel Hennes, Eugene Tarassov, Florian Strub, Vincent de Boer, Paul Muller, Jerome T Connor, Neil Burch, Thomas Anthony, et al. Mastering the game of Stratego with model-free multiagent reinforcement learning. Science , 378(6623):990-996, 2022.
- Tuomas Sandholm. Steering evolution strategically: Computational game theory and opponent exploitation for treatment planning, drug design, and synthetic biology. In Proceedings of the 29th AAAI Conference on Artificial Intelligence , pages 4057-4061, 2015.
- Samuel Sokota, Ryan D'Orazio, J. Zico Kolter, Nicolas Loizou, Marc Lanctot, Ioannis Mitliagkas, Noam Brown, and Christian Kroer. A unified approach to reinforcement learning, quantal response equilibria, and two-player zero-sum games. In Proceedings of the 12th International Conference on Learning Representations , 2023.
- Eric Steinberger. Pokerrl. https://github.com/TinkeringCode/PokerRL , 2019.
- Oskari Tammelin. Solving large imperfect information games using CFR+. arXiv preprint arXiv:1407.5042 , 2014.
- Oskari Tammelin, Neil Burch, Michael Johanson, and Michael Bowling. Solving heads-up limit texas hold'em. In Proceedings of the 24th International Conference on Artificial Intelligence , pages 645-652, 2015.
- Mingzhi Wang, Chengdong Ma, Qizhi Chen, Linjian Meng, Yang Han, Jiancong Xiao, Zhaowei Zhang, Jing Huo, Weijie J Su, and Yaodong Yang. Magnetic mirror descent self-play preference optimization. In Proceedings of the 13th International Conference on Learning Representations , 2025.
- Zifan Wang, Yi Shen, Michael Zavlanos, and Karl Henrik Johansson. No-regret learning in strongly monotone games converges to a nash equilibrium. 2023.
- Chen-Yu Wei, Chung-Wei Lee, Mengxiao Zhang, and Haipeng Luo. Linear last-iterate convergence in constrained saddle-point optimization. In Proceedings of the 9th International Conference on Learning Representations , 2021.
- Hang Xu, Kai Li, Haobo Fu, Qiang Fu, and Junliang Xing. Autocfr: learning to design counterfactual regret minimization algorithms. In Proceedings of the 36th AAAI Conference on Artificial Intelligence , volume 36, pages 5244-5251, 2022.

- Hang Xu, Kai Li, Haobo Fu, Qiang Fu, Junliang Xing, and Jian Cheng. Dynamic discounted counterfactual regret minimization. In Proceedings of the 12th International Conference on Learning Representations , 2024a.
- Hang Xu, Kai Li, Bingyun Liu, Haobo Fu, Qiang Fu, Junliang Xing, and Jian Cheng. Minimizing weighted counterfactual regret with optimistic online mirror descent. In Proceedings of the 33rd International Joint Conference on Artificial Intelligence , pages 5272-5280, 2024b.
- Hugh Zhang, Adam Lerer, and Noam Brown. Equilibrium finding in normal-form games via greedy regret minimization. In Proceedings of the 36th AAAI Conference on Artificial Intelligence , volume 36, pages 9484-9492, 2022a.
- Mengxiao Zhang, Peng Zhao, Haipeng Luo, and Zhi-Hua Zhou. No-regret learning in time-varying zero-sum games. In Proceedings of the 39th International Conference on Machine Learning , pages 26772-26808, 2022b.
- Naifeng Zhang, Stephen McAleer, and Tuomas Sandholm. Faster game solving via hyperparameter schedules. arXiv preprint arXiv:2404.09097 , 2024.
- Martin Zinkevich, Michael Johanson, Michael Bowling, and Carmelo Piccione. Regret minimization in games with incomplete information. In Proceedings of the 20th International Conference on Neural Information Processing Systems , pages 1729-1736, 2007.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claims made in the abstract and introduction accurately reflect the paper's contributions and scope

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We clearly show the assumptions used in our proof (Section 2 and 3), and discuss the primary limitation of RTCFR + in Section 6.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: We provide the full set of assumptions and a complete (and correct) proof.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We provide all the information needed to reproduce the main experimental results of this paper.

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

Answer: [No]

Justification: We will provide the code once this paper is accepted.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We provide all hyperparameters.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Yes, we do.

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

## 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We provide the details of the computer where the experiments are conducted.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: This paper only investigates the convergence of some algorithms.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This paper only investigates the convergence of some algorithms.

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

Justification: We do not release any data or models.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The code that we used is cited.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: This paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: We only use LLM for writing and editing.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A An Example of Extensive-Form Games

To illustrate the components of an EFG, we provide an example using the classic game of Matching Pennies, as depicted in its game tree representation in Figure 2. As shown in Section 2, an EFG is formally defined by the tuple G = {N , H , P, A , I , { u i }} . In this example, the set of players is N = { 0 , 1 } . The game commences at the root of the tree, which corresponds to the empty history ∅ ∈ H . The player function P ( h ) determines who moves at history h ; here, P ( ∅ ) = 0 , so the player 0 makes the first move. The actions available to the player 0 at this initial decision node are given by A ( ∅ ) = { heads, tails } .

Once the player 0 chooses an action, the game transitions to a new history. For instance, if the player 0 chooses "heads", the new history becomes ( P0:heads ) . At this stage, the player

Figure 2: A classical EFG: Matching pennies games. "P0" and "P1" represents the player 0 and 1, respectively

<!-- image -->

function dictates that it is the player 1's turn to act, i.e., P ( P0:heads ) = P ( P0:tails ) = 1 . A central concept in EFGs for modeling games with hidden information is the partition of each player's decision nodes into information sets I . In Figure 2, the dashed line connecting the player 1's two decision nodes signifies that they belong to the same infoset. This means that when the player 1 makes a choice, they are unaware of the player 0's preceding move; the histories ( P0:heads ) and ( P0:tails ) are indistinguishable to the player 1. A formal requirement is that the set of available actions must be identical for all nodes within an information set, which holds true here as the available actions are A ( P0:heads ) = A ( P0:tails ) = { heads, tails } .

After the player 1 selects an action, the game concludes, reaching a terminal history, also known as a leaf node z ∈ Z . Each leaf node is associated with a payoff vector that specifies the utility for each player, ( u 0 ( z ) , u 1 ( z )) . For example, if the sequence of actions is ( heads, heads ) , the game terminates with the payoff vector (1 , -1) , indicating a gain of 1 for the player 0 and a loss of 1 for the player 1. Conversely, if the coins do not match, as in the history ( heads, tails ) , the payoff is ( -1 , 1) . Since for any terminal history z , the payoffs for the player 0 and the player 1 are structured such that u 0 ( z ) = -u 1 ( z ) , this particular EFG is classified as a two-player, zero-sum game. This single example effectively demonstrates how an EFG captures the sequential structure, information constraints, and outcomes of a strategic interaction.

## B Related Work

Counterfactual Regret Minimization (CFR) algorithms. CFR algorithms are among the most widely used methods for solving real-world EFGs [Bowling et al., 2015, Moravˇ cík et al., 2017, Brown and Sandholm, 2018, 2019b, Pérolat et al., 2022]. The core idea of CFR is to decompose the problem of regret minimization across the entire game into subproblems within each infoset, employing a regret minimization algorithm as a local regret minimizer. The vanilla CFR algorithm was introduced by Zinkevich et al. [2007], which utilize RM [Hart and Mas-Colell, 2000] as the local regret minimizer. To enhance the performance of CFR, a common approach is to design more effective local regret minimizers, as the choice of local regret minimizer largely determines the overall CFR algorithm's efficiency. Advanced local regret minimizers are typically based on RM, including RM + [Tammelin, 2014], Discounted RM (DRM) [Brown and Sandholm, 2019a], and Predictive RM + (PRM + ) [Farina et al., 2021], which correspond to CFR + [Tammelin, 2014], Discounted CFR (DCFR) [Brown and Sandholm, 2019a], and Predictive CFR + (PCFR + ) [Farina et al., 2021], respectively. However, CFR algorithms typically achieve theoretical convergence to the set of NEs of EFGs only through the average of iterates, also be called as average-iterate convergence.

Last-iterate convergence results of CFR algorithms. Pérolat et al. [2021] provide the first lastiterate convergence result for CFR algorithms in learning an NE of EFGs by transforming the task of learning an NE of the original EFG into finding the NEs of a sequence of regularized EFGs and

ensuring the sequence of the NEs of these regularized EFGs converges to the set of NEs of the original EFG. However, their analysis assumes continuous-time feedback, a condition rarely satisfied in practical scenarios. Subsequently, Liu et al. [2023] presents the first last-iterate convergence result for CFR under the discrete-time feedback by transforming the task of learning an NE of the original EFG into finding the NEs of a sequence of perturbed regularized EFGs rather than only regularized EFGs, since the addition of perturbation introduces the smoothness of counterfactual values. Nevertheless, both algorithms do not leverage RM algorithms as the local regret minimizer, leading to a suboptimal empirical last-iterate convergence rate compared to traditional RM-based CFR algorithms that only achieve average-iterate convergence, as demonstrated in our experiments.

Last-iterate convergence results of RM algorithms. Except this paper, Cai et al. [2025], Meng et al. [2025] also investigate the last-iterate convergence of RM algorithms. However, their results mainly focus on non-parameter-free RM algorithms, whereas we considers parameter-free RM algorithms. Specifically, Cai et al. [2025], Meng et al. [2025] mainly investigate smooth RM + variants [Farina et al., 2023]. The lack of the parameter-free property in the results of Cai et al. [2025], Meng et al. [2025] makes them less applicable when solving real-world games. Although Cai et al. [2025] investigate RM + (CFR + uses RM + as the local regret minimizer), a parameter-free RM algorithm, their proof techniques related to RM + primarily follow our proof techniques. Furthermore, the results in Cai et al. [2025], Meng et al. [2025] are confined to NFGs, whereas we focus on EFGs.

Weestablish the first parameter-free last-iterate convergence for RM-based CFR algorithms in learning an NE of perturbed regularized EFGs. Notably, our parameter-free property holds for any initial accumulated counterfactual regrets not only the zero initialization in previous works [Farina et al., 2021]. While CFR + 's parameter-free property in its first theoretical convergence result [Tammelin et al., 2015] holds for any initial accumulated counterfactual regrets, this result is exclusively limited to average-iterate convergence. In contrast, our proof technique simultaneously establishes both parameter-free last-iterate (Theorem 4.1) and average-iterate convergence (Theorem F.1) for CFR + under any initial accumulated counterfactual regrets 2 . Notably, the proof techniques employed by Tammelin et al. [2015] differ fundamentally from those utilized in ours and most recent works on RM-based CFR algorithms [Farina et al., 2023, Xu et al., 2022, 2024a,b, Zhang et al., 2024]. These works, including ours, adopt the Blackwell approachability framework (as introduced in Section 2) in Farina et al. [2021] to prove the convergence of RM-based CFR algorithms, while Tammelin et al. [2015] use the potential function [Zhang et al., 2022a]. Unfortunately, as previously mentioned, the parameter-free property in Farina et al. [2021] (even including Farina et al. [2023], Xu et al. [2022, 2024a,b], Zhang et al. [2024]) holds only under the condition where the initial accumulated counterfactual regrets are zero. Lastly, experiments show that our algorithm, RTCFR + , substantially outperform existing algorithms that achieve theoretical last-iterate convergence.

In this paper, we only focus on the last-iterate convergence and do not consider the best-iterate convergence because it offers limited utility in real-world games [Anagnostides et al., 2024, Wang et al., 2023]. With the best-iterate convergence, computing the exploitability of each iteration's strategy profile is necessary to select an optimal strategy, but this task is typically challenging due to the vast size of real-world games, such as HUNL, which reaches a size of 10 170 . In contrast, the last-iterate convergence circumvents the need to compute exploitability for every iteration; it simply requires the selection of the strategy from the final iteration.

## C Discussion on the Application of RTCFR + in Large-Scale Games and Its Integration with Other Technologies

Firstly, RTCFR + can be directly applied to large-scale games without any modifications. In fact, the modifications introduced by RTCFR + over CFR+ are minimal. As demonstrated in our implementation provided in Appendix F, RTCFR + requires fewer than 30 additional lines compared to CFR + (specifically, lines 33, 40-41, 47-49, 51-55, and 62-66 of the RTCFR + implementation in Appendix H). The main limitation of applying RTCFR + to large-scale games lies in the need to tune the hyperparameters µ , γ , and T u , which can vary significantly across different games. Addressing the dependency on tuning µ , γ , and T u remains a central direction for future work. It is important to clarify, however, that this requirement originates from the RT framework itself; all existing algorithms based on the RT framework require tuning of these parameters.

2 Farina et al. [2021] also only establish parameter-free average-iterate convergence.

Secondly, integrating RTCFR + with the other technologies requires case-by-case analysis. (i) For algorithms that solely modify the game tree, such as depth-limited solving [Brown et al., 2018, 2020], impact-recall abstraction [Ganzfried and Sandholm, 2014], action abstraction [Li et al., 2024], and Vector CFR [Johanson et al., 2012], RTCFR + can be directly applied since RTCFR + only requires execution on the new game tree. This process is straightforward and presents no significant challenges. (ii) Regarding warm-start [Brown and Sandholm, 2016], while its concept of setting initial accumulated counterfactual regrets using an efficient initial strategy is insightful, current integration with RTCFR + is not feasible. Specifically, the warm-start approach in Brown and Sandholm [2016] is an enhancement tailored for the original CFR. Formally, the analysis presented on the bottom left of page four in Brown and Sandholm [2016] demonstrates that the substitute regret is given by R ′ T ( I, a ) = T ( v ′ σ ( I, a ) -v ′ σ ( I )) . This formulation implies that R ′ T ( I, a ) can be negative, a property that does not hold in CFR + and RTCFR + . (iii) As for sparsification [Farina and Sandholm, 2022], which optimizes the computation of loss gradients ( ℓ t i , the last line of Eq. (3)), RTCFR + can seamlessly integrate. This compatibility arises because RTCFR + solely requires the input of loss gradients, which then facilitates strategy updates through the update rules defined in the first four lines of Eq. (3). (iv) The pruning approach in Li and Huang [2025] can be directly integrated with RTCFR + . Since this pruning approach modifies the game tree before the algorithm execution (e.g., "permanently and correctly eliminating sub-optimal branches before the CFR begins"), it aligns with our earlier statement on game-tree modification approaches. Hence, RTCFR + can be directly applied.

## D Proof of Theorem 4.1

Proof. To prove the last-iterate convergence of CFR + in learning an NE of perturbed regularized EFGs defined in Eq. (2), we introduce the following lemmas.

Lemma D.1 (Adapted from Lemma D.4 in Sokota et al. [2023]) . For any x ∈ X , µ ≥ 0 , and γ ≥ 0 ,

<!-- formula-not-decoded -->

where ℓ x 0 = Ax 1 + µ ∇ ψ ( x 0 ) -µ ∇ ψ ( r 0 ) and ℓ x 1 = -A T x 0 + µ ∇ ψ ( x 1 ) -µ ∇ ψ ( r 1 ) .

Lemma D.2 (Proof is in Appendix E.3) . For any x ∈ X , i ∈ N , I ∈ I i , µ ≥ 0 , and γ ≥ 0 ,

<!-- formula-not-decoded -->

where ˆ v σ i ( I ) = [ˆ v σ i ( I, a ) | a ∈ A ( I )] , ˆ v σ i ( I, a ) = -ˆ ℓ x i + ∑ I ′ ∈ C i ( I,a ) ⟨ ˆ v σ i ( I ′ ) , σ i ( I ′ ) ⟩ with ˆ ℓ x 0 = Ax 1 + µ ∇ ψ ( x 0 ) -µ ∇ ψ ( r 0 ) and ˆ ℓ x 1 = -A T x 0 + µ ∇ ψ ( x 1 ) -µ ∇ ψ ( r 1 ) , as well as σ is the behavioral strategy profile associated with x .

Lemma D.3 (Proof is in Appendix E.4) . For any x , x ′ ∈ X , i ∈ N , I ∈ I i , µ ≥ 0 , and γ ≥ 0 ,

<!-- formula-not-decoded -->

where ˆ v σ i ( I ) = [ˆ v σ i ( I, a ) | a ∈ A ( I )] , ˆ v σ i ( I, a ) = -ˆ ℓ x i + ∑ I ′ ∈ C i ( I,a ) ⟨ ˆ v σ i ( I ′ ) , σ i ( I ′ ) ⟩ with ˆ ℓ x 0 = Ax 1 + µ ∇ ψ ( x 0 ) -µ ∇ ψ ( r 0 ) and ˆ ℓ x 1 = -A T x 0 + µ ∇ ψ ( x 1 ) -µ ∇ ψ ( r 1 ) , as well as σ and σ ′ are the behavioral strategy profiles associated with x and x ′ , respectively.

Lemma D.4 (Proof is in Appendix E.5) . For any ˆ x , ˆ x ′ ∈ X γ with γ &gt; 0 , i ∈ N , I ∈ I i , and µ ≥ 0 ,

<!-- formula-not-decoded -->

where ˆ σ and ˆ σ ′ are the behavioral strategy profiles associated with ˆ x and ˆ x ′ , respectively.

By substituting θ I = σ ∗ ,µ,γ, r i ( I ) = ˆ σ ∗ ,µ,γ, r i ( I ) -γ 1 into Lemma 4.2, we get

1 -α I η ⟨ ˆ m t i ( I ) , σ ∗ ,µ,γ, r i ( I ) -θ t +1 I ⟩ ≤ D ψ ( σ ∗ ,µ,γ, r i ( I ) , θ t I ) -D ψ ( σ ∗ ,µ,γ, r i ( I ) , θ t +1 I ) -D ψ ( θ t +1 I , θ t I ) . (7) Adding η ⟨-ˆ m ∗ ,µ,γ, r i ( I ) , θ t +1 I -θ t I ⟩ to each hand side of Eq. (7), we have η ⟨ ˆ m t i ( I ) , σ ∗ ,µ,γ, r i ( I ) -θ t +1 I ⟩ + η ⟨-ˆ m ∗ ,µ,γ, r i ( I ) , θ t +1 I -θ t I ⟩ ≤ D ψ ( σ ∗ ,µ,γ, r i ( I ) , θ t I ) -D ψ ( σ ∗ ,µ,γ, r i ( I ) , θ t +1 I ) + η ⟨-ˆ m ∗ ,µ,γ, r i ( I ) , θ t +1 I -θ t I ⟩ -D ψ ( θ t +1 I , θ t I ) ,

which implies

<!-- formula-not-decoded -->

where the second inequality comes from that ∀ a , b ∈ R d , ρ &gt; 0 , ⟨ a , b ⟩ ≤ ρ ∥ a ∥ 2 2 / 2 + ∥ b ∥ 2 2 / (2 ρ ) (in this case, a = ˆ m t i ( I ) -ˆ m ∗ ,µ,γ, r i ( I ) , b = θ t +1 I -θ t I , and ρ = η ), and the last inequality is from that ∀ a , b ∈ R d , ∥ a -b ∥ 2 2 / 2 = ∥ b -a ∥ 2 2 / 2 = D ψ ( a , b ) (in this case, a = θ t +1 I , and b = θ t I ). Arranging the terms in Eq. (8), we get

<!-- formula-not-decoded -->

According to the definition of ˆ m t i ( I ) , we have

<!-- formula-not-decoded -->

where the second equality comes from that

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

Let β I = π ˆ σ ∗ ,µ,γ, r i ( I ) . Continuing from Eq. (9), we get

<!-- formula-not-decoded -->

By applying Lemma 4.3, we have

<!-- formula-not-decoded -->

where ζ I = (1 -α I ) β I . Since 0 ≤ ζ I ≤ 1 (as 0 ≤ β I ≤ 1 and 0 ≤ α I ≤ 1 ), we get

<!-- formula-not-decoded -->

By applying Lemma D.1, we obtain

<!-- formula-not-decoded -->

Now, we use the smoothness of the instantaneous counterfactual regrets to transform ∥ ˆ m t i ( I ) -ˆ m ∗ ,µ,γ, r i ( I ) ∥ 2 2 into a term only related to ∥ ˆ x t -ˆ x ∗ ,µ,γ, r ∥ 2 2 . Formally, for the term ∥ ˆ m t i ( I ) -ˆ m ∗ ,µ,γ, r i ( I ) ∥ 2 2 , from the definition of ˆ m t i ( I ) and ˆ m ∗ ,µ,γ, r i ( I ) , we have

<!-- formula-not-decoded -->

where σ ∗ ,µ,γ, r i ( I ) = ˆ σ ∗ ,µ,γ, r i ( I ) -γ 1 1 -α I . By using A max = max I ∈I | A ( I ) | , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last line is from Lemma D.2. For the term ∥⟨ ˆ v t i ( I ) , σ ∗ ,µ,γ, r i ( I ) ⟩ -⟨ ˆ v ∗ ,µ,γ, r i ( I ) , σ ∗ ,µ,γ, r i ( I ) ⟩∥ 2 2 in Eq. (11), we get

<!-- formula-not-decoded -->

where the last inequality comes from ∥ σ ∗ ,µ,γ, r i ( I ) ∥ 2 2 ≤ 1 as σ ∗ ,µ,γ, r i ( I ) is in simplex. By substituting Eq. (12) and (13) into Eq. (11), as well as using A max ≥ 1 , we obtain

<!-- formula-not-decoded -->

By applying Lemma D.3 into Eq. (14), we get

<!-- formula-not-decoded -->

By applying Lemma D.4 into Eq. (15), we get

<!-- formula-not-decoded -->

By substituting Eq. (16) into Eq. (10), we have

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

Obviously, if we have

<!-- formula-not-decoded -->

By using Lemma 4.4, we have that η ⟨-ˆ m ∗ ,µ,γ, r i ( I ) , θ T +1 I ⟩ ≥ 0 . As a result, we get -D ψ ( σ ∗ ,µ,γ, r i ( I ) , θ T +1 I ) -η ⟨-ˆ m ∗ ,µ,γ, r i ( I ) , θ T +1 I ⟩ ≤ 0 . Then, we conclude that ∀ T ≥ 1

<!-- formula-not-decoded -->

which implies the asymptotic last-iterate convergence of the sequence { ˆ x 1 , ˆ x 2 , · · · , ˆ x t , · · · } to NE ˆ x ∗ ,µ,γ, r of the perturbed regularized EFG since 0 ≤ ζ I ≤ 1 (as mentioned above).

As analyzed in Farina et al. [2021], if θ 1 I = 0 , for any η &gt; 0 , the generated sequence { ˆ x 1 , ˆ x 2 , . . . , ˆ x t , . . . } remains identical, achieving the parameter-free property. In this paper, we further establish that for any initial θ 1 I ∈ R | A ( I ) | ≥ 0 and η &gt; 0 , the sequence ˆ x t converges to ˆ x ∗ ,µ,γ, r .

<!-- formula-not-decoded -->

We first prove that for the accumulated counterfactual regret sequence { θ 1 I , θ 2 I , . . . , θ t I , . . . } generated by θ 1 I ∈ R | A ( I ) | ≥ 0 and η &gt; 0 , there exists a corresponding sequence { θ 1 I ′ , θ 2 I ′ , . . . , θ t I ′ , . . . } generated by θ 1 I ′ ∈ R | A ( I ) | ≥ 0 and η ′ = µ/ (2 C 0 ) , such that the resulting strategy profile sequence { ˆ x 1 , ˆ x 2 , . . . , ˆ x t , . . . } is identical. By the update rule of CFR + defined in Eq. (3) and the analysis in Farina et al. [2021], θ t +1 I ∈ arg min θ I ∈ R | A ( I ) | ≥ 0 { ⟨-ˆ m t i ( I ) , θ I ⟩ + 1 η D ψ ( θ I , θ t I ) } can be expressed as the projection θ t +1 I = [ θ t I + η ˆ m t i ( I )] + , where [ · ] + = max( · , 0 ) . Setting θ t I ′ = η ′ θ t I /η for t ≥ 1 , it follows that θ t +1 I ′ = [ θ t I ′ + η ′ ˆ m t i ( I )] + and σ t i ( I ) = θ t I / ⟨ θ t I , 1 ⟩ = θ t I ′ / ⟨ θ t I ′ , 1 ⟩ hold [Chakrabarti et al., 2024]. Furthermore, it is evident that ∑ T t =1 ∥ ˆ x t -ˆ x ∗ ,µ,γ, r ∥ 2 2 ≤ O (1) holds independently of the value of the initial accumulated counterfactual regret.

Based on the above analysis, we conclude that (i) for any accumulated counterfactual regret sequence { θ 1 I , θ 2 I , . . . , θ t I , . . . } generated by any θ 1 I ∈ R | A ( I ) | ≥ 0 and η &gt; 0 , there exists a corresponding accumulated counterfactual regret sequence { θ 1 I ′ , θ 2 I ′ , . . . , θ t I ′ , . . . } generated by θ 1 I ′ and η ′ = µ/ (2 C 0 ) , such that the resulting strategy profile sequence { ˆ x 1 , ˆ x 2 , . . . , ˆ x t , . . . } are identical, as well as (ii) the strategy profile sequence { ˆ x 1 , ˆ x 2 , . . . , ˆ x t , . . . } generated by the accumulated counterfactual regret sequence { θ 1 I ′ , θ 2 I ′ , . . . , θ t I ′ , . . . } converges to ˆ x ∗ ,µ,γ, r . Therefore, we have that for any θ 1 I ∈ R | A ( I ) | ≥ 0 and η &gt; 0 , the generated strategy profile sequence { ˆ x 1 , ˆ x 2 , . . . , ˆ x t , . . . } converges to ˆ x ∗ ,µ,γ, r , demonstrating the parameter-free property. We complete the proof.

## E Proof of Useful Lemmas

## E.1 Proof of Lemma 4.3

Proof. From the definition of ∑ I ∈I i π σ ′ i ( I ) ⟨-v σ i ( I ) , σ i ( I ) -σ ′ i ( I ) ⟩ , we get

<!-- formula-not-decoded -->

For the term ∑ I ∈I i π σ ′ i ( I ) ⟨-v σ i ( I ) , σ ′ i ( I ) ⟩ , we have

∑

I

∈I

i

<!-- formula-not-decoded -->

Then, by substituting Eq. (18) into Eq. (17), we have

<!-- formula-not-decoded -->

We denote the initial infosets as I init i , i.e., for any I ∈ I init i , there does not exist I ′′ ∈ I i such that I ∈ C i ( I ′′ , a ′′ ) holds for a a ′′ ∈ A ( I ′′ ) . For the term ∑ I ∈I i π σ ′ i ( I ) ⟨-v σ i ( I ) , σ i ( I ) ⟩ -

π

σ

i

′

(

I

)

⟨-

v

σ

i

(

I

)

,σ

′

i

(

I

)

⟩

′

<!-- formula-not-decoded -->

Since the probability of reaching any I ∈ I init i is always 1, regardless of the strategies σ or σ ′ , we have that ∀ σ, σ ′ , and I ∈ I init i , π σ ′ i ( I ) = π σ i ( I ) . Substituting this into Eq. (20), we obtain

<!-- formula-not-decoded -->

where the last line follows from the recursion. Substituting Eq. (21) into Eq. (19), we obtain

<!-- formula-not-decoded -->

as ∀ i ∈ N , I ∈ I i , π σ i ( I ) σ i ( I, a ) = x i ( I, a ) and π σ ′ i ( I ) σ ′ i ( I, a ) = x ′ i ( I, a ) via the definition of the sequence-form strategy. It finishes the proof.

## E.2 Proof of Lemma 4.4

Proof. First, when θ I = 0 , we have that ∀ I ∈ I i , ⟨-ˆ m i ( I ) , θ I ⟩ = 0 . Next, we prove by contradiction that when θ I &gt; 0 , ∀ I ∈ I i , it holds that ⟨-ˆ m ∗ ,µ,γ, r i ( I ) , θ I ⟩ ≥ 0 . Suppose there exists one I ′ ∈ I i and θ ′ I ′ &gt; 0 such that ⟨-ˆ m ∗ ,µ,γ, r ( I ′ ) , θ ′ I ′ ⟩ &lt; 0 . We construct a new strategy σ ′ i , which matches σ ∗ ,µ,γ, r i (not ˆ σ ∗ ,µ,γ, r i ) except at the infoset I ′ , where it is defined as θ ′ I ′ / ⟨ θ ′ I ′ , 1 ⟩ . For ⟨-ˆ m ∗ ,µ,γ, r ( I ′ ) , θ ′ I ′ ⟩ , we have

∗ ,µ,γ, r

<!-- formula-not-decoded -->

Since ⟨-ˆ m ∗ ,µ,γ, r ( I ′ ) , θ ′ I ′ ⟩ &lt; 0 and ∥ θ ′ I ′ ∥ 1 &gt; 0 , we have ⟨-ˆ v ∗ ,µ,γ, r i ( I ′ ) , σ ∗ ,µ,γ, r i ( I ′ ) -σ ′ i ( I ′ ) ⟩ &gt; 0 . We define ˆ σ ′ i ( I ) = (1 -α I ) σ ′ i ( I ) + γ 1 for all I ∈ I i . Additionally, we know that ˆ σ ∗ ,µ,γ, r i ( I ) = (1 -α I ) σ ∗ ,µ,γ, r i ( I ) + γ 1 for all I ∈ I i , and that ⟨-ˆ v ∗ ,µ,γ, r i ( I ′ ) , σ ∗ ,µ,γ, r i ( I ′ ) -σ ′ i ( I ′ ) ⟩ &gt; 0 . Hence, it follows that ⟨-ˆ v ∗ ,µ,γ, r i ( I ′ ) , ˆ σ ∗ ,µ,γ, r i ( I ′ ) -ˆ σ ′ i ( I ′ ) ⟩ &gt; 0 .

The correspond sequence-form strategy of ˆ σ ′ i is represented by ˆ x ′ i . According to Lemma 4.3 and the definition of NE, we get

<!-- formula-not-decoded -->

Since σ ′ i matches σ ∗ ,µ,γ, r i except at the infoset I ′ , and given that ˆ σ ′ i ( I ) = (1 -α I ) σ ′ i ( I ) + γ 1 for all I ∈ I i , as well as ˆ σ ∗ ,µ,γ, r i ( I ) = (1 -α I ) σ ∗ ,µ,γ, r i ( I ) + γ 1 for all I ∈ I i , we obtain

<!-- formula-not-decoded -->

where ˆ x ′ i is the sequence-form strategy profile associated with ˆ σ i . By the definition of ˆ x ′ i , it follows that ˆ x ′ i ∈ X γ i . However, from the definition of NE, as shown in Eq. (22), ⟨ ˆ ℓ x ∗ ,µ,γ, r i , ˆ x ∗ ,µ,γ, r i -ˆ x ′ i ⟩ ≤ 0 , which contradicts the result in Eq. (23). Therefore, there exists no I ′ ∈ I i and θ ′ I ′ &gt; 0 such that ⟨-ˆ m ∗ ,µ,γ, r ( I ′ ) , θ ′ I ′ ⟩ &lt; 0 . Consequently, when θ I &gt; 0 for all I ∈ I i , it holds that ⟨-ˆ m ∗ ,µ,γ, r i ( I ) , θ I ⟩ ≥ 0 .

Through the discussion of the above two situations, we complete the proof.

## E.3 Proof of Lemma D.2

Proof. From the definition of ˆ v σ i ( I ) , we get

<!-- formula-not-decoded -->

where the last line is from recursion. Continuing from the above inequality, we get

<!-- formula-not-decoded -->

where ℓ x 0 = Ax 1 and ℓ x 1 = -A T x 0 . By substituting Eq. (25) into Eq. (24), we have

<!-- formula-not-decoded -->

It completes the proof.

## E.4 Proof of Lemma D.3

Proof. From the definition of ˆ v σ i ( I ) and ˆ v σ ′ i ( I ) , we have

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

For the term ∥⟨ ˆ v σ i ( I ′ ) , σ i ( I ′ ) ⟩ - ⟨ ˆ v σ i ( I ′ ) , σ ′ i ( I ′ ) ⟩∥ 1 in Eq. (26), we get

<!-- formula-not-decoded -->

where the last line comes from Lemma D.2. For the term ∥⟨ ˆ v σ i ( I ′ ) , σ ′ i ( I ′ ) ⟩ - ⟨ ˆ v σ ′ i ( I ′ ) , σ ′ i ( I ′ ) ⟩∥ 1 in Eq. (26), we get

<!-- formula-not-decoded -->

where the last line comes from ∥ σ ′ i ( I ′ ) ⟩∥ 1 ≤ 1 . By substituting Eq. (27) and (28) into Eq. (26), we obtain

<!-- formula-not-decoded -->

where the last line is from recursion. For the term ∥ ˆ ℓ x i -ˆ ℓ x ′ i ∥ 1 in Eq. (29), we get

<!-- formula-not-decoded -->

where ℓ x 0 = Ax 1 and ℓ x 1 = -A T x 0 . By substituting Eq. (30) into Eq. (29), we get

<!-- formula-not-decoded -->

where the second line is from ∀ b, c ∈ R , ( b + c ) 2 ≤ 2 b 2 +2 c 2 (in this case, b = ( L + µ ) ∥ x -x ′ ∥ 1 and c = ( P +2 µD ) ∥ σ i -σ ′ i ∥ 1 ). It completes the proof.

## E.5 Proof of Lemma D.4

Proof. From the definition of ∥ ˆ σ -ˆ σ ′ ∥ , we get i i 1 ∥ ˆ σ i -ˆ σ ′ i ∥ 1

<!-- formula-not-decoded -->

For the term ∥ ˆ x i ( I, a )ˆ x ′ i ( ρ I ) -ˆ x i ( I, a )ˆ x i ( ρ I ) ∥ 1 in Eq. (31), we have

<!-- formula-not-decoded -->

For the term

∥

ˆ

x

i

(

I, a

)ˆ

x

′

i

(

ρ

I

)

-

ˆ

x

i

(

I, a

)ˆ

x

i

(

ρ

I

)

∥

1

in Eq. (31), we have

<!-- formula-not-decoded -->

By substituting Eq. (32) and (33) into Eq. (31), we have

∥ ˆ σ i -ˆ σ ′ i ∥ 1 = ∑ I ∈I i ∑ a ∈ A ( I ) 1 ˆ x i ( ρ I )ˆ x ′ i ( ρ I ) (ˆ x i ( I,a ) ∥ ˆ x ′ i ( ρ I ) -ˆ x i ( ρ I ) ∥ 1 +ˆ x i ( ρ I ) ∥ ˆ x i ( I,a ) -ˆ x ′ i ( I,a ) ∥ 1 ) = ∑ I ∈I i ∑ a ∈ A ( I ) ( ˆ x i ( I,a ) ˆ x i ( ρ I )ˆ x ′ i ( ρ I ) ∥ ˆ x ′ i ( ρ I ) -ˆ x i ( ρ I ) ∥ 1 + ˆ x i ( ρ I ) ˆ x i ( ρ I )ˆ x ′ i ( ρ I ) ∥ ˆ x i ( I,a ) -ˆ x ′ i ( I,a ) ∥ 1 ) . Since ˆ x i ( I, a ) / ˆ x i ( ρ I ) = ˆ σ i ( I, a ) ≤ 1 , we obtain ∥ ˆ σ i -ˆ σ ′ i ∥ 1 ≤ ∑ I ∈I i ∑ a ∈ A ( I ) ( 1 ˆ x ′ i ( ρ I ) ∥ ˆ x ′ i ( ρ I ) -ˆ x i ( ρ I ) ∥ 1 + 1 ˆ x ′ i ( ρ I ) ∥ ˆ x i ( I,a ) -ˆ x ′ i ( I,a ) ∥ 1 ) ≤ ∑ I ∈I i ∑ a ∈ A ( I ) 1 γ H ( ∥ ˆ x ′ i ( ρ I ) -ˆ x i ( ρ I ) ∥ 1 + ∥ ˆ x i ( I,a ) -ˆ x ′ i ( I,a ) ∥ 1 ) , where the last inequality comes from ˆ x i ( I ) ≤ 1 /γ H for all i ∈ N , I ∈ I i , and ˆ x i ∈ X γ i (this follows from the facts that H denotes the maximum number of actions taken by all players along any path from the root to a leaf node and the probability of selecting each action is guaranteed to be greater than γ in perturbed EFGs). For the term ∑ I ∈I i ∑ a ∈ A ( I ) 1 γ H ∥ ˆ x ′ i ( ρ I ) -ˆ x i ( ρ I ) ∥ 1 , we get

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

It finishes the proof.

## F Our Parameter-Free Average-Iterate Convergence of CFR +

Now, we extend the proof of CFR + in Farina et al. [2021] via our proof approach in Appendix D to demonstrate that for all η &gt; 0 , CFR + 's average-iterate convergence holds for all θ 1 I ∈ R | A ( I ) | ≥ 0 not only for θ 1 I = 0 . This result is significant because it implies that even when the strategies generated during the initial iterations are discarded, CFR + remains achieving average-iterate convergence. Specifically, since average-iterate convergence holds for all θ 1 I ∈ R | A ( I ) | ≥ 0 , θ t I can be treated as a new θ 1 I , ensuring that CFR + enjoys average-iterate convergence for all η &gt; 0 after iteration t . Indeed, discarding the initial phase strategies is a common technique to improve the empirical convergence rate of CFR + [Steinberger, 2019].

Theorem F.1. Assuming all players follow the update rule of CFR + with any θ 1 I ∈ R | A ( I ) | ≥ 0 and η &gt; 0 , the average strategy profile ¯ x T = ∑ T t =1 x t T converges to the set of NEs of the perturbed regularized EFGs defined in Eq. (2) with any γ ≥ 0 and µ ≥ 0 as T →∞ .

Proof. By substituting θ I = σ i ( I )= ˆ σ i ( I ) -γ 1 1 -α I ∈ ∆ | A ( I ) | γ with ˆ σ i ( I ) ∈ ∆ | A ( I ) | γ into Lemma 4.2, we get

<!-- formula-not-decoded -->

According to the definition of ˆ m t i ( I ) , we have

<!-- formula-not-decoded -->

where the second equality comes from that

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

Continuing from Eq. (34), we have

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

Therefore, we get

<!-- formula-not-decoded -->

Continuing from Eq. (34), we have

<!-- formula-not-decoded -->

By applying Lemma 4.3, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˆ x i is the sequence-form strategy corresponding to ˆ σ i . Using ξ I to denote (1 -α I ) π ˆ σ i ( I ) , we get

<!-- formula-not-decoded -->

Lemma F.2 (Adapted from Lemma 11 of Wei et al. [2021]) . . If the player i follow the update rule of CFR + , with η &gt; 0 then for any I ∈ I i and t ≥ 1 , we have

<!-- formula-not-decoded -->

By substituting Lemma F.2 into Eq. (35), we get

<!-- formula-not-decoded -->

Assuming ∥ ˆ m t i ( I ) ∥ 2 2 ≤ M , we have

<!-- formula-not-decoded -->

According to the analysis in Appendix D, we have that for any accumulated counterfactual regret sequence { θ 1 I , θ 2 I , . . . , θ t I , . . . } generated by any θ 1 I ∈ R | A ( I ) | ≥ 0 and η &gt; 0 , there exists a corresponding accumulated counterfactual regret sequence { θ 1 I ′ , θ 2 I ′ , . . . , θ t I ′ , . . . } generated by θ 1 I ′ ∈ R | A ( I ) | ≥ 0 and η ′ &gt; 0 , such that the resulting strategy profile sequence { ˆ x 1 , ˆ x 2 , . . . , ˆ x t , . . . } are identical, where θ t I ′ = η ′ θ t I /η . To analysis the convergence rate of the accumulated counterfactual regret sequence { θ 1 I ′ , θ 2 I ′ , . . . , θ t I ′ , . . . } , from Eq. (36), we have

<!-- formula-not-decoded -->

By substituting θ t I ′ = η ′ θ t I /η into Eq. (37), we get

<!-- formula-not-decoded -->

From the fact that ∀ a , b ∈ R d , ∥ a -b ∥ 2 2 / 2 = ∥ b -a ∥ 2 2 / 2 = D ψ ( a , b ) , by using a = σ i ( I ) and b = η ′ θ 1 I η , we get

<!-- formula-not-decoded -->

As σ i ( I ) ∈ ∆ | A ( I ) | , we have ∥ σ i ( I ) ∥ 2 2 ≤ 1 . In addition, ∥ η ′ θ 1 I ∥ 2 2 / (2 η ′ η 2 ) = η ′ ∥ θ 1 I ∥ 2 2 / (2 η 2 ) . Continuing from Eq. (38), we get

<!-- formula-not-decoded -->

Table 2: Sizes of the games.

| Game            | #Histories   | #Infosets   | #Terminal histories   |   #Depth | #Max size of infosets   |
|-----------------|--------------|-------------|-----------------------|----------|-------------------------|
| Kuhn Poker      | 58           | 12          | 30                    |        6 | 2                       |
| Leduc Poker     | 9,457        | 936         | 5,520                 |       12 | 5                       |
| Battleship (3)  | 732,607      | 81,027      | 552,132               |        9 | 7                       |
| Liar's Dice (4) | 8,181        | 1,024       | 4,080                 |       12 | 4                       |
| Liar's Dice (5) | 51,181       | 5,120       | 25,575                |       14 | 5                       |
| Liar's Dice (6) | 294,883      | 24,576      | 147,420               |       16 | 6                       |
| Goofspiel (4)   | 1,077        | 162         | 576                   |        7 | 14                      |
| Goofspiel (5)   | 26,931       | 2,124       | 14,400                |        9 | 46                      |
| Goofspiel (6)   | 969,523      | 34,482      | 518,400               |       11 | 230                     |
| Subgame 3       | 398,112,843  | 69,184      | 261,126,360           |       10 | 1,980                   |
| Subgame 4       | 244,005,483  | 43,240      | 158,388,120           |        8 | 1,980                   |

We use M θ η to denote max( ∥ θ 1 I ∥ 2 2 / (2 η 2 ) , M ) . In addition, as 0 ≤ (1 -α I ) ≤ 1 and 0 ≤ π ˆ σ i ( I ) ≤ 1 , we have 0 ≤ ξ I ≤ 1 . Therefore, we get

<!-- formula-not-decoded -->

## G Additional Experiments

Sizes of the Games. Before introducing our additional experiments, we present the sizes of the games used in our study, as detailed in Table 2. In this table, #Histories denotes the total number of histories within the game tree, whereas #Infosets represents the count of information sets. The term #Terminal histories indicates the number of leaf nodes, and #Depth refers to the game's tree depth, defined as the maximum sequence of actions in any single history. Finally, #Max size of infosets signifies the largest number of histories contained within a single infoset.

Performance of RTCFR + under simultaneous decrease of µ and γ . we present the results for RTCFR + with modifications in line 8 where µ × (1 -ς ) , γ ← γ × 0 . 5 , and r ← ˆ x T u +1 , with ς = 1e -16 , as shown in Figure 3. We denote this variant as "RTCFR + V2". Our findings reveal that the empirical convergence performance of RTCFR + and RTCFR + V2 is similar.

Performance of RTCFR + under reset accumulated regrets as 0 . we examine the performance of RTCFR + that resets θ 1 I to 0 , which is denoted as "Unstable RTCFR + " in Figure 3. The parameters of Unstable RTCFR + are same as RTCFR + in Section 5. We observe that Unstable RTCFR + never converges across all tested games.

Comparison with average-iterate convergence CFR algorithms. We compare the last-iterate convergence performance of RTCFR + with the average-iterate performance of CFR + , PCFR + , and DCFR. The experimental results are shown in Figure 4. With fine-tuning, RTCFR + outperforms the average-iterate performance of CFR+, PCFR+, and DCFR in nearly all tested games, except for Liar's Dice (6). Even without fine-tuning, RTCFR + achieves superior performance to the average-iterate of CFR+ + , PCFR+, and DCFR in 5 out of the evaluated 9 games (Kuhn Pker, Leduc Poker, Liar's Dice (4), Liar's Dice (5), and Goofspiel (4)). In addition, as shown in Figure 4, even when considering only CFR + , PCFR + , and DCFR, no single algorithm consistently outperforms the other two across all games.

Convergence rates in the initial phase. We now present the results of our algorithms RTCFR + and RTPCFR + , alongside CFR + , R-NaD, Reg-CFR, OMWU, OGDA, PCFR + , and DCFR, over the first 1000 iterations. The results are shown in Figure 5. Consistent with the results in Figures 1, RTCFR + , RTPCFR + , and PCFR + demonstrate superior performance compared to the other algorithms. However, no single algorithm without fine-tuning outperforms all others across all games.

Performance of RTCFR + in HUNL Subgames. We now present the convergence rate of RTCFR + within HUNL Subgames, particularly the ones open-sourced by Libratus [Brown and Sandholm,

Figure 3: Last-iterate convergence rates of RTCFR + , RTCFR + V2, and Unstable RTCFR + .

<!-- image -->

2018]. We compare RTCFR + with CFR + , PCFR + , and DCFR + . Given the immense size of HUNL Subgames, we implement the tested algorithm using vector CFR. We employ the open-source code from Poker RL [Steinberger, 2019, Xu et al., 2024b], which supports vector CFR and Subgames from Libratus, specifically Subgame 3 and Subgame 4. The comparison of RTCFR + and the lastiterate convergence performance of CFR + , PCFR + , and DCFR + is illustrated in Figure 6, while the comparison of RTCFR + and the average-iterate convergence performance of CFR + , PCFR + , and DCFR + is depicted in Figure 7. RTCFR + exceeds the last-iterate convergence performance of CFR + and PCFR + across both HUNL Subgames. Additionally, in Subgame3, RTCFR + also surpasses the average-iterate convergence performance of CFR + and PCFR + . It is worth noting that CFR + and PCFR + do not provide a last-iterate convergence guarantee. For DCFR, RTCFR + , as well as CFR + and PCFR + , underperform in both last-iterate and average-iterate convergence performance. We speculate this is because DCFR is fine-tuned specifically for the tested HUNL Subgames, unlike the other evaluated algorithms.

Performance of RTCFR + under different hyperparameters. We investigate the convergence rates of RTCFR + under various hyperparameter settings. Specifically, we focus on the impact of µ and T u on the convergence rates, as we observe that γ only needs to be set to a sufficiently small value. The tested ranges for µ and T u are [1e -4 , 5e -4 , 1e -3 , 5e -3 , 1e -2 , 5e -2 , 1e -1 , 5e -1] and [10 , 50 , 100 , 500 , 1000] , respectively. Experimental results reveal that the performance of RTCFR + is primarily contingent upon the value of µ . To elucidate this dependency, we discuss the performance implications of varying µ values. Specifically, for small µ values, CFR + encounters difficulties in

Figure 4: Comparison with classical average-iterate convergence CFR algorithms.

<!-- image -->

accurately learning an NE of perturbed regularized EFGs. Consequently, this challenge persists irrespective of the value of T u , enabling that learning an NE of perturbed regularized EFGs becomes impossible. As a result, attaining an NE of the original game becomes impracticable for any T u value, which is also consistent with the experimental results. Conversely, when µ is optimal, neither too small nor too large, this condition enables CFR + to learn sufficiently accurate approximate an NE of perturbed regularized EFGs. These allow RTCFR + to achieve commendable performance. However, for large µ values, although CFR + are capable of learning the exact NE of perturbed regularized EFGs, the requisite number of reference strategy updates becomes excessively large. Hence, we observe that with large µ values, a smaller T u yields better performance. Based on these analyses, we advocate for the prioritization of determining µ 's value, followed by the value of T u , when practically applying our algorithm.

Figure 5: Last-iterate convergence rates over the first 1000 iterations.

<!-- image -->

Figure 6: Comparison with the last-iterate convergence performance of CFR + , PCFR + , and DCFR in HUNL Subgames.

<!-- image -->

Figure 7: Comparison with the average-iterate convergence performance of CFR + , PCFR + , and DCFR in HUNL Subgames.

<!-- image -->

Figure 8: Last-iterate convergence rates of RTCFR + with µ = 0 . 0001 .

<!-- image -->

Figure 9: Last-iterate convergence rates of RTCFR + with µ = 0 . 0005 .

<!-- image -->

Figure 10: Last-iterate convergence rates of RTCFR + with µ = 0 . 001 .

<!-- image -->

Figure 11: Last-iterate convergence rates of RTCFR + with µ = 0 . 005 .

<!-- image -->

Figure 12: Last-iterate convergence rates of RTCFR + with µ = 0 . 01 .

<!-- image -->

Figure 13: Last-iterate convergence rates of RTCFR + with µ = 0 . 05 .

<!-- image -->

Figure 14: Last-iterate convergence rates of RTCFR + with µ = 0 . 1 .

<!-- image -->

Figure 15: Last-iterate convergence rates of RTCFR + with µ = 0 . 5 .

<!-- image -->

## H Implementation of RTCFR +

In this section, we present a detailed description of the implementation of RTCFR + , which is adapted from the open-source implementation of CFR + by LiteEFG [Liu et al., 2024].

```
1 import LiteEFG 2 class RTCFRPlusGraph(LiteEFG.Graph): 3 def __init__(self, gamma=1e-10, mu=1e-3, shrink_iter=100): # default parameters 4 super().__init__() 5 self.timestep = 0 6 self.shrink_iter = shrink_iter # shrink_iter is T_u 7 8 # Initialization of RTCFR+ 9 with LiteEFG.backward(is_static=True): 10 ev = 1.0 * LiteEFG.const(1, 0.0) 11 # unperturbed_strategy is \sigma 12 self.unperturbed_strategy = LiteEFG.const(self. action_set_size , 1.0 / self.action_set_size) 13 # perturbed_strategy is \hat{\sigma} 14 self.strategy = LiteEFG.const(self.action_set_size , 1.0 / self.action_set_size) 15 # regret_buffer is \bm{\theta} 16 self.regret_buffer = LiteEFG.const(self. action_set_size , 0.0) 17 18 # ref_strategy is \bm{r} 19 self.ref_strategy = LiteEFG.const(self.action_set_size , 1.0 / self.action_set_size) 20 # the following three variables are used to compute \ nabla \psi(\bm{r}), note that self.ref_reach_prob(I ) = \nabla \psi(\bm{r})(I) 21 self.ref_reach_prob = LiteEFG.const(self. action_set_size , 1.0) 22 self.parent_reach_prob = LiteEFG.const(self. action_set_size , 1.0) 23 self.parent_to_child_prob = LiteEFG.const(self. action_set_size , 1.0) 24 25 self.iteration = LiteEFG.const(1, 0) 26 self.mu = LiteEFG.const(1, mu) 27 self.gamma = LiteEFG.const(1, gamma) 28 self.alpha_I = self.gamma*self.action_set_size 29 30 with LiteEFG.backward(color=0): 31 self.iteration.inplace(self.iteration+1) 32 # to compute the \hat{\bm{v}}_i^t(I) defined in (4) 33 gradient = LiteEFG.aggregate(ev, aggregator="sum") + self.utility -self.mu*(self.reach_prob*self. strategy -self.ref_reach_prob*self.ref_strategy) 34 # to compute the \langle \hat{\bm{v}}_i^t(I), \sigma^ t_i(I) \rangle defined in (4) 35 ev.inplace(LiteEFG.dot(gradient , self. unperturbed_strategy)) 36 # gradient -ev is the instantaneous counterfactual regret \hat{\bm{m}}_i^t(I ) defined in (4) 37 self.regret_buffer.inplace(LiteEFG.maximum(self. regret_buffer + gradient -ev, 0.0)) 38 39 # to get \sigma^{t+1}_i(I) 40 self.unperturbed_strategy.inplace(LiteEFG.normalize( self.regret_buffer , p_norm=1.0, ignore_negative= True)) 41 # to employ PCFR+ to solve the perturbed regularized EFGs , please use the following line
```

```
42 # self.unperturbed_strategy.inplace(LiteEFG.normalize( self.regret_buffer + gradient -ev, p_norm=1.0, ignore_negative=True)) 43 # to get \hat{\sigma}^{t+1}_i(I) 44 self.strategy.inplace(LiteEFG.normalize((1 -self. alpha_I)*self.unperturbed_strategy + self.gamma , p_norm=1.0, ignore_negative=True)) 45 46 # update gamma and the reference strategy profile 47 with LiteEFG.backward(color=1): 48 self.gamma.inplace(self.gamma * 0.5) 49 self.ref_strategy.inplace(self.strategy * 1.0) 50 51 with LiteEFG.forward(color=2): 52 # to compute \nabla \psi(\bm{r}) after updating the reference strategy profile 53 self.parent_reach_prob.inplace(LiteEFG.aggregate(self. ref_reach_prob , "sum", object="parent", player=" self", padding=1)) 54 self.parent_to_child_prob.inplace(LiteEFG.aggregate( self.ref_strategy , "sum", object="parent", player=" self", padding=1)) 55 self.ref_reach_prob.inplace(self.parent_reach_prob* self.parent_to_child_prob) 56 57 58 print("===============Graph␣is␣ready␣for␣RTCFR +===============") 59 60 def update_graph(self, env : LiteEFG.Environment) -> None: 61 self.timestep += 1 62 if self.timestep==1: 63 env.update(self.strategy , upd_color=[2]) 64 if self.timestep % self.shrink_iter == 0: 65 env.update(self.strategy , upd_color=[1]) 66 env.update(self.strategy , upd_color=[2]) 67 env.update(self.strategy , upd_color=[0], upd_player=1) 68 env.update(self.strategy , upd_color=[0], upd_player=2) 69 else: 70 env.update(self.strategy , upd_color=[0], upd_player=1) 71 env.update(self.strategy , upd_color=[0], upd_player=2) 72 73 def current_strategy(self, type_name="last-iterate") -> LiteEFG.GraphNode: 74 return self.strategy
```