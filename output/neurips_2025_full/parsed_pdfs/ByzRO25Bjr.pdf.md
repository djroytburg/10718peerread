## Deployment Efficient Reward-Free Exploration with Linear Function Approximation

## Zihan Zhang

zihanz@cse.ust.hk

HKUST

Jason D. Lee

jasondlee88@gmail.com UC Berkeley

Lin F. Yang

linyang@ee.ucla.edu UCLA

## Yuxin Chen

yuxinc@wharton.upenn.edu University of Pennsylvania

## Simon S. Du

ssdu@cs.washington.edu University of Washington

## Ruosong Wang

ruosongwang@pku.edu.cn Peking University

## Abstract

We study deployment-efficient reward-free exploration with linear function approximation, where the goal is to explore a linear Markov Decision Process (MDP) without revealing the reward function, while minimizing the number of distinct policies implemented during learning. By 'deployment efficient', we mean algorithms that require few policies deployed during exploration - crucial in real-world applications where such deployments are costly or disruptive. We design a novel reinforcement learning algorithm that achieves near-optimal deployment efficiency for linear MDPs in the reward-free setting, using at most H exploration policies during execution (where H is the horizon length), while maintaining sample complexity polynomial in feature dimension and horizon length. Unlike previous approaches with similar deployment efficiency guarantees, our algorithm's sample complexity is independent of the reachability or explorability coefficients of the underlying MDP, which can be arbitrarily small and lead to unbounded sample complexity in certain cases - directly addressing an open problem from prior work. Our technical contributions include a data-dependent method for truncating stateaction pairs in linear MDPs, efficient offline policy evaluation and optimization algorithms for these truncated MDPs, and a careful integration of these components to implement reward-free exploration with linear function approximation without sacrificing deployment efficiency.

## 1 Introduction

In real-world reinforcement learning applications, deploying new policies often incurs significant cost. For example, in robotics [Kober et al., 2013], deploying a new policy requires hardware-level operations, which can involve lengthy delays. In medical settings [Almirall et al., 2012, 2014, Lei et al., 2012], frequent policy changes are unrealistic, as each deployment typically requires a separate approval process involving domain experts. Similarly, in recommendation systems [Theocharous et al., 2015], deploying a new policy can take weeks due to mandatory internal testing to ensure safety and effectiveness. In all these scenarios, while switching policies frequently-especially based on instantaneous data, as standard RL algorithms require-is infeasible, it is often possible to run

many experiments in parallel once a policy is deployed. This highlights the need for RL algorithms that learn effective policies while minimizing the number of policy deployments.

Empirically, the notion of deployment efficiency was first proposed by Matsushima et al. [2020], while a formal definition of deployment complexity was recently introduced by Huang et al. [2022]. Intuitively, deployment complexity measures the total number of policy deployments by an RL algorithm, under the constraint that the interval between policy switches-i.e., the number of trajectories collected before switching-is fixed in advance. Under this notion, a line of recent work has developed provably efficient RL algorithms [Huang et al., 2022, Qiao et al., 2022, Qiao and Wang, 2022] in various settings. In the tabular case where the state space is discrete and of small size, Qiao et al. [2022] designed an RL algorithm with O ( H ) policy deployments, where H is the horizon length. Huang et al. [2022], Qiao and Wang [2022] studied deployment complexity in the context of RL with linear function approximation (i.e., linear MDP [Yang and Wang, 2019, Jin et al., 2023]). Specifically, their algorithms achieve sample complexity polynomial in the feature dimension d and horizon length H , with deployment complexity of O ( dH ) or O ( H ) . Huang et al. [2022] further showed that any RL algorithm for linear MDPs must incur a deployment complexity of at least ˜ Ω( H ) . 1

Although the aforementioned works provide important insights into the deployment complexity of reinforcement learning for linear MDPs, achieving the nearly optimal O ( H ) deployment complexity remains challenging. Existing algorithms that attain this guarantee either operate in the tabular setting [Qiao et al., 2022]-which is unsuitable for large or continuous state spaces-or rely on strong assumptions such as the reachability assumption [Huang et al., 2022] or the explorability assumption [Qiao and Wang, 2022]. Roughly speaking, these assumptions require that all directions in the feature space can be explored by some policy. Such conditions are quite restrictive and significantly limit the applicability of these algorithms. In particular, they typically assume a lower bound on a reachability coefficient v min , and the sample complexity of existing algorithms with O ( H ) deployment complexity depends polynomially on 1 v min . In the tabular setting, this assumption is equivalent to requiring that every state can be reached with a non-negligible probability by some policy. However, in general linear MDPs, the reachability coefficient can be arbitrarily small, rendering the sample complexity effectively infinite for such algorithms.

To address this limitation, we investigate the following fundamental question:

Is it possible to design RL algorithms for linear MDPs that achieve nearly optimal deployment complexity and polynomial sample complexity, without relying on additional assumptions such as reachability or explorability?

This question was explicitly raised in prior work [Huang et al., 2022, Qiao and Wang, 2022] and was left as an open problem. Huang et al. [2022] conjectured that achieving O ( H ) deployment complexity would necessarily require additional structural assumptions like reachability or explorability.

Our Contribution. In this paper, we resolve the above question by designing a new algorithm for linear MDPs with deployment complexity H . Our algorithm achieves polynomial sample complexity for any linear MDP and does not rely on additional assumptions such as reachability or explorability. Moreover, it operates in the reward-free exploration setting [Jin et al., 2020, Wang et al., 2020a, Chen et al., 2022, Wagenmaker et al., 2022, Zhang et al., 2021b, Li et al., 2024, 2023], where the reward function is not revealed during the exploration phase. This reward-free property further enhances the practicality of our approach in settings where reward signals are unavailable or costly to obtain. An informal statement of our main theoretical guarantee is summarized in the following theorem.

Theorem 1 (Informal version of Theorem 4) . For reward-free exploration in linear MDPs, there is an algorithm (Algorithm 1) with deployment complexity H and sample complexity polynomial in d , H , 1 /ϵ , and log(1 /δ ) , such that with probability 1 -δ , for all linear reward functions, the algorithm returns a policy with suboptimality at most ϵ . Here, d is the feature dimension and H is the horizon length.

Combined with the existing hardness result from Huang et al. [2022], our new result in the above Theorem provides a complete answer to the deployment complexity of RL for linear MDPs. It shows that additional assumptions such as reachability or explorability, previously conjectured to be necessary, are in fact not required to achieve nearly optimal deployment complexity.

1 Throughout this paper, we use ˜ O and ˜ Ω to suppress logarithmic factors.

Table 1: Comparison with the most related works.

|                     | Sample Complexity                        | Deployment Complexity   |
|---------------------|------------------------------------------|-------------------------|
| Huang et al. [2022] | poly ( d,H, 1 ϵ , log( 1 δ ) , 1 v min ) | H                       |
| Zhao et al. [2023]  | ˜ O ( d 2 H 3 ϵ 2 )                      | ˜ O ( dH )              |
| This work           | ˜ O ( d 15 H 15 ϵ 5 )                    | H                       |

## 2 Related Work

There is a large body of literature on the sample complexity of RL. We refer readers to Agarwal et al. [2019], Chi et al. [2025] for more thorough reviews, and focus on the most relevant work here.

Deployment Efficiency and Other Notions of Adaptivity. The notion of deployment efficiency was first proposed in the empirical work [Matsushima et al., 2020], while its formal definition was first defined by Huang et al. [2022]. Under this notion, Huang et al. [2022], Qiao et al. [2022], Qiao and Wang [2022] designed provably efficient RL algorithms in various settings. As mentioned ealier, in order to achieve a nearly optimal deployment complexity, existing algorithms either work in the tabular setting, or rely on additional reachability assumption or explorability assumption which we strive to avoid in this work. Zhao et al. [2023] designed deployment efficient RL algorithms for function classes with bounded eluder dimension. However, even for linear functions, the deployment complexity of the algorithm by Zhao et al. [2023] is ˜ O ( dH ) , which is far from being optimal.

The notion of deployment efficiency is closely related to the low switching setting [Bai et al., 2019, Zhang et al., 2020c, Gao et al., 2021, Kong et al., 2021, Qiao et al., 2022, Wang et al., 2021]. We refer readers to prior work [Huang et al., 2022, Qiao et al., 2022] for a detailed comparison between these two different notions. Roughly speaking, in the low switching setting, the agent decides whether to update the policy or not after collecting each trajectory. On the other hand, the notion of deployment efficiency requires the interval between policy switching to be fixed, and therefore, deployment efficient RL algorithms are easier to implement in practice. The low switching setting was also studied for other sequential decision-making problems including bandits [Abbasi-Yadkori et al., 2011, Cesa-Bianchi et al., 2013, Simchi-Levi and Xu, 2019, Ruan et al., 2021].

Reward-free Exploration. The notion of reward-free exploration was first proposed by Jin et al. [2020]. In this setting, the agent first collects trajectories from an unknown environment without any pre-specified reward function. After that, a specific reward function is given, and the goal is to use samples collected during the exploration phase to output a near-optimal policy for the given reward function. The sample complexity of reward-free exploration was studied and improved in a line of work [Kaufmann et al., 2021, Ménard et al., 2021, Zhang et al., 2020b] A similar notion called task-agnostic exploration was consider by Zhang et al. [2020a], Li et al. [2024, 2023]. For linear MDPs, the first polynomial sample complexity for reward-free exploration was obtained by Wang et al. [2020a]. Later, the sample complexity was improved by Zanette et al. [2020], Wagenmaker et al. [2022]. Reward-free exploration was also considered in other RL settings including linear mixture MDPs [Chen et al., 2022, Zhang et al., 2021a] and RL with non-linear function approximation [Chen et al., 2022].

Technical Comparison with Existing Algorithms. Finally, we compare our new algorithm with existing algorithms with O ( H ) deployment complexity [Qiao et al., 2022, Qiao and Wang, 2022] from a technical point of view. A more detailed overview of our new technical ingredients is given in Section 4. To achieve O ( H ) deployment complexity in the tabular setting, Qiao et al. [2022] applied absorbing MDP to ignore those 'hard to visit' states. In this work, similar ideas are used, though we work in the linear MDP setting which is much more complicated and requires a more careful treatment. In order to design an algorithm with O ( H ) deployment complexity in linear MDPs under the explorability assumption, Qiao and Wang [2022] showed how to solve a variant of G-optimal experiment design in an offline manner. In this work, we also use offline RL to build exploration policies in linear MDPs. However, the lack of the explorability assumption raises substantial more technical challenges which necessitates more involved algorithms and analysis.

## 3 Preliminaries

In this section, we introduce the basic definitions of MDPs and the assumptions used in our analysis. We use ∆( X ) to denote the set of probability distributions over a set X , and [ N ] to denote the set { 1 , 2 , . . . , N } for a positive integer N .

Episodic MDPs. A finite-horizon episodic Markov Decision Process (MDP) is defined by the tuple ( S , A , r, P, H, s ini ) , where S × A denotes the state-action space, r : S × A × [ H ] → [0 , 1] is the reward function, 2 P : S × A × [ H ] → ∆( S ) is the transition kernel, H is the episode horizon, and s ini ∈ S is the initial state. 3

Apolicy π = { π h : S → ∆( A ) } H h =1 is a collection of mappings from the state space S to probability distributions over the action space A , one for each time step h ∈ [ H ] . We say that π is a deterministic policy if π h ( s ) assigns probability one to a single action for all h and s .

In each episode, the learner starts from the initial state s 1 = s ini and proceeds as follows: at step h = 1 , . . . , H , the learner observes the current state s h , selects an action a h according to π h ( s h ) , receives a reward r h = r h ( s h , a h ) , and transitions to the next state s h +1 according to the transition kernel P h ( · | s h , a h ) . Fixing a policy π , we define the Q -function and the value function as follows: ∀ ( s, a ) ∈ S × A , h ∈ [ H ] ,

<!-- formula-not-decoded -->

The optimal Q -function and value function are defined by:

<!-- formula-not-decoded -->

By the Bellman optimality conditions, we have,

<!-- formula-not-decoded -->

Linear Function Approximation. We assume that both the reward function and the transition kernel lie within a known low-dimensional subspace, a setting commonly referred to as a linear MDP [Yang and Wang, 2019, Jin et al., 2023].

Assumption 2 (Linear MDP [Jin et al., 2023]) . Let { ϕ h ( s, a ) } ( s,a ) ∈S×A , h ∈ [ H ] be a collection of known feature vectors such that max s,a ∥ ϕ h ( s, a ) ∥ 2 ≤ 1 . For each h ∈ [ H ] , there exist vectors θ h ∈ R d and d measures µ h = ( µ 1 h , µ 2 h , . . . , µ d h ) over the state space S , representing the reward and transition kernels respectively, such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, we assume ∥ ∥ ∫ s ∈S v ( s ) dµ h ( s ) ∥ ∥ 2 ≤ √ d for any mapping v from S to [ -1 , 1] .

<!-- formula-not-decoded -->

Under Assumption 2, both the reward function and the transition kernel are linear in a shared set of d -dimensional features. This structure enables effective dimensionality reduction, especially when d ≪ SA .

Reward-free Exploration. We now introduce the framework of reward-free exploration. This setting consists of two phases: the exploration phase (see Algorithm 1) and the planning phase (see Algorithm 5). In the exploration phase, the learner interacts with the environment-without access to any reward signal-to collect a dataset D . In the planning phase, given any reward function { r h } h ∈ [ H ] satisfying Assumption 2, the learner is required to output an ϵ -optimal policy with probability at least 1 -δ , where ϵ is the accuracy parameter and δ is the failure probability.

Deployment-efficient Reward-free Exploration. We now present the definition of deployment complexity for reward-free exploration.

2 We assume the reward is deterministic for simplicity.

3 We may also assume the initial state s 1 is drawn from some fixed but unknown distribution d ini , which can be modeled by setting the transition from s ini to follow d ini .

Definition 3 (Huang et al. [2022]) . An algorithm is said to have deployment complexity K in linear MDPs if the following holds: given an arbitrary linear MDP satisfying Assumption 2, and for any accuracy parameter ϵ &gt; 0 and confidence level δ ∈ (0 , 1) , the algorithm performs at most K policy deployments and collects L trajectories per deployment, subject to the following constraints:

- (a) With probability at least 1 -δ , for any reward kernel { θ h } h ∈ [ H ] satisfying Assumption 2, the learner returns an ϵ -optimal policy π under this reward kernel, i.e.,

<!-- formula-not-decoded -->

where the expectation E π is taken over trajectories { s h , a h } H h =1 generated by executing policy π .

- (b) The number of trajectories per deployment, L , is polynomial in the problem parameters, i.e., L = poly ( d, H, 1 ϵ , log 1 δ ) . Moreover, L must be fixed a priori and cannot be adjusted adaptively between deployments.

Notations. For positive semidefinite (PSD) matrices A and B , we write A ⪯ B if B -A is PSD, i.e., B dominates A . We define the truncation operator T ( A,B ) as

<!-- formula-not-decoded -->

which represents the largest scaling of A that is still dominated by B . For each h ∈ [ H ] and v ∈ R S , we define θ h ( v ) := µ ⊤ h v , where µ h is the transition kernel. We also denote by 1 s the |S| -dimensional one-hot vector with a 1 in the s -th position.

## 4 Technical Overview

In this section, we give an overview of the technical challenges behind achieving Theorem 1, together our new ideas for tackling these challenges.

The Layer-by-layer Approach. Similar to existing algorithms with O ( H ) deployment complexity [Huang et al., 2022, Qiao et al., 2022, Qiao and Wang, 2022], our new algorithm is based on a layer-by-layer approach. For each layer 1 ≤ h ≤ H , based on an offline dataset obtained during previous iterations, our algorithm designs an exploration policy (a mixture of deterministic policies) for layer h , collect an offline dataset using the exploration policy, and then proceed to the next layer. Since we only use a single exploration policy for each layer, and there are H layers, the deployment complexity would consequently be H . Following such an approach, datasets obtained for previous layers will be used for the purpose of policy optimization and policy evaluation for later layers, and therefore, the dataset should be able to cover all directions in the feature space. Therefore, we must carefully design the exploration strategy, so that for any direction that can be reached by some policy, our exploration strategy could also reach that direction up to an appropriate competitive ratio. By repeatedly sample trajectories following the exploration strategy, we would get a dataset that is sufficient for the purpose of policy optimization and policy evaluation for later layers.

Dealing with Infrequent Directions. The main technical issue associated with the above approach, is that there could be directions that cannot be reached frequently by any policy. In such a case, it is unrealistic to require such a direction to be reachable by the exploration policy. Existing algorithms with O ( H ) deployment complexity [Huang et al., 2022, Qiao and Wang, 2022] avoids such an issue by assuming that any direction can be reached sufficiently frequently by some policy, in which case designing an exploration policy that can reach any direction in the feature space is feasible. However, since we do not assume explorability or reachability of the underlying MDP as in prior work [Huang et al., 2022, Qiao and Wang, 2022], we must handle those infrequent directions carefully.

If one simply chooses to ignore such infrequent directions, the error accumulated for handling such directions would in fact blow up exponentially, rendering the final sample complexity exponential in the feature dimension d or the horizon length H . In fact, such an issue occurs even in the simpler tabular setting. In the tabular setting, an infrequent direction is equivalent to a state-action pair unreachable by any policy, and in order to handle such states, prior work [Qiao et al., 2022] applied absorbing MDP to ignore those 'hard to visit' states. More specifically, once the algorithm detects

some state unreachable by any policy, that state would be directed to a dummy state in the absorbing MDP. Since we only direct states that are hard to visit to dummy states, the error accumulated during the whole process would be additive as we have more layers, which gives a polynomial sample complexity. Indeed, this is a high-level approach of the algorithm in Qiao et al. [2022].

On the other hand, for the linear MDP setting without the reachability assumption, handling infrequent directions is much more complicated. In the tabular setting, designing exploration policies is relatively simple since we can simply plan a policy for each individual state. On the other hand, for the linear MDP setting, we need to build the exploration policy in an iterative manner. Given directions that can be reached by the current exploration policy, we need to set the reward function appropriately to encourage exploring currently unreachable directions. More concretely, suppose the Λ = E [ ϕϕ ⊤ ] is the information matrix induced by the current exploration policy, for each state-action pair ( s, a ) with feature ϕ ( s, a ) , the reward function r ( s, a ) would be set to ϕ ( s, a ) ⊤ Λ -1 ϕ ( s, a ) . We then plan a new policy for the current quadratic reward function, and test whether new policy can indeed reach some new direction, both by utilizing the offline dataset. We proceed to the next layer if the algorithm can no longer find any new reachable direction. The total number of directions found during the whole process can be shown to be small, using a standard potential function argument based on the determinant of the information matrix. To test whether the new policy can indeed reach some new direction, we need to estimate its information matrix Λ = E [ ϕϕ ⊤ ] , again by using the offline dataset.

Note that by assuming reachability or explorability of the feature space, we no longer need to build the exploration policy iteratively since the whole feature space can be reached and therefore one can resort to approaches based on optimal experiment design. Indeed, this is the main idea behind previous work [Qiao and Wang, 2022]. However, such an approach critically relies on reachability or explorability of the feature space, which is one of the main technical challenges we aim to tackle.

Handling Bias Induced by Infrequent Directions. As mentioned, we heavily rely on the offline dataset obtained in previous layers for the purpose the offline policy optimization (planning for the quadratic reward function) and offline policy evaluations (for estimating the information matrix). Moreover, since we do not assume reachability of the feature space, there are always directions that cannot be reached by the exploration policy, and therefore, it is impossible for the offline dataset to cover the whole feature space. Imperfect coverage of the offline dataset will introduce additional error when conducting policy optimization and policy evaluation, due to the bias induced by infrequent directions. Although the error accumulated during offline policy optimization can be handle relatively easily, since a global argument based on comparing the groundtruth MDP and the MDP after ignoring infrequent directions would suffice, the error accumulated during offline policy evaluation is much more severe since the estimated information matrices would be used for deciding the next quadratic reward function. If not handled properly, the error will accumulate multiplicatively as we proceed to the next layer, rendering the final sample complexity exponential. Again, we note that by assuming reachability or explorability of the feature space as in prior work [Qiao and Wang, 2022], such an issue will not occur since the offline dataset would cover the whole feature space.

To handle such an issue, our new idea is to make sure the error of offline policy evaluation for estimating information matrices is always multiplicative w.r.t. the information matrix to be evaluated . More specifically, during the evaluation algorithm, if we encounter some state-action pair with feature ϕ = ϕ ( s, a ) , to ensure a multiplicative estimation error, we would add ϕϕ ⊤ to the evaluation result Λ only when ϕ ⊤ Λ -1 ϕ is small. However, this will introduce another chicken-and-egg situation: without knowing the groundtruth information matrix Λ , it is impossible to test whether ϕ ⊤ Λ -1 ϕ is small or not. To handle this, we use another iterative process to estimate the information matrix. Initially, the information matrix is set to be the identity matrix. In each iteration, in order to test whether ϕ ⊤ Λ -1 ϕ is small or not, we use the information matrix Λ obtained in the previous iteration, adding up ϕϕ ⊤ for those ϕ that passed the test to form the new information matrix, and proceed to the next iteration. We stop the whole iteration process if the two information matrices obtained in two consecutive iterations are close enough in a multiplicative sense. By using another potential function argument based on the determinant of the information matrix, it can shown that the iterative process stops with small number of rounds. Such an idea is another major technical contribution of this paper.

Handling Dependency Issues by Independent Copies. As discussed ealier, our final algorithm involves two iterative processes, and since the results of different iterations all rely on the same offline dataset, these results are subtly coupled with each other. Fortunately, such dependency issues are relatively easy to handle, as we can simply make independent copies of the offline dataset by

following the exploration policy and repeatedly sampling trajectories with fresh randomness. We denote each independent copy as a sub-dataset, which will be explained in more details in Section 5.

Our final algorithm is a careful combination of all ideas mentioned above.

## 5 Algorithms

In this section, we describe our algorithms for achieving Theorem 1. The parameter settings are postpone to Appendix A due to space limitation.

Datapoint and Sub-dataset. The typical approach for handling linear MDPs is to treat { ϕ h ( s, a ) , ˜ s } as a datapoint, where for a state s , ˜ s is the next state obtained by taking action a at level h . In our algorithm, we further assign a weight w to each datapoint to balance its importance in the whole dataset. As a result, one datapoint in our algorithm has form { ϕ h ( s, a ) , ˜ s, w } . We remark that the weight w is determined immediately once { ϕ h ( s, a ) , ˜ s } is collected.

In our algorithm, we conduct linear regression for multiple times, each time using a group of N independent datapoints. Here, N is a parameter to be decided. We denote these N independent datapoints as a sub-dataset, which has form { ϕ h,i = ϕ h ( s i , a i ) , ˜ s h,i , λ h,i } i ∈ [ N ] . To keep the statistical independence between different linear regression instances, we collect multiple independent copies of sub-datasets, so that the data used by different linear regression instances are independent.

Exploration phase: Algorithm 1. In the exploration phase, our algorithm collects samples in a layer-by-layer manner, and each layer uses a single deployment. In each layer, we assume that enough information about previous layers has been learned and focuse on learning the current layer. For the current layer, Policy-Design is called to design the exploration policy based on existing samples, and Policy-Execution is called to execute the exploration policy and collect new samples.

In each call of Policy-Design , there are m offline policy optimization sub-problems (see Line 6 of Algorithm 2) and m offline policy evaluation sub-problems (see Line 11 of Algorithm 2). As mentioned, we collect multiple independent copies of datasets, and use a group of independent copies datasets to solve each sub-problem. More precisely, we collect (2 m 2 +1) · H independent copies for each dataset to solve the 2 mH sub-problems, where each dataset consists of N datapoints. Due to page limitation, the detail about how to collect samples is deferred to Algorithm 7 in the appendix.

Policy-Design (Algorithm 2). Given datasets in the first h -1 layers, now we consider learning the h -th layer. The learner first designs reward function with form r h ( s, a ) ← min { ϕ ⊤ h ( s, a )Λ -1 ϕ h ( s, a ) , 1 } , where Λ is the current information matrix. We hope to update Λ as

<!-- formula-not-decoded -->

where π old is a near-optimal policy w.r.t. the reward r old = min { ϕ ⊤ h Λ -1 old ϕ h , 1 } . By iteratively running this process, we will obtain some Λ so that max π E π [ min { ϕ ⊤ h Λ -1 ϕ h , 1 } ] is small. However, as discussed in Section 4, due to the infrequent directions, it is inappropriate to add E π old [ ϕ h ϕ ⊤ h ] to Λ directly. Here, we need to truncate the infrequent directions in the distribution π old , and evaluate the truncated matrix with the offline datasets. Below we explain how to address this by Algorithm 3.

Matrix-Eval (Algorithm 3). In Algorithm 3, the input is a policy π and a group of datasets. The goal is to truncate the infrequent directions under π , and evaluate the information matrix after the truncation. To describe the high-level ideas, we assume D is an distribution over R d and the goal is to truncated the infrequent direction under D . For simplicity, we assume that D is known, so that one can compute Λ = E D [ ϕϕ ⊤ ] and those infrequent directions ϕ such that ϕ ⊤ Λ -1 ϕ is large. The next step is to re-scale ϕ , i.e., replace ϕ with w ( ϕ ) · ϕ such that w 2 ( ϕ ) ϕ ⊤ Λ -1 ϕ is small. However, after truncation, the new information matrix would be Λ new = E ϕ ∼ D [ w 2 ( ϕ ) ϕϕ ⊤ ] ⪯ Λ , which means that a frequent direction under Λ might turn to be an infrequent direction under Λ new . A straightforward idea is to repeat this process until Λ converges to some fixed point. Let F (Λ) = E ϕ ∼ D [ T ( ϕϕ ⊤ , c 1 Λ) ] where c 1 is the threshold for truncation and T is the operator defined in (2). By iteratively applying F ( · ) and noting that F ( · ) is non-increasing and the set of bounded PSD matrices is compact, the sequence { F ( n ) (Λ) } n ≥ 1 will converge to some Λ ∗ so that F (Λ ∗ ) = Λ ∗ , in which case no more truncation is needed and hence, infrequent directions no longer exist. One might be worried that the zero matrix is also a fixed point of F ( · ) in which case the truncation is meaningless. Fortunately, by choose c 1 properly large, we can show that Pr ϕ ∼ D [ ϕ ⊤ (Λ ∗ ) -1 ϕ ≥ c 1 ] = O ( ϵ ) , where epsilon

is the desired accuracy. This means only a small portion of directions are truncated. When D is unknown, we could draw samples from D to estimate E D [ T ( ϕϕ ⊤ , Λ)] and run the same iterative process. Incorporating this idea with linear regression, we devise Algorithm 3 and 4 to evaluate the truncated information matrix efficiently.

In the planning phase, we employ standard backward planning for linear MDPs (e.g., Algorithm 5 Planning and Algorithm 6 Planning-R ). See Appendix D for more details.

Computational Efficiency. We remark that the time complexity of our algorithm is polynomial in d, H, 1 /ϵ and the number of actions A . In comparison, the algorithm in Qiao and Wang [2022] is computationally inefficient, and the algorithm in Huang et al. [2022] suffers time complexity depending on the realization parameter. We refer the readers to Appendix E for more details.

̸

```
Algorithm 1 Exploration 1: Initialization: D h ←∅ , ˇ Λ h ← I for h ∈ [ H ] ; 2: for h = 1 , 2 , . . . , H do 3: { { π i,h } i m =1 , ˇ Λ h } ← Policy-Design ( h, {D h τ ( j ) } τ ∈ [ h -1] ,j ∈ [2 m 2 ] , { ˇ Λ τ } τ ∈ [ h -1] ) ; 4: // Roll out the policy and collect the datapoints. Each D τ h ( j ) constructs a sub-dataset for the h -th layer; 5: {D τ h ( j ) } j ∈ [2 m 2 +1] ,τ ∈ [ H ] ← Policy-Execution ( h, { π i,h } i m =1 , ˇ Λ h ) ; 6: end for 7: return : {D h h (2 m 2 +1) } h ∈ [ H ] and { ˇ Λ h } h ∈ [ H ] Algorithm 2 Policy-Design Input: horizon h ∈ [ H ] , block matrices { ˇ Λ τ } τ ∈ [ h -1] , sub-datasets { ϕ τ,i ( j ) , ˜ s τ,i ( j ) , λ τ,i ( j ) } i ∈ [ N ] for τ ∈ [ h -1] and j ∈ [2 m 2 ] ; Initialization: Λ 0 h = ζ I ; for ℓ = 1 , 2 , . . . , m do r ℓ h ( s, a ) ← min { ϕ h ( s, a ) ⊤ (Λ ℓ -1 h ) -1 ϕ h ( s, a ) , 1 } for all ( s, a ) ; r ℓ τ ( s, a ) ← 0 for τ = h and all ( s, a ) ; { π ℓ , v ℓ h } ← Planning-R ( h, r ℓ := { r ℓ τ } τ ∈ [ H ] , { ϕ τ,i ( m 2 + ℓ ) , ˜ s τ,i ( m 2 + ℓ ) , λ τ,i ( m 2 + ℓ ) } i ∈ [ N ] ,τ ∈ [ h -1] , { s 1 ,i ( m 2 + ℓ ) } N i =1 , { ˇ Λ τ } τ ∈ [ h -1] ) ; // Let Y τ,i ( a : b ) denote { Y τ,i ( j ) } b j = a for a ≤ b for Y = ϕ, ˜ s, λ and s 1 ; ˇ D ← { ϕ τ,i (( ℓ -1) m -1 : ℓm ) , ˜ s τ,i (( ℓ -1) m -1 : ℓm ) , λ τ,i (( ℓ -1) m -1 : ℓm ) } i ∈ [ N ] ,τ ∈ [ h -1] ; // Feed independent sub-datasets to Matrix-Eval ; { ¯ Λ ℓ h , ˇ Λ ℓ h } ← Matrix-Eval ( h, { ˇ Λ τ } τ ∈ [ h -1] , π ℓ , ˇ D ) ; Λ ℓ h ← Λ ℓ -1 h + ¯ Λ ℓ h ; end for return: { π i,h } i m =1 and ˇ Λ h ← Λ m h .
```

```
Algorithm 3 Matrix-Eval 1: Input: horizon h ∈ [ H ] , block matrices { ˇ Λ τ } h -1 τ =1 , policy π , sub-datasets { ϕ τ,i ( j ) , ˜ s τ,i ( j ) , λ τ,i ( j ) } τ ∈ [ h -1] ,i ∈ [ N ] ,j ∈ [ m ] ; 2: Λ ← I ; 3: for j = 1 , 2 , . . . , m do 4: // Estimate the truncated matrix with independent sub-datasets; 5: ˆ F 0 ← Truncated-Matrix-Eval ( h, π, { ˇ Λ τ } h -1 τ =1 , Λ , { ϕ τ,i ( j ) , ˜ s τ,i ( j ) , λ τ,i ( j ) } τ ∈ [ h -1] ,i ∈ [ N ] ) ; 6: if ˆ F 0 + ζ 2 x I ⪰ 1 2 Λ then 7: break and return { ˆ F 0 + ζ 2 x I , Λ } ; 8: else 9: Λ ← ˆ F 0 ; 10: end if 11: end for
```

```
Algorithm 4 Truncated-Matrix-Eval 1: Input: horizon h , policy π , block matrices { ˇ Λ τ } h -1 τ =1 , truncation matrix Λ , sub-datasets { ϕ τ,i , ˜ s τ,i , λ τ,i } τ ∈ [ h -1] ,i ∈ [ N ] ; 2: ˆ F h ( s ) ← T ( ϕ h ( s, π h ( s )) ϕ ⊤ h ( s, π h ( s )) , f 1 Λ) for s ∈ { ˜ s h -1 ,i } i ∈ [ N ] ; 3: for τ = h -1 , h -2 , ..., . . . , 1 do 4: X τ ← ∑ N i =1 λ 2 τ,i ϕ τ,i ϕ ⊤ τ,i + z I ; 5: for s ∈ { ˜ s τ -1 ,i } i ∈ [ N ] do 6: ϕ ← ϕ τ ( s, π τ ( s )) ; 7: if ϕ ⊤ ˇ Λ -1 τ ϕ ≥ 1 then 8: ˆ F τ ( s ) ← 0 ; 9: else 10: ˆ F τ ( s ) ← ϕ ⊤ X -1 τ ∑ N i =1 λ 2 τ,i ϕ τ,i ˆ F τ +1 (˜ s τ,i ) + 2 x Λ; 11: end if 12: end for 13: end for 14: return : ˆ F 0 := ˆ F 1 ( s ini ) ;
```

## 6 Analysis

In this section, we present the formal version of the main theorem and sketch its proof.

Theorem 4. By running Algorithm 1, the learner collects samples so that with probability 1 -δ , for any reward kernel { θ h } h ∈ [ H ] satisfying Assumption 2, the learner can return an ϵ -optimal policy π with Algorithm 5, i.e.,

<!-- formula-not-decoded -->

Moreover, Algorithm 1 uses O ( H ) deployments and ˜ O ( d 15 H 15 ϵ 5 ) samples.

Although we achieve reachability-independent sample complexity, the current dependencies on d, H and 1 /ϵ are far from being optimal, especially compared to the bound in Qiao and Wang [2022]. The reason is that the technical difficulty changes significantly when allowing dependency on the reachability parameter. The core challenge in deployment-efficient linear MDPs arises from the fact that the linear regression problem becomes ill-conditioned when the reachability parameter λ is very small. In the reachability-dependent methods (e.g., Qiao and Wang [2022]), one can pay O (1 /λ ∗ ) episodes to collect samples { ϕ i } i ≥ 1 such that the information matrix ∑ ϕ i ϕ ⊤ i is well-conditioned. Meanwhile, in the reachability-independent methods, we need to identify the ill-conditioned directions and avoid these directions in linear regression. This step would be even harder given the constraint in deployments, which requires offline evaluation of the information matrix.

Proof of Theorem 4. We first analyze the deployment complexity and sample complexity.

Deployment complexity. For each h = 1 , 2 , . . . , H , there is one deployment in Line 5. Therefore, the number of deployments is H .

Sample complexity. Algorithm 1 calls Algorithm 2 H times, each requiring (2 m 2 +1) N trajectories, resulting in a total sample complexity of H · H · (2 m 2 +1) N = ˜ O ( d 15 H 15 ϵ 5 ) .

To finish the proof, we use the following lemma to prove the optimality of the learned policy. See full proof in Appendix C.9

Lemma 5. With probability 1 -δ , for any reward kernel θ ∈ { θ h } H h =1 satisfying Assumption 2, Planning ( θ, { ϕ h,i , ˜ s h,i , λ h,i } N i =1 } h ∈ [ H ] , { ˇ Λ h } h ∈ [ H ] ) (see Algorithm 5) returns an ϵ -optimal policy, where { ϕ h,i , ˜ s h,i , λ h,i } N i =1 } h ∈ [ H ] and { ˇ Λ h } h ∈ [ H ] is the output of Algorithm 1.

To prove Lemma 5, a central lemma is introduced as follows, which states that the output sub-dataset of Algorithm 1 could efficiently cover all policies.

Lemma 6. Recall that ˇ Λ τ is the block matrix output by Policy-Design in Line 3 in the τ -th iteration for τ ∈ [ h -1] . With probability 1 -δ 2 -δ 2 H , for any sub-dataset of Algorithm 1 for the h -th layer { ϕ h,i , ˜ s h,i , λ h,i } i ∈ [ N ] , we have

- (i). max π Pr π [ ϕ ⊤ h ˇ Λ -1 h ϕ h &gt; 1 , ϕ ⊤ τ ˇ Λ -1 τ ϕ τ ≤ 1 , ∀ τ ∈ [ h -1] ] ≤ ϵ 8 H 2 for all h ∈ [ H ] ;
- (ii). ∑ N i =1 λ 2 τ,i ϕ τ,i ϕ ⊤ τ,i + z I ⪰ N 8 m ˇ Λ h for all h ∈ [ H ] ;
- (iii). λ 2 h,i ϕ ⊤ h,i ˇ Λ -1 h ϕ h,i ≤ f 1 for all h ∈ [ H ] and i ∈ [ N ] .

In proving Lemma 6, we use induction to construct a truncated MDP with information matrices { ˇ Λ τ } τ ≥ 1 . The three conditions in Lemma 6 serve the following purposes:

- (i). To properly bound the truncation probability.
- (ii). To ensure each ˇ Λ τ is well-covered.
- (iii). To rescale each sample for compatibility with the current information matrix ˇ Λ τ .

The proof of Lemma 6 is postponed to Appendix C.1 due to space limitation.

## 7 Conclusion

In this work, we design a new RL algorithm whose sample complexity is polynomial in the feature dimension and horizon length, while achieving nearly optimal deployment complexity for linear MDPs. Moreover, our algorithm works under the reward-free exploration setting, and does not require any additional assumptions on the underlying MDP. In our new algorithm and analysis, we propose new methods to truncate state-action pairs in a data-dependent manner, and design efficient offline algorithms for evaluating information matrices. Given our new results, an interesting future direction is to generalize our new techniques to other RL problems. For example, for function classes with bounded eluder dimension [Wang et al., 2020b, Kong et al., 2021, Zhao et al., 2023] , it would be interesting to design RL algorithm with nearly optimal O ( H ) deployment complexity and polynomial sample complexity without relying on any additional assumptions.

## Acknowledgments

YC is supported in part by the Sloan Research Fellowship, the ONR grant N00014-22-1-2354, and the NSF grant CCF-2221009. JDL acknowledges support of Open Philanthropy, NSF IIS-2107304, NSF CCF-2212262, ONR Young Investigator Award, NSF CAREER Award 2144994, and NSF CCF-2019844. SSD acknowledges the support of NSF IIS-2110170, NSF DMS-2134106, NSF CCF-2212261, NSF IIS-2143493, NSF CCF-2019844, NSF IIS-2229881, and the Sloan Research Fellowship.

## References

- Yasin Abbasi-Yadkori, Dávid Pál, and Csaba Szepesvári. Improved algorithms for linear stochastic bandits. Advances in neural information processing systems , 24, 2011.
- Alekh Agarwal, Nan Jiang, Sham M Kakade, and Wen Sun. Reinforcement learning: Theory and algorithms. CS Dept., UW Seattle, Seattle, WA, USA, Tech. Rep , 32:96, 2019.
- Daniel Almirall, Scott N Compton, Meredith Gunlicks-Stoessel, Naihua Duan, and Susan A Murphy. Designing a pilot sequential multiple assignment randomized trial for developing an adaptive treatment strategy. Statistics in medicine , 31(17):1887-1902, 2012.
- Daniel Almirall, Inbal Nahum-Shani, Nancy E Sherwood, and Susan A Murphy. Introduction to smart designs for the development of adaptive interventions: with application to weight loss research. Translational behavioral medicine , 4(3):260-274, 2014.

- Yu Bai, Tengyang Xie, Nan Jiang, and Yu-Xiang Wang. Provably efficient q-learning with low switching cost. Advances in Neural Information Processing Systems , 32, 2019.
- Nicolo Cesa-Bianchi, Ofer Dekel, and Ohad Shamir. Online learning with switching costs and other adaptive adversaries. Advances in Neural Information Processing Systems , 26, 2013.
- Jinglin Chen, Aditya Modi, Akshay Krishnamurthy, Nan Jiang, and Alekh Agarwal. On the statistical efficiency of reward-free exploration in non-linear rl. Advances in Neural Information Processing Systems , 35:20960-20973, 2022.
- Yuejie Chi, Yuxin Chen, and Yuting Wei. Statistical and algorithmic foundations of reinforcement learning. arXiv preprint arXiv:2507.14444 , 2025.
- Minbo Gao, Tianle Xie, Simon S Du, and Lin F Yang. A provably efficient algorithm for linear markov decision process with low switching cost. arXiv preprint arXiv:2101.00494 , 2021.
- Jiawei Huang, Jinglin Chen, Li Zhao, Tao Qin, Nan Jiang, and Tie-Yan Liu. Towards deploymentefficient reinforcement learning: Lower bound and optimality. arXiv preprint arXiv:2202.06450 , 2022.
- Chi Jin, Akshay Krishnamurthy, Max Simchowitz, and Tiancheng Yu. Reward-free exploration for reinforcement learning. In International Conference on Machine Learning , pages 4870-4879. PMLR, 2020.
- Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. Mathematics of Operations Research , 48(3):14961521, 2023.
- Emilie Kaufmann, Pierre Ménard, Omar Darwiche Domingues, Anders Jonsson, Edouard Leurent, and Michal Valko. Adaptive reward-free exploration. In Algorithmic Learning Theory , pages 865-891. PMLR, 2021.
- Jack Kiefer and Jacob Wolfowitz. The equivalence of two extremum problems. Canadian Journal of Mathematics , 12:363-366, 1960.
- Jens Kober, J Andrew Bagnell, and Jan Peters. Reinforcement learning in robotics: A survey. The International Journal of Robotics Research , 32(11):1238-1274, 2013.
- Dingwen Kong, Ruslan Salakhutdinov, Ruosong Wang, and Lin F Yang. Online sub-sampling for reinforcement learning with general function approximation. arXiv preprint arXiv:2106.07203 , 2021.
- Huitan Lei, Inbal Nahum-Shani, Kevin Lynch, David Oslin, and Susan A Murphy. A" smart" design for building individualized treatment sequences. Annual review of clinical psychology , 8(1):21-48, 2012.
- Gen Li, Wenhao Zhan, Jason D Lee, Yuejie Chi, and Yuxin Chen. Reward-agnostic fine-tuning: Provable statistical benefits of hybrid reinforcement learning. Advances in Neural Information Processing Systems , 36:55582-55615, 2023.
- Gen Li, Yuling Yan, Yuxin Chen, and Jianqing Fan. Minimax-optimal reward-agnostic exploration in reinforcement learning. In The Thirty Seventh Annual Conference on Learning Theory , pages 3431-3436. PMLR, 2024.
- Tatsuya Matsushima, Hiroki Furuta, Yutaka Matsuo, Ofir Nachum, and Shixiang Gu. Deploymentefficient reinforcement learning via model-based offline optimization. arXiv preprint arXiv:2006.03647 , 2020.
- Pierre Ménard, Omar Darwiche Domingues, Anders Jonsson, Emilie Kaufmann, Edouard Leurent, and Michal Valko. Fast active learning for pure exploration in reinforcement learning. In International Conference on Machine Learning , pages 7599-7608. PMLR, 2021.
- Dan Qiao and Yu-Xiang Wang. Near-optimal deployment efficiency in reward-free reinforcement learning with linear function approximation. arXiv preprint arXiv:2210.00701 , 2022.

- Dan Qiao, Ming Yin, Ming Min, and Yu-Xiang Wang. Sample-efficient reinforcement learning with loglog (t) switching cost. In International Conference on Machine Learning , pages 18031-18061. PMLR, 2022.
- Yufei Ruan, Jiaqi Yang, and Yuan Zhou. Linear bandits with limited adaptivity and learning distributional optimal design. In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing , pages 74-87, 2021.
- David Simchi-Levi and Yunzong Xu. Phase transitions and cyclic phenomena in bandits with switching constraints. Advances in Neural Information Processing Systems , 32, 2019.
- Georgios Theocharous, Philip S Thomas, and Mohammad Ghavamzadeh. Ad recommendation systems for life-time value optimization. In Proceedings of the 24th international conference on world wide web , pages 1305-1310, 2015.
- Joel A Tropp. User-friendly tail bounds for sums of random matrices. Foundations of computational mathematics , 12(4):389-434, 2012.
- Andrew J Wagenmaker, Yifang Chen, Max Simchowitz, Simon Du, and Kevin Jamieson. Rewardfree rl is no harder than reward-aware rl in linear markov decision processes. In International Conference on Machine Learning , pages 22430-22456. PMLR, 2022.
- Ruosong Wang, Simon S Du, Lin Yang, and Russ R Salakhutdinov. On reward-free reinforcement learning with linear function approximation. Advances in neural information processing systems , 33:17816-17826, 2020a.
- Ruosong Wang, Russ R Salakhutdinov, and Lin Yang. Reinforcement learning with general value function approximation: Provably efficient approach via bounded eluder dimension. Advances in Neural Information Processing Systems , 33:6123-6135, 2020b.
- Tianhao Wang, Dongruo Zhou, and Quanquan Gu. Provably efficient reinforcement learning with linear function approximation under adaptivity constraints. Advances in Neural Information Processing Systems , 34:13524-13536, 2021.
- Lin Yang and Mengdi Wang. Sample-optimal parametric q-learning using linearly additive features. In International conference on machine learning , pages 6995-7004. PMLR, 2019.
- Andrea Zanette, Alessandro Lazaric, Mykel J Kochenderfer, and Emma Brunskill. Provably efficient reward-agnostic navigation with linear value iteration. Advances in Neural Information Processing Systems , 33:11756-11766, 2020.
- Weitong Zhang, Dongruo Zhou, and Quanquan Gu. Reward-free model-based reinforcement learning with linear function approximation. Advances in Neural Information Processing Systems , 34: 1582-1593, 2021a.
- Xuezhou Zhang, Yuzhe Ma, and Adish Singla. Task-agnostic exploration in reinforcement learning. Advances in Neural Information Processing Systems , 33:11734-11743, 2020a.
- Zihan Zhang, Simon S Du, and Xiangyang Ji. Nearly minimax optimal reward-free reinforcement learning. arXiv preprint arXiv:2010.05901 , 2020b.
- Zihan Zhang, Yuan Zhou, and Xiangyang Ji. Almost optimal model-free reinforcement learningvia reference-advantage decomposition. Advances in Neural Information Processing Systems , 33: 15198-15207, 2020c.
- Zihan Zhang, Simon Du, and Xiangyang Ji. Near optimal reward-free reinforcement learning. In International Conference on Machine Learning , pages 12402-12412. PMLR, 2021b.
- Heyang Zhao, Jiafan He, and Quanquan Gu. A nearly optimal and low-switching algorithm for reinforcement learning with general function approximation. arXiv preprint arXiv:2311.15238 , 2023.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main contribution of this work is developing a deployment efficient algorithm for linear MDPs.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: In the conclusion section, we have discussed the limitations of the work and possible future directions to overcome these limitations.

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

Justification: All assumptions are clearly discussed. Full proofs are also provided.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA] .

Justification: This paper does not include experiments.

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

Answer: [NA] .

Justification: This paper does not include experiments.

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

Answer: [NA] .

Justification: This paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA] .

Justification: This paper does not include experiments.

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

Answer: [NA] .

Justification: This paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conforms with the NeurIPS Code of Ethics in every respect.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA] .

Justification: This work focuses on the fundamental aspects of reinforcement learning, and there is no foreseeable societal impact.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer:[NA] .

Justification: This paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer:[NA] .

Justification: This paper does not use existing assets.

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

Answer: [NA] .

Justification: This paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA] .

Justification: This paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA] .

Justification: This paper does not involve crowdsourcing nor research with human subjects. Guidelines:

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
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/ LLM ) for what should or should not be described.

## A Additional Parameter Settings and Notations

Assume d, H ≥ 40 , ϵ ≤ 1 40 . Set x = 1 1000 d 2 H , f 1 = 320 dH 2 ϵ , ζ = ϵ 5 10000 d 5 H 15 , ξ = ( ϵ 10 d 2 H 2 ) 10 , z = 100000 ϵ 2 d 4 H 5 , m = 32000 d 4 H 3 ϵ , N = 10 9 d 7 H 7 log ( dH ϵδ ) ϵ 3 . For a symmetric matrix A and a PSD matrix B , we write | A | ⪯ B iff B + A ⪰ 0 and B -A ⪰ 0 . We also present a table of notations as follows.

Table 2: Additional Notations.

| Notation                          | Comments                                                                            |
|-----------------------------------|-------------------------------------------------------------------------------------|
| P h ( ·&#124; s,a )               | the transition probability for the triple ( h, s,a )                                |
| r h ( s,a )                       | the reward expectation for the triple ( h, s,a )                                    |
| ϕ h ( s,a )                       | the d -dimensional feature vector for the triple ( h, s,a )                         |
| µ h                               | the probability transition kernel be such that P h ( ·&#124; , s,a ) = µϕ h ( s,a ) |
| θ h ( v )                         | the d -dimensional payoff vector defined as µ ⊤ h v                                 |
| T ( · , · )                       | the truncation function                                                             |
| N                                 | the number of datapoints in one dataset                                             |
| { ϕ τ , ˜ s τ ,λ }                | one sample from the τ -th layer                                                     |
| { ϕ τ,i , ˜ s τ,i ,λ τ,i } N i =1 | an independent dataset from the τ -th layer                                         |
| ζ                                 | the regularization parameter                                                        |
| ξ                                 | the discretization parameter                                                        |
| E 1 ( ϕ, v )                      | the concentration event for ϕ and value v w.r.t. an independent dataset             |
| E 2 ( ϕ, f )                      | the concentration event for ϕ and matrix value f w.r.t. an independent dataset      |

## B Technical Lemmas

Lemma 7 (General Equivalence Theorem in Kiefer and Wolfowitz [1960]) . For any bounded subset X ⊂ R d , there exists a distribution K ( X ) supported on X , such that for any ϵ &gt; 0 , it holds that

<!-- formula-not-decoded -->

Furthermore, there exists a mapping π G , which maps a context X to a distribution over X such that

<!-- formula-not-decoded -->

When supp( X ) has a finite size, π G ( X ) could be implemented within poly( | supp( X ) | , log(1 /ϵ )) time.

Lemma 8. Assume 0 ≤ κ ≤ 0 . 1 . Let Λ 0 = ζ I . For each i ≥ 1 , let D i be a distribution over R d satisfying that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

Then we have that for any n ≥ 1 .

Proof. Fix i ≥ 1 . Note that (4) is equivalent to

<!-- formula-not-decoded -->

Let W := E ϕ ∼ D i [ T ( ϕϕ ⊤ , Λ i -1 ) ] ⪯ E ϕ ∼ D i [ ϕϕ ⊤ ] . By definition, it holds that W ⪯ Λ i -1 and W +Λ i -1 ⪯ 2Λ i -1 . We then have that

<!-- formula-not-decoded -->

The proof is completed by taking sum over i from 1 to n .

## B.1 Concentration Inequalities

Lemma 9. Let X 1 , X 2 , ..., X n be a group of zero-mean matrices such that -Λ ⪯ X i ⪯ Λ with probability 1 for all i ∈ [ N ] . Let w 1 , w 2 , ..., w n be a group of reals. With probability 1 -δ ,

<!-- formula-not-decoded -->

Proof. Without loss of generality, we assume Λ = I . For 0 ≤ t ≤ 1 max i | w i | , define

<!-- formula-not-decoded -->

Then we have that

<!-- formula-not-decoded -->

where the first inequality is by Lieb's inequality (see Theorem 3.2, Tropp [2012]) and the second inequality is by Fact 10, 11 and 12. As a result, we learn that E [ E n ] ≤ E [ E 0 ] = d , which means that with probability 1 -δ/ 2 , the maximal eigenvalue of ∑ k i =1 w i X i is at most 2 √∑ n i =1 w 2 i log(2 d/δ ) + 2 max i | w i | log(2 d/δ ) . Similar arguments work for the other side. The proof is completed.

Fact 10. Assume X is a stochastic symmetric matrix and -I ⪯ X ⪯ I and E [ X ] = 0 . It then holds that

<!-- formula-not-decoded -->

for any 0 ≤ t ≤ 1 .

Proof. By definition, we learn that

Taking expectation we learn that

<!-- formula-not-decoded -->

Fact 11. Assume X and Y are two positively definite matrices such that X ⪯ Y . It then holds that log( X ) ⪯ log( Y ) .

Proof. Note that for any m ≥ 0 , it holds that

<!-- formula-not-decoded -->

Because X ⪯ Y , it holds that -( X + t I ) t ⪯ -( Y + t I ) -1 for any t ≥ 0 . Then for any m ≥ 0 ,

<!-- formula-not-decoded -->

Fix λ &gt; 0 and choose m ≥ 1 λ ∥ Y ∥ ∞ . We have that log( Y + m I ) ⪯ log( m (1 + λ )) I and log( X + m I ) ⪰ log( m (1 -λ )) I . As a result, for any λ &gt; 0 , we learn that

<!-- formula-not-decoded -->

which implies log( X ) ⪯ log( Y ) .

Fact 12. Let X,Y be two symmetric matrices and X ⪯ 0 . It then holds that

<!-- formula-not-decoded -->

Proof. It suffices to verify that Trace(( X + Y ) k ) ≤ Trace( Y k ) for each k ≥ 2 , which is a direct result from Löwner-Heinz theorem.

## C Missing Lemmas and Proofs

## C.1 Proof of Lemma 6

We will prove by induction over the layers. Fix h ∈ [ H ] and assume the three conditions in Lemma 6 holds for the first h -1 layers. To facilitate the presentation of the proof, we first introduce the notion of truncated MDP.

Truncated MDP. We define the truncated MDP M h -1 by redirecting all state-action pairs ( s, a ) to a dummy state at level τ if ϕ τ ( s, a ) ⊤ ˇ Λ -1 τ ϕ τ ( s, a ) &gt; 1 for τ ∈ [ h -1] . More precisely, a trajectory { ( s τ , a τ ) } H τ =1 under the original MDP M is mapped to { ( s 1 , a 1 ) , . . . , ( s k , a k ) , z , . . . , z } under M h -1 . Here k ≤ h -1 is the smallest integer such that ϕ ⊤ k ( s k , a k ) ˇ Λ -1 k ϕ k ( s k , a k ) &gt; 1 and z is the dummy state. If ϕ ⊤ k ( s k , a k ) ˇ Λ -1 k ϕ k ( s k , a k ) ≤ 1 for all k ∈ [ h -1] , the trajectory is unchanged.

In the following, we re-define E [ · ] and Pr[ · ] to be the expectation and probability under M h -1 . We verify the three conditions as follows.

<!-- formula-not-decoded -->

Condition (i). By Lemma 15, with probability 1 -δ 8 H , max π E π [ min { ϕ ⊤ h ˇ Λ -1 h ϕ h , 1 } ] ≤ ϵ 8 H 2 , which implies that max π Pr π [ ϕ ⊤ h ˇ Λ -1 h ϕ h &gt; 1 ] ≤ ϵ 8 H 2 . The proof is finished by noting the above inequality in the truncated MDP M h -1 is equivalent to (i).

Condition (ii). By Lemma 19 , with probability 1 -δ 16 H , it holds that ∑ N i =1 λ 2 h,i ϕ h,i ϕ ⊤ h,i + z I ⪰ N 8 m ˇ Λ h for all sub-datasets { ϕ h,i , ˜ s h,i , λ h,i } N i =1 .

Condition (iii). To verify the third condition, it suffices to note the definition λ h,j = min { √ f 1 ϕ ⊤ h,i ˇ Λ -1 h ϕ h,j , 1 } (See Algorithm 7).

The proof is finished.

## C.2 Statement and Proof of Lemma 13

Lemma 13. Fix h ∈ [ H ] . Recall x = 1 100 d 2 H ≥ 60 √ md log ( dH ϵδ ) N . Define F h ( s ) := ˆ F h ( s ) = T ( ϕ h ( s, π h ( s )) ϕ ⊤ h ( s, π h ( s )) , f 1 Λ) . For τ = h -1 , h -2 , . . . , 1 , we define F τ ( s ) = E s ′ ∼ P τ,s,πτ ( s ) [ F τ +1 ( s ′ ) · I [ ϕ ⊤ τ ( s, π τ ( s )) ˇ Λ -1 τ ϕ ( s, π τ ( s )) ≤ 1]] and F 0 = F 1 ( s 1 ) = F 1 ( s ini ) .

Let ˆ F 0 be the output of the Algorithm 4 with input Λ and a group of independent sub-datasets { ϕ τ,i , ˜ s τ,i , λ τ,i } τ ∈ [ h -1] ,i ∈ [ N ] . we have that

<!-- formula-not-decoded -->

Proof. It is obvious that F τ ( s ) is PSD for any proper τ and s . We prove by induction that

<!-- formula-not-decoded -->

for any 1 ≤ τ ≤ h and s ∈ { ˜ s τ -1 ,i } i ≥ 1 .

For τ = h , we have that ˆ F τ ( s ) = F τ ( s ) for any s ∈ S . Fix ℓ ≥ 2 and assume that (5) holds for τ = ℓ .

For s such that ϕ ℓ -1 ( s, π ℓ -1 ( s )) ˇ Λ -1 ℓ -1 ϕ ( s, π ℓ -1 ( s )) &gt; 1 , we have that ˆ F ℓ -1 ( s ) = F ℓ -1 ( s ) = 0 , where (5) holds trivially. Below we assume ϕ ℓ -1 ( s, π ℓ -1 ( s )) ˇ Λ -1 ℓ -1 ϕ ( s, π ℓ -1 ( s )) ≤ 1 . Recall that X τ = ∑ N i =1 λ 2 ℓ -1 ,i ϕ ℓ -1 ,i ϕ ⊤ ℓ -1 ,i + z I . By definition, we have that for s ∈ { ˜ s ℓ -2 ,i } i ≥ 1

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

By the induction assumption, we have that

<!-- formula-not-decoded -->

By Lemma 14, with probability 1 -δ 16 mH 2 it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the second term ∆ (2) ℓ -1 ( s ) , by the induction condition, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Putting all together and noting that x ≤ 1 100 dH , we learn that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof of (5) is finished.

Note that

<!-- formula-not-decoded -->

Using the induction condition, for any s ∈ S it holds that

<!-- formula-not-decoded -->

As a result,

<!-- formula-not-decoded -->

As a result, we obtain that

<!-- formula-not-decoded -->

The proof is finished.

## C.3 Statement and Proof of Lemma 14

Lemma 14. Fix f : S → R d 2 such that 0 ⪯ f ( s ) ⪯ Λ , ∀ s ∈ S for some PSD matrix Λ . Let { ϕ τ,i , ˜ s τ,i , λ τ,i } N i =1 be a sub-dataset from the τ -th layer. Assume { ϕ τ,i , ˜ s τ,i , λ τ,i } N i =1 is independent of f . Let X τ = ∑ N i =1 λ 2 τ,i ϕ τ,i ϕ ⊤ τ,i + z I . Then with probability 1 -δ 16 mH 2

<!-- formula-not-decoded -->

holds for any ϕ ∈ R 2 such that ∥ ϕ ∥ 2 ≤ 1 and ϕ ⊤ ˇ Λ -1 τ ϕ ≤ 1 .

Proof. By the induction assumption (i) and (iii) about the sub-dataset { ϕ τ,i , ˜ s τ,i , λ τ,i } N i =1 in Lemma 6, we have that X τ ⪰ N 8 m ˇ Λ τ for 1 ≤ τ ≤ h -1 and max i ϕ ⊤ τ,i X -1 τ ϕ τ,i ≤ f 1 . By Lemma 17, with probability 1 -δ 16 mH 2 , we have that

<!-- formula-not-decoded -->

## C.4 Statement and Proof of Lemma 15

Lemma 15. Recall the definition of ˇ Λ h = Λ m h in Algorithm 1. With probability 1 -δ 8 H , it holds that

<!-- formula-not-decoded -->

Proof. Recall the definition of { Λ ℓ h } m ℓ =0 , { ¯ Λ ℓ h } m ℓ =1 and { ˇ Λ ℓ h } m ℓ =1 in Algorithm 2. It then holds that Λ ℓ h = Λ ℓ -1 h + ¯ Λ ℓ h for 1 ≤ ℓ ≤ m . By the stop condition in Line 6, we have that ¯ Λ ℓ h ⪰ ˇ Λ ℓ h for 1 ≤ ℓ ≤ m . Let y ℓ = max π E π [ min { ϕ ⊤ h (Λ ℓ h ) -1 ϕ h , 1 }] . Then y ℓ is non-increasing in ℓ because Λ ℓ h is non-decreasing in ℓ . Let y = y m = max π E π [ min { ϕ ⊤ h ˇ Λ -1 h ϕ h , 1 } ] .

By Lemma 16 and Lemma 18, with probability

<!-- formula-not-decoded -->

Case i: y -B -d f 1 (1 -3 Hx ) ≥ y 4 . Recall that Λ ℓ h = Λ ℓ -1 h + ¯ Λ ℓ h for 1 ≤ ℓ ≤ m . By Lemma 13 we have that

<!-- formula-not-decoded -->

On the other hand, by (16), we have that

<!-- formula-not-decoded -->

By Lemma 8 with the D ℓ as the distribution of ϕ h · √ (1 -3 Hx ) · min {√ f 1 ϕ ⊤ h ( ˇ Λ ℓ h ) -1 ϕ h , 1 } under π ℓ and κ = y 10 ≤ 0 . 1 , we have that

<!-- formula-not-decoded -->

Using Lemma 13, we have that ¯ Λ ℓ h ⪯ 3 I and thus log(det(Λ m h )) ≤ d log(3 m ) . On the other hand, we have that log(det(Λ 0 h )) = d log( ζ ) , which means that my 40 ≤ d log(3 m/ζ ) . Therefore, we have that y ≤ 40 d log(3 m/ζ ) m ≤ ϵ 8 H 2 .

Case ii: y -B -d f 1 (1 -3 Hx ) &lt; y 4 . In this case, we have that y ≤ 4 3 B + 2 d f 1 ≤ ϵ 8 H 2 .

## C.5 Statement and Proof of Lemma 16

Lemma 16. Let B = 2 √ H 2 log(1 /δ ) N +2 H log(1 /δ ) N +2 H ( 32 √ md log ( dH ϵδ ) N + 32 md √ f 1 log ( dH ϵδ ) N ) . Let { V i 0 , π i } be the output of Opt with input reward as r i . With probability 1 -δ 8 mH ,

<!-- formula-not-decoded -->

Proof. Assume w ∈ R S satisfying ∥ w ∥ ∞ ≤ 1 . Let θ τ ( w ) = µ ⊤ τ w . By the induction condition ( i ) , we have that X τ ⪰ N 8 m ˇ Λ τ for τ ∈ [ h -1] .

By Lemma 17 and the induction condition (iii) that λ 2 τ,i ϕ ⊤ τ,i ˇ Λ -1 τ ϕ τ,i ≤ f 1 , with probability 1 -δ 16 mH 2 , we have that

<!-- formula-not-decoded -->

for all ϕ such that ∥ ϕ ∥ 2 ≤ 1 and ϕ ⊤ ˇ Λ -1 τ ϕ ≤ 1 .

Let { v τ ( s ) } and { v ∗ τ ( s ) } denote respectively the value function under the policy π i and the optimal value function. Let v 0 = v 1 ( s ini ) and v ∗ 0 = max π E π [ r i h ( s h ) ] . Because r i τ ( s, a ) ∈ [0 , 1] for any proper ( s, a, τ ) , we learn that v τ ( s ) , v ∗ τ ( s ) , v 0 , v ∗ 0 ∈ [0 , 1] . Recall the definition of { V τ ( s ) } in Algorithm 6. We next prove by induction that V τ ( s ) ≥ v ∗ τ ( s ) ≥ v τ ( s ) for any s ∈ S and 1 ≤ τ ≤ h . For τ = h , the inequality is trivial. Assume V τ ( s ) ≥ v τ ( s ) for any ℓ ≤ τ ≤ h . By (18) with w = V ℓ ( · )

<!-- formula-not-decoded -->

when ϕ ⊤ ℓ -1 ( s, a ) ˇ Λ -1 ℓ -1 ϕ ℓ -1 ( s, a ) ≤ 1 . In the case ϕ ⊤ ℓ -1 ( s, a ) ˇ Λ -1 ℓ -1 ϕ ℓ -1 ( s, a ) &gt; 1 , we have that

<!-- formula-not-decoded -->

because P ℓ -1 ,s,a = 1 z .

Therefore, we have that

<!-- formula-not-decoded -->

By Bernstein's inequality, with probability 1 -δ 16 mH , it holds that

<!-- formula-not-decoded -->

To bound the gap max π E π [ r i h ( s h ) ] -E π i [ r i h ( s h ) ] , direct computation gives that

<!-- formula-not-decoded -->

where (21) is by plugging ϕ τ,s τ ,a τ = ϕ and w = V τ +1 ( · ) into (18):

<!-- formula-not-decoded -->

## C.6 Statement and Proof of Lemma 17

Lemma 17. [Matrix concentration] Fix v ∈ R S such that ∥ v ∥ ∞ ≤ 1 and f : S → R d 2 such that 0 ⪯ f ( s ) ⪯ Λ , ∀ s ∈ S for some Λ . Let { ϕ τ,i , ˜ s τ,i , λ τ,i } N i =1 be a sub-dataset independent of v and f from the τ -th layer. Let X τ = ∑ N i =1 λ 2 τ,i ϕ τ,i ϕ ⊤ τ,i + z I . With probability 1 -δ 16 mH 2 , it holds that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

for any ϕ such that ∥ ϕ ∥ 2 ≤ 1 .

Proof. Let Φ( ξ ) be an ξ -net of the d -dimensional unit ball w.r.t. L 2 norm. Recall that ξ = ( ϵ 10 d 2 H 2 ) 10 . Then log( ξ ) ≤ 20 log( dH/ϵ ) . Let

<!-- formula-not-decoded -->

Then Pr[ E ( ϕ, v )] ≤ 2 δ by Bernstein's inequality. Assume ∪ ϕ ∈ Φ( ξ ) E 1 ( ϕ, v ) holds. Then for any ϕ ∈ R d , letting ψ be the nearest neighbor of ϕ in Φ( ξ ) , it holds that

<!-- formula-not-decoded -->

Noting that | Φ( ξ ) | ≤ ( d/ξ ) d , we have that Pr[ ∪ ϕ ∈ Φ( ξ ) ] E 1 ( ϕ, v ) ≤ 2( d/ξ ) d δ . By replacing δ with δ 16 mH | Φ( ξ ) | , with probability 1 -2 δ , it holds that

<!-- formula-not-decoded -->

for any ϕ such that ∥ ϕ ∥ 2 ≤ 1 .

Define E 2 ( ϕ, f ) to be the event where

<!-- formula-not-decoded -->

holds. We then show that Pr[ E 2 ( ϕ, f )] ≤ 2 δ .

<!-- formula-not-decoded -->

where we define ϵ τ,i = E s ′ ∼ P τ,s,a [ f ( s ′ )] -f (˜ s τ,i ) with ( s, a ) being the state-action pair such that ϕ τ ( s, a ) = ϕ τ,i . Noting that -Λ ⪯ ϵ τ,i ⪯ Λ with probability 1 , we have that

<!-- formula-not-decoded -->

holds with probability 1 -δ . In a similar way, with probability 1 -δ , we have

<!-- formula-not-decoded -->

To bound the second term zϕ ⊤ X -1 τ µ ⊤ τ f in (22), we have

<!-- formula-not-decoded -->

for any v ∈ R S such that ∥ v ∥ ∞ ≤ 1 . As a result, we have ∥ zϕ ⊤ X -1 τ µ ⊤ τ ∥ 1 ≤ √ ϕ ⊤ X -1 τ ϕ . Noting that 0 ⪯ f ( s ) ⪯ Λ for all s ∈ S , we have that

<!-- formula-not-decoded -->

By (22), (23), (24) and (26), we have that

<!-- formula-not-decoded -->

The proof is finished. Assume ∪ ϕ ∈ Φ( ξ ) E 2 ( ϕ, f ) holds. Fix ϕ and let ψ be the nearest neighbor of ϕ in Φ( ξ ) . We then have that

<!-- formula-not-decoded -->

We then bound the three terms in (28) separately. For the first term, we have that | ( ϕ -ψ ) ⊤ µ ⊤ τ v | ≤ ξ √ d for any v ∈ R S such that ∥ v ∥ ∞ ≤ 1 . As a result, we have that ∥ µ τ ( ϕ -ψ ) ∥ 1 ≤ ξ √ d , which implies that

<!-- formula-not-decoded -->

For the second term, we have that

<!-- formula-not-decoded -->

for any v ∈ R S such that ∥ v ∥ ∞ ≤ 1 . Using similar arguments, we learn that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

By ∪ ϕ ∈ Φ( ξ ) E 2 ( ϕ, f ) , we could bound the third term as

<!-- formula-not-decoded -->

Putting (29), (30) and (31) together, we learn that

<!-- formula-not-decoded -->

The proof is finished by replacing δ with δ 16 mH | Φ( ξ ) | .

## C.7 Statement and Proof of Lemma 18

Lemma 18. By running Algorithm 3, we have the following claims: (1) The iteration in line 3 ends in 10 d log ( 2 x v +1 ) rounds; (2) Let Λ end be the final value of Λ . Then it holds that

<!-- formula-not-decoded -->

Proof. Fix π . Let ˆ F 0 be the output of Algorithm 4 with input ( h, { ˇ Λ τ } h -1 τ =1 , Λ , D ) where D is a group of valid sub-datasets. Since h and { ˇ Λ τ } h -1 τ =1 are fixed in the context, we write ˆ F 0 = ˆ F 0 (Λ) as a (stochastic) function of Λ . We also define the expected truncated matrix as

<!-- formula-not-decoded -->

Number of iterations. Let Λ i be the value of Λ after the i -th iteration. Suppose there are T iterations. For 1 ≤ i ≤ T , we have that Λ i = ˆ F 0 (Λ i -1 ) . By Lemma 13, we have that

<!-- formula-not-decoded -->

Then we prove by induction that

<!-- formula-not-decoded -->

where C i = (1 + 11 Hx ) i for 1 ≤ i ≤ T . For i = 1 , we learn that Λ 0 = I and Λ 1 = ˆ F 0 ( I ) ⪯ (1 + 3 Hx ) F 0 ( I ) + 4 Hx I ⪯ (1 + 7 Hx ) I . For i ≥ 2 , by the induction and the fact that F 0 ( a Λ) ≤ aF 0 (Λ) for a ≥ 1 , we have that

<!-- formula-not-decoded -->

By (33) and (35), we have that

<!-- formula-not-decoded -->

The proof of (34) is finished.

By the update rule, we learn that

<!-- formula-not-decoded -->

Let ˇ Λ i = Λ i + ζ 2 x I for i ≥ 0 . Then we learn that

<!-- formula-not-decoded -->

As a result, the maximal eigenvalue of ˇ Λ -1 / 2 i -1 ˇ Λ i ˇ Λ -1 / 2 i -1 is at most (1 + 11 Hx ) i , while the minimal eigenvalue of ˇ Λ -1 / 2 i -1 ˇ Λ i ˇ Λ -1 / 2 i -1 is at most 1 2 . Then we have that

<!-- formula-not-decoded -->

By noting that d log( ζ/ 2 x ) ≤ log(det( ˇ Λ i )) and log(det( ˇ Λ 0 )) ≤ d log(1 + ζ/ 2 x ) , we learn that for any 1 ≤ j ≤ T

<!-- formula-not-decoded -->

As a result, it holds that

<!-- formula-not-decoded -->

for any 1 ≤ j ≤ T . Solving the quadratic inequality, we learn that T ≤ 10 d log ( 2 x ζ +1 ) .

Truncation probability. By definition, we have Λ end = Λ T . Note that Λ end ⪰ (1 -3 Hx ) F 0 (Λ end ) and F 0 (Λ end ) = E π [ T ( ϕ h ϕ ⊤ h , f 1 Λ end ) ] . We then have that

<!-- formula-not-decoded -->

On the other hand, by noting that

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

## C.8 Statement and Proof of Lemma 19

Lemma 19. Recall that z = 100000 ϵ 2 d 2 H 5 . Let D h = { ϕ h,i , ˜ s h,j , λ h,i } N i =1 be the one sub-dataset in in Line 9, Algorithm 7. With probability 1 -δ 16 m 2 H 2 , it holds that

<!-- formula-not-decoded -->

Proof. Let X i h and Y i h be respectively the final value of Λ and ˆ F 0 in the i -th call of Algorithm 3 in Algorithm 2 for the h -th round. Let I h = I [ ϕ τ ( s τ , π τ ( s τ )) ⊤ ˇ Λ -1 τ ϕ τ ( s τ , π τ ( s τ )) &lt; 1 , ∀ 1 ≤ τ ≤ h -1 ] . By Lemma 13 it holds that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Because ˇ Λ h ⪰ 1 2 X i h

<!-- formula-not-decoded -->

Also noting that λ h,i ϕ h,i ϕ ⊤ h,i ⪯ f 1 ˇ Λ h , using Lemma 9, we have that, with probability 1 -δ 16 mH 2 ,

<!-- formula-not-decoded -->

The proof is completed by re-arranging (37).

## C.9 Proof of Lemma 5

Let Θ be a dH -dimensional grid with distance ϵ 8 dH . Let Proj Θ ( · ) be the projection function to Θ by projecting each dimension to the grid. It is obvious that if θ = { θ h } h ∈ [ H ] satisfies that ∥ θ h ∥ 2 ≤ d, ∀ h ∈ [ H ] , then ∥ Proj Θ ,h ( θ ) ∥ 2 ≤ 2 d, ∀ h ∈ [ H ] .

It suffices to show that for any kernel { θ h } h ∈ [ H ] ∈ Θ , the output policy is 3 4 ϵ -optimal. Assume the conditions in Lemma 6 holds. Let ˇ M be the final truncated MDP M H . Then we have that

<!-- formula-not-decoded -->

As a result, for any π and reward function r such that ∥ r ∥ ∞ ≤ 1 , we have that

<!-- formula-not-decoded -->

Fix reward kernel θ = { θ h } h ∈ [ H ] ∈ Θ . We continue the analysis by assuming the ground MDP is ˇ M . Let π be the returned policy and π ∗ be the optimal policy. Let { V ∗ h,θ ( s ) , Q ∗ h,θ ( s, a ) } and { V π h,θ ( s ) , Q π h,θ ( s, a ) } be respectively the optimal value function and the value function of π . In particular, we let V ∗ 0 ,θ = V ∗ 1 ,θ ( s ini ) . Let { V h,θ ( s ) , Q h,θ ( s, a ) } be the value of { V h ( s ) , Q h ( s, a ) } in Algorithm 5 with input reward kernel as θ . Let V 0 ,θ = V 1 ,θ ( s ini ) and V π 0 ,θ = V π 1 ,θ ( s ini ) . When θ is clear from the context, we omit θ in the subscript.

We then have that

<!-- formula-not-decoded -->

We then prove by induction that V ∗ h ( s ) -V h ( s ) ≤ ( H -h ) · ϵ 8 H for all s ∈ S and h ∈ [ H ] . The inequality is trivial for h = H . Now we assume it is correct for all h ≥ ℓ . Let X τ = ∑ N i =1 λ 2 τ,i ϕ τ,i ϕ ⊤ τ,i + z I for τ ∈ [ H ] . Recall that Φ( ξ ) is an ξ -net of the d -dimensional unit ball. Fix ϕ ∈ Φ( ξ ) with ∥ ϕ ∥ 2 ≤ 1 and V ∈ R S with ∥ V ∥ ∞ ≤ H . By Bernstein's inequality (1-dimensional case of Lemma 9), with probability 1 -δ 4 H | Φ( ξ ) |·| Θ | , it holds that

<!-- formula-not-decoded -->

With a union bound over ϕ ∈ Φ( ξ ) , we learn that, with probability 1 -δ 4 H | Θ | ,

<!-- formula-not-decoded -->

for any ϕ such that ∥ ϕ ∥ 2 ≤ 1 and ϕ ⊤ ˇ Λ h ϕ ≤ 1 . Note that V h +1 ,θ ( · ) is determined by θ = { θ h } h ∈ [ H ] and the sub-datasets after the h -th layer (non-inclusive). With a union bound over θ ∈ Θ , we learn that: with probability 1 -δ 4 ,

<!-- formula-not-decoded -->

for any ϕ such that ∥ ϕ ∥ 2 ≤ 1 , ϕ ⊤ ˇ Λ h ϕ ≤ 1 and θ ∈ Θ . Then we have that

<!-- formula-not-decoded -->

As a result, we learn that V ∗ 0 -V 0 ≤ ϵ 8 . For the second term ( V 0 -V π 0 ) in (38), using similar arguments, we have that

<!-- formula-not-decoded -->

Putting all together, with probability 1 -δ 2 , we have that V ∗ 0 ,θ -V π 0 ,θ ≤ ϵ 4 ≤ 5 ϵ 8 for all θ ∈ Θ . As a result, π is at least a 3 4 ϵ -optimal policy under the original MDP M . The proof is completed.

## D Missing Algorithms

In this section, we present and explain the missing algorithms. Let Range [ a,b ] ( x ) = a I x &lt; a + x I [ a ≤ x ≤ b ] + b I [ x &gt; b ] for fixed a, b ∈ R and x ∈ R .

Planning (Algorithm 5). This algorithm is used to compute the optimal policy given a group of datasets. The planning method combines backward planning with linear regression. A key distinction is that the feature is clipped based on block matrices. Here Θ denotes a dH -dimensional grid with distance ϵ 8 dH , and Proj Θ ( · ) denotes the projection operator to Θ by projecting each dimension to the grid. We refer the readers to Appendix C.9 for the effectiveness of this algorithm.

Planning-R (Algorithm 6). This algorithm is used to compute the near-optimal policy given a fixed reward function. This algorithm is similar to Planning (Algorithm 5), except that the reward function is given as input (it is possible that the reward function is non-linear).

Policy-Execution (Algorithm 7). This algorithm is used to collect multiple copies of the datasets. The efficiency of the collected dataset is explained in Lemma 19.

```
Algorithm 5 Planning Input: reward kernel θ = { θ h } h ∈ [ H ] , sub-datasets { ϕ h,i , ˜ s h,i , λ h,i } i ∈ [ N ] ,h ∈ [ H ] , block matrices { ˇ Λ h } h ∈ [ H ] ; Initialization: θ ← Proj Θ ( θ ) ; V H +1 ( s ) ← 0 for all s ∈ S ; for h = H,H -1 , . . . , 1 do for ( s, a ) ∈ S × A ; do ϕ ← ϕ h ( s, a ) Q h ( s, a ) ← { ϕ ⊤ θ h + ϕ ⊤ ( ∑ N i =1 λ 2 h,i ϕ h,i ϕ ⊤ h,i + z I ) -1 ∑ N i =1 λ 2 h,i ϕ h,i V h +1 (˜ s h,i ) , ϕ ⊤ ˇ Λ -1 h ϕ ≤ 1; 0 , else ; Q h ( s, a ) ← Range [0 ,H ] ( Q h ( s, a )) ; end for for s ∈ S do V h ( s ) ← max a Q h ( s, a ) ; π h ( s ) ← arg max a Q h ( s, a ) ; end for end for return: π ←{ π h } h ∈ [ H ] .
```

## E Computational Efficiency

In this section, we present the time complexity of our algorithms. In the rest of the analysis, we use the fact that the time cost of computing the inverse of a d -dimensional PSD matrix is O ( d 3 ) .

## Algorithm 6 Planning-R

Input: horizon h , reward function r , sub-datasets { ϕ τ,i , ˜ s τ,i , λ τ,i } i ∈ [ N ] ,τ ∈ [ h -1] , block matrices

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## end for

<!-- formula-not-decoded -->

## end for end for

<!-- formula-not-decoded -->

## Algorithm 7 Policy-Execution

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 6: Run π to observe the feature ϕ h,j and the next state ˜ s h,j ;

<!-- formula-not-decoded -->

- 8: end for

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 10: end for
- 12: end for

<!-- formula-not-decoded -->

Truncated-Matrix-Eval (Algorithm 4). Firstly, the truncation operator T ( · ) could be implemented with time O ( d 3 ) . Then the total computational cost of this algorithm is bounded by O ( H ( Nd 2 + d 3 )) = O ( NHd 2 ) .

Matrix-Eval (Algorithm 3). The computational cost of this algorithm is at most O ( m ) multiplies that of Truncated-Matrix-Eval (Algorithm 4), which is O ( mNHd 2 ) .

Planning (Algorithm 5) and Planning-R (Algorithm 6). These two algorithms shares similar structure, with computational cost O ( HANd 2 ) to compute the action give the current state.

Policy-Design (Algorithm 2). The computational cost of this algorithm is at most O ( m ) multiplies that of Matrix-Eval (Algorithm 3) and Planning-R (Algorithm 6), which is bounded by O ( m 2 NHd 2 ) .

Policy-Execution (Algorithm 7). The time cost of this algorithm is simply O ( m 4 N 2 H 2 Ad 2 ) .

Exploration (Algorithm 1). By the above results, the total computation cost of this algorithm is O ( m 4 N 2 H 2 d 2 A ) = ˜ O ( d 32 H 28 A ϵ 10 ) .