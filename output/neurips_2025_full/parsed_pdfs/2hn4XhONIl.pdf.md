## Spectral Learning for Infinite-Horizon Average-Reward POMDPs

## Alessio Russo

DEIB, Politecnico di Milano alessio.russo@polimi.it

## Alberto Maria Metelli

DEIB, Politecnico di Milano albertomaria.metelli@polimi.it

## Marcello Restelli

DEIB, Politecnico di Milano marcello.restelli@polimi.it

## Abstract

We address the learning problem in the context of infinite-horizon average-reward POMDPs. Traditionally, this problem has been approached using Spectral Decomposition (SD) methods applied to samples collected under non-adaptive policies, such as uniform or round-robin policies. Recently, SD techniques have been extended to accommodate a restricted class of adaptive policies such as memoryless policies . However, the use of adaptive policies has introduced challenges related to data inefficiency, as SD methods typically require all samples to be drawn from a single policy. In this work, we propose Mixed Spectral Estimation , which generalizes spectral estimation techniques to support a broader class of belief-based policies . We solve the open question of whether spectral methods can be applied to samples collected from multiple policies, and we provide finite-sample guarantees for our approach under standard observability and ergodicity assumptions. Building on this data-efficient estimation method, we introduce the Mixed Spectral UCRL algorithm. Through a refined theoretical analysis, we demonstrate that it achieves a regret bound of r O p ? T q when compared to the optimal policy, without requiring full knowledge of either the transition or the observation model. Finally, we present numerical simulations that validate the theoretical analysis of both the proposed estimation procedure and the Mixed Spectral UCRL algorithm.

## 1 Introduction

In Reinforcement Learning (RL) [31], an agent interacts with an unknown or partially known environment to maximize the long-term sum of rewards. This approach has been successfully used in a variety of problems [23, 28, 8] under the assumption of fully observing the state of the environment. However, less attention has been paid to the more realistic scenario where the agent only receives partial and noisy observations from the environment, a problem which can be modeled through the Partially Observable Markov Decision Process (POMDP) [35] formalism. This setting can be used to represent various real-world applications such as autonomous driving [18], resource allocation [7], or financial settings [6]. Dealing with POMDPs is notably a challenging task both ( i ) statistically since it requires estimating the latent model parameters, and ( ii ) computationally since computing the optimal policy for a POMDP is intractable even when the model parameters are known [24].

In this work, we tackle the infinite-horizon average-reward POMDP formulation. In the past works, the learning problem in this setting has been addressed using Spectral Decomposition (SD) methods [2, 1]. In particular, the standard approach consists of deploying fully explorative policies (e.g., roundrobin or uniform) for data collection and then leveraging SD techniques for subsequent model

estimation [11, 32]. A different approach is proposed in [3] where spectral strategies are extended to samples collected from adaptive memoryless policies . 1 However, the model estimation they propose requires all samples to be drawn from a unique policy, which introduces data inefficiency issues since samples collected with older policies cannot be reused for model estimation. In addition, their approach is limited to stochastic policies under which each action can be chosen with a minimum positive probability ι ą 0 . By inspecting the limitations of current works, an important question arises: Can we apply spectral techniques on samples collected from multiple adaptive policies to improve the sample-efficiency of online learning algorithms for POMDPs?

Contributions. In this paper, we address this question and we provide the following contributions:

- We extend the spectral estimation procedure to the larger class of stationary belief-based policies .
- We answer the previous question affirmatively and propose a procedure, Mixed Spectral Estimation , with finite-sample guarantees for estimating the POMDP parameters (Section 5).
- We plug this novel estimation approach into a regret minimization algorithm, Mixed Spectral UCRL , and we show that we can indeed avoid using stochastic policies required in previous works. By focusing on instances satisfying the common one-step reachability assumption (Assumption 6.1), our algorithm is the first to achieve a regret of order r O p ? T q 2 competing against the optimal belief-based policy, hence improving over the state-of-the-art regret of order r O p T 2 { 3 q (Section 6).
- We provide numerical simulations showing both the effectiveness of the estimation procedure and the performance of our Mixed Spectral UCRL algorithm (Section 7).

## 2 Preliminaries

In this section, we provide the necessary background for the subsequent discussion. In the following, we will use ∆ p X q to denote the simplex over a finite set X , σ S p X q to denote the S -th singular value of matrix X , and X : to denote its Moore-Penrose pseudo-inverse.

Partially Observable MDP. A Partially Observable Markov Decision Process (POMDP) [35] is defined by a tuple Q -p S , A , O , T , O , ν , r q with S being a finite state space ( S -| S | ), A a finite action space ( A -| A | ) and O a finite observation space p O -| O |q . T ' t T a u a P A denotes a collection of transition matrices T a P R S ˆ S for every a P A . Each transition matrix T a p¨| s q P ∆ p S q defines the distribution of the next state when the agent takes action a in state s P S . O P R O ˆ S denotes the observation matrix O p¨| s q P ∆ p O q that represents the distribution over observations when the agent is in state s . ν P ∆ p S q denotes the distribution over the initial state, while r : O Ñr 0 , 1 s is the known reward function, mapping each observation to a finite reward such that r p o q is the reward received when the agents observe o P O . In a POMDP, states are hidden and the agent can only see its own actions and the observations. At each step t P N , the agent is in an unknown state s t , it receives an observation o t determined by O p¨| s t q and a reward r p o t q , then chooses an action a t and the environment transitions into a new state s t ` 1 according to T a t p¨| s t q . Then, the process repeats.

Policies in POMDPs. A policy π : ' p π t q 8 t ' 0 is a sequence of decision rules prescribing the action to play. We use H t -p O ˆ A q t ´ 1 ˆ O to denote the space of histories up to time t . A deterministic policy π t : H t Ñ A is such that π t p h q P A is the action chosen when history h P H t is observed.

From POMDP to Belief MDP. When the observation and the transition models are known, it is possible to build a belief vector b t P B (with B -∆ p S q ) from the observed history h t : ' p o j , a j q t ´ 1 j ' 0 ' o t , where ' denotes the sequence concatenation operator, as b t p s q : ' Pr p s t ' s | h t q , representing the probability that the true state is s having observed history h t . The update rule of the belief b t is determined using Bayes' theorem as:

<!-- formula-not-decoded -->

By using this notion of belief, we can transform the POMDP into a belief MDP [17] (which is a continuous-state MDP even if the original POMDP is tabular), which is used to address the POMDP

1 Under a memoryless policy, the choice over the next action a t is conditioned on the last observation o t only.

2 The notation r O p¨q disregards logarithmic terms.

learning problem. For an initial belief b P B , the average reward of the infinite-horizon belief MDP is defined as: ρ π b : ' lim sup T Ñ`8 p 1 { T q E r ř T ´ 1 t ' 0 r p o t q| b 0 ' b qs . When the underlying MDP is weakly-communicating, it has been shown [5] that the optimal average reward ρ ˚ : ' sup π : B Ñ ∆ p A q ρ π b is independent of the initial belief b and the following Bellman equation admits a unique solution:

<!-- formula-not-decoded -->

where g p b q -ř s P S ř o P O b p s q O p o | s q r p o q denotes the expected reward under belief b , while P p¨| b, a q is a probability measure over the next belief. 3 Finally, v : B Ñ R represents the bias function and quantifies the cumulative deviation of rewards w.r.t. ρ ˚ when starting from b [21].

## 3 Related Works

POMDP Learning. Learning in POMDPs is known to be challenging both from a statistical and a computational perspective. When the observation model does not provide enough information to identify the latent states, we refer to the POMDP as hard . These intractable instances can be ruled out by introducing a full-rank assumption on the observation model. A quantitative version of this assumption was first introduced in [16] and is formalized as a lower bound α ą 0 to the minimum singular value of the observation model, namely σ S p O q ě α . The instances satisfying this assumption can be efficiently learned and define the class of α -weakly revealing instances.

Weakly-Revealing POMDPs. The weakly-revealing assumption has been used both in the episodic [16, 19] and the infinite-horizon average-reward setting. By focusing on the latter, some works employed the simplifying assumption of having partial knowledge of the environment , in particular of the observation model. Among them, [13] provide a Bayesian regret of order O p T 2 { 3 q when compared against the optimal policy, while a recent work from [26] proposes the Action-wise OAS-UCRL algorithm, which employs an estimation procedure with finite-sample guarantees that leverages the knowledge of the observation model to learn the transition model. They reach a r O p ? T q regret guarantee when compared against the optimal policy. Several works have instead addressed the problem of fully learning the model parameters [11, 3, 34]. The standard approach relies on SD methods [1] for learning the latent variable model. In particular, [3] are the first to adapt SD methods to samples collected under the adaptive class of memoryless policies. They consider stochastic policies where each action is chosen with a positive probability ι ą 0 at each step and propose the SM-UCRL algorithm, which achieves a r O p ? T { ι 2 q regret guarantee when compared against this (less powerful) policy class. A different approach is taken in [32] where the regret is computed against the stronger class of deterministic ( ι ' 0 ) belief-based policies. They present the SEEU algorithm, which alternates between purely exploratory and purely exploitative phases. During exploration, samples are collected using a round-robin policy over the available actions, after which SD is applied to recover model parameters. Their algorithm achieves r O p T 2 { 3 q regret when compared against the optimal class of belief-based policies.

The introduction of our estimation strategy addresses two limitations of the aforementioned works. First, unlike the SEEU algorithm [32], we do not need to separate exploration and exploitation phases, as we can leverage samples collected during the exploitation phase to refine model estimates. Second, unlike the SM-UCRL [3], we are able to reuse samples from different policies, hence eliminating the need for stochastic policies ( ι ą 0 ) that foster continuous coverage of the action space. We refer to Table 1 for a comparison of our work with those mentioned above and to Appendix H for a more extensive discussion on the matter.

## 4 Problem Formulation

We consider the infinite-horizon average-reward POMDP setting described in Section 2. Specifically, we consider the undercomplete setting [16], where the number of states is less than or equal to the number of observations ( S ď O ). Our focus is on learning the POMDP parameters represented by the observation model O and the transition model T ' t T a u a P A . We consider the class of

3 We provide a precise definition of this quantity in the Notation section of Appendix C.

Table 1: Table comparing the SM-UCRL , SEEU and the Mixed Spectral UCRL algorithm.

| Property                                         | SM-UCRL   | SEEU            | Mixed Spectral UCRL   |
|--------------------------------------------------|-----------|-----------------|-----------------------|
| No assumption on minimum entry of obs. model     | ✓         | ✗               | ✓                     |
| No assumption on minimum entry of trans. model   | ✓         | ✗               | ✗                     |
| No assumption on minimum action probability      | ✗         | ✗               | ✓                     |
| Works with memoryless policies                   | ✓         | ✗               | ✓                     |
| Works with belief-based policies                 | ✗         | ✗               | ✓                     |
| Sample reuse with different policies             | ✗         | ✗               | ✓                     |
| Compares against the optimal belief-based policy | ✗         | ✓               | ✓                     |
| Regret w.r.t. optimal belief-based policy        | O p T q   | ˜ O p T 2 { 3 q | ˜ O p ? T q           |

belief-based policies π : B Ñ A , and we use P to denote such a set of policies. Before stating the main assumptions, we introduce some relevant quantities.

Let d π,b 0 t p s, a q -Pr p s t ' s, a t ' a | π, b 0 q be the t-step state-action distribution induced by policy π P P , with b 0 P B being the initial belief. Under mild regularity conditions (e.g., when the underlying MDP is weakly-communicating), a unique limiting distribution d π 8 p s, a q -lim t Ñ8 d π,b 0 t p s, a q P ∆ p S ˆ A q exists (see Proposition 5.1 in [25]) and it is independent of the initial belief b 0 . From the quantity just defined, we derive the stationary action distribution d π 8 P ∆ p A q defined as d π 8 p a q -ř s P S d π 8 p s, a q . Let us now introduce the conditional state distribution ω p a,π q P ∆ p S q defined as ω p a,π q s -d π 8 p s | a q ' d π 8 p s, a q{ d π 8 p a q , which is well-defined when d π 8 p a q ą 0 .

The following assumptions represent the natural extension to the POMDP setting of the assumptions commonly employed for learning in (uncontrolled) settings (i.e., Hidden Markov Models [1]).

Assumption 4.1 ( α -weakly Revealing Condition ) . There exists α ą 0 such that σ S p O q ě α .

This assumption quantifies the extent to which the received observations help in identifying the underlying hidden states. It is equivalent to the more common full-rank assumption largely adopted in problems involving the learning of Latent Variable Models [3, 12, 34]. It was first introduced in this form in [16] and then extensively employed in successive related works [19, 20, 26]. It has been shown that, without this assumption, learning becomes intractable [9].

Assumption 4.2 ( Invertibility ) . For every action a P A , the transition matrix T a is invertible.

This second assumption implies that for any state-action pair p s, a q P S ˆ A , its next-state distribution T a p¨| s q cannot be recovered as a linear combination of the next-state distribution of the other stateaction pairs. This condition is crucial for achieving identifiability and is widely used in the SD and POMDP literature [1, 3, 32, 34, 11].

Assumption 4.3 ( Per-Action Ergodicity ) . For any policy π P P , a unique limiting state-action distribution d π 8 p s, a q exists. Moreover, for every action a , if d π 8 p a q ą 0 , then ω p a,π q s ą 0 @ s P S .

Assumption 4.3 extends the standard non-degeneracy assumption [1] employed under SD techniques. The motivation behind this assumption lies in the fact that SD approaches are applied for each action a separately. Hence, in order to fully recover the transition model T a , all states should be visited with positive probability when taking action a (i.e., ω p a,π q s ą 0 ). In Appendix H, we show how related works [3, 32] tackling the POMDP setting rely on assumptions that subsume Assumption 4.3. A simple example when this assumption holds is when the transition matrices t T a u a P A have all positive entries, as we shall see in Section 6 (Assumption 6.1).

A discussion on the reasons why some of these assumptions are instead not required in the episodic setting is provided in Appendix H.2.

Learning Objective. Our goal is to find the policy attaining Equation (2) in the policy class P . Our learning objective is to minimize the cumulative regret after T P N time steps, defined as:

<!-- formula-not-decoded -->

where ρ ˚ represents the average reward obtained by the policy satisfying Equation (2), while r p o t q is the reward obtained from the observation received by playing policy π t played at time t .

We remark that solving Equation (2) and computing such an optimal policy is known to be computationally intractable. Various methods have been devised to provide an approximately optimal policy. Most of them focus on devising clever discretizations of the belief space and then solve the discretized instance [33, 27, 29]. In this work, however, we do not focus on this planning problem, but following a common approach in the POMDP literature [32, 3, 34, 13], we assume access to an optimization oracle capable of providing the optimal policy for a given POMDP model.

## 5 The POMDP Estimation Procedure

In this section, we present an adaptation of the common multi-view model employed for latent parameter estimation when using SD techniques [1, 3, 32].

## 5.1 The Multi-View Model

We now introduce a model-based strategy to estimate the parameters of the unknown POMDP which adapts the approach of [3]. For each step t P r 1 , T ´ 2 s 4 in which a t ' a P A , we construct three views containing the observations in three consecutive steps centered in t , i.e., o t ´ 1 , o t , o t ` 1 P O . Let us use (bold) o t P t 0 , 1 u O to denote the one-hot encoded vector corresponding to observation o t and similarly for the two remaining views o t ´ 1 and o t ` 1 . We further use vectors v p a q ν,t P R O with ν P t 1 , 2 , 3 u to refer to the three different view vectors when conditioned on a t ' a , and such that v p a q 1 ,t ' o t ´ 1 , v p a q 2 ,t ' o t and v p a q 3 ,t ' o t ` 1 respectively. Given a policy π P P , we define three view matrices V p a,π q ν P R O ˆ S with ν P t 1 , 2 , 3 u associated with action a P A , as follows:

<!-- formula-not-decoded -->

It can be observed that the three views are independent when conditioning on both s t and a t . We also denote with µ p a,π q ν,s ' V p a,π q ν p¨ , s q the s -th column of matrix V p a,π q ν .

Remark 5.1 . By inspecting the three different view matrices separately, we can observe that for the second view matrix it holds that V p a,π q 2 ' O , hence it does not depend on either action a or policy π . Differently, for the third view matrix, it can be shown that V p a,π q 3 ' OT J a , hence it is independent of policy π . Finally, the first view matrix V p a,π q 1 depends on both the action and employed policy. 5

Given this multi-view model, the following result from [1] applies:

Proposition 5.2. (Adapted from [3]) Let ν, ν 1 P t 1 , 2 , 3 u , π P P be a policy, and K p a,π q ν,ν 1 ' E ' v p a,π q ν b v p a,π q ν 1 ı be the covariance matrix between views v p a,π q ν and v p a,π q ν 1 , where b denotes the tensor product, and denote with the superscript : the Moore-Penrose pseudo-inverse. We define a modified version of the first and second views as:

<!-- formula-not-decoded -->

Then, the second and third moments of the modified views have a spectral decomposition as:

<!-- formula-not-decoded -->

where the expectations are w.r.t. the conditional state distribution ω p a,π q s defined in Section 4.

When Assumptions 4.1, 4.2 and 4.3 hold, the three view matrices V p a,π q ν P R O ˆ S with ν P t 1 , 2 , 3 u associated with each action a P A and policy π P P are full-column rank and a unique spectral decomposition exists [1]. As a consequence, the original model parameters can be recovered. In particular, this can be performed by exploiting the following known relations between the columns of

4 We exclude the first ( t ' 0 ) and the last ( t ' T ´ 1 ) steps.

5 For the detailed expression of V p a,π q 1 , we refer to Appendix A.

the different view matrices:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By applying SD techniques for each action a separately, we obtain estimates of the third view matrix V p a,π q 3 , hence of its columns µ p a,π q 3 ,s . Finally, when such estimates are available, the columns µ p a,π q 1 ,s and µ p a,π q 2 ,s of the remaining view matrices can be estimated by inverting Equations (5) and (6).

## 5.2 The Mixed Spectral Estimation Procedure

We now show how we combine samples coming from multiple policies, thus overcoming the limitations of existing approaches and leading to our novel Mixed Spectral Estimation . We define a set of L different trajectories of samples Γ -t τ l u L ´ 1 l ' 0 such that the l -th trajectory is generated from policy π l P P and is defined as τ l ' tp o l j , a l j qu N l ´ 1 j ' 0 . Additionally, we introduce the related set T p a q l ' t t P r 1 , N l ´ 2 s s.t. a l t ' a u which contains the time steps when action a is selected in the l -th trajectory. Let n p a q l ' | T p a q l | denote its cardinality. For each t P T p a q l , we construct the three corresponding views p v p a,l q 1 ,t , v p a,l q 2 ,t , v p a,l q 3 ,t q ' p o t ´ 1 , o t , o t ` 1 q , where the superscript l refers to the trajectory collected using π l . Our approach uses views from all the L trajectories to define new covariance matrices K p a,L q ν,ν 1 with ν, ν 1 P t 1 , 2 , 3 u and ν ‰ ν 1 . These are weighted versions of the original covariance matrices and are defined as follows:

<!-- formula-not-decoded -->

where N p a q L -ř L ´ 1 l ' 0 n p a q l , while ω p a,l q : ' ω p a,π l q P ∆ p S q denotes the conditional state distribution determined by policy π l and action a . We show that the following result holds when combining multiple policies. Its proof is deferred to Appendix A.

Theorem 5.3. Let Γ -t τ l u L ´ 1 l ' 0 be a set of trajectories collected using the set of policies t π l u L ´ 1 l ' 0 . We define a modified version of the first and second views as:

<!-- formula-not-decoded -->

where the covariance matrices are defined in Equation (7) . Let ω p a,L q -p 1 { N p a q L q ř L ´ 1 l ' 0 n p a q l ω p a,l q , then, the second and third moments of the modified views have a spectral decomposition as:

<!-- formula-not-decoded -->

where the expectations are w.r.t. the conditional state distributions ω p a,l q s .

This theorem shows that when the views v p a,l q 1 and v p a,l q 2 are modified using the weighted covariance matrices K p a,L q ν,ν 1 defined in Equation (7) instead of the covariance matrices K p a,l q ν,ν 1 associated with policy π l , the new second and third order moments have a spectral decomposition whose conditional state distribution ω p a,L q is an average of the original conditional state distributions, each one weighted proportionally by the cardinality n p a q l . Importantly, as discussed in Remark 5.1, the columns µ p a q 3 ,s of the third view matrix do not depend on the employed policies but only on action a , hence in Theorem 5.3, we do not report the dependence on the mixture of the L policies. The independence of the third view matrix from the employed policies plays a crucial role in proving Theorem 5.3.

Algorithm Pseudocode. The estimation procedure of the quantities described above, and of the estimated POMDP parameters, is described in the Mixed Spectral Estimation approach presented in Algorithm 1. For each action a , the view vectors are computed for all the L policies, and they are used to compute the mixture covariance matrices (Line 8). Given the new covariance matrices, the

## Algorithm 1 Mixed Spectral Estimation .

- 1: Input: Trajectory set Γ -t τ l u L ´ 1 l ' 0 where for each l we have τ l ' tp o l j , a l j qu N l ´ 1 j ' 0
- 2: Output: Estimated Observation model p O and Transition model t p T a u a P A
- 3: for a P A do
- 4: for l P r 0 , L ´ 1 s do
- 5: Construct views v p a,l q 1 ,t ' o t ´ 1 , v p a,l q 2 ,t ' o t , v p a,l q 3 ,t ' o t ` 1 for any t P T p a q l
- 6: end for
- 8: Compute covariance matrices for ν, ν 1 P t 1 , 2 , 3 u :
- 7: Compute N p a q L ' ř L ´ 1 l ' 0 n p a q l

<!-- formula-not-decoded -->

- 9: Compute modified views:

<!-- formula-not-decoded -->

- 10: Compute second and third moments:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 11: p V p a q 3 ' TENSORDECOMPOSITION p x M p a,L q 2 , x M p a,L q 3 q
- 12: Compute p V p a q 2 inverting Eq. (6)
- 13: end for
- 14: Define a ˚ P argmax a P A N p a q L
- 15: for a P A do
- 16: Match the columns of each p V p a q 2 with p V p a ˚ q 2
- 17: Permute the columns of p V p a q 3 using the same permutation adopted for p V p a q 2
- 18: end for
- 19: Compute p O according to Eq. (9)
- 20: for a P A do
- 21: Compute p T a according to Eq. (10)
- 22: end for

modified views are computed for each t P T p a q l with l P r 0 , L ´ 1 s (Line 9). The modified views are then used to compute second and third-order moments (Line 10), and a tensor decomposition routine 6 (line 11) is run for each action separately, thus obtaining the estimated view matrix p V p a q 3 . By inverting Equation (6), we are able to derive an estimate of the second view matrix p V p a q 2 . As noted in Remark 5.1, the second view matrices are identical across all actions, thus satisfying V p a q 2 ' O for any action a . Since spectral methods recover the columns of the original view matrices up to a permutation of the hidden states s [1], this equivalence allows us to align the columns of the different p V p a q 2 by appropriately permuting them, thus ensuring that the represented states are ordered consistently, as also done in [3]. To do that, we define a ˚ P argmax a P A N p a q L and choose p V p a ˚ q 2 as the reference view that the other views should match. 7 It is possible to show that when the estimation of each view is sufficiently accurate, the correct permutation can be found for each p V p a q 2 . When the permutation step is completed, the observation and transition model are computed as:

<!-- formula-not-decoded -->

6 We adopt the Robust Tensor Power (RTP) method from [1] as tensor decomposition strategy.

7 This way, for each action a , the columns of p V p a q 2 are permuted to minimize the 1 -norm error w.r.t. p V p a ˚ q 2 .

<!-- formula-not-decoded -->

where N L -ř a P A N p a q L . Thus, the estimated observation matrix is obtained as a weighted combination of the second view matrices p V p a q 2 , while each transition matrix is recovered by inverting the relation presented in Remark 5.1 and using the observation matrix computed as in Equation (9). The computational complexity of the presented approach is discussed in Appendix I. Algorithm 1 enjoys the following guarantees, which are proved in Appendix B.

Theorem 5.4. Let p O and t p T a u a P A be the observation and transition model estimated using Algorithm 1, respectively. Let Assumptions 4.1 and 4.2 hold and let Assumption 4.3 be true for any π l with l P r 0 , L ´ 1 s . Let δ P p 0 , 1 {p 3 SA qq , then for a sufficiently large number of samples N p a q L holding for every action a P A , with probability at least 1 ´ 3 SAδ , it holds that:

<!-- formula-not-decoded -->

a P A

<!-- formula-not-decoded -->

We highlight that Theorem 5.4 requires a minimum number of samples N p a q L for each action a (this number should satisfy Equation (38) reported in Appendix B), which depends on the set of L trajectories. Nevertheless, it places no restrictions on the length of the individual trajectories τ l , allowing for certain trajectories not to contain a specific action a . This aspect will be significant for proving the regret guarantees of our Mixed Spectral UCRL approach.

## 6 Mixed Spectral UCRL

The Mixed Spectral Estimation procedure can be easily combined with an optimistic strategy resembling the UCRL approach for MDPs [14]. We call this new algorithm Mixed Spectral UCRL , and we describe its workflow in Algorithm 2. During the first episode, we use a uniform policy π 0 (Line 3) to collect a sufficient amount of samples for each action a P A in order to provide a first estimate of the POMDP parameters. The whole interaction horizon is divided into episodes of different lengths. At the beginning of each new episode l , all samples collected up to that moment are used to estimate the new POMDP parameters according to Algorithm 1 (Line 7). Based on the estimated POMDP p Q l , we build a high-probability confidence set C l p δ l q of admissible POMDPs according to the bounds defined in Theorem 5.4, using a varying confidence level δ l -δ {p 3 SAl 3 q (Line 8). The optimistic policy and the associated POMDP are then computed at the beginning of episode l according to the program:

<!-- formula-not-decoded -->

where ρ p π, r Q q is the average reward of policy π in the POMDP instance r Q . As specified in Section 4, we assume access to an oracle to solve Equation (11). Then, each episode terminates

## Algorithm 2 Mixed Spectral UCRL .

- 1: Input: Confidence level δ , length of initial episode T 0 , total horizon T
- 2: Initialize: t Ð 0 , l Ð 0 , belief b 0 uniform over states, Trajectory set Γ ' tu
- 3: Build trajectory τ 0 from uniform policy π 0 for T 0 steps
- 4: Γ Ð Γ Yt τ 0 u
- 5: t Ð T 0 , l Ð 1 , Set N p a q 1 Ð n p a q 0 @ a P A
- 6: while t ă T do
- 7: Run Algorithm 1 using trajectory set Γ and obtain estimates p O and p T ' t p T a u a P A
- 8: Build a confidence set C l p δ l q of admissible POMDPs
- 9: Compute policy π l and optimistic Q l (Eq. 11)
- 10: τ l Ðpq , n p a q l Ð 0 for all a P A
- 11: Observe o t , get reward r t Ð r p o t q
- 12: Update belief b t using Equation (1)
- 13: Set a t Ð π l p b t q
- 14: while t ă T or n p a t q l ă N p a t q l do
- 15: Execute a t , Set n p a t q l Ð n p a t q l ` 1
- 16: Observe o t ` 1 , get reward r p o t ` 1 q
- 17: Update belief to b t ` 1 using Equation (1) and estimated p O and p T a t
- 18: Set a t ` 1 Ð π l p b t ` 1 q
- 19: τ l Ð τ l 'p o t , a t q

20:

Set

t

Ð

t

`

1

- 21: end while
- 22: Γ Ð Γ Yt τ l u
- 23: Set N p a q l ` 1 Ð N p a q l ` n p a q l @ a P A
- 24: Set l Ð l ` 1
- 25: end while

when there exists an action a P A such that the number of times n p a q l it has been chosen during the l -th episode exceeds the total number of times N p a q l it has been chosen since the beginning (Line 14).

## 6.1 Regret Analysis

Before proceeding with the analysis of the regret of the Mixed Spectral UCRL algorithm, we remark that when the estimates of the POMDP parameters are accurate enough, the belief vector p b t computed at each step t using the estimated parameters is close to the real belief b t . To the best of our knowledge, the results in the literature [32, 34, 26, 10, 15] that relate the belief error } p b t ´ b t } 1 with the estimation error of the model parameters all hold under the following one-step reachability assumption.

Assumption 6.1. ( Minimum Value Transition Model ) The smallest value in the transition matrices satisfies ϵ : ' min s,s 1 P S a P A T a p s 1 | s q ą 0 .

Note that Assumption 6.1 implies the Per-Action Ergodicity (Assumption 4.3). The regret for Mixed Spectral UCRL can be expressed as follows. Its proof is deferred to Appendix C.

Theorem 6.2. Under Assumptions 4.1, 4.2 and 6.1, let δ P p 0 , 1 { 2 q . If the Mixed Spectral UCRL algorithm is run for a sufficiently large number of steps T , with probability at least 1 ´ 2 δ , it suffers regret bounded as:

<!-- formula-not-decoded -->

where r ζ p L q -min l Pr 0 ,L ´ 1 s ζ p l q and ζ p l q is defined as in Theorem 5.4. D bounds the span 8 of the bias function appearing in Equation (2) and is defined in Proposition G.1.

This algorithm overcomes the limitations of SM-UCRL since it does not require a constantly exploring policy, and removes the need for a phased algorithm as done for SEEU . By efficiently reusing samples from different policies, we enhance the online learning of POMDPs by improving the current regret guarantee of r O p T 2 { 3 q established by the SEEU algorithm.

## 7 Numerical Simulations

In this section, we analyze the estimation error of the Mixed Spectral Estimation approach under different belief policies and we show the performance in terms of regret of the Mixed Spectral UCRL algorithm when compared against state-of-the-art approaches. Further experiments and simulation details are provided in Appendix J. 9

Mixed Spectral Estimation Algorithm. This first set of experiments studies the estimation error achieved by the Mixed Spectral Estimation algorithm. In particular, we evaluate our method on a POMDP instance with sizes described in Figure 1. The estimation error is measured using the Frobenius norm of the observation matrix and the transition matrices (one per action). Figure 1 reports the average results over 10 runs. The simulation splits the interaction horizon into 10 episodes of equal length, and for each episode, we use a different belief-based policy for data collection. As observed in the figure, the total error decreases as the number of collected samples increases, demonstrating that our approach is able to efficiently combine data from different policies.

Regret Comparison with state-of-the-art Algorithms. In this second set of experiments, we compare our Mixed Spectral UCRL algorithm with SEEU [32] and SM-UCRL [3]. The regret is measured w.r.t. the oracle whose policy satisfies Equation (2) and has full knowledge of the model parameters. As observed in Figure 2, the SM-UCRL algorithm experiences the highest regret since ( i ) it does not reuse samples across episodes, ( ii ) it relies on the weaker class of stochastic ( ι ą 0 ) memoryless policies. This forced exploration leads to constantly selecting suboptimal actions, hence

8 The span of the bias function is defined as: span p v q : ' max b P B v p b q ´ min b P B v p b q .

9 The codebase can be found at https://github.com/alesnow97/Spectral\_Learning\_POMDP.git .

Figure 1: Estimation error of the Mixed Spectral Estimation on a POMDP with S ' 4 , A ' 3 and O ' 4 . (10 runs, 95 %c.i.).

<!-- image -->

Figure 2: Regret comparison on a POMDP with S ' 3 , A ' 3 , O ' 4 (10 runs, 95 %c.i.).

<!-- image -->

resulting in higher regret. We also observe that the Mixed Spectral UCRL algorithm outperforms the SEEU algorithm. This result is in line with the theoretical guarantees, as the regret of SEEU scales with r O p T 2 { 3 q . Besides the alternating exploration-exploitation phases, the inferior performance of SEEU can also be attributed to its reduced sample efficiency since its estimates only rely on data collected during the exploration phase, hence discarding those collected during the exploitation phase. Finally, in Appendix J, we present a regret experiment where Assumption 6.1 is violated in order to show the robustness of our approach with respect to the failure of this assumption.

## 8 Conclusions and Future Directions

In this work, we tackled the problem of learning using spectral methods in the infinite-horizon average-reward POMDP setting. We showed that spectral techniques can be extended to belief-based policies and, through our Mixed Spectral Estimation approach, we answered positively to the open question of whether it is possible to combine samples coming from different adaptive policies. We provided finite-sample guarantees for the devised estimation algorithm, and we showed that the error of the different parameters conveniently scales with respect to the number of employed samples. We combined the new estimation algorithm with an optimistic approach, Mixed Spectral UCRL , and provided the first algorithm achieving a r O p ? T q regret order when compared against the optimal belief-based policy, by leveraging the new sample reuse strategy, and a suitable episode stopping condition. Finally, we validated both our approaches through numerical simulations, and we showed that our approach has improved performance over state-of-the-art algorithms. As a future step, we will study whether it is possible to relax some of the assumptions employed in this work, such as the one-step reachability (i.e., Assumption 6.1).

## Acknowledgements

This paper is supported by FAIR (Future Artificial Intelligence Research) project, funded by the NextGenerationEU program within the PNRR-PE-AI scheme (M4C2, Investment 1.3, Line on Artificial Intelligence).

## References

- [1] Animashree Anandkumar, Rong Ge, Daniel Hsu, Sham M. Kakade, and Matus Telgarsky. Tensor decompositions for learning latent variable models. J. Mach. Learn. Res. , 15(1):2773-2832, jan 2014.
- [2] Animashree Anandkumar, Daniel Hsu, and Sham M. Kakade. A method of moments for mixture models and hidden markov models. In Shie Mannor, Nathan Srebro, and Robert C. Williamson, editors, Proceedings of the 25th Annual Conference on Learning Theory , volume 23 of Proceedings of Machine Learning Research , pages 33.1-33.34, Edinburgh, Scotland, 25-27 Jun 2012. PMLR.
- [3] Kamyar Azizzadenesheli, Alessandro Lazaric, and Animashree Anandkumar. Reinforcement learning of pomdps using spectral methods. In Vitaly Feldman, Alexander Rakhlin, and Ohad

Shamir, editors, 29th Annual Conference on Learning Theory , volume 49 of Proceedings of Machine Learning Research , pages 193-256, Columbia University, New York, New York, USA, 23-26 Jun 2016. PMLR.

- [4] Kazuoki Azuma. Weighted sums of certain dependent random variables. Tohoku Mathematical Journal , 1967.
- [5] Dimitri P. Bertsekas. Dynamic Programming and Optimal Control, 3rd Edition . Athena Scientific, Belmont, MA, 2005.
- [6] Ramaprasad Bhar and Shigeyuki Hamori. Hidden Markov Models: Applications to Financial Economics , volume 40 of Advanced Studies in Theoretical and Applied Econometrics . Springer Science+Business Media, New York, NY, 2004.
- [7] Joseph L. Bower and Clark G. Gilbert, editors. From Resource Allocation to Strategy . Oxford University Press, Oxford, UK, 2005.
- [8] Noe Casas. Deep deterministic policy gradient for urban traffic light control. CoRR , abs/1703.09035, 2017.
- [9] Fan Chen, Huan Wang, Caiming Xiong, Song Mei, and Yu Bai. Lower bounds for learning in revealing POMDPs. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 5104-5161. PMLR, 23-29 Jul 2023.
- [10] Yohann De Castro, Elisabeth Gassiat, and Sylvain Le Corff. Consistent estimation of the filtering and marginal smoothing distributions in nonparametric hidden markov models. IEEE Transactions on Information Theory , 63(8):4758-4777, 2017.
- [11] Zhaohan Daniel Guo, Shayan Doroudi, and Emma Brunskill. A pac rl algorithm for episodic pomdps. In Arthur Gretton and Christian C. Robert, editors, Proceedings of the 19th International Conference on Artificial Intelligence and Statistics , volume 51 of Proceedings of Machine Learning Research , pages 510-518, Cadiz, Spain, 09-11 May 2016. PMLR.
- [12] Daniel Hsu, Sham M. Kakade, and Tong Zhang. A spectral algorithm for learning hidden markov models. Journal of Computer and System Sciences , 78(5):1460-1480, 2012. JCSS Special Issue: Cloud Computing 2011.
- [13] Mehdi Jafarnia Jahromi, Rahul Jain, and Ashutosh Nayyar. Online learning for unknown partially observable mdps. In Gustau Camps-Valls, Francisco J. R. Ruiz, and Isabel Valera, editors, Proceedings of The 25th International Conference on Artificial Intelligence and Statistics , volume 151 of Proceedings of Machine Learning Research , pages 1712-1732. PMLR, 28-30 Mar 2022.
- [14] Thomas Jaksch, Ronald Ortner, and Peter Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research , 11(51):1563-1600, 2010.
- [15] Bowen Jiang, Bo Jiang, Jian Li, Tao Lin, Xinbing Wang, and Chenghu Zhou. Online restless bandits with unobserved states. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 15041-15066. PMLR, 23-29 Jul 2023.
- [16] Chi Jin, Sham M. Kakade, Akshay Krishnamurthy, and Qinghua Liu. Sample-efficient reinforcement learning of undercomplete pomdps. In Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual , 2020.
- [17] Vikram Krishnamurthy. Partially Observed Markov Decision Processes: From Filtering to Controlled Sensing . Cambridge University Press, 2016.

- [18] Jesse Levinson, Jake Askeland, Jan Becker, Jennifer Dolson, David Held, Soeren Kammel, J. Zico Kolter, Dirk Langer, Oliver Pink, Vaughan Pratt, Michael Sokolsky, Ganymed Stanek, David Stavens, Alex Teichman, Moritz Werling, and Sebastian Thrun. Towards fully autonomous driving: Systems and algorithms. In 2011 IEEE Intelligent Vehicles Symposium (IV) , 2011.
- [19] Qinghua Liu, Alan Chung, Csaba Szepesvari, and Chi Jin. When is partially observable reinforcement learning not scary? In Po-Ling Loh and Maxim Raginsky, editors, Proceedings of Thirty Fifth Conference on Learning Theory , volume 178 of Proceedings of Machine Learning Research , pages 5175-5220. PMLR, 02-05 Jul 2022.
- [20] Qinghua Liu, Praneeth Netrapalli, Csaba Szepesvári, and Chi Jin. Optimistic MLE - A generic model-based algorithm for partially observable sequential decision making. CoRR , abs/2209.14997, 2022.
- [21] Sridhar Mahadevan. Average reward reinforcement learning: Foundations, algorithms, and empirical results. Mach. Learn. , 22(1-3):159-195, 1996.
- [22] Lingsheng Meng and Bing Zheng. The optimal perturbation bounds of the moore-penrose inverse under the frobenius norm. Linear Algebra and its Applications , 432(4):956-963, 2010.
- [23] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Proceedings of The 33rd International Conference on Machine Learning , Proceedings of Machine Learning Research, pages 1928-1937. PMLR, 2016.
- [24] Elchanan Mossel and Sébastien Roch. Learning nonsingular phylogenies and hidden markov models. In Harold N. Gabow and Ronald Fagin, editors, Proceedings of the 37th Annual ACM Symposium on Theory of Computing, Baltimore, MD, USA, May 22-24, 2005 , pages 366-375. ACM, 2005.
- [25] Alessio Russo, Alberto Maria Metelli, and Marcello Restelli. Efficient learning of pomdps with known observation model in average-reward setting. CoRR , abs/2410.01331, 2024.
- [26] Alessio Russo, Alberto Maria Metelli, and Marcello Restelli. Achieving r O p ? T q regret in average-reward pomdps with known observation models. In Yingzhen Li, Stephan Mandt, Shipra Agrawal, and Emtiyaz Khan, editors, Proceedings of The 28th International Conference on Artificial Intelligence and Statistics , volume 258 of Proceedings of Machine Learning Research , pages 4168-4176. PMLR, 03-05 May 2025.
- [27] Naci Saldi, Serdar Yüksel, and Tamás Linder. On the Asymptotic Optimality of Finite Approximations to Markov Decision Processes with Borel Spaces. Mathematics of Operations Research , 42(4):945-978, November 2017.
- [28] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. CoRR , abs/1707.06347, 2017.
- [29] Hiteshi Sharma, Mehdi Jafarnia-Jahromi, and Rahul Jain. Approximate relative value learning for average-reward continuous state mdps. In Ryan P. Adams and Vibhav Gogate, editors, Proceedings of The 35th Uncertainty in Artificial Intelligence Conference , volume 115 of Proceedings of Machine Learning Research , pages 956-964. PMLR, 22-25 Jul 2020.
- [30] Le Song, Animashree Anandkumar, Bo Dai, and Bo Xie. Nonparametric estimation of multiview latent variable models. In Eric P. Xing and Tony Jebara, editors, Proceedings of the 31st International Conference on Machine Learning , volume 32 of Proceedings of Machine Learning Research , pages 640-648, Bejing, China, 22-24 Jun 2014. PMLR.
- [31] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction . The MIT Press, second edition, 2018.
- [32] Yi Xiong, Ningyuan Chen, Xuefeng Gao, and Xiang Zhou. Sublinear regret for learning pomdps. CoRR , abs/2107.03635, 2021.

- [33] Huizhen Yu and Dimitri P. Bertsekas. Discretized approximations for POMDP with average cost. CoRR , abs/1207.4154, 2012.
- [34] Xiang Zhou, Yi Xiong, Ningyuan Chen, and Xuefeng GAO. Regime switching bandits. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems , volume 34, pages 4542-4554. Curran Associates, Inc., 2021.
- [35] K.J Åström. Optimal control of markov processes with incomplete state information. Journal of Mathematical Analysis and Applications , 10(1):174-205, 1965.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes] .

Justification: The abstract and the introduction reflect the original contribution of the paper. Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] .

Justification: The presented Mixed Spectral UCRL approach holds under the one-step reachability condition (Assumption 6.1 in Section 6) which holds true under quite stochastic environments. This limitation is also highlighted in Table 1. Another limitation is the assumption of using an oracle for the computation of the optimal policy (Section 4), which is a common assumption in this field of research. Various approximation methods are available for computing this policy.

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

Answer: [Yes] .

Justification: All the theoretical claims reported in this work are supported by complete proofs. The proofs of Theorems 5.3, 5.4 and 6.2 are reported in Appendix A, B and C respectively. The auxiliary claims used in the proofs are also reported in the Appendix, together with their associated proofs or references. The employed assumptions are clearly stated and justified.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes] .

Justification: The paper provides the description and pseudocode of both the Mixed Spectral Estimation and Mixed Spectral UCRL algorithms, together with the hyperparameters used for the experiments which are clearly reported in Appendix J. The description of the POMDP instances and the way they are generated are fully described in Section 7 and Appendix J.

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

Answer: [Yes] .

Justification: The released code provides scripts for running both the experiments on the estimation error of the POMDP parameters and also the experiments on the regret, which compare against the different baselines.

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

Answer: [Yes] .

Justification: Information about the experimental settings are reported in Section 7. More details on the way instances are generated, the employed hyperparameters, and the reasons behind their choice are provided in Appendix J. Further details are contained in the submitted code.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes] .

Justification: Both the experiments on the estimation error and the experiments on the regret are repeated multiple times, and the confidence level of the presented results is reported in the plotted figures. Their calculation is reported in the released code.

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

Answer: [Yes] .

Justification: The information on the employed hardware is reported in Appendix J.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes] .

Justification: We checked the guidelines and we are compliant with them.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA] .

Justification: This work is mainly theoretical, and its goal is to advance the field of Machine Learning. We do not see any negative societal impact.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA] .

Justification: The paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA] .

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

Answer: [Yes] .

Justification: Yes, the released code is well documented.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA] .

Justification: The paper does not involve crowdsourcing.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA] .

Justification: The paper does not involve research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA] .

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Appendix Organization

We provide here an outline of the Appendix.

- Section A, B and C present the proofs of the three theorems reported in the main paper.
- Section D provides some auxiliary results employed for the proof of Theorem 5.4. They are mostly related to the guarantees derived from the application of Tensor Decomposition methods.
- Section E gives an overview of the Symmetrization and Whitening steps, which are implemented on the third-order tensor before applying Tensor Decomposition techniques. It also introduces useful quantities that are used throughout the appendix.
- Section F provides a new bound relating the sum of successive belief errors with the error in the estimated model parameters.
- Section G presents a miscellanea of useful results.
- Section H compares our work from a theoretical perspective with the related works of [3] and [32], and compares spectral approaches with Maximum-likelihood estimation techniques.
- Section I discusses the computational complexity of the Mixed Spectral Estimation method.
- Finally, Section J provides experimental performances of POMDP instances of different characteristics, together with details about the numerical simulations presented in the main paper.

## A Proof of Theorem 5.3

In this section, we provide the proof of Theorem 5.3. For clarity, we report its statement here.

Theorem 5.3. Let Γ -t τ l u L ´ 1 l ' 0 be a set of trajectories collected using the set of policies t π l u L ´ 1 l ' 0 . We define a modified version of the first and second views as:

<!-- formula-not-decoded -->

where the covariance matrices are defined in Equation (7) . Let ω p a,L q -p 1 { N p a q L q ř L ´ 1 l ' 0 n p a q l ω p a,l q , then, the second and third moments of the modified views have a spectral decomposition as:

<!-- formula-not-decoded -->

where the expectations are w.r.t. the conditional state distributions ω p a,l q s .

Proof. Before proceeding, it is relevant to highlight the relation between the view matrices. We use V p a,l q 1 , V p a,l q 2 and V p a,l q 3 to define the views associated with policy π l and action a . We further recall that under the α -weakly revealing assumption ( 4.1) and the invertibility assumption of the transition matrices ( 4.2), the view matrices are always full-column rank [3]. We define the following quantity:

<!-- formula-not-decoded -->

with p π l p¨| a q P ∆ p A q being a probability distribution induced by policy π l and conditioned on action a . As observed in [25], this distribution always exists under the employed assumptions. In particular, p π l p a 1 | a q denotes the probability of having chosen action a 1 in a previous time step (say t ´ 1 ) conditioned on the fact that action a is taken in the successive time step (say t ). Intuitively, T a,π l represents the mixture transition matrix defining the state transition from a previous step ( t ´ 1 ) to a successive one t when action a 1 is chosen in t ´ 1 by policy π l and the next action chosen by the policy in step t is a . Let us also recall that ω p a,l q represents the state distribution induced by policy π l and conditioned on action a such that ω p a,l q s is the probability of being in state s when choosing

action a . Using the definition in (12), we can also define the state distribution at the previous time step ( t ´ 1 ) as:

<!-- formula-not-decoded -->

with ξ p a,l q P ∆ p S q and such that ξ p a,l q s represents the probability that state s is visited in the previous time step ( t ´ 1 ) conditioned on having chosen action a in t .

Having defined the previous state distribution ξ p a,l q in Eq. (13) and inspired by the multi-view model on Markov Chains of [1], we can now express the views using the following relations:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From the relations stated above, we observe that the second view V p a,l q 2 corresponds to the observation model, thus it depends neither on the action nor on the employed policy. Hence, we may refer to it simply as V 2 . The third view depends on the action a but not on the employed policy, so we may refer to it also using V p a q 3 . Finally, the first view depends on both the action a and on quantities related to the employed policy π l .

Let us now recall the definition of the covariance matrix associated with a single policy π l , as reported in Proposition 5.2. In particular, we will use the notation K p a,l q ν,ν 1 to highlight that the covariance matrix depends on policy π l P P , thus distinguishing it from the mixture covariance (in bold) K p a,L q ν,ν 1 resulting from the combination of L different policies.

I) Analysis of Covariance Matrix K p a,L q 3 , 2 . We start by considering the covariance matrix K p a,l q 3 , 2 P R O ˆ O obtained from a single policy π l :

<!-- formula-not-decoded -->

where ω p a,l q P ∆ p S q is the state distribution conditioned on action a and diag ` ω p a,l q ˘ P R S ˆ S represents a diagonal matrix whose diagonal values correspond to w p a,l q . Let us now recall the definition of the mixed covariance matrix in Equation (7). The following holds:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in line 19 we used V p a,l q 3 ' V p a q 3 for any l , and V p a,l q 2 ' V 2 for any a and l , hence highlighting the independence of both view matrices from the used policy π l . In line 21 we introduced the new state distribution ω p a,L q P ∆ p S q such that ω p a,L q -p 1 { N p a q L q ř L ´ 1 l ' 0 n p a q l ω p a,l q .

II) Analysis of Covariance Matrix K p a,L q 3 , 1 . Let us now consider a similar relation for the covariance matrix K p a,L q 3 , 1 P R O ˆ O combining L different policies. We have that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in the last line V p a,L q 1 -O ˆ 1 N p a q L ř L ´ 1 l ' 0 n p a q l T J a,π l diag ` ξ p a,l q ˘ ˙ J diag ` ω p a,L q ˘ ´ 1 the mixed first view matrix.

III) Analysis of Covariance Matrix K p a,L q 2 , 1 . By applying similar steps to those employed for covariance matrix K p a,L q 3 , 1 , we are able to show that:

<!-- formula-not-decoded -->

We are now ready to provide the proofs for the second and third moments. For simplicity, we will just provide the proof for the second moment matrix M p a,L q 2 since the proof for the third moment tensor M p a,L q 3 follows analogous steps.

Proof for the second Moment matrix M p a,L q 2 . The relation for the mixed second moment matrix is defined as follows.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where line 29 holds since r v p a,l q 1 b r v p a,l q 2 ' r v p a,l q 1 ´ r v p a,l q 2 ¯ J , while line 34 holds for the relations of covariance matrices found in the above points.

The simplification steps made from line 34 to line 35 are done considering that the multiplication of a matrix and its pseudoinverse while projecting along the smaller space of size S produces I S , an identity matrix of rank S . In particular, by applying the definition of the Moore-Penrose inverse of a matrix, we have that V : 2 ' p V J 2 V 2 q ´ 1 V J 2 . Since the pseudo-inverse of a transpose corresponds to the transpose of the pseudo-inverse, we get that ` V J 2 ˘ : ' V 2 p V J 2 V 2 q ´ 1 . Hence, the expression in line 34 can be simplified as:

<!-- formula-not-decoded -->

Similar steps also lead to ´ V p a,L q 1 ¯ : V p a,L q 1 ' I S .

Finally, the last equivalence in line 36 concludes the proof.

## B Proof of Theorem 5.4

Theorem 5.4. Let p O and t p T a u a P A be the observation and transition model estimated using Algorithm 1, respectively. Let Assumptions 4.1 and 4.2 hold and let Assumption 4.3 be true for any π l with l P r 0 , L ´ 1 s . Let δ P p 0 , 1 {p 3 SA qq , then for a sufficiently large number of samples N p a q L holding for every action a P A , with probability at least 1 ´ 3 SAδ , it holds that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We recall that Spectral Decomposition techniques are separately applied for each action a P A and each of them outputs estimates of the third view V p a q 3 . From the columns µ 3 ,s of the third view, estimates of the columns µ 2 ,s of the second view matrix can be computed by inverting Equation (6). We remark that the second view is equal for all actions a and it corresponds to the observation matrix O . Since we require that the number of samples N p a q L satisfies conditions in Equation (95) and (102), Lemma D.1 can be used to bound the error of the columns µ 2 ,s of the second view matrix, thus having:

<!-- formula-not-decoded -->

holding with probability at least 1 ´ 3 δ , and with ϵ p a,L q M defined in Lemma D.1.

Condition for Column Permutation The next step of algorithm 1 consists in permuting the view matrices p V p a,L q 2 10 for each action a in order to minimize the 1-norm error with respect to view matrix p V p a ˚ ,L q 2 where a ˚ P arg max a P A N p a q L 11 . The permutation found for each estimated matrix p V p a,L q 2 is

Guarantees on the permutation are achieved when each column µ p a,L q 2 ,s is estimated sufficiently well. Let us denote with d O -min s,s 1 P S ,s ‰ s 1 } O p¨| s q ´ O p¨| s 1 q} 1 the minimum distance between columns of then applied as well to the associated third view p V p a,L q 3 .

O . As observed in [3], when the estimation error is lower than d O { 4 , the columns can be permuted without error. Hence, we derive here the minimum sample condition such that the estimation error of each column (reported in D.1) is bounded by d O { 4 :

<!-- formula-not-decoded -->

By combining the condition above with those required for the bound of Lemma D.1, we obtain:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

10 Differently from the notation used in the pseudocode of the Algorithm, here we add the superscript L to the second view, thus specifying that the estimate depends on L policies.

11 This choice is motivated by the fact that, without knowledge of the parameters characterizing the different ϵ p a,L q M , we assume that the view presenting the lowest error is the one associated with the action that has been chosen the highest number of times.

Bound on the Observation Model Error After the permutation operation, we can finally combine the obtained view matrices p V p a,L q 2 as shown in Equation (9) to obtain a unique matrix p V p L q 2 such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with N L ' ř a P A N p a q L . Let us denote with p µ p L q 2 ,s the s -th column of view matrix p V p L q 2 . From the bound defined in Equation (37) and using a union bound argument, we finally get with probability at least 1 ´ 3 Aδ that:

where:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We notice that a further A term appears in (39) as a result of the union bound, and we stress that the minimization over the singular values is done considering both ν P t 1 , 2 , 3 u and a . Since the result in (39) is independent of the single column s , we can easily extend it to the whole observation matrix and finally get:

<!-- formula-not-decoded -->

holding with probability at least 1 ´ 3 SAδ . By simplifying the notation and highlighting the most relevant terms in the bound, we get:

<!-- formula-not-decoded -->

with C O being a suitable constant and ζ p L q being defined as:

<!-- formula-not-decoded -->

Bound on the Transition Model Error By following Algorithm 1, the s -th row of each estimated transition matrix p T a is computed as p T a p s, ¨q ' p O : p µ p a,L q 3 ,s . Let us analyze its associated error. We have:

<!-- formula-not-decoded -->

Let us now analyze the different terms separately. Concerning the term (a), we use i) › › › µ p a,L q 3 ,s › › › 2 ď 1 and ii) we first apply Proposition D.6 on the spectral norm of the pseudo-inverse of matrix p O and then we use Proposition D.7 to bound › › › O : ´ p O : › › › 2 and obtain:

<!-- formula-not-decoded -->

Analogously, for the second term (b), we apply i) Proposition D.6 to bound } p O : } 2 and ii) we use Lemma D.2 to bound the error of the estimated view vector, thus obtaining:

<!-- formula-not-decoded -->

Since Proposition D.6 holds under the condition } O ´ p O } 2 ď p 1 { 2 q σ S p O q , we require a minimum number of samples N L based on the bound in 42. It should satisfy:

<!-- formula-not-decoded -->

The conditions defined in 38 together with the one just stated above on the total number of samples N L determine the sufficient conditions for the theorem to hold.

Going back to the bound on the estimated transition matrix, by combining the results reported so far, we get with probability at least 1 ´ 3 SAδ :

<!-- formula-not-decoded -->

where C 1 T is a suitable constant term, while we used here a new quantity r ϵ p a,L q M for which it holds both r ϵ p L q M ď r ϵ p a,L q M and ϵ p a,L q M ď r ϵ p a,L q M since it is defined as:

<!-- formula-not-decoded -->

scaling with rate 1 { N p a q L differently from the rate 1 { N L of r ϵ p L q M defined in Equation (40). Since this bound holds for any row of the transition matrix, we can derive the error on the whole transition matrix as:

holding with probability at least 1 ´ 3 SAδ and presenting an additional ? S term. By simplifying notation and highlighting the most relevant terms in the bound, we get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where C T is a suitable constant and ζ p L q is defined as in Eq. (43). This last step concludes the proof.

## C Proof of Theorem 6.2

This section will present the proof for Theorem 6.2, showing the regret guarantees of the Mixed Spectral UCRL algorithm. This result makes use of Theorem 5.4 related to the estimation guarantees of the Mixed Spectral Estimation approach presented in Algorithm 1, and it makes use of the new bound on the belief error provided in Lemma F.1. Some steps of this analysis are inspired by the work of [34].

## Notation

Before proceeding, we need to define some useful quantities that will be employed throughout the proof.

Let us define vector ϕ P R S of expected rewards. Its elements are such that:

<!-- formula-not-decoded -->

From the quantity defined above, we have that the expected reward given a belief b t at time t is:

<!-- formula-not-decoded -->

The real transition and observation model of the POMDP instance Q are defined respectively as T ' t T a u a P A and O .

We will use instead p T l ' t p T a,l u a P A and p O l to denote the transition model and observation models estimated by the Mixed Spectral Estimation procedure at the beginning of episode l , while we will use T l ' t T a,l u a P A and O l to denote the optimistic transition and observation model returned as output by the oracle and actually used during episode l . In a similar way, we will denote the estimated and optimistic POMDP instances at episode l with p Q l and Q l respectively. We use ρ l to denote the optimal average reward for the optimistic POMDP Q l .

We introduce the deterministic function H p b t , a t , o t ` 1 q which returns the belief at the next step b t ` 1 given the action a t and the next observation o t ` 1 according to the Bayes' rule defined in (1). We define a similar function H l p b t , a t , o t ` 1 q which transforms the belief using the optimistic observation model O l and transition model T a t ,l used during the l -th episode.

The probability distribution over the next observation o t ` 1 given belief b t and action a t is defined by:

<!-- formula-not-decoded -->

where e o is the standard basis vector in t 0 , 1 u O corresponding to observation o P O . The probabilities here are computed according to the transition model T a t related to the chosen action and the observation model O of POMDP Q . With P l p o t ` 1 | b t , a t q we denote the analogous probability computed using the observation and transition models of the optimistic POMDP Q l .

The same probability distribution holds over the next belief given the current belief, and it is defined as:

<!-- formula-not-decoded -->

We will use U l to denote a similar measure defined with respect to the observation and transition models of the optimistic POMDP Q l .

We will use E l to characterize the time intervals belonging to the l -th episode, from which we exclude the first and the last interval (this is done since the first and last samples of an interval are not used for SD). Hence, we will have that the number of samples from the l -th episode that will be used for SD is n l ' | E l | .

Having defined the employed notation, we report here the statement of the theorem.

Theorem 6.2. Under Assumptions 4.1, 4.2 and 6.1, let δ P p 0 , 1 { 2 q . If the Mixed Spectral UCRL algorithm is run for a sufficiently large number of steps T , with probability at least 1 ´ 2 δ , it suffers regret bounded as:

<!-- formula-not-decoded -->

where r ζ p L q -min l Pr 0 ,L ´ 1 s ζ p l q and ζ p l q is defined as in Theorem 5.4. D bounds the span 12 of the bias function appearing in Equation (2) and is defined in Proposition G.1.

Proof. Let us recall here the definition of the regret as reported in (3):

<!-- formula-not-decoded -->

12 The span of the bias function is defined as: span p v q : ' max b P B v p b q ´ min b P B v p b q .

where we consider an expectation E taken w.r.t. the true transition model T ' t T a u a P A and the true observation model O ' t O a u a P A . The quantity F t ´ 1 denotes the filtration defined with respect to the events that occurred up to time t ´ 1 . The second term in the summation defines a martingale. Indeed, by denoting the stochastic process as:

<!-- formula-not-decoded -->

we observe that X t defines a martingale. By applying now the Azuma-Hoeffding inequality [4], with probability at least 1 ´ δ { 4 we have:

<!-- formula-not-decoded -->

We can further observe that since the belief b t is conditioned on the filtration F t ´ 1 , we have:

<!-- formula-not-decoded -->

where vector ϕ is defined in Equation (47), while function g is defined in Equation (48). We recall that the belief b t is computed using the true model parameters. Using analogous notation, we will denote the expected instantaneous reward assuming to have updated the belief using the optimistic transition model T a,l and observation model O l as:

<!-- formula-not-decoded -->

From the quantities defined above, we can rewrite the first term of Equation (49) as:

<!-- formula-not-decoded -->

where we recall that the belief is updated using the actions taken by the played policy. By following the procedure described in the Mixed Spectral UCRL algorithm, at the beginning of each episode l , an optimistic POMDP Q l is chosen from the set of possible POMDPs determined by the confidence region C l p δ l q . We recall that the optimistic POMDP Q l is defined by the optimistic transition model T l ' t T a,l u a P A and the optimistic observation model O l provided by the oracle. Since the bound for the estimated transition and observation models provided in Theorem 5.4 holds jointly with probability at least 1 ´ 3 SAδ , we can also observe that P p Q P C l p δ l qq ě 1 ´ 3 SAδ l . Let us now consider two possible events: the good event which considers the case where for all episodes l , the true POMDP is contained in the confidence sets C l p δ l q and the failure event which denotes the complementary event.

By setting the confidence level used for the l -th episode as δ l -δ 3 SAl 3 , the probability of the failure event can now be bounded as:

<!-- formula-not-decoded -->

From the result above, we can observe that the good event holds with probability at least 1 ´ 3 2 δ . When this is the case, we have that ρ ˚ ď ρ l for any l since the optimal average reward is taken from the optimistic POMDP Q l .

We can now bound the regret under the good event during the different L episodes as:

<!-- formula-not-decoded -->

where we have rewritten the summation by highlighting the different L episodes. In particular, for each episode l we use interval E l that excludes the first and the last timestamp of that episode, while the term 2 L appearing in the first inequality is obtained by assuming to pay maximum regret for each pair of samples not contained in each E l .

In the second inequality instead, we explicit the length T 0 of the first episode for which we assume to pay maximum regret: the ´ 2 term is due to the fact that the first and the last timestamps of the first episode are already counted in the 2 L term. Finally, the last equality expresses the length of the first episode as the sum of the counts of the chosen actions, and adds and subtracts the quantity g l p b l t q -r J O l b l t .

For what will follow, we will focus on the term Ψ .

## Analysis of ( Ψ )

Let us restate the term Ψ defined above.

<!-- formula-not-decoded -->

We will now focus on analyzing the first and the second term separately.

Analysis of the First Term of Ψ (line 54) Let us use the Bellman equation reported in Equation (2) for the optimistic belief MDP, and the definition of the probability distribution U over the next belief defined in the Notation section. The following relations hold:

<!-- formula-not-decoded -->

The equation above allows us to write that:

<!-- formula-not-decoded -->

where the first equality is obtained from the Bellman Equation, while the last equality derives from adding and subtracting the term x U p¨| b l t , a t q , v l p¨qy for each time step t . We recall that U p¨| b l t , a t q defines the probability distribution over the belief at the next step t ` 1 under the true POMDP instance Q , while U l p¨| b l t , a t q represents this probability distribution under the optimistic instance Q l . For the term p a q in 55, we have:

<!-- formula-not-decoded -->

where the term p a. 1 q is obtained by observing that the sum on the first line reduces to a telescopic summation. For each episode l , the terms appearing in this summation are respectively the difference between the value of the bias function of the belief in the first timestamp (denoted s l ) and the last plus one (denoted e l ` 1 ) timestamp appearing in E l .

The term p a. 2 q is instead obtained by observing that:

<!-- formula-not-decoded -->

By using Proposition G.1, we can easily see that the span of the bias function defined as span p v l q : ' max b P B v l p b q ´ min b P B v l p b q can be bounded by D { 2 with D being a finite quantity. Hence, we can write that:

<!-- formula-not-decoded -->

For the term p a. 2 q , we can observe that it defines a martingale. By applying analogous results as those used for bounding 50, we get with probability at least 1 ´ δ { 4 that:

<!-- formula-not-decoded -->

By combining the bounds for p a. 1 q and p a. 2 q , we obtain with probability at least 1 ´ δ { 4 :

d

<!-- formula-not-decoded -->

We can now proceed in bounding the term p b q appearing in 55. Let us recall the definition of the function H p b t , a t , o t ` 1 q and P p o t ` 1 | b t , a t q defined in the Notation section. The following relations hold:

<!-- formula-not-decoded -->

where in the first equality we have decoupled the stochasticity induced by the observation from the deterministic update of the belief b 1 at the next step through the H and H l functions. Let us now analyze the different terms separately.

<!-- formula-not-decoded -->

The last inequality is instead obtained from Corollary F.2 which bounds the one-step error of the belief vector when updated using the estimated observation and transition matrices. Constants C 2 and C 3 are instead defined in Lemma F.1. Concerning the term p b. 2 q , we have:

<!-- formula-not-decoded -->

By combining the results obtained for p b. 1 q and p b. 2 q , we are able to bound the term p b q as:

<!-- formula-not-decoded -->

Finally, we can combine the results defined in lines 59 and 62 on p a q and p b q to finally bound the first term of Ψ (line 54) and obtain with probability at least 1 ´ δ { 4 :

<!-- formula-not-decoded -->

where we used that n l ' | E l | denotes the cardinality of the interval E l , and we also recall that ř a P A n p a q l ' n l .

## Analysis of the Second Term of Ψ (line 54)

We can now focus on the second term appearing in the summation of 54. We have that:

<!-- formula-not-decoded -->

Let us now consider the last term appearing in the last inequality. It can be bounded by using the result appearing in Lemma F.1. In particular, we have:

<!-- formula-not-decoded -->

with constants C 1 , C 2 and C 3 defined in Lemma F.1. From the results above, we obtain the following result for the second term of Ψ

:

<!-- formula-not-decoded -->

## Merge of Obtained Results and Final Bound

Let us recall the definition of the regret in line 49 and let us observe that it can be bounded using the bound on the martingale in line 50 and the bound on line 53. We have just seen how line 64 and 65 allow us to bound the term Ψ in 53. By combining everything, we get:

<!-- formula-not-decoded -->

Let us now focus on the quantities appearing in p c q and p d q . We have:

<!-- formula-not-decoded -->

where for the first term of the inequality on the first line we used ř a P A n p a q 0 ' n 0 and Theorem 5.4. Here, we recall that N l represents the number of samples used for the model estimation for the l -th

episode. In line 67 we defined r ζ p L q -min l ζ p l q , while the last line simply follows by observing that N L ď T .

We can apply similar considerations to bound the term p e q . In particular:

<!-- formula-not-decoded -->

where the last but one inequality follows by recalling that N L ' ř a P A N p a q L . From the result obtained in 68 and 70, we rewrite the bound on the regret reported in line 66 as:

<!-- formula-not-decoded -->

holding with probability at least 1 ´ 2 δ , obtained by using a union bound on the bound of the two martingales (each one holding with probability at least 1 ´ δ { 4 ) and on the bound of the optimistic model which holds with probability at least 1 ´p 3 { 2 q δ , as reported in Eq. (52).

The last step of the proof consists in observing that, for the stopping condition employed by the algorithm, the number of total episodes can be bounded as L ď A log p T { A q . Finally, the regret expression can be simplified by highlighting the dependencies on the main terms as follows:

<!-- formula-not-decoded -->

This final step concludes the proof.

## D Auxiliary Results for the Proof of Theorem 5.4

In this section, we will provide auxiliary results required for the proof of Theorem 5.4. They are based on previous results on learning Hidden Markov Models (HMM) and POMDPs by [1] and [3]. We carefully adapt the results to the Mixed Spectral Estimation strategy presented in Algorithm 1.

Lemma D.1 ( Error Bound of µ p a,L q 2 ,s ) . Let p V p a,L q 2 be the second view estimated using Algorithm 1 when the set of policies t π l u L ´ 1 l ' 0 is used to interact with the environment, and let p µ p a,L q 2 ,s P ∆ p O q be its s -th column. If N p a q L satisfies the conditions in Equation (95) and (102) , then with probability at

least 1 ´ 3 δ , we have:

<!-- formula-not-decoded -->

with ϵ p a,L q M defined as in Equation (98) of Lemma D.4.

Proof. Let us recall that each column µ p a,L q 2 ,s of the second view matrix V p a,L q 2 can be obtained from µ p a,L q 3 by inverting Equation (6). We can thus write the following:

<!-- formula-not-decoded -->

The terms in 72 can be bounded by using i) Lemma D.3 for the concentration bound of empirical estimates for › › › K p a,L q 2 , 1 ´ x K p a,L q 2 , 1 › › › 2 , ii) Proposition D.5 for › › › › ´ K p a,L q 3 , 1 ¯ : ´ ´ x K p a,L q 3 , 1 ¯ : › › › › , iii)

2 Lemma D.2 for › › › µ p a,L q 3 ,s ´ p µ p a,L q 3 ,s › › › 2 , iv) › › › K p a,L q 2 , 1 › › › 2 ď 1 , v) › › › › ´ K p a,L q 3 , 1 ¯ : › › › › ď 1 { σ S p K p a,L q 3 , 1 q and vi)

› › › µ p a,L q 3 ,s › › › ď 1

2 2 . Thus we have:

<!-- formula-not-decoded -->

holding with probability at least 1 ´ 3 δ . The last inequality follows from observing that each of the first two terms is ď ϵ p a,L q M { σ S p K p a,L q 3 , 1 q .

Lemma D.2 ( Error Bound for µ p a,L q 3 ,s ) . Let p V p a,L q 3 be the third view estimated in Algorithm 1 when the set t π l u L ´ 1 l ' 0 of policies is used to interact with the environment, and let p µ p a,L q 3 ,s P ∆ p O q be its s -th column. If N p a q L satisfies the condition in Equation (95) reported in Lemma D.4 then, with probability at least 1 ´ 2 δ , we have:

<!-- formula-not-decoded -->

with ϵ p a,L q M defined as in Equation (98) of Lemma D.4.

Proof. The theoretical guarantees on the estimation quality of the third view p V p a,L q 3 are related to the guarantees provided by Spectral Decomposition approaches.

In past works such as [1] and [30], it has been shown that among the different spectral algorithms, those relying on tensor decomposition are more sample efficient. Our approach relies on the Robust Tensor Power (RTP) method presented in [1], which is applied to the symmetrized and whitened thirdorder moment tensor. We will now denote the steps required to transform the empirical estimates and

provide them to the RTP algorithm. The definition of some of the quantities that are used throughout this proof, together with the employed notation, is discussed in Section E.

Let us consider now the empirical matrices and tensors (without symmetrization) defined as:

<!-- formula-not-decoded -->

We observe that the definition of the non-symmetrized matrix Ă M p a,L q 2 coincides with the one of K p a,L q 1 , 2 . These non-symmetrized versions 13 indeed differ from the symmetrized one M p a,L q 3 and M p a,L q 3 presented in Theorem 5.3.

Using the multilinear map notation introduced in Section E, we define the symmetrized and whitened tensor as Ă M p a,L q 3 p W p a,L q 1 , W p a,L q 2 , W p a,L q 3 q P R S ˆ S ˆ S , where W p a,L q 1 P R O ˆ S , W p a,L q 2 P R O ˆ S and W p a,L q 3 P R O ˆ S are the corresponding symmetrization-whitening matrices for each of the tensor dimensions. By using Lemma D.4, it is possible to show that for a sufficient number of samples N p a q L , the error ϵ p a,L q M on the estimated symmetrized and whitened tensor x Ă M p a,L q 3 p x W p a,L q 1 , x W p a,L q 2 , x W p a,L q 3 q can be bounded with probability at least 1 ´ δ as:

<!-- formula-not-decoded -->

From Lemma D.4, we can also observe that when a sufficient number of samples N p a q L is used, the estimation properties of the RTP method are guaranteed. In particular, let us denote with ` p r µ p a,L q 3 ,s , p r ω p a,L q s ˘ s P S the set of robust eigenvector/eigenvalue pairs provided as output by RTP. Then, from [1], with probability at least 1 ´ 2 δ the following holds: 14

›

›

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let us now denote with ϵ p a,L q 3 -} µ p a,L q 3 ,s ´ p µ p a,L q 3 ,s } 2 the error of the s -th column of the third view matrix V p a,L q 3 .

We recall that in order to obtain the estimate p µ p a,L q 3 ,s from the corresponding robust eigenvector/eigenvalue pair ` p r µ p a,L q 3 ,s , p r ω p a,L q s ˘ given as output by RTP, we have to de-whiten vector p r µ p a,L q 3 ,s which can done by the following relation:

<!-- formula-not-decoded -->

where we defined p B P R O ˆ S as the Moore-Penrose inverse of ´ x W p a,L q 3 ¯ J . The equation above is obtained by inverting the first Equation appearing in (116), which relates the robust eigenvector/eigenvalue pair of the whitened tensor with that of the non-whitened counterpart.

13 We use symbol r to denote the non-symmetrized quantities Ă M 2 and Ă M 3 in order to distinguish them from the symmetrized ones M 2 and M 3 .

14 To be more precise, the statement refers to a permutation of the found eigenvector/eigenvalue pairs satisfying the condition above. However, to avoid clutter, we consider that the bounds are defined for the correct permutation of these estimates.

Let us now analyze the error ϵ p a,L q 3 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We can bound the error of each term separately. Let us start with (a). For the first term of (a), we have:

<!-- formula-not-decoded -->

where the first equality follows from the fact that r µ p a,L q 3 ,s is a unit vector, while the last equality follows from the definition in Equation (116) linking the original eigenvalue ω p a,L q s with the one of the whitened tensor r ω p a,L q s . For the second term of (a), we have:

<!-- formula-not-decoded -->

where the result directly follows from Equation (112) in Proposition D.8.

Let us now consider the term (b). We have:

<!-- formula-not-decoded -->

where the inequality in line 84 follows from } p B } 2 ď 1 .

By combining the expressions in 81, 82 and 87, with probability at least 1 ´ 2 δ , we get:

<!-- formula-not-decoded -->

where the last inequality is obtained by observing that the first term of the summation is ď ϵ p a,L q M . This last expression completes the proof.

Lemma D.3 ( Concentration Bounds for Covariance Matrices obtained from Multiple Policies ) . Let t π l u L ´ 1 l ' 0 policies interact with a POMDP Q generating trajectories Γ ' t τ l u L ´ 1 l ' 0 . Let Assumption 4.3 hold for each action a P A and for each policy π l P P . Then, for any ν, ν 1 P t 1 , 2 , 3 u and ν ‰ ν 1 , with probability at least 1 ´ δ , the following holds:

<!-- formula-not-decoded -->

For the tensor case, for r ν, ν 1 , ν 2 s being any permutation of the set t 1 , 2 , 3 u , with probability at least 1 ´ δ , it holds:

<!-- formula-not-decoded -->

where r G -max l Pr 0 ,L ´ 1 s G p π l q and r η -min l Pr 0 ,L ´ 1 s η p π l q . Here, 1 ď G p π l q ă 8 is the geometric ergodicity constant of the Markov Chain obtained from policy π l and 0 ď η p π l q ă 1 represents the related contraction coefficient.

Proof. The proof of this lemma follows from standard concentration bounds on HMM when adapted to the observations conditioned on a specific action a . Let us first observe that the covariance matrix obtained from policy π l is exactly defined as:

<!-- formula-not-decoded -->

and we can define an analogous quantity for the tensor case as:

<!-- formula-not-decoded -->

where we recall that n p a q l -| T p a q l | . By applying Theorem 13 in [3], when a single policy π l is used, the error on the quantities defined above can be bounded as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability at least 1 ´ δ . In this version of the proof, differently from what done in [3], we bound the distance by assuming that the expectation defining both K p a,l q ν,ν 1 and K p a,l q ν,ν 1 ,ν 2 is defined with respect to the initial (arbitrary) state distribution, which may be different from the stationary one 15 .

Since we assume to have multiple policies interacting with the environment, our objective is to provide a bound for a mixing covariance matrix and a mixing tensor, respectively denoted as:

»

fi

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

15 Indeed, for Spectral decomposition techniques to be applied, it is not required that the moments are defined with respect to the stationary state distribution.

We will study the error for the mixed covariance matrices. The same steps will hold for the tensor case. We have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in line 93, we use the new terms r G -max l Pr 0 ,L ´ 1 s G p π l q and r η -min l Pr 0 ,L ´ 1 s η p π l q . We finally observe that the bound on the mixture covariance matrix presents a further term ? L in the bound due to the application of the union bound. The final result follows by substituting the definition of the covariance matrix in the statement of the lemma.

## D.1 Minimum Number of Samples Required for Applying Tensor Decomposition

Lemma D.4. Let Ă M p a,L q 2 and Ă M p a,L q 3 be defined as in Equations (73) . Let Assumptions 4.1, 4.2 and 4.3 hold. Then, if the number of samples satisfies:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

then the following relation holds:

<!-- formula-not-decoded -->

Hence, this condition allows applying the RTP approach on the estimated tensor x Ă M p a,L q 3 ´ x W p a,L q 1 , x W p a,L q 2 , x W p a,L q 3 ¯ , as prescribed in Proposition D.8.

Proof. We recall here the result in Proposition D.8 which allows us to provide a bound on the estimation error ϵ p a,L q M of matrix x Ă M p a,L q 3 ´ x W p a,L q 1 , x W p a,L q 2 , x W p a,L q 3 ¯ . We have:

<!-- formula-not-decoded -->

where this last inequality uses concentration results on the empirical estimates of Ă M p a,L q 2 and Ă M p a,L q 3 (Lemma D.3), and holds with probability at least 1 ´ δ .

In order to successfully apply the RTP method on the estimated tensor, the estimation error ϵ p a,L q M should be reasonably small. In particular, the result in Equation (97) holds under the assumption that i) › › › › Ă M p a,L q 2 ´ x Ă M p a,L q 2 › › › › 2 ď 1 2 σ S p Ă M p a,L q 2 q , as prescribed in Proposition D.8. In addition, from [2], it is required that ii) ϵ p a,L q M ď C ? for some constant C . From condition i), we require that:

S

<!-- formula-not-decoded -->

while for condition ii), it surely holds when each of the terms appearing in (98) is upper bounded by C {p 2 ? S q under a suitable constant C , namely:

<!-- formula-not-decoded -->

From the previous bounds, we obtain respectively:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By rearranging the results reported in Equations (99), (100) and (101), we get the final result of the lemma on the minimum number of samples required for the condition 96 to hold.

## D.2 Auxiliary Propositions

Proposition D.5. Let x K p a,L q 3 , 1 be an empirical estimate of K p a,L q 3 , 1 obtained using N p a q L samples. Then if:

<!-- formula-not-decoded -->

then with probability at least 1 ´ δ , the covariance matrix p K p a,L q 3 , 1 is invertible and it holds that:

<!-- formula-not-decoded -->

Proof. Since x K p a,L q 3 , 1 ' 1 N p a q L ř L ´ 1 l ' 0 n p a q l E ' v p a,l q 3 b v p a,l q 1 ı , we can apply lemma D.3 and get

<!-- formula-not-decoded -->

Let us consider the condition:

<!-- formula-not-decoded -->

By denoting with σ S p K p a,L q 3 , 1 q the minimum singular value of matrix K p a,L q 3 , 1 we have › › › › ´ K p a,L q 3 , 1 ¯ ´ 1 › › › › 2 ' 1 { σ S p K p a,L q 3 , 1 q . By using the bound in 103, it is easy to show that this condition( 104) is verified with probability 1 ´ δ when:

<!-- formula-not-decoded -->

Under condition (104), we can state the following:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where line 106 derives from Lemma E.4 in [2], while line 107 is obtained by substituting at the denominator the condition in 104.

Proposition D.6. (From [3]) Let W P R Y ˆ X and x W P R Y ˆ X with Y ě X be any pair of matrices such that x W ' W ` E for a suitable error matrix E and let σ X p x W q be the X -th singular value of matrix x W . If the error matrix is such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then we can derive the following:

Proof. Given that x W is a perturbation of the true matrix W , we can use Weyl inequality to have a bound on the difference of the minimum singular value:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since we have assumed that the perturbation is not too large, we can safely invert this bound to obtain:

<!-- formula-not-decoded -->

where the last inequality follows from the precondition on the perturbation error (109). Hence, we can derive the final result as:

<!-- formula-not-decoded -->

Proposition D.7. (From [22]) Let W and x W be any pair of matrices such that x W ' W ` E for a suitable error matrix E . Then we have:

<!-- formula-not-decoded -->

with } ¨ } 2 denoting the spectral norm.

Proposition D.8 (From [3]) . Let Ă M p a q 2 -E r v p a q 1 b v p a q 2 s and Ă M p a q 3 -E r v p a q 1 b v p a q 2 b v p a q 3 s be the matrices associated with action a P A , with the expectations defined by policy π P P . Let also denote with Ă M p a q 3 p W p a q 1 , W p a q 2 , W p a q 3 q the symmetrized and whitened third-moment tensor, as defined in Section E. If Assumptions 4.1, 4.2 and 4.3 hold, then, under the condition 16

<!-- formula-not-decoded -->

the two following statements hold:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## E Symmetrization and Whitening

This section shows how the symmetrization and the whitening steps can be used for the quantities defined in this work. To reduce clutter, we will avoid using the apices a and L in this section.

## Notation

We will stick here with the notation used in [1]. Let us denote a p -th order tensor as A P Â p i ' 1 R n i . When n 1 ' n 2 ' ¨ ¨ ¨ ' n p ' n , we can simply write A P b p R n . For a vector v P R n let us use

16 The requirements on the minimum number of samples needed to satisfy (110) are reported in Lemma D.4.

which leads to

v b p -v b v b¨¨¨ b v P Â p R n to denote its p -th order tensor.

We can consider A to be a multilinear map when it holds that for a set of matrices t V i P R n ˆ m i : i P r p su , the p i 1 , i 2 , . . . , i p q -th entry in of the tensor A p V 1 , V 2 , . . . , V p q P R m 1 ˆ m 2 ˆ¨¨¨ˆ m p is

<!-- formula-not-decoded -->

So, if A is a matrix p p ' 2 q , then we have:

<!-- formula-not-decoded -->

## Symmetrization

Let us now denote with v 1 P R d 1 , v 2 P R d 2 and v 3 P R d 3 the three view vectors, and let V 1 P R d 1 ˆ k , V 2 P R d 2 ˆ k and V 3 P R d 3 ˆ k be the associated view matrices, with k ď d ν for ν P t 1 , 2 , 3 u 17 . We use µ ν,i to denote the i -th column of the view matrix V ν . Let us consider the second moment Ă M 2 P R d 1 ˆ d 2 and third moment Ă M 3 P R d 1 ˆ d 2 ˆ d 3 of the three views as follows:

<!-- formula-not-decoded -->

Our objective is to represent these views as the second-order tensor and the third-order tensor with respect to view v 3 . In order to achieve this result, we need to modify the views v 1 and v 2 by making use of the covariance matrices as follows:

<!-- formula-not-decoded -->

with R 1 P R d 1 ˆ d 3 and R 2 P R d 2 ˆ d 3 being the rotation matrices of the views v 1 and v 2 respectively. Using notation in Equation (113), it is possible to show that the symmetized version M 2 P R d 3 ˆ d 3 can be defined as:

<!-- formula-not-decoded -->

## Whitening

When the symmetrization step is concluded, the third-order matrix needs to be whitened in order to run the Robust Tensor Power (RTP) method on it. The whitening transformation is defined through the matrix W P R d 3 ˆ k and is such that:

<!-- formula-not-decoded -->

with M 2 being the symmetrized matrix defined above and I P R k ˆ k is the identity matrix. From the relations above, we also have:

<!-- formula-not-decoded -->

which introduces the symmetrization-whitening matrices W 1 P R d 1 ˆ k and W 2 P R d 2 ˆ k . Since the third view does not need to be symmetrized but only whitened, we have W 3 -W P R d 3 ˆ k .

Let us now define:

and we observe that:

<!-- formula-not-decoded -->

17 In our POMDP setting, we have d 1 ' d 2 ' d 3 ' O and k ' S .

<!-- formula-not-decoded -->

from which we also observe that r µ 3 ,i P R k are orthonormal vectors.

We can now define the symmetrized and whitened tensor Ă M 3 p W 1 , W 2 , W 3 q P R k ˆ k ˆ k as:

<!-- formula-not-decoded -->

where the first equality follows from analogous considerations as those in Equation (115).

The decomposition expressed in the last equality allows representing tensor Ă M 3 p W 1 , W 2 , W 3 q in terms of the orthonormal eigenvectors r µ 3 ,i and the related eigenvalues r ω i . In this form, the tensor can be provided as input to the RTP method [1]. The RTP method will then provide as output an estimate of the robust eigenvector/eigenvalue pairs p r µ 3 ,i , r ω i q for each i P r k s .

Finally, the original eigenvector/eigenvalue pairs p µ 3 ,i , ω i q can be recovered by inverting the Equations in (116).

## F Belief Vector Concentration Bound

We present here Lemma F.1 that will be fundamental for proving the regret result of the Mixed Spectral UCRL algorithm.

Lemma F.1. Let Q be a POMDP instance satisfying Assumption 6.1. Let p O and p T ' t p T a u a P A be the estimate of the observation and transition model and let T ' tp o t , a t qu T t ' 0 be a trajectory generated while interacting with the environment. We have that:

<!-- formula-not-decoded -->

where C 1 , C 2 , C 3 are finite constants, while n p a q represents the number of times each action a P A is chosen during the interaction.

Proof. Wedenote with p b t and b t the estimated and real belief vector at time t updated using Equation 1, using respectively the estimated and real transition model. From the belief decomposition reported in [10], we derive that the belief error bound at time t is:

<!-- formula-not-decoded -->

where η -1 ´ ϵ 1 ´ ϵ , while c o is a finite constant based on both the transition and the observation model such that c o -min o P O min a P A min s P S ř s 1 P S T a p s 1 | s q O p o | s 1 q which is always positive thanks to

Assumption 6.1.

We proceed by bounding 118 as:

<!-- formula-not-decoded -->

where the inequality simply follows by observing that } p b 0 ´ b 0 } 2 ď } p b 0 ´ b 0 } 1 ď 2 .

This bound shows that the error in the belief vector depends on the sequence of actions and the contribution in the error of each action scales geometrically with time. Using the relations above, let

us now bound the sum of belief errors over T ` 1 different time steps:

<!-- formula-not-decoded -->

where the constant 2 is obtained by bounding the first term } p b 0 ´ b 0 } 1 ď 2 . Let us now focus on the terms p a q and p b q .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Differently, the term c can be bounded by using the result from [26] (see their Lemma D.1) and we obtain that:

<!-- formula-not-decoded -->

where n p a q represents the number of times action a P A is chosen during the interaction, while the last step follows by using the definition of η .

By combining the results in p a q , p b q and p c q , we get:

<!-- formula-not-decoded -->

where in the last line we simply substituted the definition of η into the bound. The final result of the lemma simply follows by defining the constants

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From the considerations reported above, we can derive the following corollary for the one-step belief error.

Corollary F.2. (One-step Belief Bound) Let Q be a POMDP instance satisfying Assumption 6.1. Let us denote with p O , T a q and p p O , p T a q respectively the real and estimated model parameters related to action a . Starting from a common belief vector b 0 , and choosing action a P A , the one-step error in the estimated belief vector can be bounded as:

<!-- formula-not-decoded -->

where constants C 2 and C 3 are defined in line 120.

Proof. The proof of this corollary easily follows from the bound in 118 by using t ' 1 and having that b 0 ' p b 0 .

## G Miscellanea of Useful Results

This section is devoted to the presentation of some useful results used throughout the work.

The first one is taken from [34] and relates the maximum span of the bias function span p v q with a finite constant D .

Proposition G.1 (Uniform bound on the bias span from [34]) . Let us assume to have a POMDP instance that can be rewritten as a belief MDP . If Assumption 6.1 holds, then for ρ, v satisfying the Bellman Equation (2) , we have the span of the bias function span p v q : ' max b P B v p b q ´ min b P B v p b q is bounded by D p ϵ q , where:

<!-- formula-not-decoded -->

Hence, this proposition ensures that span p v q is bounded by D ' D p ϵ { 2 q for any bias functions v associated with a belief MDP derived from a POMDP instance Q .

This second result is used in the bound of Theorem 6.2.

Lemma G.2 (Lemma 19 in [14]) . For any sequence of numbers y 0 , . . . , y n ´ 1 with 0 ď y k ď Y k and Y k -max t 1 , ř k ´ 1 i ' 0 y i u :

<!-- formula-not-decoded -->

## H Comparison with Related Literature

We provide here a detailed comparison of our Mixed Spectral UCRL with respect to the SEEU and the SM-UCRL algorithms tackling the infinite-horizon average reward setting (Section H.1), while we devote Section H.2 to a discussion on the differences of our formulation with respect to Maximum-Likelihood approaches typically used in episodic settings.

## H.1 Comparison with Algorithms in the Infinite-horizon setting

We provide here a comparison in terms of assumptions and theoretical guarantees of our Mixed Spectral UCRL algorithm with other algorithms in the literature that tackle this setting. Some key aspects are reported in Table 1. In particular:

Comparison with SEEU [32]. Our approach strictly improves over the SEEU algorithm both in terms of assumptions and results. Indeed, unlike SEEU , our algorithm does not require an assumption on the minimum values of the observation model . Additionally, we introduce the sample reuse strategy for adaptive policies, leading to an improved sample efficiency which, together with a more refined theoretical analysis, also translates to an improved regret bound with respect to the interaction horizon, from r O p T 2 { 3 q to r O p ? T q .

Comparison with SM-UCRL [3]. Similarly, we also make improvements over the SM-UCRL algorithm. Indeed, the SM-UCRL algorithm employs stochastic memoryless policies which are known to suffer linear regret when compared against the optimal POMDP policy. The employed policy class includes those policies for which each action can be chosen with a minimum probability ι ą 0 at

every time step. By introducing our sample reuse strategy, we improve sample efficiency, and we are not obliged to continuously choose every action since we can use those observed in the past, hence being able to eliminate stochastic policies and allowing for ι ' 0 .

On the other hand, our approach employs the stronger class of belief-based policies. This comes at the cost of requiring an assumption on the minimum value of the transition model (as also done in SEEU ) in order to bound the error of the estimated belief vector, as explained in Section 6 of the main paper.

Both SEEU and SM-UCRL subsume Assumption 4.3. We show here how both the SEEU and the SM-UCRL algorithms rely on assumptions that imply our Assumption 4.3. In particular:

- the SEEU algorithm directly employs the one-step reachability assumption (our Assumption 6.1) for learnability. Differently, we use the weaker Assumption 4.3 for learning the model parameters, and then require the stronger one-step reachability assumption to ensure guarantees for the Mixed Spectral UCRL algorithm.
- the SM-UCRL algorithm assumes standard ergodicity assumptions (not conditioned on action) but restricts to the class of stochastic policies ( ι ą 0 ). Under this set of stochastic policies and the ergodicity assumption, the state-action distribution d π 8 p s, a q always exists and satisfies d π 8 p s, a q ą 0 for any p s, a q P S ˆ A . Consequently, the conditional state distribution ω p a,π q is always well-defined (since, under the considered policy class, d π 8 p a q ą 0 for any a P A ) and its elements are always strictly positive, hence satisfying Assumption 4.3.

Finally, we remark that the set of Assumptions 4.1, 4.2 and 4.3 employed in our work constitute the minimum working assumptions for learning in the infinite-horizon average-reward POMDP setting .

## H.2 Comparison between Spectral Decomposition and Maximum-likelihood Approaches

Besides Spectral Decomposition techniques, other methods can be used for parameter estimation. Among the most common, we highlight those based on Maximum-Likelihood estimation mainly adopted in the episodic setting, such as the OOM-UCB [19] or the Optimistic-MLE [20] algorithms. We describe below the two key differences between these approaches:

1. MLE-based methods lack Estimation Guarantees for Latent Variable Models, differently from Spectral Methods.

MLE-based methods are not guaranteed to recover the original parameters ( O , T ) when estimating latent variable models, such as HMMs or POMDPs. In contrast, Spectral Decomposition methods provide finite-sample guarantees for such models and represent the most computationally efficient methods for estimating such models. Notably, MLE-based approaches are used to learn an alternative POMDP parametrization known as the Observable Operator Model (OOM) for which finite-sample guarantees can be derived by only employing the α -weakly revealing condition. Crucially, it is important to highlight that knowledge of the Observable Operators does not alone allow recovering the original POMDP parameters p O , T q for which instead different techniques (Spectral Decomposition) and further assumptions (invertibility of the transition matrices and ergodicity-like conditions) are needed to ensure estimation guarantees.

## 2. MLE-based approaches typically addresses the finite-horizon setting, while our focus is on the infinite-horizon one.

The difference between the two settings also lies in the class of optimal policies. Indeed, while the best policy in the finite-horizon case depends on the sequence of observations and actions of limited length (bounded by the episode length H ) and does not rely on a notion of belief state, the optimal policy for the infinite-horizon case depends on maintaining and updating a belief vector over the hidden states. Since belief updates rely on the Bayes' rule, which in turn requires estimates of both the observation and transition models, we need to use estimation methods with finite-sample guarantees (such as Spectral Methods) to recover the model parameters. This is in contrast to the finite-horizon setting, where guarantees on the policy suboptimality can be related to the quality of OOM estimates.

## I Discussion on Computational Complexity

We discuss here the computational complexity of the Mixed Spectral Estimation procedure. The computational complexity of this approach is comparable with the estimation approaches used both by SEEU and SM-UCRL since all of them rely on the underlying tensor decomposition . The overall computational complexity of the method scales as O p A max t O 3 , S 5 log S uq , where:

- The complexity scales linearly with the number of actions since SD is performed separately for each action a P A ,
- The first term in the max arises from inverting the covariance matrices having order O appearing in Equation (6),
- The second term comes from the RTP strategy introduced in [1], which is used as a subroutine by the Mixed Spectral Estimation strategy. This method operates on a symmetric and whitened three-order tensor 18 with dimension R S ˆ S ˆ S . Hence, each operation requires O p S 3 q computations, and, assuming each eigenvector is computed from roughly O p S q initializations, with O p log S q power iterations per initialization, the total time for obtaining the S different eigenvector/eigenvalue pairs is O p S 5 log S q . Some optimization techniques can reduce this complexity to O p S 4 q .

We refer to [1] for a more detailed discussion on this matter.

## J Additional Simulations and Simulation Details

This section provides details about the numerical simulations reported in the main paper. The simulations illustrated in this work have been run on an 88 Intel(R) Xeon(R) CPU E7-8880 v4 @ 2.20GHz CPUs with 94 GB of RAM.

The code can be found at https://github.com/alesnow97/Spectral\_Learning\_POMDP.git .

Transition and Observation Model Generation. For the generation of the different POMDPs, we adopted a similar approach to the one followed in [25]. The matrices of both the observation and transition models are randomly generated, and successive modifications are applied:

- Transition model T a : we set a minimum value for each cell of the matrix that should be at least ϵ ' 1 {p 10 S q .
- Observation model O : for each state, we define a subset of observations that may be observed with higher probability with respect to the others. This caveat improves the informativeness of the observation model, hence avoiding matrices with zero (or close to zero) minimum singular values.

## J.1 Simulations on Estimation Error of the Mixed Spectral Estimation Algorithm

In this section, we report further experiments on estimation errors of POMDP instances of different sizes. In particular, we analyze the behavior of our estimation approach with both smaller and larger instances with respect to the one presented in the main paper. The results are presented in Figure 3 and are expressed in terms of the Frobenius norm.

For the experiment on the left, we measured the estimation error over 10 different episodes, each one having size 10 5 steps. Since the considered POMDP is smaller with respect to the others ( S ' 3 , A ' 2 , O ' 5 ), fewer samples are required to achieve good model estimates.

For the experiment on the right, we consider a larger POMDP instance ( S ' 5 , A ' 5 , O ' 5 ) and we run our simulation across 30 episodes, each one of length 1 . 2 ˚ 10 6 steps. As expected, the estimation process in this case has more noise, but a decrease in the estimation error is evident across the different episodes.

How Policies Vary across Episodes. The change of belief-policies across the different episodes is implemented in the following way.

18 See Appendix E for details.

Figure 3: Frobenius norm of the estimation error of two different POMDP instances. For the instance on the left we have S ' 3 , A ' 2 , O ' 5 , for the one on the right S ' 5 , A ' 3 , O ' 5 . (10 runs, 95 %c.i.).

<!-- image -->

Figure 4: Regret comparison on a POMDP with S ' 3 , A ' 3 , O ' 4 violating Assumption 6.1 (10 runs, 95 %c.i.).

<!-- image -->

- ( i ) Each policy has an internal transition and observation model that it uses to update its belief. When the episode changes, we change as well these components. We remark that these models are only used for the internal update of the belief and are independent of the transition and observation model of the interacting POMDP instance.
- ( ii ) Each policy has an internal vector r P R O of rewards associated to each observation. At each step, the chosen action is the one maximizing the expected reward in the next time step. When the episode changes, we change as well the internal reward vector r . As a last point, in order to ensure enough exploration of all actions, the policy has a minimum probability of choosing every action at each time step.

## J.2 Simulations and Details on Regret Experiments

For the experiments on the regret, we adopted the following hyperparameters for the different algorithms.

- Mixed Spectral UCRL : length of initial episode T 0 ' 3 ˚ 10 5 ;
- SM-UCRL : length of initial episode T 0 ' 3 ˚ 10 5 , minimum action probability ι ' 0 . 02 ;
- SEEU : length of exploration phase τ 1 ' 10 5 , length of initial exploitation phase τ 2 ' 3 ˚ 10 5 . At each new episode l , the length of the exploitation phase is computed as ? l ` 1 τ 2 , as defined in the original work.

Concerning the computation of the optimal policy, for both the SEEU and the Mixed Spectral UCRL algorithm, we adopted the following approach. Since there is uncertainty in the model parameters, the Extended Value Iteration algorithm [14] should be used to find a robust policy. However, in practice, since we are in the POMDP setting, our approach consists in sampling multiple POMDPs within the confidence region C l p δ l q , discretize the belief space of each of the corresponding belief MDPs , find the corresponding best policy by using Value Iteration on each discretized MDP, and finally return the best among them. Similar approaches are also employed in [3]. For the considered simulations,

we adopted a discretization step size of 0.04.

Since the SM-UCRL algorithm relies on memoryless policies, we applied a similar sampling procedure and then directly the Value Iteration algorithm, replacing the state space with the observation space.

By following the suggestions in [3], we replaced the theoretical bounds with smaller values. This approach is commonly used in experimental comparisons in these settings and generally results in either a regret with larger multiplicative constants or guarantees holding with a lower probability.

Regret Experiment Violating Assumption 6.1. Our belief is that Assumption 6.1 can be relaxed in practice while still guaranteeing sublinear regret, however it is hard to remove it from a technical perspective.

To corroborate our intuition, we run new regret experiments on a POMDP instance that violates Assumption 6.1. The experimental results are shown in Figure 4 and demonstrate how the tested algorithms (both our Mixed Spectral UCRL and SEEU) show regret results that align with their theoretical guarantees, hence showing robustness to failure of this assumption.