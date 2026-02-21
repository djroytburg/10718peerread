## Reinforcement Learning with Imperfect Transition Predictions: A Bellman-Jensen Approach

## Chenbei Lu

Institute for Interdisciplinary Information Sciences Tsinghua University

## Zaiwei Chen

Edwardson School of Industrial Engineering Purdue University

## Tongxin Li

School of Data Science The Chinese University of Hong Kong (Shenzhen)

## Chenye Wu ∗

School of Science and Engineering The Chinese University of Hong Kong (Shenzhen)

## Abstract

Traditional reinforcement learning (RL) assumes the agents make decisions based on Markov decision processes (MDPs) with one-step transition models. In many real-world applications, such as energy management and stock investment, agents can access multi-step predictions of future states, which provide additional advantages for decision making. However, multi-step predictions are inherently high-dimensional: naively embedding these predictions into an MDP leads to an exponential blow-up in state space and the curse of dimensionality. Moreover, existing RL theory provides few tools to analyze prediction-augmented MDPs, as it typically works on one-step transition kernels and cannot accommodate multi-step predictions with errors or partial action-coverage. We address these challenges with three key innovations: First, we propose the Bayesian value function to characterize the optimal prediction-aware policy tractably. Second, we develop a novel BellmanJensen Gap analysis on the Bayesian value function, which enables characterizing the value of imperfect predictions. Third, we introduce BOLA (Bayesian Offline Learning with Online Adaptation), a two-stage model-based RL algorithm that separates offline Bayesian value learning from lightweight online adaptation to real-time predictions. We prove that BOLA remains sample-efficient even under imperfect predictions. We validate our theory and algorithm on synthetic MDPs and a real-world wind energy storage control problem.

## 1 Introduction

Reinforcement Learning (RL) [1] has emerged as a powerful framework for sequential decisionmaking, achieving remarkable success across diverse domains [2-5]. Classical RL formulates

∗ Corresponding author, chenyewu@yeah.net

## Adam Wierman

Computing &amp; Mathematical Sciences Caltech

decision-making as a Markov Decision Process (MDP), where an agent seeks to maximize expected cumulative rewards in a stochastic environment. A central premise of this framework is that once the agent has accurately captured the environment model (e.g., the transition dynamics), it can, in principle, compute an optimal policy. In model-based RL, this involves explicitly learning the transition kernel and reward function to solve the optimal policy [6]. Even in model-free RL, agents implicitly learn the environment through value function or policy learning [7].

However, in many real-world applications, agents can access even richer information: the prediction of future transition realizations. Rather than relying solely on expected transition dynamics, these realization-level predictions specify exact future states, which can reduce or even eliminate the environments inherent stochasticity and enable more effective decision-making. For example, in financial markets, accurate multi-step price forecasts can substantially improve trading strategies [8], while in energy systems, reliable predictions of renewable energy generation allow the system operators to schedule the power generation sources more efficiently [9].

Despite their potential, incorporating multi-step transition predictions into MDPs faces three key challenges. First, the predictions over a multi-step horizon are inherently high-dimensional: augmenting the state with these predictions expands the state space exponentially, making standard solutions computationally intractable (see Section 3 for more details). Second, even if the augmented MDP can be solved, existing theory lacks formal tools to quantify the benefits of multi-step transition predictions, particularly when they are inaccurate or only cover a subset of actions. Third, in the absence of strong assumptions on function approximation [1, 10-13], RL's sample complexity scales at least linearly with the size of the stateaction space [14, 15], and the exponential state expansion also induces an exponential blow-up in the required samples [16]. Addressing these challenges is essential for the rigorous integration of transition predictions into RL. See Appendix A for a detailed literature review.

To overcome these challenges, our contributions can be summarized as follows:

Tractable Optimal Policy for MDPs with Transition Predictions. We introduce a low-dimensional Bayesian value function that integrates multi-step transition predictions into the value evaluation, which enables a tractable characterization of the optimal prediction-aware policy.

Characterization of the Value of Imperfect Predictions using Bellman-Jensen Gap. We introduce the Bellman-Jensen Gap framework, a novel analytical tool that decomposes the advantage of multistep predictions into a recursive sum of local Jensen gaps in the Bayesian value function. Building on this framework, we characterize the value of imperfect predictions and show how it can close the performance gap to the offline optimal policy in Theorem 4.1.

Prediction-Aware Algorithm with Improved Sample Complexity. We propose BOLA, a two-stage model-based RL algorithm that combines offline Bayesian value estimation with online integration of real-time predictions. We prove that BOLA avoids the exponential sample complexity and, given high-quality predictions, is more sample efficient than classical model-based RL [14, 15]. Our analysis relies on tailored error-decomposition and telescoping bounds to control multi-step transition errors.

The remainder of this paper proceeds as follows. Section 2 formalizes the prediction-augmented MDP framework. Based on this formulation, Section 3 introduces a tractable Bayesian value function to characterize the optimal policy, circumventing the curse of dimensionality. Subsequently, Section 4 establishes the Bellman-Jensen Gap to theoretically quantify the value of imperfect predictions. This analysis directly motivates Section 5, where we present the BOLA algorithm with provable sample efficiency guarantees. Section 6 then provides empirical validation on a wind energy storage problem. Finally, Section 7 concludes with a discussion of limitations and future work. Complete proofs and additional details are included in the appendices.

## 2 Markov Decision Processes with Transition Predictions

In this section, we formally introduce the framework for MDPs augmented with transition predictions.

We begin by introducing the definition of a discounted infinite-horizon MDP, which is specified by the tuple M = ( S , A , P, r, γ ) . Here, S and A are the finite state and action spaces, respectively; P is the transition kernel, i.e., P ( · | s, a ) denotes the distribution of the next state given that action a

is taken at state s ; r : S × A → [0 , 1] is the reward function; and γ ∈ (0 , 1) is the discount factor. Since we focus on finite MDPs, assuming bounded rewards is without loss of generality. The model parameters of the MDP (i.e., the transition kernel P and the reward function r ) are unknown to the agent, but the agent can interact with the environment by observing the current state, selecting actions, and receiving the resulting next state and reward.

## 2.1 MDPs with Transition Predictions

We extend the classical MDP framework by incorporating imperfect predictions of future transition dynamics. Formally, consider an MDP with transition predictions characterized by the tuple M p = ( S , A , P, r, γ, K, A -, ε ) , where ( K, A -, ε ) capture the prediction structure. Specifically, K denotes the finite prediction horizon; A -⊆ A specifies the subset of actions for which transition outcomes can be predicted; and ε quantifies the associated prediction errors.

We first consider an ideal case that the prediction is accurate. At discrete time steps t = 0 , K, 2 K,... , the agent receives a batch of predicted transitions for the next K steps, denoted as σ ∗ = ( σ ∗ 1 , σ ∗ 2 , . . . , σ ∗ K ) , and each σ ∗ k is a binary matrix of size |S||A -| × |S| , where each row corresponds to a ( s, a ) pair with a ∈ A -and is a one-hot vector indicating the predicted next state.

Given an accurate one-step transition prediction σ ∗ k , the conditional transition probabilities depend on whether the action taken falls within the predictable action subset A -. This subset captures actions for which reliable prediction models are available, allowing the agent to exploit future information. In contrast, actions outside A -must rely solely on the underlying transition dynamics P ( s ′ | s, a ) of the environment. Accordingly, the conditional transition model with predictions is:

<!-- formula-not-decoded -->

To preserve the Markov property of the underlying MDP, we impose two natural conditions on each onestep prediction matrix σ ∗ k within the K -step forecast:

- Independence and stationarity . Each σ ∗ k is drawn i.i.d. from a fixed distribution. This ensures that every predicted transition remains stationary with respect to the transition kernel. Also, the prediction depends only on the current stateaction pair and not on any prior history.
- Consistency . In expectation, the accurate prediction exactly recovers the true transition kernel:

<!-- formula-not-decoded -->

However, exact prediction is not always attainable in practice. Thus, we assume the agent only receives inaccurate predictions denoted as σ = ( σ 1 , σ 2 , . . . , σ K ) ∈ Q K , where Q K denotes the space of inaccurate prediction σ , and each σ k ∈ [0 , 1] |S||A -|×|S| is a stochastic matrix indicating predicted transition probabilities for the state-action pairs at step k in the future. Each prediction σ k may differ from the true future transition σ ∗ k due to prediction error. We model this discrepancy as σ k = σ ∗ k + ε k , k = 1 , . . . , K , where ε k is a random error matrix drawn from a distribution f ε k | σ ∗ k .

This formulation captures a broad class of imperfect predictions, allowing us to study how finitehorizon, partial, and inaccurate forecasts can be leveraged for improved decision-making in MDPs.

Remark: Our model differs fundamentally from prior work such as [17, 18], which treats predictions as noisy estimates of the transition kernel and aims only to match standard MDP performance. In contrast, we model ideal predictions as concrete, one-hot realizations with certain consistency property in Eq. (2), enabling us to leverage realization-level information to surpass classic MDP performance. Furthermore, unlike these methods, we explicitly address partial action predictability , an open challenge identified in [17].

## 2.2 Decision-Making and Optimization Objectives

We adopt a fixed-horizon planning protocol [19], where the agent makes decisions at discrete time points t = 0 , K, 2 K,... . At each decision point, after observing the current state s t and receiving the prediction batch σ , the agent selects an action sequence a = ( a 0 , . . . , a K -1 ) ∈ A K according to a

policy π : S×Q K → ∆( A K ) , where Q K denotes the prediction space and ∆( A K ) denotes the probability simplex over K -step action sequences. This setting models how agents dynamically plan decisions over a finite prediction horizon. The agent seeks to maximize the expected cumulative reward by selecting an optimal policy, defined as π ∗ = arg max π E π [ ∑ ∞ t =0 γ t r ( s t , a t ) | s 0 = s, σ 0 = σ ] for all s ∈ S , σ ∈ Q K .

## 3 Bayesian Value Function and Prediction-Aware Optimal Policy

In this section, we develop a tractable formulation for decision-making with multi-step imperfect transition predictions.

Motivation: A straightforward strategy for using predictions is to treat ( s, σ ) as the new state and solve a standard MDP over this extended state space. However, this approach quickly becomes intractable. A K -step prediction σ = ( σ 1 , . . . , σ K ) consists of K transition matrices, each of size |S||A| × |S| , resulting in an exponentially large state space size of at least |S| K |S||A| . Moreover, since σ is typically noisy and continuous, its support can be uncountably infinite. As a result, the augmented value function must satisfy an infinite-dimensional Bellman equation, making classical solutions impractical even for K = 1 .

In summary, while predictions have the potential to improve performance, naively augmenting the state space with raw prediction vectors leads to intractable computation. To address this issue, we next introduce a Bayesian value function that enables a tractable, prediction-aware characterization of the optimal policy.

## 3.1 Bayesian Value Function and Optimal Policy Structure

To avoid explicit state augmentation, we instead formulate a Bayesian value function defined over the original state space. The key idea is to take an expectation over the prediction distribution, thereby shifting the complexity into an outer integral while preserving a tractable structure. Formally, we define the Bayesian value function as:

<!-- formula-not-decoded -->

This Bayesian value function represents the expected cumulative reward when each decision is made after drawing a K -step prediction σ . We call it Bayesian because we marginalize over the distribution of σ , thereby accounting for forecast uncertainty in the value estimate. Importantly, the policy can condition on the realized σ , yet the value function itself remains defined solely over the original state space. This preserves tractability by avoiding an explicit statespace augmentation. The optimal Bayesian value is then:

<!-- formula-not-decoded -->

By constructing an auxiliary MDP that incorporates the predictions and linking it with the optimal Bayesian value function, we derive the corresponding Bellman optimality equation (see Appendix B for the proof).

Theorem 3.1 (Bellman Optimality Equation for Bayesian Value Function) . The optimal Bayesian value function V Bayes , ∗ K, A -, ε is the unique solution to the following fixed-point equation:

<!-- formula-not-decoded -->

Here in Eq. (5), a 0: t -1 = ( a 0 , . . . , a t -1 ) and σ 1: t = ( σ 1 , . . . , σ t ) denote the sequences of actions and predictions, respectively, and P ( s t | s 0 , a 0: t -1 , σ 1: t ) is the multi-step transition probability from initial state s 0 to state s t after t steps under the sequences of actions and predictions a 0: t -1 and σ 1: t , which satisfies the following recursive relation:

<!-- formula-not-decoded -->

The recursive form in Eq. (5) captures how predictions guide near-term planning over horizon K , with long-term value rolled into V Bayes , ∗ K, A -, ε ( s K ) . Importantly, the corresponding Bellman operator is a contraction mapping with parameter γ K under the infinity norm, which guarantees the existence and uniqueness of the solution and enables efficient fixed-point computation.

The optimal Bayesian value function directly yields the optimal policy, as characterized below. The proof is provided in Appendix C.

Corollary 3.1 (Optimal Policy with Bayesian Value Function and Transition Predictions) . The optimal policy π ∗ ( · | s, σ ) with K -step transition predictions σ satisfies:

<!-- formula-not-decoded -->

This result shows that the optimal prediction-aware policy can be computed via a finite-horizon planning over σ , followed by terminal reward using the Bayesian value function V Bayes , ∗ K, A -, ε . In effect, we have reduced the original infinite-horizon problem with high-dimensional predictions to a special form of fixedhorizon planning [19], which is tractable without explicitly augmenting the state space.

## 4 Analyzing the Value of Predictions

In this section, we examine how access to transition predictions improves decision-making in MDPs. Classical MDPs face a structural limitation: their value functions involve deeply nested max -overE operations, which force agents to commit to fixed policies based on expected dynamics. We show that transition predictions alleviate this limitation by enabling a localized reordering of the max and E operators, allowing actions to adapt to realized transitions. We use a Bellman-Jensen Gap analysis on the Bayesian value function to characterize the value of predictions.

## 4.1 Bellman-Jensen Gap

We introduce the Bellman-Jensen Gap by comparing the following value functions.

Bellman Expansion of Optimal Value Function. By recursively applying the Bellman optimality equation for classical discounted MDPs, the value function can be expressed in the following nested form [20]:

<!-- formula-not-decoded -->

where σ ∗ t denotes the transition realization of s t at time t . Each expectation E σ ∗ t is equivalent to taking the expectation over the next state s t ∼ P ( · | s t -1 , a t -1 ) , corresponding to the transition dynamics governed by σ ∗ t . This formulation results in a deeply nested max -overE structure, where the agent must choose an action that is optimal in expectation, without the ability to anticipate and adapt to future information.

In contrast, if the agent had access to perfect predictions of future transitions, it could defer action selection until those transitions are known, which allows a localized reordering of the max and E operators. We use a one-step prediction case to illustrate it:

Operator Reordering with One-Step Prediction. Recall the Bayesian value function with K = 1 , which can be expanded into the following recursive form:

<!-- formula-not-decoded -->

Observe that, with one-step prediction, each E σ ∗ t operator is moved to the outer side of the neighborhood max a t -1 operator. Intuitively, it provides the agent the ability to make decisions according to the transition prediction σ ∗ t at time t . Mathematically, this localized reordering creates a local

Jensen gap by exploiting Jensen's inequality due to E σ [max a f ( s, a ; σ )] ≥ max a E σ [ f ( s, a ; σ )] , where discrete maximization is a convex function, and f ( s, a ; σ ) denotes the expected return under state s with action a and prediction σ . Since the Bayesian value function contains infinitely many such operator reordering in a recursive manner, we term it as the Bellman-Jensen Gap .

Maximal Bellman-Jensen Gap with Infinite-Step Prediction. Such Bellman-Jensen Gaps reach the maximum when the prediction horizon is infinite. Formally, let V Bayes , ∗ off ∈ R |S| denote the offline optimal Bayesian value function, where the agent has exact knowledge of all future transitions:

<!-- formula-not-decoded -->

where σ ∗ 1: k is the sequence of realized transition kernels and a 0: k -1 is the action sequence over horizon k . The existence of the limit is shown in Appendix D.

Observe that, with infinitely long accurate prediction, all E σ ∗ operators appear outside of any max a operator, which indicates that the agent can make the decision with full information of all future information, yielding the maximal Bellman-Jensen Gap defined as follows:

Definition 4.1 (Maximal Bellman-Jensen Gap) . For any state s ∈ S , we define the Maximal BellmanJensen Gap as ∆( s ) := V Bayes , ∗ off ( s ) -V ∗ MDP ( s ) , which quantifies the greatest possible performance gain from knowing exact future transitions.

The maximal Bellman-Jensen Gap characterizes the fundamental benefit that predictive information can offer in MDPs. It upper-bounds the value of any prediction by capturing the intrinsic benefits of operator reordering in the value function. Note that, this analytical framework naturally extends to other types of predictions. For example, by redefining σ to represent the prediction on reward realizations, the same Bellman-Jensen Gap analysis applies.

## 4.2 Closing the BellmanJensen Gap with Imperfect Predictions

We now leverage the BellmanJensen gap framework to analyze how imperfect predictions narrow the performance gap to the offline oracle. In particular, we derive explicit bounds on the suboptimality of a policy that uses finite-horizon, inaccurate, and partial action-coverage predictions.

The following theorem provides a finite-horizon performance bound that decomposes the suboptimality into three interpretable components, each capturing a distinct structural limitation. The proof is provided in Appendix E.

Theorem 4.1 (Bellman-Jensen Performance Bound) . Given any prediction with horizon K ≥ 1 , predictable action set A -⊆ A and prediction errors ε , the performance gap between the predictionaware policy and the offline optimal policy satisfies:

<!-- formula-not-decoded -->

A 3 : loss due to partial action predictability

where C 1 and C 2 are absolute constants, glyph[epsilon1] j denotes the prediction error at step j , defined as the Wasserstein-1 distance between the predicted and true transition distributions; parameter θ 2 max = max s, a 0: t ,t σ 2 ( r ( s t , a t | s 0 = s, a 0: t )) captures the variability of the reward, and σ ( · ) denotes the sub-Gaussian parameter.

Interpretation. Theorem 4.1 shows that the performance gap decomposes into three terms. The first term A 1 quantifies the performance loss due to the finite prediction horizon K . The factor γ K reflects that the benefit of predictions decreases exponentially with the horizon length, implying that even short-term predictions can capture significant potential improvement. When K - → ∞ , this loss term

diminishes. The (1 -γ ) 6 / 5 exponent arises from a refined dyadic horizon decomposition argument controlling the dependence on the discount factor (see Lemmas E.2 and E.3 for details). The term √ log |A| shows the number of actions slightly increases this gap, as a larger action space makes it statistically harder to identify the optimal action under uncertainty.

The second term A 2 captures the impact of prediction errors and disappears as the predicted transitions become accurate. Notably, it highlights that errors in subsequent steps have progressively smaller effects on overall performance, aligning with practical intuition. When glyph[epsilon1] j = glyph[epsilon1] for all j , we have A 2 = O ( glyph[epsilon1]/ (1 -γ ) 2 ) , which is independent of K , indicating that this term is primarily governed by the average prediction error, rather than the length of the prediction horizon.

The third term A 3 arises from partial action predictability and vanishes when all actions are predictable (i.e., A -= A ). It is scaled by √ θ 2 max , indicating that greater reward uncertainty amplifies the Bellman-Jensen Gap. When A -= ∅ , this term will not blow up and simplifies to O ( √ log |A| θ 2 max (1 -γ ) -3 2 ) .

Corollary 4.1. Given any prediction horizon K ≥ 1 , if the predictions are perfectly accurate with glyph[epsilon1] j = 0 for all 1 ≤ j ≤ K , and all actions are predictable with A -= A , then the maximal performance gap satisfies max s ∈S ( V Bayes , ∗ off ( s ) -V Bayes , ∗ K, A -, ε ( s )) ≤ O ( γ K √ K ) .

This result demonstrates that sufficiently accurate predictive informationeven over a finite horizoncan dramatically reduce the fundamental Bellman-Jensen Gap, bringing the agents performance significantly closer to the offline oracle benchmark. It characterizes the theoretical upper bound on the improvement that predictive signals can offer, revealing an exponential decay in the gap with horizon length K , up to a sublinear √ K correction term.

## 5 BOLA: Bayesian Offline Learning with Online Adaptation

Building on the theoretical understanding of the prediction-aware policy, in this section, we present a practical model-based algorithm for implementing the prediction-aware optimal policy.

The key insight from Theorem 3.1 and Corollary 3.1 is that optimal decisions can be achieved by combining short-horizon planning with a precomputed Bayesian value as the terminal function, which can effectively leverage predictive information without explicitly expanding the state space. This motivates the design of BOLA , a two-stage approach that cleanly separates offline learning from online adaptation to predictions.

## 5.1 BOLA Algorithm Overview

Wepropose BOLA (Bayesian Offline Learning with Online Adaptation), a model-based reinforcement learning algorithm designed to exploit transition predictions for efficient decision-making. BOLA decomposes learning and planning into two stages: (1) Offline Stage: Estimate the Bayesian value function V Bayes , ∗ K, A -, ε ( s ) from samples by solving the Bellman equation in Eq. (5), and (2) Online Stage: At each decision point, observe real-time transition predictions σ and compute the optimal short-horizon action sequence using Eq. (7).

Offline Bayesian Value Function Learning. To implement the prediction-aware Bellman operator from Eq. (5), we adopt a model-based learning approach inspired by classical MDPs. Specifically, we estimate the key quantities required to compute the Bayesian value function V Bayes , ∗ K, A -, ε ( s ) via value iteration. These include: (1) the reward function r ( s, a ) , (2) the distribution over K -step transition predictions P ( σ ) , and (3) the multi-step transition kernel P ( s ′ | s, a , σ ) .

Importantly, the recursive structure in Eq. (6) allows us to avoid estimating the full K -step transition model directly. Instead, it suffices to estimate one-step transition probabilities P ( s ′ | s, a, σ ) for predictable actions a ∈ A -, and standard MDP transitions P ( s ′ | s, a ) for actions a / ∈ A -.

We assume access to a generative model [21, 22]. For each state-action pair ( s, a ) with a / ∈ A -, the generative model allows us to generate N 1 independent next-state samples, denoted by { s i ( s,a ) } N 1 i =1 .

For estimating the prediction distribution, we assume there exists a prediction oracle defined as follows:

Assumption 5.1 (Prediction Oracle) . The agent has access to a prediction oracle O pred that, upon query, returns independent samples σ i ∼ P ( σ ) , where σ i = ( σ i 1 , . . . , σ i K ) represents a K -step transition prediction vector drawn from the underlying distribution P ( σ ) .

Under this assumption, we draw N 2 independent prediction samples from the oracle, denoted by { σ i = ( σ i 1 , . . . , σ i K ) } N 2 i =1 , which are then used to estimate P ( σ ) via empirical frequencies.

Using these samples, we estimate the relevant probabilities by empirical frequency:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The reward function r ( s, a ) is obtained by sampling from each state-action pair ( s, a ) once.

With all model components estimated, we apply value iteration on the prediction-augmented Bellman operator in Eq. (5) to compute an approximate Bayesian value function ̂ V Bayes , ∗ K, A -,ε . The contraction property of the Bellman operator ensures that this fixed-point iteration converges.

Online Adaptation. At each decision point, BOLA receives a K -step transition prediction vector σ = ( σ 1 , . . . , σ K ) . Given the current state s , BOLA first evaluates the multi-step transition probabilities by incorporating the prediction σ , and then solves the optimal action sequence through Eq. (7) using the precomputed Bayesian value function ̂ V Bayes , ∗ K, A -,ε .

This scheme enables real-time adaptation without solving high-dimensional value functions online. By combining offline long-term terminal value estimation with short-horizon prediction-aware planning [23], BOLA avoids state space augmentation and maintains computational tractability. Algorithm 1 summarizes the BOLA procedure.

## Algorithm 1 BOLA: Bayesian Offline Learning with Online Adaptation

- 1: Sample N 1 times from each ( s, a ) ∈ S × A \ A -and get samples { s i ( s,a ) } N 1 i =1 ;
- 2: Sample N 2 empirical predictions { σ i } N 2 i =1 from the prediction oracle O pred ;
- 3: Estimate transition model ̂ P ( s ′ | s, a ) and distribution ̂ P ( σ ) according to Eq. (11)-(12);
- 4: Solve the estimated Bayesian value function ̂ V Bayes , ∗ K, A -,ε using value iteration;
- 5: for each decision timestep do
- 6: Observe the current state s and the transition prediction σ ;
- 7: for t = 1 to K do
- 8: Compute the transition probabilities P ( s t | s, a 0: t -1 , σ 1: t ) using Eq. (6);
- 9: end for
- 10: Determine the optimal policy π ∗ ( · | s, σ ) by solving Eq. (7);
- 11: end for

## 5.2 Sample Complexity Guarantees

This section presents the sample complexity guarantees of Algorithm 1, more specifically, the learning of the Bayesian value function. The proof of the following theorem is presented in Appendix F.

Theorem 5.1. For any given MDP, any confidence level δ ∈ (0 , 1) , any desired accuracy level glyph[epsilon1] ∈ (0 , 1 1 -γ ) , a tradeoff parameter α ∈ (0 , 1) , and prediction horizon K ≥ 1 , let D 1 be the number of samples drawn from the generative model, and D 2 be the number of samples drawn from the prediction oracle O pred . If

<!-- formula-not-decoded -->

where C 1 , C 2 are absolute constants, then with probability at least 1 -δ , the learned Bayesian value function satisfies

<!-- formula-not-decoded -->

Theorem 5.1 quantifies the sample complexity of BOLA, explicitly capturing the interplay between environment sampling and predictive adaptation. In particular, BOLAs total sample requirements decompose into two distinct regimes:

Environmentinteraction samples D 1 . The first requirement D 1 , arising from direct environment interactions, scales with |A| - |A -| , the number of unpredictable actions. This leads to a sample complexity which is strictly smaller than the classical dependence of O ( |S||A| glyph[epsilon1] -2 ) [14, 24], since increased predictability reduces the environment sampling burden. In the extreme case where all actions in the prediction horizon are predictable ( A -= A ), the dominant term of D 1 vanishes altogether (except for the reward function learning cost |S||A| ), as the predictive model fully specifies the transition dynamics within the prediction horizon.

Prediction Oracle Samples D 2 . The second term D 2 represents the samples required from the predictive model, which exhibits a distinct scaling behavior: as the prediction horizon K grows, the required number of samples decreases due to the stronger contraction factor γ K in the Bayesian Bellman equation. When K ≥ O (log( 1 γ )) , this term improves to (1 -γ ) -2 , which is lower than that in model-based RL [15]. This highlights that when predictions are both comprehensive and with long enough horizon, BOLA can achieve lower sample complexity than classical MDP approaches.

Trade-off between the Two Sample Sources. Together, these two sampling regimes reveal a tradeoff parameterized by α : increasing α (more environment interaction) raises the environment sample requirement D 1 to O ((1 -α ) -2 ) , while reducing the required number of samples D 2 from the prediction oracle. One can choose a reasonable α to trade off the sample requirements.

## 6 Numerical Studies

Although our emphasis is on theory and finitesample guarantees, we provide a small-scale empirical case study to demonstrate practical relevance. Specifically, we evaluate BOLA on a windfarm storage control task, where the operator minimizes energy imbalance penalties by charging or discharging a battery based on wind mismatch, price signals, and current state of charge (see Figure 1 for the setup, Appendix G for more details, and Appendix H for additional experiments).

Specifically, Figure 2(a) illustrates the cumulative cost reduction achieved by BOLA under different prediction horizons, compared against a baseline MDP policy without prediction. As the prediction horizon K increases from 1 to 4, the cost reduction consistently improves, confirming that longer foresight enables the agent to better anticipate upcoming mismatches and price fluctuations. Notably, the largest marginal improvement occurs at K = 1 , indicating that even a short look-ahead can significantly enhance decision-making performance. Figure 2(b) illustrates the robustness of BOLA under increasing levels of relative prediction error. As the prediction error increases, the performance of all methods declines approximately linearly, which aligns with our theoretical analysis. Notably, longer prediction horizons yield greater cost savings under perfect or low-noise forecasts but also exhibit greater sensitivity to prediction errors. In contrast, shorter horizons are more stable and

Figure 1: Wind Farm Storage Control

<!-- image -->

Figure 2: Storage Control Performance. (a) Cumulative cost reduction for different prediction horizons: longer prediction horizon yields greater savings over the no-prediction baseline. (b) Robustness to prediction noise: cost savings decline roughly linearly with error, yet all predictive policies outperform the baseline even at 30% noise.

<!-- image -->

degrade more gradually as noise increases. Nevertheless, all prediction-based policies consistently outperform the no-prediction baseline, even when the relative error reaches 30%.

## 7 Conclusion

In this work, we study the theoretical value of transition predictions in sequential decision-making. We propose a prediction-augmented MDP framework, characterize the benefit of predictions via the Bellman-Jensen Gap, and develop a tractable model-based RL algorithm with sample complexity guarantees. A natural future direction is to extend our results to the model-free setting.

Limitations and Future Directions. In prediction-augmented MDP, we consider fixed-horizon planning where the agent receives a K -step prediction and plans a K -step sequence of actions. For sequential decision-making problems with predictions, another popular framework is called the receding-horizon control, where the agent receives a K -step prediction but only plans a single-step action instead of a sequence of K actions. Intuitively, using receding-horizon control could be more beneficial to the agent than fixed-horizon planning, since the agent does not have to commit to a sequence of actions and can adaptively choose actions based on the new realizations of the states and the predictions. Further investigating the advantage of prediction-augmented MDPs with receding-horizon control is among the future directions of this work. On the theoretical side, we have established the first upper bounds on BOLAs sample complexity, but it remains open whether these can be tightened or matched by lower bounds. In particular, refined variance-based techniques (e.g., refined concentration for the multi-step Bayesian operator) may yield stronger guarantees.

## Acknowledgments

We sincerely thank the anonymous area chair and reviewers for their insightful feedback. We are grateful to Hongyu Yi for proofreading the paper and to Laixi Shi for helpful discussions.

C. Wu's work was supported in part by the National Natural Science Foundation of China under Grant 72271213, the Shenzhen Science and Technology Program under Grant RCYX20221008092927070. A. Wierman's work was supported in part by NSF grants CNS-2146814, CPS-2136197, CNS2106403, and NGSDI-2105648.

## References

- [1] Richard S Sutton and Andrew G Barto. Reinforcement Learning: An Introduction . MIT press, 2018.
- [2] Łukasz Kaiser, Mohammad Babaeizadeh, Piotr Miłos, Bła˙ zej Osi´ nski, Roy H Campbell, Konrad Czechowski, Dumitru Erhan, Chelsea Finn, Piotr Kozakowski, Sergey Levine, et al. Modelbased reinforcement learning for Atari. In International Conference on Learning Representations .
- [3] Jens Kober, J Andrew Bagnell, and Jan Peters. Reinforcement learning in robotics: A survey. The International Journal of Robotics Research , 32(11):1238-1274, 2013.
- [4] Ammar Haydari and Yasin Yılmaz. Deep reinforcement learning for intelligent transportation systems: A survey. IEEE Transactions on Intelligent Transportation Systems , 23(1):11-32, 2020.
- [5] Xin Chen, Guannan Qu, Yujie Tang, Steven Low, and Na Li. Reinforcement learning for selective key applications in power systems: Recent advances and future challenges. IEEE Transactions on Smart Grid , 13(4):2935-2958, 2022.
- [6] Thomas M Moerland, Joost Broekens, Aske Plaat, Catholijn M Jonker, et al. Model-based reinforcement learning: A survey. Foundations and Trends® in Machine Learning , 16(1):1-118, 2023.
- [7] Leslie Pack Kaelbling, Michael L Littman, and Andrew W Moore. Reinforcement learning: A survey. Journal of artificial intelligence research , 4:237-285, 1996.
- [8] Arthur Charpentier, Romuald Elie, and Carl Remlinger. Reinforcement learning in economics and finance. Computational Economics , pages 1-38, 2021.
- [9] Adil Ahmed and Muhammad Khalid. A review on the selected applications of forecasting models in renewable power systems. Renewable and Sustainable Energy Reviews , 100:9-21, 2019.
- [10] John Tsitsiklis and Benjamin Van Roy. Analysis of temporal-difference learning with function approximation. Advances in neural information processing systems , 9, 1996.
- [11] Jalaj Bhandari, Daniel Russo, and Raghav Singal. A finite time analysis of temporal difference learning with linear function approximation. In Conference on learning theory , pages 1691-1692. PMLR, 2018.
- [12] Rayadurgam Srikant and Lei Ying. Finite-time error bounds for linear stochastic approximation and TD-learning. In Conference on Learning Theory , pages 2803-2830. PMLR, 2019.
- [13] Zaiwei Chen, John-Paul Clarke, and Siva Theja Maguluri. Target network and truncation overcome the deadly triad in Q-learning. SIAM Journal on Mathematics of Data Science , 5(4): 1078-1101, 2023.
- [14] Mohammad Gheshlaghi Azar, Rémi Munos, and Hilbert J Kappen. On the sample complexity of reinforcement learning with a generative model. In Proceedings of the 29th International Coference on International Conference on Machine Learning , pages 1707-1714, 2012.
- [15] Gen Li, Yuting Wei, Yuejie Chi, Yuantao Gu, and Yuxin Chen. Breaking the sample size barrier in model-based reinforcement learning with a generative model. Advances in neural information processing systems , 33:12861-12872, 2020.
- [16] Chenbei Lu, Laixi Shi, Zaiwei Chen, Chenye Wu, and Adam Wierman. Overcoming the curse of dimensionality in reinforcement learning through approximate factorization. In Forty-second International Conference on Machine Learning (ICML 2025) , 2025.
- [17] Ziyi Zhang, Guannan Qu, et al. Predictive control and regret analysis of non-stationary MDP with look-ahead information. Transactions on Machine Learning Research .

- [18] Lixing Lyu, Jiashuo Jiang, and Wang Chi Cheung. Efficiently solving discounted MDPs with predictions on transition matrices. Preprint arXiv:2502.15345 , 2025.
- [19] Carlos E Garcia, David M Prett, and Manfred Morari. Model predictive control: Theory and practicea survey. Automatica , 25(3):335-348, 1989.
- [20] EN Barron and H Ishii. The bellman equation for minimizing the maximum cost. Nonlinear Anal. Theory Methods Applic. , 13(9):1067-1090, 1989.
- [21] Michael Kearns, Yishay Mansour, and Andrew Y Ng. A sparse sampling algorithm for nearoptimal planning in large Markov decision processes. Machine learning , 49:193-208, 2002.
- [22] Sham Machandranath Kakade. On the sample complexity of reinforcement learning . University of London, University College London (United Kingdom), 2003.
- [23] Philip Allmendinger. Planning theory . Bloomsbury Publishing, 2017.
- [24] Gen Li, Laixi Shi, Yuxin Chen, Yuantao Gu, and Yuejie Chi. Breaking the sample complexity barrier to regret-optimal model-free reinforcement learning. Advances in Neural Information Processing Systems , 34:17762-17776, 2021.
- [25] Alekh Agarwal, Sham Kakade, and Lin F Yang. Model-based reinforcement learning with a generative model is minimax optimal. In Conference on Learning Theory , pages 67-83. PMLR, 2020.
- [26] Mohammad Gheshlaghi Azar, Rémi Munos, and Hilbert J Kappen. Minimax PAC bounds on the sample complexity of reinforcement learning with a generative model. Machine learning , 91:325-349, 2013.
- [27] Aaron Sidford, Mengdi Wang, Xian Wu, Lin Yang, and Yinyu Ye. Near-optimal time and sample complexities for solving Markov decision processes with a generative model. Advances in Neural Information Processing Systems , 31, 2018.
- [28] Mohammad Gheshlaghi Azar, Ian Osband, and Rémi Munos. Minimax regret bounds for reinforcement learning. In International Conference on Machine Learning , pages 263-272. PMLR, 2017.
- [29] Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory , pages 2137-2143. PMLR, 2020.
- [30] Yujia Jin and Aaron Sidford. Towards tight bounds on the sample complexity of average-reward MDPs. In International Conference on Machine Learning , pages 5055-5064. PMLR, 2021.
- [31] Sharan Vaswani, Lin Yang, and Csaba Szepesvári. Near-optimal sample complexity bounds for constrained MDPs. Advances in Neural Information Processing Systems , 35:3110-3122, 2022.
- [32] Pascal Poupart and Nikos Vlassis. Model-based bayesian reinforcement learning in partially observable domains. In Proc Int. Symp. on Artificial Intelligence and Mathematics, , pages 1-2, 2008.
- [33] Masatoshi Uehara, Ayush Sekhari, Jason D Lee, Nathan Kallus, and Wen Sun. Provably efficient reinforcement learning in partially observable dynamical systems. Advances in Neural Information Processing Systems , 35:578-592, 2022.
- [34] Qinghua Liu, Praneeth Netrapalli, Csaba Szepesvari, and Chi Jin. Optimistic mle: A generic model-based algorithm for partially observable sequential decision making. In Proceedings of the 55th Annual ACM Symposium on Theory of Computing , pages 363-376, 2023.
- [35] Michael O'Gordon Duff. Optimal Learning: Computational procedures for Bayes-adaptive Markov decision processes . University of Massachusetts Amherst, 2002.
- [36] Stephane Ross, Brahim Chaib-draa, and Joelle Pineau. Bayes-adaptive pomdps. Advances in neural information processing systems , 20, 2007.

- [37] Arthur Guez, David Silver, and Peter Dayan. Efficient bayes-adaptive reinforcement learning using sample-based search. Advances in neural information processing systems , 25, 2012.
- [38] Ibrahim El Shar and Daniel Jiang. Lookahead-bounded Q-learning. In International Conference on Machine Learning , pages 8665-8675. PMLR, 2020.
- [39] Anna Winnicki, Joseph Lubars, Michael Livesay, and R Srikant. The role of lookahead and approximate policy evaluation in reinforcement learning with linear value function approximation. Operations Research , 73(1):139-156, 2025.
- [40] Yonathan Efroni, Gal Dalal, Bruno Scherrer, and Shie Mannor. Beyond the one-step greedy approach in reinforcement learning. In International Conference on Machine Learning , pages 1387-1396. PMLR, 2018.
- [41] Aviv Rosenberg, Assaf Hallak, Shie Mannor, Gal Chechik, and Gal Dalal. Planning and learning with adaptive lookahead. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 37, pages 9606-9613, 2023.
- [42] Yonathan Efroni, Mohammad Ghavamzadeh, and Shie Mannor. Online planning with lookahead policies. Advances in Neural Information Processing Systems , 33:14024-14033, 2020.
- [43] Noah Golowich and Ankur Moitra. Can Q-learning be improved with advice? In Conference on Learning Theory , pages 4548-4619. PMLR, 2022.
- [44] Tongxin Li, Yiheng Lin, Shaolei Ren, and Adam Wierman. Beyond black-box advice: Learningaugmented algorithms for MDPs with Q-value predictions. Advances in Neural Information Processing Systems , 36, 2024.
- [45] Yingying Li and Na Li. Leveraging predictions in smoothed online convex optimization via gradient-based algorithms. Advances in Neural Information Processing Systems , 33: 14520-14531, 2020.
- [46] Yingying Li, Guannan Qu, and Na Li. Online optimization with predictions and switching costs: Fast algorithms and the fundamental limit. IEEE Transactions on Automatic Control , 66(10): 4761-4768, 2020.
- [47] Niangjun Chen, Anish Agarwal, Adam Wierman, Siddharth Barman, and Lachlan LH Andrew. Online convex optimization using predictions. In Proceedings of the 2015 ACM SIGMETRICS International Conference on Measurement and Modeling of Computer Systems , pages 191-204, 2015.
- [48] Yiheng Lin, Gautam Goel, and Adam Wierman. Online optimization with predictions and non-convex losses. Proceedings of the ACM on Measurement and Analysis of Computing Systems , 4(1):1-32, 2020.
- [49] Chenkai Yu, Guanya Shi, Soon-Jo Chung, Yisong Yue, and Adam Wierman. The power of predictions in online control. Advances in Neural Information Processing Systems , 33: 1994-2004, 2020.
- [50] Runyu Zhang, Yingying Li, and Na Li. On the regret analysis of online LQR control with predictions. In 2021 American Control Conference (ACC) , pages 697-703. IEEE, 2021.
- [51] Yiheng Lin, Yang Hu, Guanya Shi, Haoyuan Sun, Guannan Qu, and Adam Wierman. Perturbation-based regret analysis of predictive control in linear time varying systems. Advances in Neural Information Processing Systems , 34:5174-5185, 2021.
- [52] Yiheng Lin, Yang Hu, Guannan Qu, Tongxin Li, and Adam Wierman. Bounded-regret MPC via perturbation analysis: Prediction error, constraints, and nonlinearity. Advances in Neural Information Processing Systems , 35:36174-36187, 2022.
- [53] Luc Mercier and Pascal Van Hentenryck. Performance analysis of online anticipatory algorithms for large multistage stochastic integer programs. In IJCAI , pages 1979-1984, 2007.
- [54] Martin L Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming . John Wiley &amp; Sons, 2014.

- [55] Dimitri Bertsekas and John N Tsitsiklis. Neuro-Dynamic Programming . Athena Scientific, 1996.
- [56] Stefan Banach. Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales. Fund. math , 3(1):133-181, 1922.
- [57] Walter Rudin et al. Principles of Mathematical Analysis , volume 3. McGraw-hill New York, 1964.
- [58] Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science , volume 47. Cambridge University Press, 2018.
- [59] Gabriel Peyré and Marco Cuturi. Computational optimal transport. Foundations and Trends® in Machine Learning , 11(5-6):355-607, 2019.
- [60] California ISO. Electricity Price Data. https://www.energyonline.com/Data/ , 2021. Online; accessed on 20 December 2022.
- [61] Chenbei Lu, Hongyu Yi, Jiahao Zhang, and Chenye Wu. Self-improving online storage control for stable wind power commitment. IEEE Transactions on Smart Grid , 2024.
- [62] Ian Osband and Benjamin Van Roy. Near-optimal reinforcement learning in factored MDPs. Advances in Neural Information Processing Systems , 27, 2014.
- [63] Ruosong Wang, Russ R Salakhutdinov, and Lin Yang. Reinforcement learning with general value function approximation: Provably efficient approach via bounded Eluder dimension. Advances in Neural Information Processing Systems , 33:6123-6135, 2020.
- [64] William Koch, Renato Mancuso, Richard West, and Azer Bestavros. Reinforcement learning for UAV attitude control. ACM Transactions on Cyber-Physical Systems , 3(2):1-21, 2019.
- [65] Gaoyuan Xu, Jian Shi, Jiaman Wu, Chenbei Lu, Chenye Wu, Dan Wang, and Zhu Han. An optimal solutions-guided deep reinforcement learning approach for online energy storage control. Applied Energy , 361:122915, 2024.
- [66] Chenbei Lu, Nan Gu, Wenqian Jiang, and Chenye Wu. Sample-adaptive robust economic dispatch with statistically feasible guarantees. IEEE Transactions on Power Systems , 39(1): 779-793, 2023.
- [67] Lin Lin, Xin Guan, Yu Peng, Ning Wang, Sabita Maharjan, and Tomoaki Ohtsuki. Deep reinforcement learning for economic dispatch of virtual power plant in internet of energy. IEEE Internet of Things Journal , 7(7):6288-6301, 2020.
- [68] Giancarlo Mantovani and Luca Ferrarini. Temperature control of a commercial building with model predictive control techniques. IEEE transactions on industrial electronics , 62(4): 2651-2660, 2014.
- [69] Shiying Huang, Peng Li, Ming Yang, Yuan Gao, Jiangyang Yun, and Changhang Zhang. A control strategy based on deep reinforcement learning under the combined wind-solar storage system. IEEE Transactions on Industry Applications , 57(6):6547-6558, 2021.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction reflect the papers contributions and scope.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We provide the limitations in the conclusion.

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

Justification: We provide them in the Appendix

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We have provided detailed parameters and settings in Appendix.

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

Justification: We included code and data in supplementary files.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.

- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/public/ guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so No is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https://nips.cc/public/ guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We have provided all the experimental settings in our experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: They are correctly defined.

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

Justification: They are included in the Appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We confirm this issue.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We justify the practical relevance and societal impacts in the introduction part.

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

Justification: This work does not involve releasing pretrained models or sensitive datasets, and thus no additional safeguards for high-risk model or data release are applicable.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We mentioned the source of data.

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

Justification: We provided the code and data.

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

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or nonstandard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: We use LLMs to polish some writings.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Related Literature

Classical Model-Based RL and Sample Complexity. Our work builds on the modelbased RL paradigm, which separates learning into model estimation and policy planning. Classical modelbased RL for finite MDPs has been extensively studied and achieves minimax-optimal sample complexity guarantees [14, 25-29]. Recent extensions of sample complexity analyses cover broader classes of problems, including average-reward MDPs [30], constrained MDPs [31], and partially observable MDPs (POMDPs) [32-34]. In contrast, we tackle a new settingMDPs augmented with multistep transition predictionsand establish the first finitesample guarantees (Theorem 5.1) under both environment and prediction sampling. Notably, when the prediction horizon satisfies K = Ω(log(1 /γ )) , the stronger contraction factor γ K leads to a strictly lower sample complexity than the classical bound [14, 15], highlighting how predictive information can fundamentally accelerate learning.

RL with Look-ahead Prediction and Planning . Integrating multi-step look-ahead forecasts into reinforcement learning has driven strong empirical gains. Related work ranges from Bayes-Adaptive methods [35-37] and look-ahead Q-learning [38, 39] to policy-iteration variants [40, 41] and realtime dynamic programming [42]. However, these methods focus primarily on empirical gains and offer little theoretical support on some key points: they neither quantify the value added by multi-step predictions nor guarantee robustness when forecasts are noisy or partial, and they do not address the sample complexity of learning with such predictions. We fill these gaps by introducing the BellmanJensen framework, which is the first to deliver rigorous performance and robustness guarantees for multi-step, imperfect transition predictions in RL. We also propose BOLA, which comes with finite-sample complexity bounds for RL with transition predictions.

MDPs with Predictions. Recent work has explored integrating predictions into MDPs in several ways. Some methods leverage Q -value predictions to to speed up learning or refine suboptimal policies [43, 44]. There are also some more related work focusing on incorporating estimates of the transition kernel to boost performance in nonstationary environment [17] or improve sample efficiency [18]. However, these approaches target matching the performance of an idealized MDP with perfect models. By contrast, we use realizationbased, multistep transition forecasts to surpass the inherent limits of classical online MDP solutions, driving performance closer to the offlineoptimal benchmark even under imperfect and partial predictions (Theorem 4.1).

Online Optimization and Control with Predictions. Leveraging predictive information is a common strategy in broader online optimization and control contexts, notably within Online Convex Optimization (OCO) and Model Predictive Control (MPC). For example, recent studies [45-47] demonstrate exponentially decaying regret when leveraging predictions in Smoothed Online Convex Optimization (SOCO). Lin et al. [48] have identified conditions under which predictions significantly enhance performance in general online optimization settings. Similarly, prediction-driven improvements have been established in linear-quadratic control frameworks, yielding exponential regret reduction [49, 50], and have been successfully extended to MPC settings, even with time-varying constraints [51, 52]. Mercier et al. [53] investigate prediction in a general online optimization setting, offering useful insights, while leaving open questions on the formal analysis on the value of prediction and on broader prediction settings. Compared to this extensive literature on deterministic or structured linear settings, prediction-augmented stochastic models such as MDPs have received relatively limited attention due to their inherent complexity and lack of closed-form solutions. Our work addresses this gap by providing rigorous theoretical foundations, clear sample complexity characterizations, and practical algorithms that effectively integrate predictive realizations into sequential stochastic decision-making.

## Appendices

## B Proof of Theorem 3.1

The proof is divided into 3 steps. First, we construct an auxiliary MDP ˜ M and show that its optimal policy ˜ π ∗ is the same as the optimal policy π ∗ of the prediction-augmented MDP. Then, we show that the optimal value function satisfies our proposed Bellman equation of the prediction-augmented MDP. Finally, we show that the Bellman operator associated with our Bellman equation is a contraction mapping, which implies that it has a unique fixed point solution.

Step 1: Construction of the Auxiliary MDP ˜ M . In a prediction-augmented MDP, the agent essentially makes decisions based on both the current state s and the received prediction σ . Therefore, we can incorporate the prediction into the state and define an auxiliary MDP based on it. Specifically, let ˜ M = ( ˜ S , ˜ A , ˜ r, ˜ P, ˜ γ ) be an MDP with state space ˜ S , action space ˜ A , reward function ˜ r , transition kernel ˜ P , and discount factor γ . The state space ˜ S is the product space of the state space of the original MDP and the domain of the prediction σ , i.e., ˜ s = ( s, σ ) ∈ ˜ S := S × Q , where σ = ( σ 1 , σ 2 , ..., σ K ) . The action space ˜ A is the K -product space of the action space of the original MDP, i.e., a = ( a 0 , a 1 , · · · , a K -1 ) ∈ ˜ A := A K . For any ˜ s ∈ ˜ S and a ∈ ˜ A , the reward function ˜ r ( · , · ) is defined to be the K -step expected discounted reward of the original MDP, that is,

<!-- formula-not-decoded -->

where s 0 = s and s t ∼ P ( · | s, a 0: t -1 , σ 1: t ) for all t ≥ 1 . For any (˜ s = ( s, σ ) , a ) and ˜ s ′ = ( s ′ , σ ′ ) , the transition probability ˜ P (˜ s ′ | ˜ s, a ) is defined as

<!-- formula-not-decoded -->

Finally, the discount factor of the auxiliary MDP satisfies ˜ γ = γ K .

For the policy π : ˜ S × ˜ A - → ∆( ˜ A ) (where ∆( ˜ A ) denotes the probability simplex on ˜ A ). The corresponding value function ˜ V π of the auxiliary MDP is defined as

<!-- formula-not-decoded -->

The auxiliary MDP is with an infinite state space and a finite action space. The Bellman optimality equation remains valid provided the bounded rewards, the measurable transition kernel, and a discount factor ˜ γ ∈ [0 , 1) . These conditions ensure the Bellman operator is a ˜ γ -contraction on the space of bounded measurable functions [54, 55].

Therefore, the optimal value function ˜ V ∗ is the unique solution to the following Bellman optimality equation:

<!-- formula-not-decoded -->

In addition, any policy ˜ π ∗ satisfying

<!-- formula-not-decoded -->

for all ˜ s = ( s, σ ) is an optimal policy.

Since the auxiliary MDP has the same problem structure (state, action, transition, reward) as the prediction-augmented MDP, an optimal policy ˜ π ∗ of the auxiliary MDP is also an optimal policy π ∗ of the prediction-augmented MDP.

Step 2: Establishing the Bellman Equation. Recall from Section 3.1 that we defined the Bayesian value function V Bayes ,π ( s ) = E σ [ V π K, A -,ε ( s, σ )] = E σ [ ˜ V π (˜ s )] , where ˜ s = ( s, σ ) . Using the previous identity in Eq. (14), we have that under the optimal policy π ∗ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the term ∫ ˜ s ∈ ˜ S ˜ P ( d ˜ s ′ | ˜ s, a ) ˜ V ∗ (˜ s ′ ) , we use the independence between prediction σ and state s and get:

<!-- formula-not-decoded -->

Combining the previous two equations, we finally have:

<!-- formula-not-decoded -->

for all s ∈ S . This leads to the Bellman optimality equation in Eq. (5).

Step 3: The Uniqueness of the Solution to the Bayesian Bellman Equation. It is easy to verify that our Bellman equation is a fixed-point equation with a contractive fixed-point operator, where the contraction factor is γ K . Therefore, by the Banach fixed-point theorem [56], the solution to our Bellman equation is unique.

## C Proof of the Corollary 3.1

This is a direct corollary of Theorem 3.1. Recall that, the optimal policy π ∗ is consistent with the optimal policy of the auxiliary MDP ˜ M defined in Section B. Hence, combining Eq. (15) and Eq. (17) yields the optimal policy:

<!-- formula-not-decoded -->

## D Proof of the Existence of V Bayes , ∗ off ( s )

For any state s ∈ S and k ≥ 1 , we define the following truncated optimal value function with k -step accurate transition prediction σ ∗ 1: k :

<!-- formula-not-decoded -->

We now show the sequence { V ∗ off ,k ( s ) } k is (1) monotonically increasing with k and (2) bounded.

For monotonically increasing, since V ∗ off ,k +1 ( s ) can be represented by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality is due to r ( s, a ) ≥ 0 for any state-action pair ( s, a ) .

For the bounded property, it is clear that V ∗ off ,k ( s ) ≤ ∑ k -1 t =0 γ t ≤ 1 1 -γ for any k . Hence, the sequence { V ∗ off ,k ( s ) } k is monotonically increasing and bounded, implying V Bayes , ∗ off ( s ) = lim k →∞ V ∗ off ,k ( s ) exists for any s . This concludes our proof.

## E Proof of Theorem 4.1

Let V Bayes , ∗ ∞ , A -, 0 ( s ) := lim k →∞ V Bayes , ∗ k, A -, 0 ( s ) for all s ∈ S . The following lemma verifies that V Bayes , ∗ ∞ , A -, 0 ( s ) is indeed well-defined.

Lemma E.1 (Proof in Appendix E.1) . lim k →∞ V Bayes , ∗ k, A -, 0 ( s ) exists and is unique.

To bound the difference between V Bayes , ∗ off ( s ) and V Bayes , ∗ K, A -, ε ( s ) , we perform the following decomposition:

<!-- formula-not-decoded -->

Here, T 1 denotes the loss incurred by the finite prediction window K , T 2 captures the loss stemming from partial action predictability, and T 3 accounts for the error introduced by prediction errors. We bound these three terms as follows.

Step 1: Bounding the Term T 1 using Dyadic Horizon Decomposition. We begin by analyzing the Bellman-Jensen Gap due to the finite prediction horizon K . Since the value function sequence { V Bayes , ∗ K, A -, 0 ( s ) } converges as k →∞ , any subsequence must also converge to the same limit. Therefore, for any s ∈ S , we make a dyadic horizon decomposition as follows:

<!-- formula-not-decoded -->

Based on this decomposition, we only need to provide an upper bound to the value function gap when doubling the prediction window, which is stated in the following lemma.

Lemma E.2 (Proof in Appendix E.2) . For any prediction window K ≥ 1 , we have

<!-- formula-not-decoded -->

where C is an absolute constant.

By repeatedly using Lemma E.2, we have the following lemma that bounds the term T 1 in Eq. (19). Lemma E.3 (Proof in Appendix E.3) . There exists an absolute constant C 0 &gt; 0 such that the following inequality holds for all K ≥ C 0 :

<!-- formula-not-decoded -->

where C 1 is an absolute constant.

Step 2: Bounding the term T 2 . For any s ∈ S , V Bayes , ∗ off ( s ) and V Bayes , ∗ ∞ , A -, 0 ( s ) are based on full prediction σ ∗ and partial one σ with predictable action a ∈ A -, satisfying:

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

The following lemma further bounds the Bellman-Jensen Gap due to partial action-coverage. Lemma E.4 (Proof in Appendix E.4) . For any predictable action set A -⊆ A , we have:

<!-- formula-not-decoded -->

where C 2 is an absolute constant, θ 2 max = max s, a 0: t ,t σ 2 ( r ( s t , a t | s 0 = s, a 0: t )) , where σ ( · ) denotes the sub-Gaussian parameter. Parameter θ 2 max then captures the variability of the reward.

Step 3: Bounding the Term T 3 . T 3 captures the value decay due to prediction errors. We Lemma E.5 (Proof in Appendix E.5) . For any prediction horizon K and predictable action set A -we have:

,

<!-- formula-not-decoded -->

where glyph[epsilon1] j := W ( d ) 1 ( P ∗ j , ̂ P j ) denotes the Wasserstein1 distance between the distributions P ∗ j , ̂ P j of the j -step predictive model under accurate and inaccurate prediction σ ∗ j and ˆ σ j , respectively, measured with respect to the base metric d ( σ j , σ ′ j ) = W 1 ( σ j , σ ′ j ) , i.e., the Wasserstein1 distance between single-step predictive distributions.

Step 4: Putting pieces together. Our last step is to combine the bounds we obtained for the terms T 1 , T 2 and T 3 to get the final results:

<!-- formula-not-decoded -->

This concludes the proof.

## E.1 Proof of Lemma E.1

First of all, since we work with bounded rewards, it is easy to see that V Bayes , ∗ k, A -, 0 ( s ) ≤ 1 / (1 -γ ) for any k ≥ 0 and s ∈ S . Next, we show that { V Bayes , ∗ k, A -, 0 ( s ) } k ≥ 1 is a Cauchy sequence, which implies its convergence [57].

Recall the Bellman equation for V Bayes , ∗ k +1 , A -, 0 :

<!-- formula-not-decoded -->

for any s ∈ S and k ≥ 0 . Therefore, we can truncate the tail term and have

<!-- formula-not-decoded -->

In addition, since the residual is positive, we have V Bayes , ∗ k +1 , A -, 0 ( s ) -V Bayes , ∗ k, A -, 0 ( s ) ≥ 0 . Together, they imply

<!-- formula-not-decoded -->

Similarly, we have

<!-- formula-not-decoded -->

and V Bayes , ∗ k, A -, 0 ( s ) -V Bayes , ∗ k, A -, 0 ( s ) ≥ -γ k . Together, they imply

<!-- formula-not-decoded -->

Combining Eq. (23) and (24), we have by triangle inequality that

<!-- formula-not-decoded -->

Now, for any glyph[epsilon1] &gt; 0 , choosing k such that 2 γ k (1 -γ ) 2 ≤ glyph[epsilon1] , then, for any m,n ≥ k (assuming without loss of generality that m ≥ n ), we have

<!-- formula-not-decoded -->

Therefore, for any s ∈ S , { V ∗ + k ( s ) } k ≥ 0 is a Cauchy sequence.

## E.2 Proof of Lemma E.2

To bound the value improvement from doubling the prediction window, we use a sub-Gaussian moment-generating function bound combined with a tailored Jensen-based analysis. For simplicity of presentation, we define an auxiliary Q -function, denoted by ˜ Q ∗ + K ( s, a , σ ) . Specifically, for all s ∈ S , a ∈ A K , and σ ∈ Q K ,

<!-- formula-not-decoded -->

Thus, for any s ∈ S and K ≥ 1 , we have

<!-- formula-not-decoded -->

This treatment allows us to focus on the value improvement from additional transition prediction of σ K +1:2 K . For any σ 1:2 K ∈ Q 2 K , we bound the term max a ˜ Q ∗ 2 K ( s, a 0:2 K -1 , σ 1:2 K ) -max a ˜ Q ∗ K ( s, a 0: K -1 , σ 1: K ) as follows:

<!-- formula-not-decoded -->

Applying the max-difference inequality on Eq. (26), we can conclude:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining Eq. (25) and (27), and take the maximal on states s on both sides, we have a simplified bound in a trajectory-wise reward form:

<!-- formula-not-decoded -->

where V ∗ s ∈ R ( |S||A| ) K is a trajectory-wise reward vector, defined as:

<!-- formula-not-decoded -->

The matrix P s ( σ K +1:2 K ) ∈ R |A| K × ( |S||A| ) K , with each entry denote the probability of visiting a state-action trajectory ( s K +1 , s K +2 , ..., s 2 K , a K , a K , ..., a 2 K -1 ) given initial state s , the action vector a K :2 K -1 and transition σ K +1:2 K . The matrix P s ( σ K +1:2 K ) = E σ K +1:2 K [ P s ( σ K +1:2 K )] .

Let X s, a K +1:2 K , σ K +1:2 K = ( P s ( σ K +1:2 K ) -P s ) V ∗ s | a K :2 K -1 , which denotes the a K :2 K -1 -th entry of the vector ( P s ( σ K +1:2 K ) -P s ) V ∗ s . We can verify that X s, a K +1:2 K , σ K +1:2 K is the cumulative discounted reward, whose absolute value is bounded by 1 1 -γ . For simplicity, we denote X s, a K +1:2 K , σ K +1:2 K by X s, a , σ . Therefore, X s, a , σ is a sub-Gaussian random variable, which implies

<!-- formula-not-decoded -->

Where θ 2 s, a ,K denotes the sub-Gaussian parameter [58] of X s, a , σ . Therefore, using Jensen's inequality and the monotonicity of exponential functions, we have for all λ :

<!-- formula-not-decoded -->

Taking the logarithm on both sides of the previous inequality, we have

<!-- formula-not-decoded -->

By choosing λ = √ 2 K log |A| max a θ 2 s, a ,K in Eq. (29), we have

<!-- formula-not-decoded -->

Substituting Eq. (30) into Eq. (28) yields:

<!-- formula-not-decoded -->

Due to the boundedness of the reward function, the sub-Gaussian max s, a θ 2 s, a ,K of trajectory can be upper bounded by:

<!-- formula-not-decoded -->

where C is an absolute constant.

Combining (32) and (31) yields our result.

## E.3 Proof of Lemma E.3

Using Lemma E.2 in Eq. (20) and , we have:

<!-- formula-not-decoded -->

Next, we focus on bounding the term ∑ ∞ i =0 ( γ K · 2 i √ 2 i ) . Denote a i = γ K · 2 i √ 2 i . Consider the index i ∗ = 7 20 log 2 ( 1 1 -γ ) , for any i ≥ i ∗ , we notice that:

<!-- formula-not-decoded -->

Hence, for any i ≥ i ∗ , we have:

<!-- formula-not-decoded -->

Summing a i from i = glyph[ceilingleft] i ∗ glyph[ceilingright] to infinity yields that:

<!-- formula-not-decoded -->

For the sum of a i from i = 0 to glyph[ceilingleft] i ∗ glyph[ceilingright]1 , we have:

<!-- formula-not-decoded -->

Then the desired sum satisfies:

<!-- formula-not-decoded -->

Hence, we can conclude that:

<!-- formula-not-decoded -->

where C 1 is an absolute constant.

## E.4 Proof of Lemma E.4

The desired gap satisfies:

<!-- formula-not-decoded -->

where the last inequality follows by exchanging the limit and expectation with the finite summation. Then we only need to bound each Q t separately. Specifically, for each Q t , it can be rewritten into:

<!-- formula-not-decoded -->

where r ∈ R |S||A| denotes the reward vector, with the ( s, a ) -th entry equal to r ( s, a ) ; P s, σ ∗ 1: t ∈ R |A| t +1 ×|S||A| and P s, σ 1: t ∈ R |A| t +1 ×|S||A| denote the random transition probability matrices from the initial state s to the state-action pair ( s t , a t ) under the action sequence a 0: t , specified by the transition predictions σ ∗ 1: t and σ 1: t , respectively.

Note that P s, σ 1: t differs from P s, σ ∗ 1: t only due to partial predictability. We can observe that the form in Eq. (34) is with the similar form as Eq. (28) in Appendix E.2. We can follow the same routine in the proof to Lemma E.2 in Appendix E.2 to handle Eq. (34). We can verify that for any action sequence a 0: t ∈ ( A -) t +1 , the corresponding rows of P s, σ 1: t and P s, σ ∗ 1: t are identical. Formally, for all σ ∗ 1: t , we have

<!-- formula-not-decoded -->

Hence, under any fixed t , the two t -step kernels P off , 1: t and P + , 1: t agree on every row corresponding to an action sequence in ( A -) t +1 . Since there are |A -| t +1 such sequences, the number of remaining mismatched rows is |A| t +1 - |A -| t +1 . Adding the trivial all-zero case yields at most |A| t +1 -|A -| t +1 +1 distinct nonzero differences between the two kernels.

Applying Lemma E.2 to each such mismatched row, we obtain for all t ≥ 1 :

<!-- formula-not-decoded -->

where C 2 is an absolute constant and θ 2 max = max s, a 0: t ,t σ 2 ( r ( s t , a t | s 0 = s, a 0: t )) , where σ ( · ) denotes the sub-Gaussian parameter. Parameter θ 2 max indicates the maximal variance of the reward.

Summing over t = 1 , 2 , . . . for Eq. (36) then gives

<!-- formula-not-decoded -->

This completes the proof.

## E.5 Proof of Lemma E.5

For clarity of presentation, for any Bayesian value function V Bayes ∈ R |S| and K -step transition prediction σ ∈ Q K , we define:

<!-- formula-not-decoded -->

Applying the Bayesian Bellman equation on T 3 , we have:

<!-- formula-not-decoded -->

where P ∗ 1: K and ̂ P 1: K denote the distributions of accurate and inaccurate K -step transition prediction σ .

Taking the absolute value on both sides and selecting the state s that maximizes it yields:

<!-- formula-not-decoded -->

Now we focus on ∣ ∣ ∣ E σ ∼P ∗ 1: K [ R ( s, V Bayes , ∗ K, A -, 0 , σ ) ] -E σ ∼ ̂ P 1: K [ R ( s, V Bayes , ∗ K, A -, 0 , σ ) ]∣ ∣ ∣ . The difference in this term comes from the prediction errors, which we use the Kantorovich-Rubinstein inequality to bound:

Lemma E.6 (Kantorovich-Rubinstein Inequality [59]) . Let ( X , d ) be a Polish metric space, and let µ, ν be two probability measures over X . Let f : X → R be a measurable function that is L -Lipschitz with respect to d , i.e.,

<!-- formula-not-decoded -->

Then the difference in expectations satisfies

<!-- formula-not-decoded -->

where W 1 ( µ, ν ) is the Wasserstein-1 distance defined by

<!-- formula-not-decoded -->

and Π( µ, ν ) denotes the set of all couplings (joint distributions) with marginals µ and ν .

We first show the perturbation sensitivity. A perturbation in σ j changes only the terms involving { s t } for t ≥ j . Its impact on the reward-sum ∑ K -1 t = j γ t r ( s t , a t ) is bounded by ∑ K -1 t = j γ t ≤ γ j -γ K 1 -γ , while its impact on the terminal term γ K V Bayes , ∗ K, A -, 0 ( s K ) is γ K max s | V Bayes , ∗ K, A -, 0 ( s ) | ≤ γ K 1 -γ . Hence, for each j ,

<!-- formula-not-decoded -->

Hence, we introduce the distance metric d ( σ , σ ′ ) = ∑ K j =1 γ j W 1 ( σ j ,σ ′ j ) 1 -γ . Under d , the above inequality simply says ∣ ∣ R ( s, V Bayes , ∗ K, A -, 0 , σ ) -R ( s, V Bayes , ∗ K, A -, 0 , σ ′ ) ∣ ∣ ≤ d ( σ , σ ′ ) , and R is 1-Lipschitz.

By the KantorovichRubinstein inequality in Lemma E.6, for two measures P ∗ 1: K , ̂ P 1: K on the product space,

<!-- formula-not-decoded -->

Since P ∗ 1: K = ⊗ K j =1 P ∗ j and ̂ P 1: K = ⊗ K j =1 ̂ P j , where P ∗ j and ̂ P j denote the distribution of σ j when prediction is accurate and inaccurate, respectively. One checks:

<!-- formula-not-decoded -->

Thus, we have:

<!-- formula-not-decoded -->

## F Proof of Theorem 5.1

Based on the proposed algorithm, the required sample amounts D 1 and D 2 satisfy:

<!-- formula-not-decoded -->

The first term D 1 represents the sample complexity of estimating the environment model. The term N 1 |S| ( |A| - |A -| ) denotes the learning cost transition kernel ̂ P ( s ′ | s, a ) for s ∈ S and a ∈ A \ A -, where |S| ( |A| - |A -| ) denotes the number of sampled entries, and N 1 denotes the number of times each entry is sampled. Learning the cost function only requires to sample each state action pair once with |S||A| samples. The second term D 1 directly equals the number of samples N 2 from the prediction oracle.

We need to determine appropriate values for N 1 and N 2 to ensure that the estimation error of ̂ V Bayes , ∗ K, A -,ε is smaller than glyph[epsilon1] .

Let's first analyze the structure of the error. For any state s ∈ S , the estimate Bayesian value function ̂ V Bayes , ∗ K, A -,ε ( s ) satisfies:

<!-- formula-not-decoded -->

where the estimated multi-step transition kernel ̂ P ( s t | s, a 0: t -1 , σ 1: t ) satisfies the following recursive condition:

<!-- formula-not-decoded -->

And ̂ P ( s t | s t -1 , a t -1 , σ t ) satisfies:

<!-- formula-not-decoded -->

We can see that, any term ̂ P ( s t | s, a 0: t -1 , σ 1: t ) presents finite-sample error only caused by the estimation error of ̂ P ( s t ′ | s t ′ -1 , a t ′ -1 ) with t ′ ≤ t . And the Bayesian value function estimation

̂ V Bayes , ∗ K, A -,ε ( s ) is influenced by the estimation errors of both ̂ P ( s t | s t -1 , a t -1 ) and ̂ P ( σ ) . To highlight such dependence, we define an auxiliary value function V ( P σ , P ( s,a ) , V Bayes , s ) as

<!-- formula-not-decoded -->

Hence, ̂ V Bayes , ∗ K, A -,ε ( s ) can be denoted by ̂ V Bayes , ∗ K, A -,ε ( s ) = V ( ̂ P ( σ ) , ̂ P ( s ′ | s, a ) , ̂ V Bayes , ∗ K, A -,ε , s ) to show the dependence on estimated probabilities and Bayesian value function. So the true Bayesian value function V Bayes , ∗ K, A -, ε ( s ) = V ( P ( σ ) , P ( s ′ | s, a ) , V Bayes , ∗ K, A -, ε , s ) .

Hence, for any state s ∈ S , the maximal estimation error can be decomposed by:

<!-- formula-not-decoded -->

Here, T 21 captures the bias in the estimated Bayesian value function, T 22 quantifies the error from imperfect predictiondistribution estimation, and T 23 reflects the error in onestep transition kernel estimation.

The remaining hurdle is to bound the terms T 21 , T 22 and T 23 , respectively. Lemmas F.1, F.2 and F.3 provide the desired bounds as follows:

Lemma F.1 (Proof in Appendix F.1) . For any state s , the term T 21 satisfies:

<!-- formula-not-decoded -->

Lemma F.2 (Proof in Appendix F.2) . With probability at least 1 -δ , the term T 22 satisfies:

<!-- formula-not-decoded -->

Lemma F.3 (Proof in Appendix F.3) . With probability at least 1 -δ , the term T 23 satisfies:

<!-- formula-not-decoded -->

Combining the results in Eq. (43), (44), (45) yields that, with probability at least 1 -δ

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Injecting Eq. (46) and (47) into Eq. (39) and combining the constants yields our result.

## F.1 Proof of Lemma F.1

For any fixed state s , the only difference between the two expressions in T 21 lies in their terminal Bayesian value function. Hence, by the γ K contraction property of the Bayesian Bellman operator, we immediately obtain:

<!-- formula-not-decoded -->

This concludes our proof.

## F.2 Proof of Lemma F.2

We denote X ( σ , s ) as the cumulative reward from state s with K -step transition prediction σ :

<!-- formula-not-decoded -->

where s 0 = s . Note that, for any state s , ̂ P ( s t | s, a 0: t -1 , σ 1: t ) and V Bayes , ∗ K, A -, ε ( s K ) , X ( σ , s ) is bounded with 0 ≤ X ( σ , s ) ≤ 1 1 -γ . Hence, for any state s ∈ S , we have:

<!-- formula-not-decoded -->

To bound the distribution estimation error. We reformulate Eq. (48) into a finite-sample Hoeffding form. Specifically, we define X ( s ) = ∫ σ 1: K ∈Q K P ( d σ 1: K ) X ( σ , s ) and X i = X ( σ i , s ) . Then, we

have:

<!-- formula-not-decoded -->

where E σ ( X i ( s )) = X ( s ) for all s . Applying the Hoeffding's inequality yields:

<!-- formula-not-decoded -->

where the first inequality holds because 0 ≤ X i ( s ) ≤ 1 (1 -γ ) for all i . Letting the RHS probability term be δ |S| , we apply the union bound across all different states s ∈ S yields:

<!-- formula-not-decoded -->

## F.3 Proof of Lemma F.3

This proof focuses on bounding how errors from inaccurate single-step transition kernels propagate to the long-term cumulative reward. The key idea is to decompose the total cumulative error into multiple time-dependent components and bound each individually.

## Step 1: Decompose the estimation error of ̂ P ( s t | s, a 0: t -1 , σ 1: t ) :

For any estimation ̂ P ( s t | s, a 0: t -1 , σ 1: t ) , the term T 23 is upper bounded by:

<!-- formula-not-decoded -->

Now we have

<!-- formula-not-decoded -->

Specifically, for any t , initial state s , the action sequence a 0: k -1 and prediction σ 1: K , ∆ t can be represented as the following product form:

<!-- formula-not-decoded -->

where vector P 0 ∈ R 1 ×|S| denotes the initial state s with the s -th entry equals 1 and the other entries equal 0 , r s max ∈ R |S|× 1 denotes the vector of reward with the s -th entry equals max a r ( s, a ) . Matrices ̂ P i : i +1 ∈ R |S|×|S| and P i : i +1 ∈ R |S|×|S| denote the estimated and real state transition probabilities from state s i to state s i +1 with given action a i .

Hence, ∆ t can be further decomposed by one-step errors as follows:

<!-- formula-not-decoded -->

Similarly, the terminal value error ∆ K can be decomposed as:

<!-- formula-not-decoded -->

Combining Eq. (51), (52) with Eq. (49), (50), we can decompose total error T 23 into:

<!-- formula-not-decoded -->

## Step 2: Bound individual error terms ∆ t,i and ∆ K :

We can show the concentration behavior of | ∆ t,i | as follows

<!-- formula-not-decoded -->

where r s t ,s max = ∏ t -1 k = i P k,k +1 · r s max ∈ R |S| denotes the expected reward vector from state s t , which satisfies | r s t ,s max | ∞ ≤ 1 for any t .

With N 1 samples on each substate-action pair ( s, a ) , we can directly bound ( ̂ P i -1: i -P i -1: i ) r s t ,s max by the Hoeffding's inequality:

<!-- formula-not-decoded -->

Weneed to ensure | ∆ t,i | ≥ glyph[epsilon1] for any ( s, a ) with s ∈ S and a ∈ A\A -, thus, we take 2 exp ( -2 glyph[epsilon1] 2 N 1 ) = δ |S| ( |A|-|A -| ) and yields that, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Following exactly the same routine, since ̂ V Bayes , ∗ K, A -,ε ( s ) ∈ [0 , 1 1 -γ ] , we have that, with probability at least 1 -δ , ∆ K,i satisfies:

<!-- formula-not-decoded -->

Step 3: Combine the pieces: Now it is enough to bound ∆ . Let the probability δ in Eq. (54)-(55) be δ K 2 , we have that, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

This concludes our proof.

## G Detailed Model of Wind-Farm Storage Control

We consider a windfarm operator that must deliver a precommitted power schedule to the grid while managing the uncertainty of realtime wind generation. To avoid costly imbalances, the farm uses a battery storage system: when actual output exceeds the commitment, excess energy is stored; when output falls short, the storage discharges to make up the difference. The operators goal is to minimize cumulative penalty costs for over or underdelivery by choosing charge/discharge actions based on observed prices, the current stateofcharge, and shortterm forecasts of wind generation.

Below, we formalize this sequential decision problem as an MDP.

## G.1 MDPFormulation for WindFarm Storage Control

We model sequential storage control as an MDP ( S , A , P , r, γ ) with:

- State : s t = ( p t , ∆ t , SoC t ) , where p t is the (identical) penalty price, ∆ t is the generation mismatch between generated wind power w t and required wind power ̂ w t , and SoC t is the storages state of charge.
- Action : a t = ( v + t , v -t ) , denoting charge/discharge amounts subject to v + t v -t = 0 , capacity, and generation constraints.
- Transition : the state transition probabilities satisfy:

<!-- formula-not-decoded -->

- Reward : the reward function is the negative penalty defined as:

<!-- formula-not-decoded -->

The dynamics of the storage control problem is visualized in Figure 3.

Figure 3: Wind Farm Storage Control

<!-- image -->

Figure 4: Prediction Helps More in Low-Value States. (a): The expected return at a low-value state increases significantly with prediction horizon K (b): The return at a high-value state also improves with K , but the marginal gain is much smaller.

<!-- image -->

## G.2 Parameter Settings

In the numerical study, we utilized the California aggregate wind power generation dataset from CAISO [60] containing predicted and real wind power generation data with a 5-minute resolution spanning from January 2020 to December 2020. The penalty price equals the average electricity price of CASIO [60] with the matching resolution and periods. We set C = 10 kWh, γ = 0 . 95 . The discretization levels of p , ∆ w and SoC are set to be 10 , 10 , 21 , respectively. The action set includes 9 discretized choices ranging from charging 2 KWh to discharging 2 KWh. The other parameters follow [61].

## H Additional Experiments

In this section, we conduct experiments to validate our theoretical findings. In particular, we demonstrate the advantage of Algorithm 1 for MDPs with transition predictions, compared with model-based RL for classical MDPs. For each experiment, we randomly generate 20 MDPs with |S| = 10 and |A| = 5 , and present the average performance of both approaches.

We first verify how predictions improve expected returns over standard MDPs, as predicted by our theoretical analysis in Section 4.2. Specifically, we examine two representative statesone with the lowest value and one with the highest value under the standard MDP value function. Figure 4(a) illustrates how incorporating transition predictions enhances the value functions of these two states. Even with a short prediction horizon K = 1 , we observe notable improvements over the MDP baseline. As K increases, the improvements also increase and tend to converge, which aligns with our theoretical findings. Interestingly, we find that low-value states benefit more from predictions than high-value states. In particular, for the low-value state, the expected return improves by 5 . 43% with K = 4 . In contrast, the corresponding gain for the high-value state is 2 . 75% . This is intuitive because predictions help guide the agent toward transitions that reach higher-value regions, offering more substantial gains for states with lower initial value.

Figure 5: Sample Efficiency of Learning with Predictions. (a): Convergence of (Bayesian) value function estimation error; (b): Policy performance measured by expected return. BOLA with predictions achieves similar convergence and consistently higher performance across varying sample sizes compared to classical model-based RL.

<!-- image -->

Figure 5 compares the learning efficiency of BOLA with K = 1 and standard model-based RL. In Figure 5(a), BOLA exhibits a faster decay in the value estimation error with fewer environment samples, indicating its sample efficiency is no worse than vanilla MDPs even with K = 1 . In Figure 5(b), we observe that BOLA consistently outperforms the model-based RL baseline. Notably, once the number of samples exceeds a small threshold 500 , the policy learned via BOLA yields higher expected return than what is maximally achievable by any MDP policy without predictive information. This highlights the fundamental advantage of incorporating predictions, which enables agents to surpass the conventional performance ceiling imposed by standard MDP frameworks.

## I Extension to Splitable State Modeling

Our model can be extended to Markov Decision Processes (MDPs) with splittable states , which naturally generalize to settings with predictable trajectories. A key feature of this extension is the decomposition of each state s ∈ S into two independent components, represented as a pair s = ( s m , s d ) , where s m ∈ S m is the Markovian substate , and s d ∈ S d is the dependent substate , with S = S m ×S d .

This modeling approach provides a natural way to incorporate exogenous or independently forecastable time seriessuch as demand, weather, or price signalsinto the decision-making process. Specifically, these predictable sequences can be encoded in the Markovian substate s m , allowing the agent to plan adaptively using trajectory-level predictions without enlarging the core Markov state space. The two substates are formally defined below.

Markovian Substates. The first type of substates, denoted by s m , is used to capture externally evolving states that do not depend on the agent's action, with several important real-world examples to be discussed in Section I.1. State transitions with respect to s m are Markovian such that they only depend on the previous substate. Formally, its transition kernel satisfies

<!-- formula-not-decoded -->

Dependent Substates. The transition of this substate, denoted by s d , depends on both past substate and action (like in classical MDPs). The state transitions with respect to s are

<!-- formula-not-decoded -->

The Markovian substate and the dependent substate are assumed to have independent transitions, i.e., for any s t = ( s t m , s d t ) ∈ S , a ∈ A , the overall transition probability P ( s t +1 | s t , a t ) satisfies P ( s t +1 | s t , a t ) = P ( s t m +1 | s t m ) P ( s d t +1 | s d t , a t ) 2 . Unlike Markovian substates, a dependent substate depends on both past action and substate, making it harder to predict. Next, we introduce the prediction model of the prediction-augmented MDPs considered in this work.

2 Without the predictive model P , the MDP model with Markovian and dependent substates is essentially a special case of the factored MDP [62].

Let P = ( K, A -, σ ) denote the prediction model that provides the predictions to the future states, where K is the length of the prediction window, A -⊆ A is a predictable action set, and σ denotes the transition prediction. Specifically, given t = 0 , K, 2 K,... , let s t = ( s t m , s d t ) be the current state of the environment. Before taking actions, the agent receives a probablistic prediction of the transition σ = ( σ 1 , σ 2 , · · · , σ K ) for the next K steps, where different σ k 's are independent and sampled from an unknown probability distribution P ( σ k ) . For each k ∈ { 1 , 2 , · · · , K } , σ k captures the transitions from state s t + k -1 to state s t + k , and is of the form σ k = ( σ m k , σ d k ) , where σ m k ∈ [0 , 1] |S|×|S| is an |S| × |S| -dimensional matrix representing the transition prediction of the Markovian substate and σ d k ∈ [0 , 1] |S||A -|×|S| is an |S||A -| by |S| matrix representing the transition prediction of the dependent substate. Given a prediction σ = ( σ m , σ d ) , the transition probabilities satisfy the following.

Fully Predictable Markovian Substates. For Markovian substates, we have

<!-- formula-not-decoded -->

where σ m ( s t m , s t m +1 ) is the ( s t m , s t m +1 ) -th entry of the matrix σ m .

Partially Predictable Dependent Substates. We consider a general setting that allows for partially predictable states and actions. In particular, given a set of predictable actions A -, we have

<!-- formula-not-decoded -->

where σ d t +1 (( s d t , a t ) , s d t +1 ) denotes the entry located in the ( s d t , a t ) -th row and the s d t +1 -th column of the matrix σ d t +1 .

## I.1 Illustrative Examples

Extending classic MDPs, the prediction-augmented MDP model introduced in Section 2.1 naturally fits real-world scenarios with Markovian states. Examples include stock prices in stock market trading [8], outdoor temperatures in building thermal control [63], wind speeds in unmanned aerial vehicle (UAV) control [64], electricity prices in storage control [65], and grid electricity demands in power system economic dispatch [66].

Table 1: Real-world examples that instantiate the prediction-augmented MDP model (see Section 2.1).

|                                                                                  | Action                                                                             | Markovian Substate predictable                                            | Dependent Substate partially predictable                                           |
|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------|---------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Stock Investment VPP Operation Building HVAC Control Storage Control UAV Control | Buy/sell stocks Energy consumption Heating/cooling Charge/discharge DC motor force | Stock price Renewable generation N/A Energy mismatch Wind direction/speed | N/A Electricity price Indoor/outside temperature Battery SoC UAV position/attitude |

Stock Investment. Consider the stock investment problem for a retail investor [8]. The market stock prices are action-independent unknown time series if the trading volume is high. With relatively accurate stock price predictions, the revenue from the investment can be significantly improved.

Virtual Power Plant operation. Another example is the virtual power plant (VPP) operation problem [67], where the agent sequentially decides the energy consumption of a large-scale VPP to minimize total electricity costs. In this scenario, the renewable generation within the VPP depends solely on its previous state and can be effectively predicted. Conversely, the real-time electricity price in the electricity market depends on both its previous state and the VPP's energy consumption actions. The electricity price is partially predictable: when the VPP's energy consumption is low, its impact on market prices is minimal and predictable. However, when energy consumption is very high, the VPP becomes a market price-maker, causing market prices to fluctuate wildly and become unpredictable.

Building HVAC Control. Besides, many online decision-making problems related to sustainability exhibit predictable structures aligning with our model. For example, the control of heating, ventilation, and air conditioning (HVAC) systems relies on temperature predictions [68], battery storage

management depends on energy predictions [69]. Similar predictable components exist for the task of controlling battery storage systems.

In Table 1, we summarize the key features of real-world scenarios with Markovian substates and partially predictable dependent substates, which present challenges for modeling with classic MDPs.