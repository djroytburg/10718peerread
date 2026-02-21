## Optimal Dynamic Regret by Transformers for Non-Stationary Reinforcement Learning

## Baiyuan Chen

The University of Tokyo chenbaiyuan75@g.ecc.u-tokyo.ac.jp

## Shinji Ito

The University of Tokyo / RIKEN AIP shinji@mist.i.u-tokyo.ac.jp

## Masaaki Imaizumi

The University of Tokyo / RIKEN AIP

imaizumi@g.ecc.u-tokyo.ac.jp

## Abstract

Transformers have demonstrated exceptional performance across a wide range of domains. While their ability to perform reinforcement learning in-context has been established both theoretically and empirically, their behavior in nonstationary environments remains less understood. In this study, we address this gap by showing that transformers can achieve nearly optimal dynamic regret bounds in non-stationary settings. We prove that transformers are capable of approximating strategies used to handle non-stationary environments and can learn the approximator in the in-context learning setup. Our experiments further show that transformers can match or even outperform existing expert algorithms in such environments.

## 1 Introduction

Transformers have emerged as a powerful class of sequence models with remarkable expressive capabilities. Originally popularized in the context of natural language processing, they leverage self-attention mechanisms to in-context learn new tasks without any parameter updates (Vaswani et al., 2017; Liu et al., 2021; Dosovitskiy et al., 2021; Yun et al., 2019; Dong et al., 2018). In other words, a large transformer model can be given a prompt consisting of example input-output pairs for an unseen task and subsequently produce correct outputs for new queries of that task, purely by processing the sequence of examples and queries (Lee et al., 2022; Laskin et al., 2023; Yang et al., 2023; Lin et al., 2024). This ability to dynamically adapt via context rather than gradient-based finetuning has spurred extensive interest in understanding the theoretical expressivity of transformers and how they might learn algorithms internally during training.

Recent theoretical work has begun to analyze the various aspects of transformers. On the expressiveness front, transformers have been shown to be universal or near-universal function approximators under various conditions (Yun et al., 2020). Beyond mere approximation of static functions, researchers have demonstrated that transformers can encode and execute entire computational procedures. For example, by carefully setting a transformer's weights, one can make its forward pass simulate iterative algorithms such as gradient descent and many others (Bai et al., 2023; Cheng et al., 2024; Huang et al., 2025).

Studies have shown that transformers can approximate various reinforcement learning (RL) algorithms. The question here is whether a transformer can be trained to act as an RL algorithm when presented with an interaction history as context. Lin et al. (2024) shows that a sufficiently large transformer model, after being pre-trained on offline interaction sequences, can approximate near-

| Study          | Lin et al. (2024) (LinUCB)   | Lin et al. (2024) (TS)   | Ours                                     |
|----------------|------------------------------|--------------------------|------------------------------------------|
| Dynamic Regret | O p T q                      | O p ∆ 1 { 4 T 3 { 4 q    | O p min t ? JT, ∆ 1 { 3 T 2 { 3 ` ? T uq |

Table 1: Upper bounds on dynamic regret of transformers in the non-stationary setup. T is the time budget, and ∆ and J represent the amount and the number of changes in the environment, respectively. LinUCB suffers from O p T q dynamic regret in non-stationary environments. The rate of Thompson Sampling (TS) is from Kim and Tewari (2020), whose algorithm is approximated by transformers through the analysis of Lin et al. (2024). The bound for ours shows the optimal rate derived in Besbes et al. (2014).

optimal online RL algorithms like LinUCB and Thompson Sampling in a multi-armed bandit, as well as UCB value iteration for tabular Markov decision processes.

Despite these encouraging results, an important open challenge remains: Can transformers adapt to non-stationary environments? Most existing theoretical analyses assume a fixed task or reward distribution during the transformer's in-context inference (Mao et al., 2021; Domingues et al., 2021). However, in many real-world scenarios, the environment is non-stationary-the reward-generating process evolves over time, violating the assumptions of a static underlying model. In multi-armed bandits, for instance, the true reward probabilities of the arms might drift or change abruptly due to seasonality or shifts in user preferences. Classical algorithms for non-stationary bandits-such as sliding-window UCB, periodic resets, or change-point detection-are known to achieve sublinear dynamic regret that closely tracks the moving optimum (Besbes et al., 2014; Garivier and Moulines, 2011; Wei and Luo, 2021; Mao et al., 2021; Cheung et al., 2020). However, the behavior of transformers in non-stationary environments remains largely unexplored.

Our study. In this paper, we investigate whether a transformer, trained in a supervised in-context learning setting, can learn in non-stationary environments. Specifically, we introduce a transformer architecture designed to adapt to non-stationarity, and we analyze both the approximation error and the generalization error arising from learning it from data. Based on these results, we derive an upper bound for the dynamic regret of the transformer and show its optimality.

Our main contributions are summarized as follows:

1. Regret optimality in non-stationary bandits: We prove that a transformer acting as an incontext bandit learner can achieve cumulative regret matching the minimax optimal rate for non-stationary environments (e.g., T 2 { 3 for bounded changes) (Besbes et al., 2014). Rates are summarized in Table 1.
2. Generalization analysis under distribution shifts: Extending the offline-to-online generalization theory of Lin et al. (2024) (Lin et al., 2024), we derive conditions on pretraining data diversity and model size needed to guarantee low regret in new non-stationary environments.
3. Architectural requirements for implementation: We identify key features-depth, attention heads, and activations-that enable a transformer to forget outdated information and implement restart-style strategies akin to handling non-stationary environments.
4. Novel proof technique via internal algorithm selection: We develop a proof approach that treats the transformer's hidden state as maintaining multiple hypothesis policies and uses learned randomness in self-attention to sample among them, yielding tight regret bounds.

## 1.1 Related Works

In-context learning In-context learning (ICL) was first introduced by Brown et al. (2020), allowing large language models (LLMs) to learn tasks by using a few demonstration examples without fine-tuning or updating model parameters. Since its introduction, researchers have studied the properties and mechanisms of ICL in depth (Mao et al., 2024; Zhao et al., 2024; Mosbach et al., 2023; Singh et al., 2024; Bertsch et al., 2025). For example, Mao et al. (2024) adopts a data generation perspective, showing that skill recognition and skill learning abilities in LLMs can transfer between tasks. Zhao et al. (2024) examines the decision boundaries of LLMs in classification tasks and finds that, while in-context examples can improve accuracy, the models often display fragmented decision boundaries, meaning small input changes can lead to different outputs. Moreover, like fine-tuning,

ICL can be unstable and perform poorly on both in-domain and out-of-domain data, although finetuning tends to generalize more effectively (Mosbach et al., 2023).

Approximation capability of transformers Transformers have been widely applied to tasks such as reinforcement learning, computer vision, graph processing, and speech recognition (Chen et al., 2021; Lin et al., 2024; Liu et al., 2021; Dosovitskiy et al., 2021; Yun et al., 2019; Min et al., 2022; Dong et al., 2018). For instance, Lin et al. (2024) explores using transformers for in-context reinforcement learning, Chebotar et al. (2023) applies them to model Q-functions for multi-task policies, and Zhao et al. (2022) combines vision transformers with reinforcement learning for video captioning. While transformers have shown great potential, their limitations have also been discussed in the literature (Hahn, 2020; Bhattamishra et al., 2020).

Non-stationary Reinforcement Learning Non-stationary reinforcement learning (RL) addresses scenarios where rewards and transitions evolve over time, with the degree of difficulty often characterized by the frequency and magnitude of these changes (Auer et al., 2019; Chen et al., 2019; Cheung et al., 2020). To tackle this challenge, Wei and Luo (2021) propose a method that periodically schedules instances with a restart mechanism, while Cheung et al. (2020) advocate leveraging recent data and employing wider confidence intervals to adapt to shifting conditions. Meta-reinforcement learning approaches have also been explored in non-stationary environments (Bing et al., 2022). Other research efforts address non-stationary settings as well, although many rely on prior knowledge, such as the number or extent of environmental changes, to achieve effective adaptation (Mao et al., 2021; Li and Li, 2019; Domingues et al., 2021).

## 1.2 Notation

We use the following notations throughout the paper. r s, e s for s, e P N with s ď e denotes the integer interval t s, s ` 1 , . . . , e u . ∆ p X q denotes the space of probability distributions over a finite set X . For sets X 1 , . . . , X n , we write Â n i ' 1 X i : ' X 1 ˆ¨¨¨ ˆ X n to denote the Cartesian product. d denotes element-wise multiplication, and À denotes concatenation. } ¨ } op denotes the operator norm of a matrix. For an event E , we use 1 E r x s to denote an indicator function on E , which equals 1 if x P E , and 0 otherwise. O p¨q is Landau's Big-O notation which hides some absolute constant, and r O p¨q additionally hides logarithmic terms. r a ; b s denotes that a and b are stacked vertically.

## 2 Problem Setup

## 2.1 Learning Setup

We first define an environment of reinforcement learning following Lin et al. (2024). Let T P N be a number of rounds, and define a tuple of state/action/reward spaces t S t , A t , R t u t Pr T s , a transition model T t : S t ˆ A t Ñ ∆ p S t ` 1 q with S 0 , A 0 ' tHu , and the reward function R t : S t ˆ A t Ñ ∆ p R t q . We assume a finite action space A ' | A t | ă 8 .

Next, we define a learning scheme for the environment. Let D t ' p s j , a j , r j q t j ' 1 Ă b t j ' 1 S j ˆ A j ˆ R j be a sequence of observed state/action/reward tuples, where r t ' R t p s t , a t q . We define an algorithm alg : pb t ´ 1 j ' 1 S j ˆ A j ˆ R j q ˆ S t Ñ ∆ p A t q . Then, we obtain a distribution function over a trajectory D T as

<!-- formula-not-decoded -->

For training, we assume a base algorithm alg B that provides high-quality actions and serves as a guiding policy for training the transformer-based algorithm.

Given N i.i.d. trajectories D i T ' p s i 1 , a i 1 , r i 1 , ..., s i T , a i T , r i T q i Pr N s ' P alg 0 with an offline algorithm alg 0 , we augment each D i T by a i t ' i.i.d. alg B p¨| D i t ´ 1 , s i t q t Pr T s with a base algorithm alg B and denote it as D i T . The base algorithm can observe the full trajectory and environment to generate actions for the supervised learning of alg .

0

Using the reward function defined above, we further define non-stationary environments following Wei and Luo (2021).

Definition 1 (Non-stationary measure) . A function ∆ : r T s Ñ R is defined as a non-stationarity measure if it satisfies

<!-- formula-not-decoded -->

for all t . Given any interval I ' r s, e s , we define ∆ I : ' ř e ´ 1 τ ' s ∆ p τ q and J I : ' 1 ` ř e ´ 1 τ ' s 1 r ∆ p τ q ‰ 0 s . We denote ∆ ' ∆ r 1 ,T s and J ' J r 1 ,T s as the amount and number of changes in the environment, respectively.

Our goal is to develop a bound on the dynamic regret of transformers under the non-stationary setup. In preparation, we define the maximal achievable reward r ˚ t : ' max s P S t ,a P A t R t p s, a q at round t , its accumulated version R ˚ p T q ' ř T τ ' 1 r ˚ τ , and the expected reward R alg p T q ' E D T ' P alg ' ř T τ ' 1 r τ ı for an algorithm alg . The regret with an algorithm alg is then defined as

<!-- formula-not-decoded -->

We will show that transformers can effectively solve the non-stationary data problem.

## 2.2 Transformer

We define the architecture of a transformer. Given an input sequence H P R d ˆ n , where d is the feature dimension and n is the sequence length, consider a masked attention layer with M heads. Denote the parameters as tp V m , Q m , K m qu m Pr M s Ă p R d ˆ d q ˆ 3 . The output of the masked attention layer is denoted as H ' Attn θ p H q ' r h 1 , . . . , h n s P R d ˆ n , where each h i for i P r n s is given by:

<!-- formula-not-decoded -->

where ReLU p x q ' max t 0 , x u . With slight abuse of notation, we write h i ' Attn θ p h i , H q . Next, the attention output is fed into a feedforward MLP layer MLP θ mlp p¨q with parameter θ mlp ' p W 1 , W 2 q P R d 1 ˆ d ˆ R d 1 ˆ d , where h i ' h i ` W 2 ¨ ReLU p W 1 h i q P R d , and D 1 is the hidden dimension of the MLP layer, a transformer with L layers is then defined as TF θ p H q ' H p L q , where for ℓ P r L s :

<!-- formula-not-decoded -->

The transformer parameters are collectively denoted as θ ' p θ p 1: L q attn , θ p 1: L q mlp q . The attention parameters are θ p ℓ q attn ' tp V p ℓ q m , Q p ℓ q m , K p ℓ q m qu m Pr M s Ă R d ˆ d , and the MLP parameters are θ p ℓ q mlp ' p W p ℓ q 1 , W p ℓ q 2 q P R d 1 ˆ d ˆ R d ˆ d 1 . The norm of a transformer TF θ is denoted as

<!-- formula-not-decoded -->

where } ¨ } op is the operator norm.

## 2.3 Algorithm induced by transformers in In-Context Learning

Let s t P S t and p a t , r t q P A t ˆ R t be embedded by h : Ť t Pr T s S t Y Ť t Pr T s p A t ˆ R t q Ñ R d such that h p s t q P R d and h p a t , r t q P R d . For the input H ' r h p s 1 q , h p a 1 , r 1 q , . . . , h p a t ´ 1 , r t ´ 1 q , h p s t qs P R d ˆp 2 t ´ 1 q , the transformer outputs H ' TF θ p H q ' r h 1 , h 2 , . . . , h 2 t ´ 1 s P R d ˆp 2 t ´ 1 q . By introducing a linear mapping A P R A ˆ d to extract a distribution over A t with | A t | ' A actions, the algorithm induced by the transformer is defined as:

<!-- formula-not-decoded -->

For convenience, we denote alg θ ' alg θ p¨| D t ´ 1 , s t q .

In supervised pretraining, the transformer parameters are learned by maximizing the log-likelihood over the algorithm class t alg θ u θ P Θ :

<!-- formula-not-decoded -->

where Θ : ' t θ ' p θ p 1: L q attn , θ p 1: L q mlp qu is the finite parameter class of transformers.

## 3 Main Result

We show that under non-stationary environments, transformers trained with p θ in (6) can approximate the reduced base algorithm alg B ' E D T ' alg 0 r alg t B p¨| D T , M q| D t ´ 1 , s t s . This is achieved by bounding the dynamic regret of transformers.

## 3.1 Preparation

We first introduce definitions required for the approximation theorem. In particular, we define the covering number to quantify the complexity of a class of algorithms. This concept is standard in learning theory (Anthony and Bartlett, 2009) and has been used in analyses of transformers (Lin et al., 2024).

Definition 2 (Covering number) . A finite subset Θ 0 Ď Θ is called a ρ -cover of the algorithm class t alg θ : θ P Θ u if, for every θ P Θ , there exists θ 0 P Θ 0 such that:

<!-- formula-not-decoded -->

The covering number N Θ p ρ q is the minimal cardinality of any such ρ -cover Θ 0 .

Next, we define the distribution ratio as a measure of discrepancy between two algorithms, which is used in Lin et al. (2024).

Definition 3 (Distribution ratio) . The distribution ratio of algorithms alg 1 and alg 2 is defined as:

<!-- formula-not-decoded -->

Many reinforcement learning algorithms generate an auxiliary value at each round, for instance the upper confidence bound (UCB) in UCB-based algorithms. Let r R t p s t , a t q denote such an auxiliary value at round t , under state s t P S t and action a t P A t . For simplicity, we write r r t : ' r R t p s t , a t q . In transformer-based models, this auxiliary value can be interpreted as the output probability distribution that guides action selection. To introduce the approximation theorem, we introduce the following assumptions used in the non-stationary analysis, e.g., Wei and Luo (2021).

Assumption 1 (Non-stationarity) . For all θ P Θ , alg θ outputs an auxiliary quantity r r t P r 0 , 1 s at the beginning of each round t . There exists a non-stationarity measure ∆ and a non-increasing function ρ : r T s Ñ R such that running alg θ satisfies that without knowing ∆ r 1 ,t s , alg θ ensures with probability at least 1 ´ 1 { T :

<!-- formula-not-decoded -->

for all t P r T s , provided that ∆ r 1 ,t s ď ρ p t q and ρ p t q ě 1 { ? t . Additionally, we assume the function C p t q : ' tρ p t q is non-decreasing.

In addition, we assume that the base algorithm generating the observation sequence can be approximately realized by a transformer, which is the following assumption. This assumption has been proved under several traditional RL algorithms such as LinUCB and Thompson Sampling (Lin et al., 2024). For additional discussion of this assumption, see Appendix A.

Assumption 2 (Realizability) . Given a reinforcement learning algorithm alg B , for any ε ą 0 , there exist a parameter θ such that the following holds: there exist constants D 0 , M 0 , L 0 , D 1 0 , C 0 ą 0 , which depend on ε , such that a transformer TF θ with D ' D 0 , M ' M 0 , L ' L 0 , D 1 ' D 1 0 , ~ θ ~ ' C 0 , satisfies

<!-- formula-not-decoded -->

for all rounds t and actions a t,k .

## 3.2 Regret Bound

As our main result, we develop an upper bound on the dynamic regret of the transformer-induced policy alg p θ . This theorem quantifies how well a transformer-based policy performs in a nonstationary environment compared to an optimal baseline algorithm. The proof is provided in Appendix D.2.

Theorem 1. Let alg B be a reinforcement learning algorithm satisfying Assumptions 1 and 2. Suppose Assumption 1 holds with C p t q ' t p for some p P r 1 { 2 , 1 q , and let p θ be a solution of (6) . Assume that | r t | ď 1 holds almost surely, and define N Θ : ' N Θ pp NT q ´ 2 q . Then, for any ε ą 0 , the dynamic regret of alg p θ induced by a transformer with D ď D 0 ` 10 , M ' max t M 0 , 2 u , L ' L 0 , D 1 ' O p max t D 1 0 , T log 2 T { ε uq , ~ p θ ~ ' r O p C 0 ` a T log 2 T { ε q satisfies the following with probability at least 1 ´ max t δ, 1 { T u with R : ' R alg p θ , alg B :

<!-- formula-not-decoded -->

This bound consists of three main components T comp , T aprx , and T algo . T comp captures the effect of pretraining the transformer on N supervised samples to learn the base distribution. T aprx represents the approximation error incurred when using the transformer to imitate the base policy in the nonstationary setting. T algo corresponds to the intrinsic regret of the base algorithm in a non-stationary environment.

Theorem 1 shows that the transformer-based algorithm performs nearly optimally even in nonstationary environments, with the error diminishing as the training dataset grows and the environment stabilizes. Specifically, when the number of observations N is sufficiently large and the approximation error ε is sufficiently small, the regret bound in (8) is dominated by the third term T algo . To further clarify this, we present the following corollary, whose proof is contained in Appendix D.3.

Corollary 2. Consider the setup in Theorem 1. There exists a universal constant C such that, for N ě CT 3 log T , and ε ď T ´ 3 , the dynamic regret of alg p θ satisfies

<!-- formula-not-decoded -->

This result implies that, with a sufficiently large dataset and a sufficiently expressive transformer (small ε ), the transformer-based policy achieves the optimal dynamic regret in non-stationary environments, matching the bounds established by prior works (Wei and Luo, 2021). Intuitively, smaller ε corresponds to larger transformer architectures, as indicated in Theorem 1. Hence, with enough data and a suitably large architecture, the transformer is capable of realizing the optimal strategy that minimizes dynamic regret.

While the above bounds could also be achieved with a standard transformer, for the sake of simpler proofs we use a non-continuous transformer . Structurally, this model is equivalent to a regular transformer except that its inputs are duplicated a few times to form an augmented input sequence. The motivation for this construction, along with a discussion on how the proofs extend to standard transformers, is provided in Appendix E.2. An abstract illustration of the non-continuous transformer is shown in Figure 1.

## 4 Proof Outline

In this section, we provide an overview of the proof of Theorem 1. The most critical component is the analysis of the approximation error T aprx . To this end, we first introduce common operations and algorithmic structures used to handle non-stationary data, and then show how a transformer can approximate these operations effectively.

## 4.1 Operation for Non-Stationary Environments

In non-stationary environments, maintaining strong performance requires limiting the reliance on historical data . Existing approaches primarily employ two strategies: using a window scheduler , where only data within a sliding window is available to the model (Cheung et al., 2022; Trovo et al., 2020), and a test-and-restart mechanism , which resets the algorithm when significant changes in the environment are detected (Gomes et al., 1998; Cheung et al., 2020; Cayci et al., 2020; Wei and Luo, 2021; Mao et al., 2021). Among these approaches, the MASTER algorithm (Wei and Luo, 2021) stands out. MASTER not only achieves near-optimal performance in non-stationary environments but also does not require prior knowledge of the environment's change parameters, such as the number or magnitude of changes.

We briefly describe the window scheduler and the restarting operation in MASTER, while their formal definitions will be provided in Appendix E.1. The overview of the proof is illustrated in Figure 1.

Figure 1: Overview of the proof structure for the approximation analysis (Theorem 5). The blue box represents the algorithmic operations for handling non-stationarity, while the green box represents the transformer approximating the algorithm. Each block or sub-architecture of the transformer corresponds to a specific operation of the algorithm.

<!-- image -->

Window scheduler: Denote n ` 1 copies of trajectories D T as t D p i q T u n i ' 0 . Given window sizes t W p i q u n i ' 0 and probabilities t p p i q u n i ' 0 , we define the following two modules: (i) a masking operator, and (ii) a selection operator.

First, the masking operator σ 1 maps the input trajectory D T to a randomly masked version of the copies t D p i q T u n i ' 0 , using the window sizes and probabilities. Formally,

<!-- formula-not-decoded -->

where m p i q P t 0 , 1 u T is a binary mask for copy i defined elementwise as

<!-- formula-not-decoded -->

This operator probabilistically selects candidate time points of interest within each copy based on the window size W p i q and time position t .

Second, the selection operator σ 2 chooses a single active trajectory instance among the masked copies:

<!-- formula-not-decoded -->

where k ' min t j P t 0 , . . . , n u : p s p j q t , a p j q t , r p j q t q ‰ 0 u is the index of the first non-zero entry (orderk ). The selected instance is called 'active', while the others are 'inactive'. This ensures that only one instance is active at each round, and it is always the lowest-order scheduled instance.

Finally, the window scheduler ( WS ) is defined as:

<!-- formula-not-decoded -->

The window scheduler stochastically schedules multi-scale windows to restrict the algorithm to recent data (Figure 2).

Figure 2: Illustration of WS when n ' 3 . Purplish blocks represent instances scheduled by σ 1 , while reddish blocks represent the active instances selected by σ 2 . Reddish blocks connected by a dashed line are concatenated.

<!-- image -->

Stationary Test and Restart Mechanism: After passing through the window scheduler WS , the base algorithm interacts with the environment, receives a reward, and performs a stationary test , which outputs a change-detection signal C t P t 0 , 1 u . If a significant change in the environment is detected ( C t ' 1 ), the algorithm is restarted. We denote the output of WS as the sequence of active instances tp s 1 t , a 1 t , r 1 t qu T t ' 1 . We then define the auxiliary memory as A t : ' tp s 1 p , a 1 p , r 1 p , r r p qu t p ' 1 , where the auxiliary value r r t is added at each step.

Given the change-detection signal C t P t 0 , 1 u and auxiliary memory A t , the restart mechanism ( RM ) is defined as:

<!-- formula-not-decoded -->

where A 0 is an empty/reset auxiliary state (all counts= 0 , rewards= H ), and U A updates the auxiliary memory (e.g., appending new rewards). Typically, r r t is evaluated to determine when a restart is necessary.

It is worth noting that the sliding window and restart mechanisms used in previous works (e.g., (Cheung et al., 2022; Trovo et al., 2020; Cayci et al., 2020)) can be considered special cases of WS and RM . We discuss their equivalence in Appendix E.4. As a result, this approximation can be extended to a wide range of algorithms designed to handle non-stationarity.

## 4.2 Transformers for Approximating the Operations

Our goal is to demonstrate that transformers can approximate the above techniques and the overall algorithm. Given a base algorithm alg B , we have the following statement.

Proposition 3 (Approximating WS ) . For any small ε ą 0 , there exists a transformer TF θ p¨q with D ' O p 1 q , L ' O p 1 q , M ' O p 1 { ? ε q , D 1 ' O p a T log 2 T { ε q such that the algorithm alg θ , defined in (5) , can generate a random number drawn from a uniform distribution and satisfies

<!-- formula-not-decoded -->

This proposition shows the existence of a transformer block (called the scheduler block) that approximates the window scheduler WS . In the construction, we also include a block that extracts randomness from the data to implement the stochastic scheduling in WS . The proof of Proposition 3 is provided in Appendix E.3.

Furthermore, since RM involves only zeroing out certain values and applying a mask 1 ą 0 r¨s to detect changes, its approximation by a transformer follows directly. Details for approximating RM are included in Appendix E.1 as part of the overall proof.

Proposition 4 (Approximating RM ) . The restart mechanism RM can be implemented by a two-layer MLP with D 1 ' O p 1 q .

Finally, we consider an expert algorithm alg E , which incorporates both the window scheduler and the restart mechanism to handle the non-stationary environment. The full details of this algorithm are provided in Section C. Then, the following theorem holds:

Theorem 5. For any ε ą 0 , there exists a transformer TF θ with D ď D 0 ` 10 , M ' max t M 0 , 2 u , L ' L 0 , D 1 ' O p max t D 1 0 , T log 2 T { ε uq , ~ θ ~ ' r O p C 0 ` a T log 2 T { ε q , such that the transformer-based algorithm alg θ defined in (5) satisfies

<!-- formula-not-decoded -->

With this result, we obtain the approximation error T aprx . The entire proof is shown in Figure 1. The proofs of Theorem 5 is contained in Appendix D.4.

Together with the analyses of the training error T comp and the algorithmic regret T algo , Theorem 5 shows that transformers can effectively replicate the techniques required for handling non-stationary environments, completing the argument for Theorem 1. Full technical details are provided in the supplementary material.

## 5 Experiment

In this section, we evaluate transformers and other algorithms in a linear bandit setting. The stochastic linear bandit framework is given by M ' p w ˚ , E , A 1 , . . . , A T q . At each round t P r T s , the learner selects an action a t P R d from the set A t ' t a t, 1 , . . . , a t,A u , which may vary over time. The learner then receives a reward r t ' x a t , w ˚ y ` ε t , where ε t ' i.i.d. E and w ˚ P R d is unknown. The problem generalizes by setting s t ' A t , with the state transitioning deterministically to s t ` 1 regardless of the action.

We compare transformers against Linear UCB (LinUCB) and Thompson Sampling (TS), as well as MASTER (Wei and Luo, 2021) combined with LinUCB/TS (denoted as expert algorithms) under environments with varying degrees of non-stationarity. In our experiments, we set d ' 32 , A ' 10 , ε t ' N p 0 , 1 . 5 2 q , and w ˚ ' Unif( r 0 , 1 s d ). We consider two types of environments:

- (1) Low Non-Stationarity: Models are evaluated over 1,000 rounds, with elevated rewards in t P r 50 , 100 s Y r 350 , 400 s scaled to r t P r 3 , 4 s , and the remaining rewards in r t P r 0 , 1 s . Training data consists of 100,000 samples with normalized rewards r t P r 0 , 1 s .
- (2) High Non-Stationarity: The reward is defined as r t ' px a t , w ˚ y ` ε t q cos p 2 πbt q . For training, we generate 100,000 samples for each b P t 0 . 005 , 0 . 01 , 0 . 015 , 0 . 02 u . For evaluation, we test on unseen environments with b P t 0 . 018 , 0 . 025 u , running 200 rounds per environment.

For transformer models, we use GPT-2 with L ' 16 layers and M ' 16 attention heads, trained for 200 epochs. The objective is to assess generalization to non-stationary environments unseen during training.

From Figure 3, we observe that transformers achieve performance comparable to, and sometimes surpassing, both the expert algorithms and their MASTER variants, attaining near-optimal cumulative regret. Additional experimental results can be found in Appendix B.

## 6 Conclusion

In this study, we have shown that transformers can effectively handle non-stationary environments, achieving near-optimal performance by minimizing dynamic regret. By demonstrating that transformers can implement strategies commonly used for adapting to non-stationarity, we provide a theoretical guarantee for their dynamic regret bound. Our experiments further show that transformers can match, and in some cases outperform, existing expert algorithms. As a limitation, we evaluated transformers empirically only against two RL algorithms and their variants in the linear bandit setting. Future work should explore additional experimental setups and broader algorithm comparisons to further validate our findings.

Figure 3: Cumulative regret comparison for LinUCB, Thompson Sampling (TS), MASTER+LinUCB/TS, and transformer (TF) in linear bandits d ' 32 , A ' 10 . The first row corresponds to Low Non-Stationarity environments, while the second row shows High Non-Stationarity environments. Shading indicates the standard deviation of the regret estimates.

<!-- image -->

## Acknowledgements

Shinji Ito was supported by JSPS KAKENHI (JP25K03184). Masaaki Imaizumi was supported by JSPS KAKENHI (JP24K02904), JST CREST (JPMJCR21D2), and JST FOREST (JPMJFR216I).

## References

- Anthony, M. and Bartlett, P. L. (2009). Neural Network Learning: Theoretical Foundations . cambridge university press.
- Auer, P., Gajane, P., and Ortner, R. (2019). Adaptively tracking the best bandit arm with an unknown number of distribution changes. In Conference on Learning Theory , pages 138-158. PMLR.
- Bai, Y., Chen, F., Wang, H., Xiong, C., and Mei, S. (2023). Transformers as statisticians: Provable in-context learning with in-context algorithm selection. Advances in Neural Information Processing systems , 36.
- Bertsch, A., Ivgi, M., Xiao, E., Alon, U., Berant, J., Gormley, M. R., and Neubig, G. (2025). In-context learning with long-context models: An in-depth exploration. In Association for Computational Linguistics , pages 12119-12149.
- Besbes, O., Gur, Y., and Zeevi, A. (2014). Stochastic multi-armed-bandit problem with nonstationary rewards. Advances in Neural Information Processing systems , 27.
- Bhattamishra, S., Ahuja, K., and Goyal, N. (2020). On the ability and limitations of transformers to recognize formal languages. In Conference on Empirical Methods in Natural Language Processing , pages 7096-7116.
- Bing, Z., Lerch, D., Huang, K., and Knoll, A. (2022). Meta-reinforcement learning in non-stationary and dynamic environments. IEEE Transactions on Pattern Analysis and Machine Intelligence , 45(3):3476-3491.
- Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. (2020). Language models are few-shot learners. Advances in Neural Information Processing systems , 33.
- Cayci, S., Eryilmaz, A., and Srikant, R. (2020). Continuous-time multi-armed bandits with controlled restarts. arXiv preprint arXiv:2007.00081 .

- Chebotar, Y., Vuong, Q., Hausman, K., Xia, F., Lu, Y., Irpan, A., Kumar, A., Yu, T., Herzog, A., Pertsch, K., et al. (2023). Q-transformer: Scalable offline reinforcement learning via autoregressive q-functions. In Conference on Robot Learning , pages 3909-3928. PMLR.
- Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A., and Mordatch, I. (2021). Decision transformer: Reinforcement learning via sequence modeling. Advances in Neural Information Processing systems , 34.
- Chen, Y., Lee, C.-W., Luo, H., and Wei, C.-Y. (2019). A new algorithm for non-stationary contextual bandits: Efficient, optimal and parameter-free. In Conference on Learning Theory , pages 696726. PMLR.
- Cheng, X., Chen, Y., and Sra, S. (2024). Transformers implement functional gradient descent to learn non-linear functions in context. In International Conference on Machine Learning , pages 8002-8037. PMLR.
- Cheung, W. C., Simchi-Levi, D., and Zhu, R. (2020). Reinforcement learning for non-stationary markov decision processes: The blessing of (more) optimism. In International Conference on Machine Learning , pages 1843-1854. PMLR.
- Cheung, W. C., Simchi-Levi, D., and Zhu, R. (2022). Hedging the drift: Learning to optimize under nonstationarity. Management Science , 68(3):1696-1713.
- Domingues, O. D., M´ enard, P., Pirotta, M., Kaufmann, E., and Valko, M. (2021). A kernel-based approach to non-stationary reinforcement learning in metric spaces. In International Conference on Artificial Intelligence and Statistics , pages 3538-3546. PMLR.
- Dong, L., Xu, S., and Xu, B. (2018). Speech-transformer: a no-recurrence sequence-to-sequence model for speech recognition. In International Conference on Acoustics, Speech and Signal Processing , pages 5884-5888. IEEE.
- Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations .
- Furuya, T., de Hoop, M. V., and Peyr´ e, G. (2025). Transformers are universal in-context learners. In International Conference on Learning Representations .
- Garivier, A. and Moulines, E. (2011). Upper-confidence bound policies for non-stationary bandit problems. In Algorithmic Learning Theory , pages 174-188.
- Gomes, C. P., Selman, B., and Kautz, H. (1998). Boosting combinatorial search through randomization. In Conference on Artificial Intelligence/Innovative Applications of Artificial Intelligence , pages 431-437.
- Hahn, M. (2020). Theoretical limitations of self-attention in neural sequence models. Transactions of the Association for Computational Linguistics , 8:156-171.
- Hataya, R. and Imaizumi, M. (2024). Transformers as stochastic optimizers. In ICML 2024 Workshop on In-Context Learning .
- Huang, J., Wang, Z., and Lee, J. D. (2025). Transformers learn to implement multi-step gradient descent with chain of thought. In International Conference on Learning Representations .
- Kim, B. and Tewari, A. (2020). Randomized exploration for non-stationary stochastic linear bandits. In Conference on Uncertainty in Artificial Intelligence , pages 71-80. PMLR.
- Laskin, M., Wang, L., Oh, J., Parisotto, E., Spencer, S., Steigerwald, R., Strouse, D., Hansen, S. S., Filos, A., Brooks, E., et al. (2023). In-context reinforcement learning with algorithm distillation. In International Conference on Learning Representations .
- Lee, K.-H., Nachum, O., Yang, M. S., Lee, L., Freeman, D., Guadarrama, S., Fischer, I., Xu, W., Jang, E., Michalewski, H., et al. (2022). Multi-game decision transformers. Advances in Neural Information Processing Systems , 35.

- Li, Y. and Li, N. (2019). Online learning for markov decision processes in nonstationary environments: A dynamic regret analysis. In American Control Conference , pages 1232-1237. IEEE.
- Lin, L., Bai, Y., and Mei, S. (2024). Transformers as decision makers: Provable in-context reinforcement learning via supervised pretraining. In International Conference on Learning Representations .
- Liu, Z., Lin, Y ., Cao, Y ., Hu, H., Wei, Y ., Zhang, Z., Lin, S., and Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In International Conference on Computer Vision , pages 10012-10022.
- Mao, H., Liu, G., Ma, Y., Wang, R., and Tang, J. (2024). A data generation perspective to the mechanism of in-context learning. CoRR .
- Mao, W., Zhang, K., Zhu, R., Simchi-Levi, D., and Basar, T. (2021). Near-optimal model-free reinforcement learning in non-stationary episodic mdps. In International Conference on Machine Learning , pages 7447-7458. PMLR.
- Min, E., Chen, R., Bian, Y., Xu, T., Zhao, K., Huang, W., Zhao, P., Huang, J., Ananiadou, S., and Rong, Y. (2022). Transformer for graphs: An overview from architecture perspective. arXiv preprint arXiv:2202.08455 .
- Mosbach, M., Pimentel, T., Ravfogel, S., Klakow, D., and Elazar, Y. (2023). Few-shot fine-tuning vs. in-context learning: A fair comparison and evaluation. CoRR .
- Singh, A., Chan, S., Moskovitz, T., Grant, E., Saxe, A., and Hill, F. (2024). The transient nature of emergent in-context learning in transformers. Advances in Neural Information Processing Systems , 36.
- Streeter, M. and Golovin, D. (2008). An online algorithm for maximizing submodular functions. Advances in Neural Information Processing Systems , 21.
- Trovo, F., Paladino, S., Restelli, M., and Gatti, N. (2020). Sliding-window thompson sampling for non-stationary settings. Journal of Artificial Intelligence Research , 68:311-364.
- Vaart, A. W. v. d. and Wellner, J. A. (2023). Empirical processes. In Weak Convergence and Empirical Processes: With Applications to Statistics , pages 127-384. Springer International Publishing, Cham.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing systems , 30.
- Wei, C.-Y. and Luo, H. (2021). Non-stationary reinforcement learning without prior knowledge: An optimal black-box approach. In Conference on Learning Theory , pages 4300-4354. PMLR.
- Yang, S., Nachum, O., Du, Y., Wei, J., Abbeel, P., and Schuurmans, D. (2023). Foundation models for decision making: Problems, methods, and opportunities. arXiv preprint arXiv:2303.04129 .
- Yun, C., Rao, M., Becker, B., et al. (2020). Are transformers universal approximators of sequenceto-sequence functions? In International Conference on Machine Learning , pages 9116-9126.
- Yun, S., Jeong, M., Kim, R., Kang, J., and Kim, H. J. (2019). Graph transformer networks. Advances in Neural Information Processing systems , 32.
- Zhao, H., Chen, Z., Guo, L., and Han, Z. (2022). Video captioning based on vision transformer and reinforcement learning. PeerJ Computer Science , 8:e916.
- Zhao, S., Nguyen, T., and Grover, A. (2024). Probing the decision boundaries of in-context learning in large language models. Advances in Neural Information Processing Systems , 37.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We clearly stated the main claim in the abstract and introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We stated the limitation in the last section for discussion.

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

Justification: We have described the full assumption, statements, and proof in the main body and the appendix.

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

Justification: We have described the detailed experimental details in the corresponding section in the appendix.

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

Justification: We clearly stated the source of data. Also, we will open the source code.

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

Justification: We have clearly described the experimental details in the appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We have repeated the experimental multiple times and report the standard deviation comes from the replication.

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

Answer: [No]

Justification: Our experiments are small-scale and implementable by a small laptop. Also, we do not pursue the computational cost in this study, so the computational resource is out of our focus.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have checked the code.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: A main focus of this study is fundamental, so there is almost no effect on social impacts.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [No]

Justification: Since this paper is theoretical, the outcome does not have a righ risk for misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We have cited the source of the data which we use.

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

Answer: [No]

Justification: We have not created a new asset throughout this study.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [No]

Justification: We have not performed the crowdsourcing experiments and others.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [No]

Justification: Since this study is fundamental, there is no potential risk on this point.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [No]

Justification: We have used LLM only for the formatting purposes.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Limitation and Discussion

Regret bound In Corollary 2, the regret bound of the learned algorithm alg p θ is guaranteed when the approximation error ε is sufficiently small-which requires a sufficiently large model size-and when the dataset size N is sufficiently large. Although these requirements may seem inefficient compared to the model sizes and training datasets of traditional RL algorithms, modern LLMs are typically trained with scales that readily meet both conditions.

Proof method To establish the regret bound, we employ a non-continuous transformer, which is structurally equivalent to a standard transformer. This choice is made primarily for proof simplicity. While we also provide a brief description of how the standard transformer can be used for the proof (Appendix E.2), developing a more elegant proof directly in that setting is left for future work.

Pretraining of transformer The regret bound holds without assumptions on the non-stationarity of the training data. A natural direction for future work is to examine how training data with different types and degrees of non-stationarity influence the generality and performance of transformers.

Assumption 1 and 2 Our theoretical guarantees rest on Assumptions 1 and 2. The former has been shown to hold for a wide range of RL algorithms (Wei and Luo, 2021), whereas the latter may appear somewhat strong. We emphasize, however, that Assumption 2 builds on prior work demonstrating that transformers can emulate standard reinforcement learning algorithms, such as LinUCB and Thompson Sampling, through in-context learning (Lin et al., 2024). In addition, Furuya et al. (2025) shows that transformers are universal approximators over distributions, which further supports the plausibility of this assumption. We also emphasize that the assumption does not require the transformer to exactly replicate the expert policy. Instead, it assumes that the transformer can learn an amortized algorithm that approximates the expert's output distribution, which is a more relaxed and often more realistic requirement in practice.

## B Further Experiment Results

In our experiments, we set the confidence scaling parameter α of LinUCB to 1 , and the noise variance for Thompson Sampling (TS) to 0 . 3 .

The suboptimality of the models in Low Non-stationary environments and in High Non-stationary environments with b ' 0 . 018 (Figure 3) is shown in Figures 4 and 5. We observe that the transformer not only outperforms the expert algorithms but also maintains a consistently low suboptimality rate across rounds.

Figure 4: Suboptimality Comparisons of LinUCB, Thompson Sampling (TS), MASTER+LinUCB/TS, and transformer (TF) in Low Non-stationary environments. Linear bandit with d ' 32 , A ' 10 . Shading indicates the standard deviation of the regret estimates.

<!-- image -->

Figures 6 and 7 present the results for High Non-stationary environments with b ' 0 . 025 . We observe that, when the level of non-stationarity becomes too high, the transformer no longer outperforms the other models. This indicates that the model generalizes well to b ' 0 . 018 , but struggles

Figure 5: Suboptimality Comparisons of LinUCB, Thompson Sampling (TS), MASTER+LinUCB/TS, and transformer (TF) in High Non-stationary environments ( b ' 0 . 018 ). Linear bandit with d ' 32 , A ' 10 . Shading indicates the standard deviation of the regret estimates.

<!-- image -->

with b ' 0 . 025 , likely due to limited coverage of highly non-stationary scenarios in the training data.

Figure 6: Suboptimality and Cumulative Regret Comparisons of LinUCB, MASTER+LinUCB, and transformer (TF) in High Non-stationary environments ( b ' 0 . 025 ). Linear bandit with d ' 32 , A ' 10 . Shading indicates the standard deviation of the regret estimates.

<!-- image -->

Figure 7: Suboptimality and Cumulative Regret Comparisons of Thompson sampling (TS), MASTER+TS, and transformer (TF) in High Non-stationary environments ( b ' 0 . 025 ). Linear bandit with d ' 32 , A ' 10 . Shading indicates the standard deviation of the regret estimates.

<!-- image -->

## C MASTER Algorithm

We begin by introducing MALG (Multi-scale ALGorithm) (Wei and Luo, 2021), which extends a base reinforcement learning algorithm, ALG, by running multiple instances at different time scales.

Given an integer n and a non-increasing function ρ : r T s Ñ R , MALG operates over a time horizon of length 2 n .

At each time step τ , given an integer m , if τ is a multiple of 2 m , MALG schedules a new instance of length 2 m with probability ρ p 2 n q{ ρ p 2 m q . These are referred to as orderm instances. The start and end times of each instance are denoted by alg.s and alg.e , respectively. Among all scheduled instances, only the one with the lowest order remains active at any given time, producing an auxiliary value r r t .

After each step, MALG updates the active instance based on observed rewards R t from the environment. In this framework, all reinforcement learning algorithms generate a specific auxiliary value; for instance, in the LinUCB setting, r r t corresponds to a score that combines a reward estimate with an uncertainty term.

## Algorithm 1: MALG (Multi-scale ALG)(Wei and Luo, 2021)

```
Input: n , ρ p¨q 1 for τ ' 0 , . . . , 2 n ´ 1 do 2 for m ' n, n ´ 1 , . . . , 0 do 3 if τ is a multiple of 2 m then 4 With probability ρ p 2 n q{ ρ p 2 m q , schedule a new instance alg of ALG at scales 2 m ; 5 Run the active instance alg to output r r τ , select an action, and update with feedback.
```

Algorithm 2: MALG with Stationarity TEsts and Restarts (MASTER)(Wei and Luo, 2021)

```
Input: p ρ p¨q where p ρ p t q ' 6 p log 2 T ` 1 q log p T q ρ p t q ( T : block length) 1 Initialize t Ð 1 2 for n ' 0 , 1 , . . . do 3 Set t n Ð t and initialize an MALG (Algorithm 2) for the block r t n , t n ` 2 n ´ 1 s ; 4 while t ă t n ` 2 n do 5 Run MALG to obtain prediction r r t , select action a t , and receive reward R t ; 6 Update MALG with feedback, and set U t ' min τ Pr t n ,t s r r τ ; 7 Perform Test 1 and Test 2 (see below); 8 Increment t Ð t ` 1 ; 9 if either test returns fail then 10 restart from Line 2; 11 Test 1 : If t ' alg.e for some orderm alg and 1 2 m ř alg.e τ ' alg.s R τ ě U t ` 9 p ρ p 2 m q , return fail. 12 Test 2 : If 1 t ´ t n ` 1 ř t t n p r r τ ´ r τ q ě 3 p ρ p t ´ t n ` 1 q , return fail.
```

MASTER (MALG with Stationarity Tests and Restarts) further enhances this process by incorporating repeated checks for stability and performance. It uses two key tests: the first ensures that the average reward within any completed instance does not significantly exceed a threshold determined by the auxiliary value r r t ; the second verifies that the difference between the auxiliary output r r t and observed rewards r t remains bounded on average. If either test fails, MASTER resets the algorithm, thereby maintaining robustness under changing environmental conditions.

## D Proofs in Section 3

## D.1 Intermediate Statement

We provide the following intermediate statement to show Theorem 1.

Theorem 6. Let Assumption 1 and 2 hold. Then for any small ε ą 0 , there exists an algorithm alg p θ introduced by a transformer with

<!-- formula-not-decoded -->

satisfying

<!-- formula-not-decoded -->

## D.2 Proof of Theorem 1

As established in Wei and Luo (2021), for any reinforcement learning algorithm alg E , the combination of MASTER and alg results in a stabilized version: alg E . The dynamic regret of alg p θ can then be expressed as

<!-- formula-not-decoded -->

A can be obtained from Theorem 6, and B is bounded as follows:

<!-- formula-not-decoded -->

where C p t q ' c 1 t 1 { 2 ` c 2 with c 1 , c 2 ą 0 (Wei and Luo, 2021). Setting c 1 ' 1 and c 2 ! 1 , we obtain:

<!-- formula-not-decoded -->

## D.3 Proof of Corollary 2

It suffices to prove that

<!-- formula-not-decoded -->

According to Theorem 5, we have

Consequently, we have

<!-- formula-not-decoded -->

Thus, the cumulative distribution ratio over T rounds satisfies

<!-- formula-not-decoded -->

Since we assume ε ď T ´ 3 , it follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, following Vaart and Wellner (2023), there exist universal constants C 1 and C 2 such that

<!-- formula-not-decoded -->

where V p ε, Θ q denotes the fat-shattering dimension of Θ . Let V p Θ q be the VC-dimension of Θ ; then, for any ε ą 0 , it holds that V p ε, Θ q ď V p Θ q (Vaart and Wellner, 2023). Since Θ is a finite parameter class, there exists a constant C 3 such that V p ε, Θ q ď V p Θ q ď C 3 for all ε ą 0 . Therefore, under the assumption that N ě CT 3 log T , we obtain:

<!-- formula-not-decoded -->

which completes the proof.

## D.4 Proof of Theorem 5

As shown in Section E, a noncontinuous transformer (Definition 7) with

<!-- formula-not-decoded -->

can approximate the MASTER algorithm. Treating any reinforcement learning algorithm that uses MASTER as the expert algorithm alg E in Theorem 5, the result follows directly from Assumption 2.

## D.5 Proof of Theorem 6

By Theorem 5, we have

<!-- formula-not-decoded -->

where the last inequality uses the fact that alg p¨q ď 1 . With a slight abuse of notation, let r t,k denote the reward of the k -th action at round t . Since | r t,k | ď 1 almost surely for all t P r T s , we obtain the policy imitation error between transformers and MASTER as follows:

<!-- formula-not-decoded -->

Finally, since ε ą 0 is small, we use the approximation

<!-- formula-not-decoded -->

and then the policy imitation error becomes εAT .

Next, we leverage the result from Lin et al. (2024):

where

<!-- formula-not-decoded -->

for all t P r T s . We extend this result to non-stationary settings to derive a bound for | R alg p θ p T q ´ R alg E p T q| .

Suppose there are n 0 ` 1 orders of instances, and alg B runs for T rounds. Let k i ě 0 represent the number of rounds for orderi instances, so that k 0 ` k 1 `¨¨¨ ` k n 0 ' T . Define T i as:

<!-- formula-not-decoded -->

which is the set of rounds during which orderi instances are active. If k i ' 0 , orderi instances are inactive. If k i ą 0 , one or more orderi instances are active. Using Lemma 7, we know that running one instance during T i yields a higher regret bound (as in (15)) than running zero or multiple

instances in T i . Furthermore, Theorem 6 guarantees that the regret for a reinforcement learning algorithm at T i is bounded by εAk i . Thus, we have:

<!-- formula-not-decoded -->

Summing over all T i , we get:

<!-- formula-not-decoded -->

where the final inequality follows from the Cauchy-Schwarz inequality. By Assumption 2, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining this with the result of Theorem 6 completes the proof.

## D.5.1 An auxiliary lemma

Lemma 7 (Regret bound for multiple instances) . Let a, b, c, N, C 1 , C 2 ą 0 with a ` b ' c . The following bound holds:

<!-- formula-not-decoded -->

Proof of Lemma 7. Let 0 ă u ă 1 be such that a ' uc and b ' p 1 ´ u q c . Substituting into (16), we obtain:

<!-- formula-not-decoded -->

Denoting G : ' log p C 1 c q{ N , we rewrite the inequality as:

<!-- formula-not-decoded -->

Since 0 u 1 , we have log u 0 and log 1 u 0 . Substituting these into (17), we obtain:

ă ă ă p ´ q ă u 2 `p 1 ´ u q 2 ď 1 , which follows directly from 0 ă u ă 1 , as u 2 `p 1 ´ u q 2 ' 1 ´ 2 u p 1 ´ u q ă 1 .

Therefore,

## E Proofs in Section 4

## E.1 Structural Approximation

To match the ordern instance length as described in Appendix C, we consider a single input matrix H ' r h 1 , ..., h 2 n s P R d ˆ 2 n , where t h i u 2 n i ' 1 Ă R d . This matrix is then extended to H P R D ˆ 2 n defined as H : ' ' H p 0 q ; . . . ; H p n q ; H ˚ ‰ P R D ˆ 2 n (Figure 8) where D : ' pp d ` 5 qp n ` 1 q ` 5 q , and H ˚ contains auxiliary entries for approximation purposes. Here, H p i q : ' r h p i q 1 , . . . , h p i q 2 n s and h p i q t ' r h t ; 2 i , 0 , 0 , 0 , 0 s where the four auxiliary entries capture the order information. Details of each entry are given in (21) and (22). Once H is defined, the transformer produces the output H P R D ˆ 2 n . Note that even if the input length T is not exactly 2 n , the transformer can still replicate the operations of MASTER and generate a lengthT output by selecting the smallest 2 n greater than T and running the algorithm for T rounds.

We then define functions σ 1 and σ 2 to replicate the MALG operation: σ 1 stochastically schedules instances for each order, and σ 2 selects the instance with the lowest order to remain active in each round.

Definition 4 ( σ 1 ) . Given a non-increasing function ρ : r 2 n s Ñ R , we define σ ρ 1 : R D ˆ 2 n Ñ R D ˆ 2 n as

<!-- formula-not-decoded -->

where for each i P t 0 , ..., n u

<!-- formula-not-decoded -->

Here, B p k q is a Bernoulli random variable with parameter k , meaning B p k q ' 1 with probability k ď 1 and B p k q ' 0 otherwise.

Using σ 1 , multiple instances can be scheduled simultaneously (represented as the bluish blocks in Figure 8). Since only one instance can be active the environment at any given time, we define σ 2 to select the instance with the lowest order at each time step t .

We denote h t : ' ' h p 0 q t ; . . . ; h p n q t ı P R D for each t P r 2 n s , so that H ' r h 1 , . . . , h 2 n s . Then, σ 2 is defined as follows.

Definition 5 ( σ 2 ) . We define σ 2 : R D ˆ 2 n Ñ R D ˆ 2 n by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

to reproduce the uniqueness of an instance scheduled at every moment in MALG. Here, ˚ s denote the last five entries of h t .

After applying σ 2 , only a single instance remains active at each time t ; this active instance is always the one with the lowest order among all scheduled instances. The reddish blocks in Figure 8 illustrate this selection.

To approximate the stationary tests ( Test 1 and Test 2 ) in Algorithm 2, we define the following:

Definition 6 ( TEST ) . TEST represents the stationary tests in MASTER, involving two functions test1 and test2 that map from R D ˆ R D ˆ 2 n to R . Given p ρ defined in Algorithm 2 and an active instance in h t of order m , where h t contains m,t, U t , r r t , and r t , we define:

<!-- formula-not-decoded -->

Figure 8: Illustration of H when n ' 3 . Purplish blocks represent instances scheduled by σ 1 , while reddish blocks represent the active instances selected by σ 2 . Reddish blocks connected by a dashed line are concatenated.

<!-- image -->

<!-- formula-not-decoded -->

TEST is then defined as:

<!-- formula-not-decoded -->

Finally, we define the noncontinuous transformer. This transformer incorporates the stationarity tests and halts processing of the remaining sequence whenever a test fails. To implement this, we design a transformer module that processes each element individually, performs the required tests, and then concatenates the results.

Definition 7 (Noncontinuous Transformer) . For t P r 2 n s , denote

<!-- formula-not-decoded -->

where the parameters are θ p ℓ q attn ' t V p ℓ q m , Q p ℓ q m , K p ℓ q m u m Pr 2 n s Ă R D ˆ D and θ p ℓ q mlp ' tp W p ℓ q 1 , W p ℓ q 2 qu m Pr 2 n s Ă R D 1 ˆ D ˆ R D ˆ D 1 . After processing h t , the model interacts with the environment, observes the reward, and inserts it into the sequence: h ˚ t Ñ r h t ( (21) and (22) ). By concatenating r h t Pr 2 n s , we define the L -layer noncontinuous transformer T F θ : R D ˆ 2 n Ñ R D ˆ 2 n as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where r H p T q : ' ' r h 1 , ..., r h T , T T h T ` 1 , ..., T T h 2 n ı P R D ˆ 2 n . Here, T T P R D ˆ D is the test matrix at time T :

<!-- formula-not-decoded -->

where TEST ' TEST p h T , H p T q q . Order information ř t ´ 1 τ ' t ´ 2 i ` 1 r τ , ř i ´ 1 k ' 0 r r p k q t , r r p i q t , historical data ř t ´ 1 τ ' 1 r τ , ř t ´ 1 τ ' 1 r r τ , U t ´ 1 , and rand in (21) and (22) are reset when TEST ' 0 . r r p i q t denotes the auxiliary value generated at order i .

Interaction with the environment To illustrate the transformer's interaction with the environ- ment, we consider the following:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, ř 0 τ ' 1 r τ ' 0 and r τ ' 0 for τ ď 0 . The rand entry is a random number generated by the transformer. The vectors x p i q t P R d are used for in-context learning, and x p i q t denotes their processed forms; for example, these vectors can encode action choices in bandit problems (Lin et al., 2024).

Each entry of the vectors captures cumulative information or constants relevant to the model's operations up to time t . After passing the sequence through the MLP ` Attention module, certain orders are scheduled at each time t , while all other instances are set to 0 . The σ 2 operation ensures that only one instance remains active at a time by selecting the instance with the lowest order and zeroing out all others. If the active instance is order q ď n , then r r p q q t is assigned as r r t at time t , and the update U t ' min t U t ´ 1 , r r p q q t u is applied. Finally, the module interacts with the environment, observes the reward R t , and updates the tensor accordingly.

Notation with regard to the input sequence To avoid potential confusion, the notations defined above are summarized below:

- h t P R d : The t -th element of the input sequence.
- H ' r h 1 , . . . , h 2 n s P R d ˆ 2 n : The input sequence as a matrix.
- H ' r H p 0 q ; ¨ ¨ ¨ ; H p n q ; H ˚ s ' r h 1 ¨ ¨ ¨ h 2 n s P R D ˆ 2 n : The extended input sequence, where H p i q denotes the orderi sequence.

'

ı

- h t ' h p 0 q t ; ¨ ¨ ¨ ; h p n q t ; h ˚ t P R D : The t -th vector of the extended sequence.
- h 1 t ' ' 0 ; ¨ ¨ ¨ ; 0 ; h p k q t ; 0 ; ¨ ¨ ¨ ; 0 ; h ˚ t ı P R D : The t -th vector of the extended input after being processed by σ 2 . Here, k denotes the order of the active instance selected by σ 2 , i.e., the lowest nonzero order at time t .
- h ˚ t : The result of processing h t by the noncontinuous transformer module.
- h t : The output of the noncontinuous transformer after incorporating the observed reward at time t .

Restart The restart mechanism (line 9 of Algorithm 2) is integrated into the noncontinuous transformer. If all stationary tests pass, the matrices T T ( T P r 2 n s ) remain identity matrices, and the sequence is processed normally. If a test fails at time T , the matrices T T ( T P r 2 n s ) modify the states h T ` 1 , . . . , h 2 n : the first D ´ 7 elements-including x T , T , 2 n , and 2 i -are preserved, while the remaining seven elements are set to zero. This ensures that when a test fails, all historical information is erased, but essential identifiers and order information are retained to start a new block.

Rollout The noncontinuous transformer's architecture is more complex than the classic version, so we outline its input-output flow.

Starting with an input sequence H P R D ˆ 2 n , we extend it to H ' r H p 0 q ; ¨ ¨ ¨ ; H p n q ; H ˚ s P R D ˆ 2 n , where submatrices H p i q and H p j q are only different in several entries to record order information.

H is passed through σ 1 and σ 2 : σ 1 schedules instances for each order, and σ 2 activates only the instance with the lowest order at any given time t . As a result, at each time t , only one instance is active, and the active instance may come from a different order at each time step.

Next, each element in the sequence is processed by the traditional MLP ` Attention module. The module uses this information to interact with the environment, collects the reward, and inserts it into the sequence: h ˚ t Ñ h t ((21) and (22)). The updated sequence h t is evaluated by TEST , which performs stationary tests. If the tests pass, the block remains unchanged; otherwise, the remaining block entries are set to zero, keeping only essential information. This effectively ends the current 2 n -length block and starts a new 2 n -length block (line 2, Algorithm 2). On restart, variables like R t and f t are reinserted at their appropriate positions, following the same procedure. After processing the entire sequence, the final tensor H is produced. By gathering the rewards from all orders at each t (only one reward is nonzero at a given time), the resulting sequence H serves as the output of the noncontinuous transformer, containing cumulative rewards up to time 2 n .

## E.2 Motivation of the Non-Continuous Transformer

In this section, we first briefly demonstrate how the proofs can be adapted to a regular transformer, then explain the motivation of using the non-continuous transformer.

The core challenge in using a regular transformer lies in approximating σ 2 (Definition 5), which must select the first non-zero entry among all entries. This operation requires comparing all trajectories simultaneously to identify the first non-zero one, and it is the main reason why the augmented inputs are used.

One alternative approach is to use a regular transformer with n ` 1 heads to approximate σ 1 and σ 2 as follows:

- Layer 1 ( Compression ): Each head uses masked multi-head attention to extract pertrajectory representations, i.e., H p i q ' W comp i ¨ MaskAttn m p H q P R n D ` 1 ˆ T . Here, MaskAttn restricts information flows across heads, and the combined output is H 1 ' r H p 0 q ; ... ; H p n q s P R D ˆ T .
- Layer 2 ' K: Generate Bernoulli masks, block masks, and select the first non-zero trajectory to approximate σ 1 and σ 2 (same as the procedure in Appendix E.3.2 and E.3.3).
- Final Layer ( Decompression ): Reconstruct the selected trajectory: H 2 ' ř n m ' 0 W decomp m ¨ H p m q P R D ˆ T .

However, this method adds several analytical burdens, such as ensuring that W decomp m ¨ W comp m « I D for m P t 0 , ..., n u to preserve information, and constructing MaskAttn, which make the analysis more cumbersome.

In contrast, our proposed method is a simple and transparent preprocessing step where augmented inputs are generated via a linear transformation h t ' M t ¨ h t . This preserves the core semantics while avoiding unnecessary architectural detours. Importantly, the resulting non-continuous transformer is structurally equivalent to a standard transformer with transformed inputs.

## E.3 Approximation of WS (Proposition 3)

We divide the proof into three parts: approximating random number generation, σ 1 , and σ 2 . Throughout this section, we denote the ReLU activation function by σ p¨q .

## E.3.1 Approximating random number generation

We start by approximating 1 ą 0 r¨s . According to the definition of the ReLU function, we have

<!-- formula-not-decoded -->

When k Ñ8 , it approximates 1 r x s ' " 0 , x ď 0 1 , x ą 0

.

We denote z k : ' x k, 0 and P z p z q being the cumulative distribution function (CDF) of t z 1 , ..., z 2 n u . Following the proof in Hataya and Imaizumi (2024), we construct a 2-head attention layer to approximate the CDF.

From the approximation of 1 r¨s , P z p t q ' 1 { 2 n ř 2 n k ' 1 1 r x k, 0 ď t s can be approximated by sum of ReLU functions as

<!-- formula-not-decoded -->

where k is sufficiently large. Suppose the vector in formula 21 and 22 (left) is one of the input h t . By selecting t Q 1 , 2 , K 1 , 2 , V 1 , 2 u such that

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

Therefore, the output is

<!-- formula-not-decoded -->

where rand ' p P z p x i, 0 q can be regarded as a random variable sampled from U p 0 , 1 q .

## E.3.2 Approximating σ 1

Step 1. Generating Bernoulli masks. Define Bernoulli Masks as B i,t ' 1 r rand i,t ď ρ p 2 n q{ ρ p 2 i qs . We consider a random number rand i,t P R at order i and time t , and rand i,t , ρ p¨q P r 0 , 1 s . Following the approach in Step 1, we begin by considering the input h p i q t . This input includes rand i,t , 2 i , 2 n , and we also assume that ρ p 2 i q and ρ p 2 n q are added to h p i q t through ρ p¨q . In this case, a two-layer MLP can implement Bernoulli Masks.

First, consider the function f p x, y q ' x ¨ y P R . For x, y P r 0 , 1 s , dividing the domain r 0 , 1 s ˆ r 0 , 1 s into t 1 { ϵ u segments allows us to express it as:

<!-- formula-not-decoded -->

This form can approximate f using an MLP. Based on this strategy, the two-layer MLP is constructed as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where W 1 P R 2 t 1 { ε u ˆ D .The output is then processed by another layer:

<!-- formula-not-decoded -->

The hidden dimensions of W 1 , W 1 are O p 1 { ε q . If the input is expanded from h p i q t to H p i q , the hidden dimensions increase to O p 2 n { ε q . Further expand further to H , the hidden dimensions of W 1 , W 1 become O p n 2 n { ε q ' O p T log 2 T { ε q . The operator norms } W 1 } op , } W 1 } op scale as O p a T log 2 T { ε q . Finally, by taking k 2 Ñ8 , the error approaches zero.

Step 2: Generating block masks. By setting rand i,t ' ¨ ¨ ¨ ' rand i,t ` 2 i ´ 1 p t ' 1 p mod 2 qq , we have B i,t ' ¨ ¨ ¨ ' B i,t ` 2 i ´ 1 p t ' 1 p mod 2 qq . Since the output of Bernoulli masks can be expressed as H : ' r h 1 ¨ ¨ ¨ h 2 n s P R D ˆ 2 n , where for t P r 2 n s :

<!-- formula-not-decoded -->

we choose t Q m , K m , V m u m Pt 0 ,...,n u such that for tokens h t ,

<!-- formula-not-decoded -->

In this way, with k ă t we have

<!-- formula-not-decoded -->

and with k ' t we have

<!-- formula-not-decoded -->

Finally, by referring to the approximation of 1 ą 0 r¨s (E.3.1), we note that this indicator function can be approximated using two ReLU functions. Consequently, the block mask can be approximated by a 2 p n ` 1 q -head masked attention layer. By setting 1 { ε ' 4 p n ` 1 q 2 , we have M ' O p 1 { ? ε q

Combining this block mask with the random number generation and the Bernoulli mask completes the proof.

## E.3.3 Approximating σ 2

Based on the output of σ 1 , we can express the input of σ 2 as H : ' r h 1 ¨ ¨ ¨ h 2 n s P R D ˆ 2 n , where for t P r 2 n s :

<!-- formula-not-decoded -->

From the definition of σ 1 , we know that some h p i q t values are nonzero, while others are zero. If we assume i 1 ă ... ă i k p 0 ă k ď n q such that h p i 1 q t , ..., h p i k q t ‰ 0 while all other h p n q t values are 0 , the goal of σ 2 is to ensure that h p i 1 q t ‰ 0 and the others being 0 .

We choose t Q m , K m , V m u m Pt 0 ,...,n u such that for tokens h t ,

<!-- formula-not-decoded -->

where c 0 ą 0 is sufficiently large and ϵ 0 ą 0 is sufficiently small such that ϵ 0 ´ c 0 f j ă 0 for any f j ą 0 . In this way, with k ă t we have

<!-- formula-not-decoded -->

and with k ' t we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since ř m ´ 1 p ' 0 f p t ' 0 is equivalent to h p 0 q t ' ¨ ¨ ¨ ' h p p ´ 1 q t ' 0 , we have for each t P r T s that

<!-- formula-not-decoded -->

where h p 0 q t ' ¨ ¨ ¨ ' h p q ´ 1 q t ' 0 and h p m q t ‰ 0 . Finally, referring the approximation of 1 ą 0 r¨s (E.3.1), we can approximate 1 ą 0 r¨s using 2 ReLU functions. Therefore, σ 2 can be approximated by a 2 p n ` 1 q -head masked attention layer.

## E.4 Equivalence of WS and RM with previous works

## E.4.1 WS

Sliding windows used in previous works (e.g., (Cheung et al., 2022; Trovo et al., 2020)) can be regarded as special cases of WS . Specifically, they are equivalent to WS when WS contains only one instance, all windows are of equal size, and they are connected without overlap. Figure 9 shows the equivalence between the stochastic instance scheduler in MASTER and the sliding window schedulers from previous works.

<!-- image -->

scheduler

Figure 9: Equivalence between the stochastic instance scheduler in MASTER and the sliding window schedulers from previous works. Purplish blocks represent scheduled blocks, while reddish blocks represent active ones. Reddish blocks connected by a dashed line are concatenated. Here, T ' 8 and the sliding window size is 2 .

## E.4.2 RM

The restart strategies in previous works can be broadly categorized into two types: stochastic and deterministic (Gomes et al., 1998; Streeter and Golovin, 2008). Stochastic restarts typically involve some form of randomness or a test, whereas deterministic restarts are executed after a fixed number of rounds. Consequently, deterministic restarts can be viewed as a special case of sliding window strategies. As for stochastic restarts, given that transformers are capable of generating random numbers and MLPs can implement stationary tests, it follows from the universal approximation theorem (Definition 8) that transformers can also approximate stochastic restart mechanisms.

Definition 8 (Universal Approximation Theorem for MLPs) . Let f be a continuous function from a compact subset K Ď R n to R m . Suppose σ : R Ñ R is a non-polynomial, continuous activation function applied component-wise.

Then, for any ϵ ą 0 , there exists a single hidden-layer MLP that approximates f to within ϵ . Specifically, there exist weights A P R k ˆ n , b P R k , C P R m ˆ k , and a sufficiently large number of hidden units k such that the MLP

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

satisfies