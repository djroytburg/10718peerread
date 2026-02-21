## Q♯ : Provably Optimal Distributional RL for LLM Post-Training

Jin Peng Zhou *1 , Kaiwen Wang *1 , Jonathan Chang 4 , Zhaolin Gao 1 , Nathan Kallus 1,3 , Kilian Q. Weinberger 1 , Kianté Brantley 2 , and Wen Sun 1

1 Cornell University 2 Harvard University 3 Netflix 4 Databricks

## Abstract

Reinforcement learning (RL) post-training is crucial for LLM alignment and reasoning, but existing policy-based methods, such as PPO and DPO, can fall short of fixing shortcuts inherited from pre-training. In this work, we introduce Q♯ , a value-based algorithm for KL-regularized RL that guides the reference policy using the optimal regularized Q function. We propose to learn the optimal Q function using distributional RL on an aggregated online dataset. Unlike prior value-based baselines that guide the model using unregularized Q -values, our method is theoretically principled and provably learns the optimal policy for the KL-regularized RL problem. Empirically, Q♯ outperforms prior baselines in math reasoning benchmarks while maintaining a smaller KL divergence to the reference policy. Theoretically, we establish a reduction from KL-regularized RL to no-regret online learning, providing the first bounds for deterministic MDPs under only realizability. Thanks to distributional RL, our bounds are also variance-dependent and converge faster when the reference policy has small variance. In sum, our results highlight Q♯ as an effective approach for post-training LLMs, offering both improved performance and theoretical guarantees. The code can be found at https://github.com/jinpz/q\_sharp .

## 1 Introduction

Reinforcement learning (RL) post-training is a crucial step in training large language models (LLMs), aligning their generations with human preferences [1] and enhancing their reasoning capabilities [2, 3]. This stage typically follows supervised learning (next-token prediction), where the model is further trained to maximize expected cumulative reward while minimizing KL divergence from the reference policy π ref obtained via supervised learning. The KL penalty plays a critical role by keeping the model close to π ref , mitigating issues such as reward hacking and catastrophic forgetting.

Most state-of-the-art LLMs [4, 5, 6] are post-trained using policy-based RL methods, which update model weights via stochastic gradient descent using algorithms like RLOO [7], PPO [8], and DPO [9]. However, these methods are computationally expensive, requiring full backpropagation through the LLM during training. In this paper, we propose a more efficient alternative: a value-based RL approach that guides the generations of the reference policy π ref using a learned value function, without modifying π ref model weights. This approach is particularly attractive because, for many tasks, evaluating generations is easier than producing them [4, 10], suggesting we can use much smaller models to learn value functions for guidance. For instance, in our experiments (Section 3.2), we show that a 1B parameter value model can effectively steer and improve a 70B parameter LLM.

Existing value-based methods for LLM post-training, such as CD [11] and VAS [12], fall short of faithfully optimizing the KL-constrained RL objective. These approaches guide π ref using Q π ref -the

* Equal contribution. Correspondence to {jz563,kw437}@cornell.edu .

Figure 1: (Left) A sketch of our post-training algorithm ( Q♯ ) based on distributional RL. Q♯ alternates between learning Z ⋆ - the reward-to-go distribution of π ref - and using the induced policy to collect new data and further improve the distributional estimate. (Right) Evaluation result on the GSM8K dataset [19]. We see that Q♯ achieves both higher accuracy and lower KL compared to prior valuebased post-training algorithms [11, 12].

<!-- image -->

expected reward-to-go under π ref without KL regularization-which does not guarantee convergence to the optimal policy π ⋆,η . In contrast, under the classical KL-regularized RL framework, we show that it is provably optimal to guide π ref using Q ⋆,η , the expected reward-to-go under the optimal policy π ⋆,η , which accounts for KL regularization. This theoretical insight ensures convergence to π ⋆,η and addresses the shortcomings of previous methods. As we demonstrate empirically and theoretically, prior approaches can lead to suboptimal rewards or large KL divergence-issues that our algorithm, Q♯ , provably avoids.

Our method exploits special properties of Q ⋆,η in deterministic MDPs and iteratively trains a model to estimate it through supervised distributional learning such as MLE. The iterative training procedure is motivated by the classic imitation learning algorithm DAgger [13], which addresses covariate shift and ensures that the learned Q ⋆,η estimator remains accurate when used to guide π ref at inference time. This distributional learning approach not only enhances empirical performance but also enables second-order style regret bounds - instance-dependent bounds that adapt to the variance of the model's generation.

Q♯ differs from traditional RL methods in two key aspects. First, we avoid complex temporal difference (TD) learning [14] or Q-learning techniques [15, 16], instead relying on direct supervised learning of a fixed critic. Second, while we adopt a distributional perspective, Q♯ is conceptually simpler than classical distributional RL algorithms like C51 [17]: we directly learn outcome distributions via supervised maximum likelihood, without invoking distributional Bellman updates. We elaborate on this and related works in Appendix A. In summary, our contributions are as follows:

1. We propose Q♯ , a principled algorithm for KL-regularized RL in deterministic MDPs, which includes LLMs, based on guiding π ref with the soft Q ⋆ learned with distributional RL (Section 2.2).
2. We prove variance-dependent PAC bounds for convergence to the optimal policy, which only requires realizability in the function class (Section 4).
3. We show that value-based post-training, which includes Q♯ , can fix biases and shortcuts in a star-graph environment [18], while popular policy-based methods cannot (Section 3.1).
4. We provide extensive experiments on math reasoning tasks that validate the effectiveness of our method at maximizing reward while maintaining small KL deviations from the reference policy (Section 3.2).

## 2 Method

## 2.1 Preliminaries

We study KL-regularized reinforcement learning (RL) in deterministic Markov Decision Processes (MDPs), where large language model (LLM) post-training is a motivating special case. An MDP is defined by a state space X , action space Y , horizon H , transition kernels ( P 1 , . . . , P H ) with P h : X × Y ↦→ ∆( X ) , and known reward functions ( r 1 , . . . , r H ) where r h : X × Y → R . A policy π = ( π 1 , . . . , π H ) consists of decision rules π h : X → ∆( Y ) . For a given η &gt; 0 , the KL-regularized

value of a policy π is defined as

<!-- formula-not-decoded -->

A classical result shows that KL-regularized RL can be solved via soft Bellman equations [20]. Starting from h = H and proceeding backward, we define:

<!-- formula-not-decoded -->

This expresses the optimal policy as a softmax over Q ⋆,η h , weighted by π ref h . Moreover, Q ⋆,η h ( x, y ) is the maximal expected KL-regularized return starting from ( x, y ) at time h . We now focus on deterministic MDPs, which covers LLM post-training and other structured generation tasks such as diffusion models [21].

Assumption 2.1. The transitions P h are deterministic.

Under this assumption, the value function simplifies significantly:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Equation (3) is due to the determinism of P h , while Equation (4) follows by recursively unrolling until the final step. Note that although V ⋆,η h ( x h ) corresponds to the soft value of the optimal policy, its recursion is expressed via expectations over π ref . We summarize this in the following known result [22, 23, 21]:

Theorem 2.2. Under Assumption 2.1, we have V ⋆,η h ( x h ) = η ln E π ref [exp( η -1 ∑ t ≥ h r t ( x t , y t )) | x h ] and Q ⋆,η h ( x h , y h ) = η ln E π ref [exp( η -1 ∑ t ≥ h r t ( x t , y t )) | x h , y h ] .

This shows V ⋆,η and Q ⋆,η are simple functionals of Z ⋆ - the cumulative reward distribution of π ref -where the functional is f ( P ) = η ln E P exp( X/η ) . In other words, if we learn the cumulative reward distribution of π ref , then we can directly compute V ⋆,η and Q ⋆,η , without any dynamic programming.

This offers several benefits. First, we do not require temporal difference (TD) learning ( i.e. , bootstrapping) which is notoriously unstable with deep networks [24] and requires completeness-type assumptions to guarantee convergence in theory [25]. Second, fitting the reward-to-go distribution Z ⋆ or regressing E π ref [exp( η -1 ∑ t ≥ h r t )] is a standard supervised learning task with a fixed target, which is much more stable in practice and well-understood in theory. Notably, there is no bootstrapping or changing targets which is what renders deep RL fragile. Third, we can apply distributional RL methods, where we directly fit the distribution Z ⋆ via supervised learning ( e.g. , maximum likelihood). Importantly, our approach does not involve distributional Bellman equation nor distributional TD update, which are known to be non-contractive under certain metrics [17]. Prior work has shown that fitting Z ⋆ in this manner yields benefits in representation learning [17, 26], lower variance updates [27], and second-order bounds [28, 29].

Applicability to LLMs. Our deterministic MDP framework directly models LLM post-training as a special case [4]. The initial state x 1 corresponds to the input prompt, each intermediate state x h is the current generation prefix, and the action y h is the next token. The policy thus reflects the LLM's autoregressive decoding process. The transition function is deterministic: P h ( x h , y h ) = x h y h , which simply appends the new token to the prefix. In many post-training settings, the reward is sparse, meaning only r H is nonzero. Under this assumption, Theorem 2.2 simplifies to Q ⋆,η h ( x h , y h ) = η ln E π ref [exp( η -1 r ( x H , y H )) | x h , y h ] . For example, the reward may indicate solution correctness in math tasks or reflect user preference in dialogue, as determined by a learned reward model.

Inference with cumulative reward distribution. Let Z ⋆ denote the conditional distribution over cumulative rewards under rollouts from π ref , that is, Z ⋆ h ( x, y ) D = ∑ t ≥ h r t ( x t , y t ) | x h = x, y h =

y , where the trajectory ( x h , y h , . . . , x H , y H ) is sampled under π ref , and D = denotes equality in distribution. Combining Theorem 2.2 and Equation (2), the optimal policy can be rewritten in terms

of Z ⋆ as π ⋆,η h ( y | x ) ∝ π ref h ( y | x ) E z ∼ Z ⋆ h ( x,y ) [exp( z/η )] . This motivates defining a general family of policies induced by any distribution Z : X × Y → ∆( R ) via

<!-- formula-not-decoded -->

Since π ⋆,η = π Z ⋆ ,η , we can approximate the optimal policy by estimating Z ⋆ with ̂ Z ≈ Z ⋆ using distributional learning techniques such as maximum likelihood estimation (MLE), and then instantiating π Z,η . This forms the core of our proposed Q♯ algorithm.

## 2.2 Algorithm Q♯

We propose Q-Sharp ( Q♯ ), a distributional value-based algorithm for KL-regularized RL in deterministic MDPs. Q♯ iteratively collects data from progressively improved policies to approximate the target distribution Z ⋆ (Algorithm 1). In this section, we describe Q♯ in practical terms for deep neural networks and LLMs; in Section 4, we formalize it using online learning oracles and prove convergence under a mild realizability assumption.

## Algorithm 1 Q♯

- 1: Input: reference policy π ref .
- 2: Initialize parameters θ 1 of conditional distribution Z θ : X ×Y → ∆( R ) and dataset D h = ∅ for all h .
- 3: for k = 1 , 2 , . . . until convergence do
- 4: Let π k ← π Z θ k ,η be policy induced by Z θ k (using Equation (5)).
- 5: for i = 1 , 2 , . . . , N do
- 6: Sample a switching time h ∼ [ H ] .
- 7: Roll-in with π k for h -1 steps.
- 8: Resume trajectory with π ref from x h .
- 9: Let R t denote cumulative rewards after time t .
- 10: Add ( x t , y t , R t ) to D t , ∀ t ≥ h .
- 11: end for
- 12: Update θ k by minimizing the distributional loss on the aggregated data:

θ

k

+1

←

arg min

θ

- 13: end for
- 14: Output: Final θ k .

Let Z θ h : X × Y → ∆( R ) denote a parametric conditional distribution with parameters θ . Given a sample R ∈ R ( e.g. , drawn from Z ⋆ ) and a model prediction ˆ Z , let L ( R, ˆ Z ) be a distributional loss for training the model. We denote by θ ⋆ the parameter that minimizes the distance between Z ⋆ and Z θ . For example, if Z ⋆ h ( x, y ) is Ber( p ⋆ h ( x, y )) , we can parameterize Z θ h ( x, y ) by a neural network that outputs a scalar estimate ˆ p of p ⋆ h ( x, y ) . The natural loss in this case is binary cross-entropy (BCE): L bce ( r, ˆ p ) = -r ln ˆ p -(1 -r ) ln(1 -ˆ p ) . This binary setup is appropriate for tasks such as math or multiple-choice questions where the reward is binary. If the reward distribution has no known parametric form, one can use a non-parametric model ( e.g. , a histogram that discretizes the reward space) trained via maximum likelihood (MLE) [17]: L mle ( r, ˆ z ) = -ln ˆ z [idx( r )] , where idx( r ) returns the index of the bin containing r , and ˆ z [ i ] denotes the probability estimate for bin i . In general, Q♯ can incorporate any distributional RL loss function [30]. Once Z θ closely approximates

Z ⋆ , we instantiate a near-optimal policy π θ,η via Equation (5). In Section 4, we prove that this procedure converges to the optimal policy under a mild realizability assumption.

Then, the key idea of Q♯ is an iterative data-collection and update process. At iteration k , with current parameters θ k , we deploy the induced policy π k := π Z θ k ,η to gather new data. Specifically, we roll in with π k for h -1 steps to reach a state x h , then switch to π ref to complete the trajectory. The cumulative reward from step h to the end, denoted R h,k , is a sample from Z ⋆ h ( x h ) . We add these samples to the dataset and update θ via gradient descent on the distributional loss. This process repeats until convergence.

Our iterative approach is similar in spirit to DAgger [13], AggreVaTe [31, 32], and RLGF [33], which likewise mitigate distribution shift to ensure the learned estimator remains accurate at test time. In contrast, prior value-based methods such as CD [11] and entropy-regularized PRM [34] train their estimators only on data from π ref . While such an estimator may perform well on π ref 's distribution, it offers no guarantee of accuracy when used to steer π ref 's generation at inference time.

Comparison with CD and VAS. The most closely related value-based baselines are CD [11] and VAS [12], yet they exhibit three critical limitations. (i) Incorrect value target. Both methods re-weight π ref using Q π ref , 0 -the unregularized Q -function of π ref -thereby ignoring the KL term. As shown in Section 4, this choice can yield policies that are either sub-optimal in reward or far from π ref . Q♯

∑

E

D

[

L

(

R

h

, Z

θ

(

x

h

, y

h

))]

h

h

.

Figure 2: (a) The star-graph with degree d = 5 and path length ℓ = 5 . Next-token prediction, the standard pre-training loss, learns a sub-optimal shortcut that selects a random first node and follows it until the end. (b) Accuracies on held-out test graphs for various models. Policy-based post-training methods such as REINFORCE and RPO (a variant of DPO) still exhibit the shortcut and has test accuracy 1 /d , while our value-based Q♯ fixes the shortcut and achieves near-perfect accuracy.

<!-- image -->

instead employs the principled target Q ⋆,η and is guaranteed to converge to π ⋆,η under realizability. (ii) Offline training. CD and V AS fit their value functions on a fixed dataset, whereas Q♯ alternates data collection and updates, improving robustness to distribution shift [13, 31]. (iii) Squared-loss regression. Both baselines learn Q π ref , 0 with an ℓ 2 loss, implicitly assuming homoskedastic Gaussian rewards. Q♯ leverages distributional RL losses, which are theoretically more sample-efficient [35, 29] and empirically superior [17, 26].

Relation to actor-critic methods. Although Q♯ learns a value function, its target V ⋆,η (or Q ⋆,η ) is fixed throughout training. Standard actor-critic algorithms (e.g., PPO) continuously update V π or Q π as π evolves, and rely on bootstrap-based TD updates. In contrast, Q♯ trains the value network via distributional supervised learning (e.g., MLE), thereby avoiding the instability of changing targets.

Relation to DPO [9]. While the form of Equation 5 resembles DPO's policy expression, their derivations and scopes are fundamentally different. DPO begins from the same KL-regularized RL objective but, without exploiting the deterministic transition structure, operates at the sequence level , corresponding to the one-step case ( H = 1 ). Its policy is given by π r ( y | x ) = 1 Z ( x ) π ref ( y |

x ) exp ( 1 β r ( x, y ) ) , where y denotes a full completion and Z ( x ) is the partition function over all possible sequences. When H = 1 , our Q ⋆ reduces to the reward r , and the DPO expression naturally follows as a special case. However, the DPO partition function Z ( x ) is intractable to normalize, and practical implementations must rely on pairwise preference data (e.g., Bradley-Terry modeling) to bypass it.

Inference with multiple η . Because the learned distribution ̂ Z θ is independent of η , a single trained network can support any choice of η at inference time simply by plugging it into Equation (5).

## 3 Experiments

## 3.1 Star-Graph

We begin with the star-graph task from [18], illustrated in Figure 2(a). A star-graph G ( d, ℓ ) has d paths of length ℓ from a central node. Given a start/goal node and the graph edges, the LM must generate a valid path. Though seemingly simple, [18] showed that next-token pre-training often learns a faulty shortcut: the model picks the first node at random (correct with probability 1 /d ) and follows the path, yielding a test accuracy of only 1 /d . This highlights the limitations of next-token prediction on planning tasks. [36] also showed that the task embeds the "sparse parity" problem determining whether the sum of a binary string is even or odd - which is known to be difficult for gradient-based optimizers and is widely studied in learning theory and optimization [37, 38, 39, 40].

Can this shortcut be fixed during post-training? We evaluate REINFORCE [41], DPO [9], RPO [42], and Q♯ , reporting test accuracies in Figure 2 (b). Q♯ consistently corrects the shortcut, achieving near-perfect accuracy, even for long paths ( G (2 , 20) ) or large degrees ( G (5 , 5) ). CD [11] achieves similar performance as Q♯ . In contrast, policy-based methods like REINFORCE and RPO fail to

Table 1: Comparison of Q♯ with π ref and CD baseline on GSM8K (Left) and MATH (Right). For both Llama 3 and Llama 3.1 8B, Q♯ consistently improves both pass@1 and majority voting accuracy upon baselines while incurring minimal KL deviation.

| π ref           | Llama 3 8B   | Llama 3 8B   | Llama 3 8B   | Llama 3.1 8B   | Llama 3.1 8B   | Llama 3.1 8B   | π ref           | Llama 3 8B   | Llama 3 8B   | Llama 3 8B   | Llama 3.1 8B   | Llama 3.1 8B   | Llama 3.1 8B   |
|-----------------|--------------|--------------|--------------|----------------|----------------|----------------|-----------------|--------------|--------------|--------------|----------------|----------------|----------------|
| Methods         | π ref        | CD           | Q♯           | π ref          | CD             | Q♯             | Methods         | π ref        | CD           | Q♯           | π ref          | CD             | Q♯             |
| pass@1 ↑        | 69.1         | 77.8         | 78.4         | 82.9           | 84.5           | 85.1           | pass@1 ↑        | 25.4         | 24.9         | 27.1         | 43.9           | 45.3           | 46.7           |
| maj1@8 ↑        | 85.8         | 87.2         | 88.1         | 90.5           | 90.9           | 91.4           | maj1@8 ↑        | 34.3         | 34.3         | 37.9         | 57.0           | 59.0           | 60.1           |
| KL-Divergence ↓ | -            | 6.39         | 2.65         | -              | 7.43           | 3.67           | KL-Divergence ↓ | -            | 15.27        | 7.14         | -              | 26.8           | 8.69           |

fix the shortcut, plateauing at 1 /d accuracy. DPO performs worst, often collapsing the policy to zero accuracy by suppressing both chosen and rejected paths-a failure mode also noted by RPO. These results suggest that once shortcuts are learned, policy-based methods struggle to unlearn them, reinforcing the effectiveness of value-based approaches like Q♯ and CD for LLM post-training. Please see Appendix C for a more detailed discussion on why REINFORCE and RPO cannot fix shortcuts and implementation details.

## 3.2 Math Reasoning

Datasets. We evaluate on two mathematical reasoning benchmarks: GSM8K [19], a dataset of grade school arithmetic word problems, and MATH [43], which features more challenging high school competition problems. We split each training set 90%-10% for training and validation. Test performance is reported on the full GSM8K test set and a 500-sample subset of MATH (MATH-500), following prior work [44, 45]. In Appendix G, we also evaluate Q♯ on AIME-24 dataset.

Models. We experiment with Llama 3 [5] and Qwen 2.5 [46] model families, both of which are competitive on math reasoning tasks and span a wide range of parameter scales. Due to space constraints, we report results for Llama 3 in the main text and defer Qwen 2.5 results to Appendix G. Unless otherwise noted, the Q ⋆,η function in Q♯ is parameterized and initialized with a Llama 3.2 1B model, and we use η = 0 . 1 , which yields consistent and strong performance. We run Q♯ for two iterations, after which performance converges. Additional details on model configurations and Q♯ training are provided in Appendices D and E.

Evaluation metrics. We report single sample accuracy ( pass@1 ) and majority voting accuracy ( maj1@k ). pass@1 evaluates one sampled generation per problem against the ground truth, while maj1@k checks if the most frequent answer among k samples is correct. We use k = 8 , temperature T = 0 . 8 , and nucleus sampling p = 0 . 9 . The evaluation prompt template is provided in Appendix F.

Main results. Table 1 presents Q♯ performance on GSM8K (Left) and MATH (Right) with π ref as Llama 3 or 3.1 8B. Although both have 8B parameters, Llama 3.1 performs significantly better. Across all settings, Q♯ consistently improves over π ref , boosting pass@1 by up to 9% on GSM8K with just 1B additional parameters. We also compare against the CD baseline [11], which incorrectly uses Q π ref , 0 to guide π ref . Q♯ outperforms CD on both accuracy metrics while maintaining lower KL divergence. Overall, Q♯ Pareto-dominates CD in the KL-regularized RL setting by achieving higher reward and lower KL. We note that CD [11] and VAS [12] are concurrent work and differ only in minor aspects such as sampling strategy. Therefore, we use CD as a canonical baseline for empirical comparison. Since Q♯ is complementary to policy-based methods, we further evaluate its effectiveness when guiding a PPO-trained model, as shown in Appendix I.

Larger π ref and Q♯ sizes. We evaluate how performance scales with larger π ref and Q♯ models on MATH (Table 2). Using 70B Llama 3 and 3.1 as π ref significantly boosts baseline pass@1 (45.6% and 60.6%, respectively). Remarkably, a 1B Q♯ still improves these large models-e.g., by 2.5% pass@1 and 3.5% maj1@8 for Llama 3.1. Increasing Q♯ to 3B yields further gains, demonstrating scalability. Compared to Table 1 (right), we note that with 9B total parameters (8B π ref + 1B Q♯ ), the maj1@8 accuracy already matches the pass@1 of the 70B π ref in Table 2, suggesting a promising low-resource alternative. For Llama 3, pass@1 improves while maj1@8 slightly drops, likely due to increased generation diversity benefiting harder problems but reducing consistency on easier ones.

Q♯ as a reward model. Beyond guiding π ref generation, Q♯ 's token-level Q function can also assess how good a complete generation is among many. We compute Q ( generation , EOS ) by applying Q♯ as a reward model ( Q♯ -RM) on GSM8K and MATH, using both π ref and Q♯ generations. Table 3 reports two settings: Q♯ -RM Best of 8 (selects top-scoring sample) and Q♯ -RM maj1@8 (aggregates

Table 2: Performance of π ref and Q♯ on MATH with larger π ref and Q♯ model sizes. Q♯ of size 1B is capable of guiding a 70B π ref model. Increasing Q♯ model sizes to 3B also leads to noticeably better performance for Llama 3.1 70B.

| π ref           | Llama 3 70B   | Llama 3 70B   | Llama 3 70B   | Llama 3.1 70B   | Llama 3.1 70B   | Llama 3.1 70B   |
|-----------------|---------------|---------------|---------------|-----------------|-----------------|-----------------|
| Q♯ Model        | None          | Llama 3.2 1B  | Llama 3.2 3B  | None            | Llama 3.2 1B    | Llama 3.2 3B    |
| pass@1 ↑        | 45.6          | 46.4          | 46.7          | 60.6            | 63.1            | 64.1            |
| maj1@8 ↑        | 55.6          | 55.5          | 55.3          | 69.0            | 72.5            | 72.7            |
| KL-Divergence ↓ | -             | 3.12          | 5.15          | -               | 4.98            | 4.99            |

Figure 3: Performance tradeoff between CD and Q♯ on the GSM8K validation set. ( Left ) pass@1 vs. KL divergence: Q♯ outperforms CD with higher accuracy and lower KL. ( Right ) KL divergence across η : CD's KL increases rapidly and deviates from π ref , while Q♯ remains stable.

<!-- image -->

majority voting with scores). Q♯ -RM maj1@8 consistently improves over vanilla maj1@8 , and Best of 8 yields more than 10% gains over pass@1 for π ref . The reward model can be used on both π ref and Q♯ own generations to further improve performance, which suggests the (same) reward model has generalizability for evaluating diverse generations.

Effect of η . Figure 3 shows the performance-KL tradeoff between CD and Q♯ on the GSM8K validation set. ( Left ) Increasing KL can improve pass@1 for both methods, but Q♯ consistently achieves a better Pareto frontier. ( Right ) CD is highly sensitive to η : as η -1 increases, its KL grows rapidly and performance degrades below that of π ref . In contrast, Q♯ remains stable and requires less tuning of η .

Ablations. We ablate several design choices in Table 4 on the GSM8K and MATH validation sets using pass@1 accuracy. The 'Prefix' column tests training on all t ≥ h prefixes after switching to π ref (Algorithm 1, Line 10), as opposed to only t = h . Though this breaks IID assumptions, the increased training data improves Q♯ performance by up to 4%. We compare two parameterizations of Q ⋆,η : Q-type, which computes Q ⋆,η ( x, y ) for all y , and V-type, which predicts Q ⋆,η ( x, ˆ y ) for a specific ˆ y . V-type outperforms Q-type, likely due to its lower parameter count and per-token computation. Details are in Appendix D. We also compare distributional Q♯ with MSE-based regression, which underperforms as expected under Bernoulli rewards. Finally, more iterations of Algorithm 1 yield marginal gains, with performance saturating after two iterations, which we adopt by default.

Qualitative comparison. Figure 6 shows side-by-side generations from π ref and Q♯ on math reasoning tasks. While both models often begin with similar prefixes-consistent with Q♯ 's low KL deviationQ♯ typically corrects π ref 's mistakes and produces more coherent reasoning. This behavior reflects Q♯ 's ability to assign higher value to correct tokens, thereby steering generation more effectively at critical decision points. Additional examples are provided in Appendix K.

Beyond math reasoning. To further validate the generality of Q♯ beyond mathematical reasoning tasks, we evaluate its performance on QuALITY [47], a challenging multiple-choice reading comprehension benchmark with long-form passages drawn from Project Gutenberg. As shown in

Table 3: Performance of π ref and Q♯ with Q♯ as a reward model. The reward model can determine the best generation among all generations for a problem and consistently improves maj1@8 for π ref and Q♯ own generations.

| Setting          | Llama 3 8B GSM8K   | Llama 3 8B GSM8K   | Llama 3.1 8B MATH   | Llama 3.1 8B MATH   |
|------------------|--------------------|--------------------|---------------------|---------------------|
| Methods          | π ref              | Q♯                 | π ref               | Q♯                  |
| pass@1           | 69.1               | 78.4               | 43.9                | 46.7                |
| maj1@8           | 85.8               | 88.1               | 57.0                | 60.1                |
| Q♯ -RM Best of 8 | 85.9               | 86.0               | 54.0                | 54.0                |
| Q♯ -RM maj1@8    | 88.5               | 89.2               | 59.2                | 60.6                |

Table 4: Ablations of Q♯ (last row) on pass@1 with various configurations on the validation set of GSM8K and MATH. The improvement suggests that our design choices all contribute positively to the final performance.

| Prefix   | Type   | Opt.   |   # Iter. |   Llama 3 8B GSM8K |   Llama 3.1 8B MATH |
|----------|--------|--------|-----------|--------------------|---------------------|
| Single   | V      | Dist.  |         1 |               80.5 |                64.5 |
| All      | Q      | Dist.  |         1 |               81.4 |                66.4 |
| All      | V      | MSE    |         1 |               81.4 |                65.4 |
| All      | V      | Dist.  |         1 |               82.3 |                67.4 |
| All      | V      | Dist.  |         2 |               83.5 |                68.5 |

Appendix H, Table 6, we compare Q♯ with π ref and CD baseline using Qwen 2.5 and Llama 3.1. Specifically, Qwen 2.5 1B guides Qwen 2.5 7B and Llama 3.2 1B guides Llama 3.1 8B. Across both architectures, Q♯ consistently improves upon π ref in all evaluation metrics, demonstrating its robustness beyond the mathematical domain.

## 4 Theory

## 4.1 CD &amp; VAS are sub-optimal for KL-regularized RL

First, CD and VAS both propose to reweight π ref ( · | x ) with the unregularized Q -function of π ref :

<!-- formula-not-decoded -->

where recall that Q π ref h ( x h , y h ) = E π ref [ ∑ t ≥ h r t | x h , y h ] . Comparing with Equation (2), we can already see that π CD ,η does not match the optimal policy π ⋆,η , as Q π ref can be arbitrarily far from Q ⋆,η . In particular, π CD may fail to optimize the KL-regularized RL objective and exhibit two failure cases, which we demonstrate with a simple MDP in Figure 4. First, we show that CD fails to maximize expected reward in this MDP, even as the KL-regularizer η decays to zero.

Theorem 4.1. Under Figure 4, CD learns to always select the left sub-tree as η → 0 , which gives a sub-optimal reward of 0 . 1 , while π ⋆,η learns to always select the right sub-tree and chooses the path that gives reward 1 .

Proof. First, for CD, we have Q π ref ( x 1 , a L ) = 0 . 1 and Q π ref ( x 1 , a R ) = 0 . 05 . Hence, CD's probability of selecting the left sub-tree is p L exp(0 . 1 /η ) p L exp(0 . 1 /η )+ p R exp(0 . 05 /η ) , which converges to 1 as η → 0 . Next, for Q♯ , we have Q ⋆,η ( x 1 , a L ) = 0 . 1 and Q ⋆,η ( x 1 , a R ) = η ln(0 . 05 exp(1 /η ) + 0 . 95) . Hence, Q♯ 's probability of selecting the left sub-tree is p L exp(0 . 1 /η ) p L exp(0 . 1 /η )+ p R (0 . 05 exp(1 /η )+0 . 95) , which converges to 0 as η → 0 . Thus, CD learns the sub-optimal path.

Figure 4: A tree MDP where edges are labeled with π ref 's action probability. π ref goes to the left sub-tree w.p. p L and the right sub-tree w.p. p R , where p L , p R &gt; 0 . The left sub-tree gives r = 0 . 1 w.p. 1 . In the right sub-tree, π ref chooses reward 1 w.p. 0 . 05 and chooses reward 0 w.p. 0 . 95 .

<!-- image -->

Next, we show that CD also incurs a higher KL than Q♯ .

Theorem 4.2. Under Figure 4, CD's KL converges to ln(1 /p L ) while Q♯ 's KL converges to ln(1 /p R ) as η → 0 . Thus if p L ≪ p R , CD converges to a higher KL than Q♯ .

Proof. As shown in Theorem 4.1, CD learns to select the left sub-tree while Q♯ learns to select the right sub-tree as η → 0 . Then, the KLs simply follow by definition.

In sum, we proved that Figure 4, CD both incurs a higher KL and achieves a lower sub-optimal reward compared to Q♯ . Thus, Q♯ generally Pareto-dominates CD in the reward-KL tradeoff, which matches our empirical findings.

## 4.2 Performance Guarantee for Q♯

We prove that the learned policy by Q♯ is guaranteed to converge to the optimal policy with enough samples. This result holds in rich-observation MDPs where the size of the state space can be exponentially large or infinite, so long as the mild realizability assumption holds.

To setup, let F be a distributional function class for modeling Z ⋆ , the reward-to-go distribution under π ref . Each element of F has type f = ( f 1 , . . . , f H ) and f h : X × Y ↦→ ∆([0 , V max ]) . 1 For the purpose of analysis, we assume access to a no-regret online learning oracle for the maximum likelihood (MLE) loss, which proceeds as follows: for each iteration k = 1 , 2 , . . . , K , given any { x h,k , y h,k , R h,k } H h =1 , the oracle outputs ̂ Z k ∈ F s.t.

<!-- formula-not-decoded -->

Here, Reg mle ( K ) denotes the cumulative regret of the MLE oracle after K iterations. No-regret online learning is well-studied in the literature [48, 49] and is a standard tool when reducing decision making to supervised learning [13, 50, 35]. For example, if F is finite and satisfies realizability, then Vovk's aggregating algorithm ensures that Reg mle ( K ) ≲ ln( |F| ) [51]. 2

## Assumption 4.3 (Realizability) . Z ⋆ ∈ F .

The following algorithm is a slightly modified version of Algorithm 1 amenable for theoretical analysis. The only differences with Algorithm 1 are: (1) we use the MLE oracle to learn ̂ Z k , and (2) for purpose of local exploration, we play a random action at the switching time h before following π ref to the end of the trajectory [31].

We now state our main PAC bound for Q♯ .

Theorem 4.4. Fix any η ∈ (0 , V max ] and δ ∈ (0 , 1) . Under Assumptions 2.1 and 4.3, Algorithm 2 ensures w.p. at least 1 -δ , setting β = ln(1 /δ ) + Reg mle ( K ) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Algorithm 2 Q♯ (Theory Version)

- 1: Input: reference π ref , iteration count K , regularizer η .

```
2: Initialize ̂ Z 1 randomly. 3: for k = 1 , 2 , . . . , K do 4: Let π k ← π ̂ Z k ,η . 5: for step h = 1 , 2 , . . . , H do 6: Roll-in with π k for h -1 steps and see x h,k . 7: Play random action y h,k and transit to x h +1 ,k . 8: Resume trajectory with π ref from x h +1 ,k . 9: Let R h,k be cumulative rewards after time h . 10: end for 11: Input { x h,k , y h,k , R h,k } h ∈ [ H ] to MLE oracle. 12: Receive ̂ Z k from MLE oracle. 13: end for 14: Output: ̂ Z 1 , . . . , ̂ Z K .
```

We highlight this applies to richobservation MDPs where our only requirement for F is realizability. Our bound only scales with the function class's complexity, i.e. , ln( |F| ) , and does not contain structural complexity measures. In contrast, prior bounds in RL theory require stronger assumptions such as Bellman completeness [52, 53, 54, 55, 56, 57, 29], even in deterministic MDPs [58], and/or scale with structural complexity measures such as coverability [59, 60], eluder dimension [61, 55], and certain rank related complexity measures [62, 63, 64].

Computational efficiency. Algorithm 2

is model-free and computationally effi- cient. In contrast, prior model-free algorithms for rich-observation MDPs perform exploration with version spaces and are computationally hard [62, 65, 55, 59, 29]. Thus, Theorem 4.4 shows that Algorithm 2 achieves both statistical and computational efficiency under mild assumptions by simply operating within the KL-regularized RL framework, which is of great relevance for post-training. We remark that [66] observed similar benefits in offline RL while we study the harder online setting.

1 Suppose rewards-to-go under π ref lie in [0 , V max ] w.p. 1 .

2 a ≲ b is short for a ≤ Cb for some universal constant C .

Second-order guarantee. Thanks to the distributional perspective, Theorem 4.4 is a second-order bound [28, 29]. Its leading term O ( √ ∑ H h =1 ∑ K k =1 CV 2 h,k ( x, y )) aggregates coefficients of variation.

In the worst case this is O ( √ ∑ H h =1 E 2 h K ) but when Z ⋆ h has small or zero variance the term vanishes, leaving the lower-order O (max h ∈ [ H ] E h ln( K )) , logarithmical in K . Thus the bound adaptively improves from O ( √ K ) to O (ln( K )) in benigh instances. Interestingly, the envelope term E h is also instance-dependent; when Q ⋆,η = V max it equals 1 , eliminating the exponential dependence on η . In general, we can tolerate an η that is as small as the worst V max -Q ⋆,η under rollouts from π ref , which is reminiscent of the condition required for first-order or small-loss bounds [50, 35, 57].

Bernoulli rewards simplification. For closed-ended tasks (e.g. math or multiple-choice), the rewardto-go Z ⋆ h ( x, y ) is Bernoulli, Z ⋆ h ( x, y ) ∼ Ber ( p h ( x, y ) ) . Then the CV term can be bounded by CV h,k ≤ E π k ◦ Unif √ (1 -p h ) /p h and the envelope term becomes ∥ 1 /p h ∥ L ∞ ( π ref ) , which notably does not have exponential dependence on 1 /η . Thus, as long as the reference model π ref has sufficient probability of solving the problem, our bound can be made independent of η . Finally, we note that the distributional-realizability condition can also be weakened to mean-realizability, since the only parameter of a Bernoulli distribution is its mean; also the MLE loss reduces to the binary cross-entropy loss [50, 57]. We present the corollary below and the proof in Appendix B.1.

Corollary 4.5. Suppose reward-to-gos are Bernoulli: Z ⋆ h ( x, y ) ∼ Ber( p h ( x, y )) . Then, under the setup of Theorem 4.4, the bound can be simplified to:

<!-- formula-not-decoded -->

Remark: Modification for Regret Bound. It is possible to turn Theorem 4.4 into a regret bound by replacing random action in Line 7 of Algorithm 2 with a no-regret contextual bandit oracle, where 'context' is x h , action is y h and 'reward' is R h . This is alike the steps needed to convert AggreVaTe's PAC bound into a regret bound [31]. Our theory can be interpreted as a regret/PAC reduction from KL-regularized RL in deterministic MDPs to no-regret online learning, which mirrors the type of imitation learning guarantees obtained for AggreVaTe [31].

## 5 Limitations &amp; Conclusion

Our results focus on deterministic MDPs including LLM post-training, where the optimal action-value Q ⋆,η is a simple functional of the reference return distribution Z π ref and Theorem 2.2 applies directly. For domains with stochastic transitions such as multi-agent game playing where the next state depends on the (potentially unpredictable) behavior of the other player, Q ⋆,η need to be learned via temporal-difference methods, which typically rely on the stronger Bellman-completeness assumption and may introduce additional training instability. In summary, Q♯ offers a principled and practical avenue for post-training LLMs. It combines a distributional-RL objective with supervised regression, enjoys provable convergence under mild assumptions, and consistently surpasses prior value-based baselines on synthetic planning and math-reasoning benchmarks-achieving higher accuracy while maintaining a lower KL divergence from the reference policy.

## Acknowledgment

JPZ is supported by a grant from the Natural Sciences and Engineering Research Council of Canada (NSERC) (567916). KW is supported by a Google PhD Fellowship. ZG is supported by LinkedInCornell Grant. Wen Sun is supported by NSF IIS-2154711, NSF CAREER 2339395 and DARPA LANCER: LeArning Network CybERagents. This research is also supported by grants from the National Science Foundation NSF (IIS-1846210, IIS-2107161, and IIS-1724282, HDR-2118310), the Cornell Center for Materials Research with funding from the NSF MRSEC program (DMR-1719875), DARPA, arXiv, LinkedIn, Google, and the New York Presbyterian Hospital.

## References

- [1] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems , 30, 2017.
- [2] Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, and Aviral Kumar. Rewarding progress: Scaling automated process verifiers for llm reasoning. arXiv preprint arXiv:2410.08146 , 2024.
- [3] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 , 2025.
- [4] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems , 35:27730-27744, 2022.
- [5] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 , 2024.
- [6] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118 , 2024.
- [7] Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 reinforce samples, get a baseline for free! 2019.
- [8] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 , 2017.
- [9] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems , 36, 2024.
- [10] Jing-Cheng Pang, Pengyuan Wang, Kaiyuan Li, Xiong-Hui Chen, Jiacheng Xu, Zongzhang Zhang, and Yang Yu. Language model self-improvement by reinforcement learning contemplation. arXiv preprint arXiv:2305.14483 , 2023.
- [11] Sidharth Mudgal, Jong Lee, Harish Ganapathy, YaGuang Li, Tao Wang, Yanping Huang, Zhifeng Chen, Heng-Tze Cheng, Michael Collins, Trevor Strohman, et al. Controlled decoding from language models. arXiv preprint arXiv:2310.17022 , 2023.
- [12] Seungwook Han, Idan Shenfeld, Akash Srivastava, Yoon Kim, and Pulkit Agrawal. Value augmented sampling for language model alignment and personalization. arXiv preprint arXiv:2405.06639 , 2024.
- [13] Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics , pages 627-635. JMLR Workshop and Conference Proceedings, 2011.
- [14] Gerald Tesauro. Practical issues in temporal difference learning. Advances in neural information processing systems , 4, 1991.
- [15] Hado Van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double q-learning. In Proceedings of the AAAI conference on artificial intelligence , volume 30, 2016.
- [16] Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for offline reinforcement learning. Advances in neural information processing systems , 33:11791191, 2020.

- [17] Marc G Bellemare, Will Dabney, and Rémi Munos. A distributional perspective on reinforcement learning. In International conference on machine learning , pages 449-458. PMLR, 2017.
- [18] Gregor Bachmann and Vaishnavh Nagarajan. The pitfalls of next-token prediction. arXiv preprint arXiv:2403.06963 , 2024.
- [19] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168 , 2021.
- [20] Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, Anind K Dey, et al. Maximum entropy inverse reinforcement learning. In Aaai , volume 8, pages 1433-1438. Chicago, IL, USA, 2008.
- [21] Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky TQ Chen. Adjoint matching: Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control. arXiv preprint arXiv:2409.08861 , 2024.
- [22] Alexandre Piché, Valentin Thomas, Cyril Ibrahim, Yoshua Bengio, and Chris Pal. Probabilistic planning with sequential monte carlo methods. In International Conference on Learning Representations , 2018.
- [23] Xiner Li, Yulai Zhao, Chenyu Wang, Gabriele Scalia, Gokcen Eraslan, Surag Nair, Tommaso Biancalani, Shuiwang Ji, Aviv Regev, Sergey Levine, et al. Derivative-free guidance in continuous and discrete diffusion models with soft value-based decoding. arXiv preprint arXiv:2408.08252 , 2024.
- [24] Hado Van Hasselt, Yotam Doron, Florian Strub, Matteo Hessel, Nicolas Sonnerat, and Joseph Modayil. Deep reinforcement learning and the deadly triad. arXiv preprint arXiv:1812.02648 , 2018.
- [25] Rémi Munos and Csaba Szepesvári. Finite-time bounds for fitted value iteration. Journal of Machine Learning Research , 9(5), 2008.
- [26] Clare Lyle, Marc G Bellemare, and Pablo Samuel Castro. A comparative analysis of expected and distributional reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 33, pages 4504-4511, 2019.
- [27] Mark Rowland, Yunhao Tang, Clare Lyle, Rémi Munos, Marc G Bellemare, and Will Dabney. The statistical benefits of quantile temporal-difference learning for value estimation. In International Conference on Machine Learning , pages 29210-29231. PMLR, 2023.
- [28] Kaiwen Wang, Nathan Kallus, and Wen Sun. The central role of the loss function in reinforcement learning. arXiv preprint arXiv:2409.12799 , 2024.
- [29] Kaiwen Wang, Owen Oertell, Alekh Agarwal, Nathan Kallus, and Wen Sun. More benefits of being distributional: Second-order bounds for reinforcement learning. International Conference of Machine Learning , 2024.
- [30] Marc G Bellemare, Will Dabney, and Mark Rowland. Distributional reinforcement learning . MIT Press, 2023.
- [31] Stephane Ross and J Andrew Bagnell. Reinforcement and imitation learning via interactive no-regret learning. arXiv preprint arXiv:1406.5979 , 2014.
- [32] Wen Sun, Arun Venkatraman, Geoffrey J Gordon, Byron Boots, and J Andrew Bagnell. Deeply aggrevated: Differentiable imitation learning for sequential prediction. In International conference on machine learning , pages 3309-3318. PMLR, 2017.
- [33] Jonathan D Chang, Kiante Brantley, Rajkumar Ramamurthy, Dipendra Misra, and Wen Sun. Learning to generate better than your llm. arXiv preprint arXiv:2306.11816 , 2023.
- [34] Hanning Zhang, Pengcheng Wang, Shizhe Diao, Yong Lin, Rui Pan, Hanze Dong, Dylan Zhang, Pavlo Molchanov, and Tong Zhang. Entropy-regularized process reward model. arXiv preprint arXiv:2412.11006 , 2024.

- [35] Kaiwen Wang, Kevin Zhou, Runzhe Wu, Nathan Kallus, and Wen Sun. The benefits of being distributional: Small-loss bounds for reinforcement learning. Advances in Neural Information Processing Systems , 36, 2023.
- [36] Edward S Hu, Kwangjun Ahn, Qinghua Liu, Haoran Xu, Manan Tomar, Ada Langford, Dinesh Jayaraman, Alex Lamb, and John Langford. Learning to achieve goals with belief state transformers. arXiv preprint arXiv:2410.23506 , 2024.
- [37] Shai Shalev-Shwartz, Ohad Shamir, and Shaked Shammah. Failures of gradient-based deep learning. In International Conference on Machine Learning , pages 3067-3075. PMLR, 2017.
- [38] Boaz Barak, Benjamin Edelman, Surbhi Goel, Sham Kakade, Eran Malach, and Cyril Zhang. Hidden progress in deep learning: Sgd learns parities near the computational limit. Advances in Neural Information Processing Systems , 35:21750-21764, 2022.
- [39] Emmanuel Abbe and Colin Sandon. Polynomial-time universality and limitations of deep learning. Communications on Pure and Applied Mathematics , 76(11):3493-3549, 2023.
- [40] Yiwen Kou, Zixiang Chen, Quanquan Gu, and Sham Kakade. Matching the statistical query lower bound for k -sparse parity problems with sign stochastic gradient descent. Advances in Neural Information Processing Systems , 37:113001-113037, 2024.
- [41] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740 , 2024.
- [42] Richard Yuanzhe Pang, Weizhe Yuan, Kyunghyun Cho, He He, Sainbayar Sukhbaatar, and Jason Weston. Iterative reasoning preference optimization. arXiv preprint arXiv:2404.19733 , 2024.
- [43] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874 , 2021.
- [44] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. arXiv preprint arXiv:2305.20050 , 2023.
- [45] Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 9426-9439, 2024.
- [46] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115 , 2024.
- [47] Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi, Nikita Nangia, Jason Phang, Angelica Chen, Vishakh Padmakumar, Johnny Ma, Jana Thompson, He He, et al. Quality: Question answering with long input texts, yes! arXiv preprint arXiv:2112.08608 , 2021.
- [48] Nicolo Cesa-Bianchi and Gábor Lugosi. Prediction, learning, and games . Cambridge university press, 2006.
- [49] Francesco Orabona. A modern introduction to online learning. arXiv preprint arXiv:1912.13213 , 2019.
- [50] Dylan J Foster and Akshay Krishnamurthy. Efficient first-order contextual bandits: Prediction, allocation, and triangular discrimination. Advances in Neural Information Processing Systems , 34:18907-18919, 2021.
- [51] Vladimir G Vovk. A game of prediction with expert advice. In Proceedings of the eighth annual conference on Computational learning theory , pages 51-60, 1995.

- [52] Jinglin Chen and Nan Jiang. Information-theoretic considerations in batch reinforcement learning. In International Conference on Machine Learning , pages 1042-1051. PMLR, 2019.
- [53] Yuanhao Wang, Ruosong Wang, and Sham Kakade. An exponential lower bound for linearly realizable mdp with constant suboptimality gap. Advances in Neural Information Processing Systems , 34:9521-9533, 2021.
- [54] Dylan J Foster, Akshay Krishnamurthy, David Simchi-Levi, and Yunzong Xu. Offline reinforcement learning: Fundamental barriers for value function approximation. arXiv preprint arXiv:2111.10919 , 2021.
- [55] Chi Jin, Qinghua Liu, and Sobhan Miryoosefi. Bellman eluder dimension: New rich classes of rl problems, and sample-efficient algorithms. Advances in neural information processing systems , 34:13406-13418, 2021.
- [56] Jonathan Chang, Kaiwen Wang, Nathan Kallus, and Wen Sun. Learning bellman complete representations for offline policy evaluation. In International Conference on Machine Learning , pages 2938-2971. PMLR, 2022.
- [57] Alex Ayoub, Kaiwen Wang, Vincent Liu, Samuel Robertson, James McInerney, Dawen Liang, Nathan Kallus, and Csaba Szepesvari. Switching the loss reduces the cost in batch reinforcement learning. In Forty-first International Conference on Machine Learning , 2024.
- [58] Runzhe Wu, Ayush Sekhari, Akshay Krishnamurthy, and Wen Sun. Computationally efficient rl under linear bellman completeness for deterministic dynamics. arXiv preprint arXiv:2406.11810 , 2024.
- [59] Tengyang Xie, Dylan J Foster, Yu Bai, Nan Jiang, and Sham M Kakade. The role of coverage in online reinforcement learning. arXiv preprint arXiv:2210.04157 , 2022.
- [60] Zakaria Mhammedi, Dylan J Foster, and Alexander Rakhlin. The power of resets in online reinforcement learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [61] Daniel Russo and Benjamin Van Roy. Eluder dimension and the sample complexity of optimistic exploration. Advances in Neural Information Processing Systems , 26, 2013.
- [62] Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E Schapire. Contextual decision processes with low bellman rank are pac-learnable. In International Conference on Machine Learning , pages 1704-1713. PMLR, 2017.
- [63] Wen Sun, Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, and John Langford. Model-based rl in contextual decision processes: Pac bounds and exponential improvements over model-free approaches. In Conference on learning theory , pages 2898-2933. PMLR, 2019.
- [64] Simon Du, Sham Kakade, Jason Lee, Shachar Lovett, Gaurav Mahajan, Wen Sun, and Ruosong Wang. Bilinear classes: A structural framework for provable generalization in rl. In International Conference on Machine Learning , pages 2826-2836. PMLR, 2021.
- [65] Christoph Dann, Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E Schapire. On oracle-efficient pac rl with rich observations. Advances in neural information processing systems , 31, 2018.
- [66] Masatoshi Uehara, Nathan Kallus, Jason D Lee, and Wen Sun. Offline minimax soft-q-learning under realizability and partial coverage. Advances in Neural Information Processing Systems , 36, 2023.
- [67] Kevin Yang and Dan Klein. FUDGE: Controlled text generation with future discriminators. In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou, editors, Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , pages 3511-3535, Online, June 2021. Association for Computational Linguistics.

- [68] Stephen Zhao, Rob Brekelmans, Alireza Makhzani, and Roger Grosse. Probabilistic inference in language models via twisted sequential monte carlo. arXiv preprint arXiv:2404.17546 , 2024.
- [69] Will Dabney, Mark Rowland, Marc Bellemare, and Rémi Munos. Distributional reinforcement learning with quantile regression. In Proceedings of the AAAI conference on artificial intelligence , volume 32, 2018.
- [70] Jesse Farebrother, Jordi Orbay, Quan Vuong, Adrien Ali Taïga, Yevgen Chebotar, Ted Xiao, Alex Irpan, Sergey Levine, Pablo Samuel Castro, Aleksandra Faust, et al. Stop regressing: Training value functions via classification for scalable deep rl. arXiv preprint arXiv:2403.03950 , 2024.
- [71] Kaiwen Wang, Dawen Liang, Nathan Kallus, and Wen Sun. Risk-sensitive rl with optimized certainty equivalents via reduction to standard rl. arXiv preprint arXiv:2403.06323 , 2024.
- [72] Ke Sun, Bei Jiang, and Linglong Kong. How does return distribution in distributional reinforcement learning help optimization? arXiv preprint arXiv:2209.14513 , 2022.
- [73] Ke Sun, Yingnan Zhao, Wulong Liu, Bei Jiang, and Linglong Kong. Distributional reinforcement learning with regularized wasserstein loss. Advances in Neural Information Processing Systems , 37:63184-63221, 2024.
- [74] Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction , volume 1. MIT press Cambridge, 1998.
- [75] Alisa Liu, Xiaochuang Han, Yizhong Wang, Yulia Tsvetkov, Yejin Choi, and Noah A. Smith. Tuning language models by proxy. In First Conference on Language Modeling , 2024.
- [76] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning , pages 19274-19286. PMLR, 2023.
- [77] Wei Xiong, Hanze Dong, Chenlu Ye, Han Zhong, Nan Jiang, and Tong Zhang. Gibbs sampling from human feedback: A provable kl-constrained framework for rlhf. CoRR , 2023.
- [78] Richard Yuanzhe Pang, Weizhe Yuan, He He, Kyunghyun Cho, Sainbayar Sukhbaatar, and Jason Weston. Iterative reasoning preference optimization. Advances in Neural Information Processing Systems , 37:116617-116637, 2024.
- [79] Zhaolin Gao, Jonathan Chang, Wenhao Zhan, Owen Oertell, Gokul Swamy, Kianté Brantley, Thorsten Joachims, Drew Bagnell, Jason D Lee, and Wen Sun. Rebel: Reinforcement learning via regressing relative rewards. Advances in Neural Information Processing Systems , 37:5235452400, 2025.
- [80] Tengyang Xie, Dylan J Foster, Akshay Krishnamurthy, Corby Rosset, Ahmed Awadallah, and Alexander Rakhlin. Exploratory preference optimization: Harnessing implicit q*-approximation for sample-efficient rlhf. arXiv preprint arXiv:2405.21046 , 2024.
- [81] Jae Hyeon Cho, Minkyung Park, and Byung-Jun Lee. Vpo: Leveraging the number of votes in preference optimization. arXiv preprint arXiv:2410.22891 , 2024.
- [82] Shenao Zhang, Donghan Yu, Hiteshi Sharma, Han Zhong, Zhihan Liu, Ziyi Yang, Shuohang Wang, Hany Hassan, and Zhaoran Wang. Self-exploring language models: Active preference elicitation for online alignment. arXiv preprint arXiv:2405.19332 , 2024.
- [83] Roy Fox, Ari Pakman, and Naftali Tishby. Taming the noise in reinforcement learning via soft updates. arXiv preprint arXiv:1512.08562 , 2015.
- [84] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning , pages 1861-1870. PMLR, 2018.
- [85] Ofir Nachum, Yinlam Chow, Bo Dai, and Lihong Li. Dualdice: Behavior-agnostic estimation of discounted stationary distribution corrections. Advances in neural information processing systems , 32, 2019.

- [86] Ofir Nachum, Bo Dai, Ilya Kostrikov, Yinlam Chow, Lihong Li, and Dale Schuurmans. Algaedice: Policy gradient from arbitrary experience. arXiv preprint arXiv:1912.02074 , 2019.
- [87] Dylan J Foster, Sham M Kakade, Jian Qian, and Alexander Rakhlin. The statistical complexity of interactive decision making. arXiv preprint arXiv:2112.13487 , 2021.
- [88] Monroe D Donsker and SR Srinivasa Varadhan. Asymptotic evaluation of certain markov process expectations for large time. iv. Communications on pure and applied mathematics , 36(2):183-212, 1983.
- [89] Edward S Hu, Kwangjun Ahn, Qinghua Liu, Haoran Xu, Manan Tomar, Ada Langford, Jayden Teoh, Bryon Xu, David Yan, Dinesh Jayaraman, et al. The belief state transformer. arXiv preprint arXiv:2410.23506 , 2024.
- [90] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog , 1(8):9, 2019.
- [91] Ilya Loshchilov, Frank Hutter, et al. Fixing weight decay regularization in adam. arXiv preprint arXiv:1711.05101 , 5, 2017.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Please see the last paragraph of our introduction which includes both our claims and the sections where the claim is supported.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations in Section 5.

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

Justification: Please see Appendix B for the complete assumptions and proofs.

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

Justification: Please see Appendix C, D, E for the information needed to reproduce our results.

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

Justification: We provide our code with instructions in the supplementary materials.

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

Justification: All training and test details are provided in Section 3.1, 3.2 and Appendix C, D, E, F.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: The experiments are reported on datasets that are significantly large.

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

Justification: The compute resources used are reported in Appendix C.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in the paper conforms with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

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

Answer: [Yes]

Justification: The benchmarks and data splits are publicly available. All licenses are respected.

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

Justification: Assets will be released, and all instructions and details are included for reproduction.

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

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Related Works

From the empirical side, the most relevant works are controlled decoding (CD; [11]) and value augmented sampling (VAS; [12]). These two works both propose to guide the reference policy π ref with Q π ref , 0 , the expected reward-to-go under π ref without KL regularization. As discussed in Section 4.1, guiding with Q π ref , 0 is not principled for the KL-regularized RL problem and can lead to both sub-optimal reward and large KL from π ref . In contrast, we propose to guide π ref with Q ⋆,η , the expected reward-to-go under the optimal policy with KL regularization, which is the correct closed-form of the optimal policy. A recent work [34] proposed a process reward model (PRM) of a similar form as our Q ⋆,η , but their PRM is applied to steps instead of tokens, and they do not use distributional RL or iterative training (i.e., data aggregation).

In terms of reweighting π ref with classifier scores, FUDGE [67] is another closely related work but their derivation is based on Bayes rule and FUDGE does not solve KL-regularized RL. Sequential Monte Carlo (SMC) methods [22, 68] also reweight π ref 's distribution with a twist function, where the optimal twist function is analogous to our Q ⋆,η . One key difference is that SMC performs resampling while we directly combine logits of π ref and exp( Q ⋆,η ) to avoid importance sampling, which has higher variance. Finally, none of these prior works apply distributional RL losses [17, 69, 70, 57] or online data aggregation [13] to learn Q ⋆,η , which we showed to be beneficial in our ablations. Indeed, CD and VAS both use square loss regression over a fixed offline dataset. We also remark that risk-sensitive RL has been an important application of distributional RL [69, 71] and extending Q♯ along those lines is a promising future direction.

Wealso discuss some of the recent advances in stable distributional RL. [72] shows that the categorical distributional RL loss, which we employ for our theory and experiments, enjoys smoothness and optimization stability under a bounded logit condition. [73] introduces a Sinkhorn distributional RL loss which is a computationally efficient alternative for Wasserstein distance, and was shown to be more stable for multi-dimensional rewards. [69] proposed a KL-regularized categorical loss which they showed is empirically more stable in Atari games. However, these references all apply TD-learning with function approximation and replay buffers, which [74] identified as a deadly triad that is notoriously difficult to scale, requiring many tricks such as double Q-learning and target networks. In contrast, our work obviates the need for TD-learning or tricks such as the target network by leveraging the special form of Q ⋆ in deterministic KL-regularized MDPs, which perfectly captures the LLM post-training application we focus on.

We also cite some tangentially related works. Proxy tuning [75] and speculative decoding [76] both use a small model to guide the logit distribution of a large π ref model. Speculative decoding is focused on maximizing the large model's likelihood, which does not relate to any extrinsic rewards. In our framework, the classifier model can be any size relative to π ref , although deeper investigation into the computational benefits of using a small classifier is a promising direction for future work. We note that the star-graph problem can also be solved during pre-training by also predicting backwards via the belief state transformer [36].

Finally we discuss previous post-training methods for LLMs. First, online iterative DPO [77, 78], REBEL [79], PPO [8], etc. are based on policy gradient and require a good reset distribution which only guarantees local optimality. XPO [80], VPO [81], SELM [82], etc. treat this as an exploration setting but requires solving non-convex optimization oracles and relies on strong structure conditions such as coverability / eluder / linearity, similar to the theoretical works like [55, 59]. Instead, we approach post-training in a fundamentally different angle and solve it via simple computationally tractable regression and mle oracles, without any strong structural conditions or reset distribution assumptions.

From the theoretical side, KL-regularized RL is closely related to soft RL or maximum entropy RL which are well-studied [20, 83, 84, 22]. The optimal policy decomposition in deterministic MDPs is also known in prior works [23, 21]. Our contribution is an algorithm that provably learns Q ⋆,η using distributional RL [17] and data aggregation [13]. This enables us to prove a reduction of KL-regularized RL (in deterministic MDPs) to no-regret online learning, which ensures convergence to the optimal policy with realizability being the only assumption for function approximation. Notably we are able to avoid more stringent conditions such as completeness or structural MDP conditions which are ubiquitous in the current literature [53, 55, 56, 35, 29, 57, 59]. [66] observed similar benefits in offline RL, while we provide guarantees for the harder online RL setting.

Complementary to our online, KL-regularized setting, DualDICE [85] and AlgaeDICE [86] tackle the high-variance "curse of horizon" that arises when one performs importance weighting for long trajectories in offline RL. Both methods replace per-step importance weights with stationary-distribution density ratios, learned through a dual (Lagrangian) formulation, and have shown empirical success on low-dimensional continuous-control benchmarks, although learning is also shown to be difficult in high-dimensional control tasks [56]. Because we continually collect on-policy data and constrain updates via an explicit KL penalty-which already limits distribution shift-we do not need such ratio estimation; nonetheless, density-ratio approaches remain a promising orthogonal direction for variance reduction in purely offline LLM post-training.

We remark that our theoretical guarantees are quite similar in structure to that of AggreVaTe [31, 32], which is a reduction of imitation learning to no-regret online learning. Besides the obvious difference in problem setting, another improvement from our work is using distributional RL theory to prove second-order bounds. Notably, we are able to prove second-order bounds without any completeness assumptions that were required in [35, 28, 29].

## B Proofs

In this section, we provide the full proof for Theorem 4.4.

Theorem 4.4. Fix any η ∈ (0 , V max ] and δ ∈ (0 , 1) . Under Assumptions 2.1 and 4.3, Algorithm 2 ensures w.p. at least 1 -δ , setting β = ln(1 /δ ) + Reg mle ( K ) , we have

<!-- formula-not-decoded -->

Proof. Fix any η ∈ (0 , V max ) . Let Q h,k ( x, y ) = η ln E z ∼ ̂ Z h,k ( x,y ) exp( z/η ) denote the induced soft Q function from the distributional estimate ̂ Z k . Let π k h ( y | x ) ∝ π ref h ( y | x ) exp( Q h,k ( x, y ) /η ) denote the induced policy from Q h,k . Then,

<!-- formula-not-decoded -->

where (i) is by the performance difference lemma in the soft MDP (Lemma B.2); (ii) is by DonskerVaradhan (Lemma B.1) which proves that π k h ( x h ) = arg max π E π [ Q h,k ( x h , π ) -KL ( π ( x h ) ∥ π ref h ( x h ))] . Now, we bound the difference between the optimal and learned Q functions:

<!-- formula-not-decoded -->

where (i) is by Lemma B.4 and the fact that Z ⋆ , ̂ Z k ∈ [0 , V max ] and H h,k ( x, y ) := H ( Z ⋆ h ( x, y ) , ̂ Z h,k ( x, y )) is the Hellinger distance between the learned ̂ Z h,k and optimal Z ⋆ h .

Thus, if we let x h , y h ∼ π k ◦ h Unif( A ) denote the distribution of rolling in with π k until x h and taking a random y h ∼ Unif( A ) , then we have:

<!-- formula-not-decoded -->

The final step is to bound the summed Hellinger square terms. This can be done via Multiplicative Azuma's inequality and [87, Lemma A.14], which shows that for any δ ∈ (0 , 1) , we have ∑ h,k E π k ◦ h Unif( A ) [ H 2 h,k ( x h , y h )] ≲ ∑ h,k H 2 h,k ( x h,k , y h,k ) + ln(1 /δ ) ≲ Reg mle ( K ) + ln(1 /δ ) , which recall is exactly the definition of β . This finishes the proof of Theorem 4.4.

Lemma B.1 (Donsker-Varadhan's Variational Formula; [88]) . For any prior p ∈ ∆(Θ) , consider the KL-regularized optimization:

<!-- formula-not-decoded -->

The optimal policy π ⋆ is given by π ⋆ ( θ ) ∝ p ( θ ) exp( Q ( θ ) /η ) and it has value V ( π ⋆ ) = η ln E θ ∼ p exp( Q ( θ ) /η ) .

Lemma B.2 (Soft Performance Difference Lemma (PDL)) . For any f and π ,

<!-- formula-not-decoded -->

For any π, π ′ ,

<!-- formula-not-decoded -->

Proof. Let KL ( π h ( x h )) := KL ( π h ( x h ) ∥ π ref h ( x h )) denote KL-divergence w.r.t. π ref . Then,

<!-- formula-not-decoded -->

Lemma B.3. For any two numbers x, y ∈ [exp( a ) , exp( b )] , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. If x ≥ y , then ln( x ) -ln( y ) = ln(1+( x -y ) /y ) ≤ ( x -y ) /y . If x &lt; y , then ln( y ) -ln( x ) = -ln(1+( x -y ) /y ) . By premise, we have 0 ≥ x -y y ≥ exp( a -b ) -1 . Note that -ln(1+ z ) is convex and is thus upper bounded by the line connecting (0 , 0) and (exp( a -b ) -1 , b -a ) , i.e. , -ln(1+ z ) ≤ b -a 1 -exp( a -b ) | z | for 0 ≥ z ≥ exp( a -b ) -1 . Thus, -ln(1 + ( x -y ) /y ) ≤ b -a 1 -exp( a -b ) ∣ ∣ ∣ x -y y ∣ ∣ ∣ . Thus, we've shown that | ln( x ) -ln( y ) | ≤ max ( 1 , b -a 1 -exp( a -b ) ) | x -y y | . Finally, since x 1 -exp( -x ) ≤ 1 + x when x ≥ 0 , we have max(1 , b -a 1 -exp( a -b ) ) ≤ max(1 , 1 + b -a ) = 1 + b -a .

Lemma B.4. For any distributions p, q on [ a, b ] , we have

<!-- formula-not-decoded -->

where H 2 ( p, q ) = 1 2 ∑ z ( √ p ( z ) -√ q ( z )) 2 is the squared Hellinger distance.

Proof. By Lemma B.3, we have | ln ∑ z p ( z ) e z -ln ∑ z q ( z ) e z | ≤ (1 + b -a ) ∣ ∣ ∣ ∑ z ( p ( z ) -q ( z )) e z ∑ z q ( z ) e z ∣ ∣ ∣ . By Lemma B.5, we have that the numerator is bounded by √ Var q ( e z ) H ( p, q ) + (exp( b ) -exp( a )) H 2 ( p, q ) .

Lemma B.5 (Second-Order Lemma) . Suppose p, q are distributions on the interval [ a, b ] . Then, we have

<!-- formula-not-decoded -->

Proof. Define p ′ , q ′ as the normalized distributions on [0 , 1] , i.e. , p ′ is the law of X ′ = ( X -a ) / ( b -a ) where X ∼ p . Then, we have

<!-- formula-not-decoded -->

where the ≲ step is due to the second-order lemma of [28].

## B.1 Case of Bernoulli reward-to-go

In this section, we focus on problems where Z ⋆ h ( x, y ) = Ber( p h ( x, y )) is a Bernoulli distribution, which is common for closed-ended problems such as math or multiple choice. Here, the envelope term can be bounded as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Fix x, y and let p = p h ( x, y ) . Then, it suffices to show that

<!-- formula-not-decoded -->

This is indeed true because

<!-- formula-not-decoded -->

We can also bound the coefficient of variance in terms of the Bernoulli parameter.

Lemma B.7. If Z ⋆ h ( x, y ) = Ber( p h ( x, y )) , then for all η &gt; 0 , we have

<!-- formula-not-decoded -->

Proof. Fix x, y and let p = p h ( x, y ) . Then, the variance term is:

<!-- formula-not-decoded -->

Thus, the CV is:

<!-- formula-not-decoded -->

Figure 5: Full results for star-graph experiments. The empty bar for G (5 , 5) and G (3 , 8) are for DPO, which pushed down both the chosen and reject paths resulting in 0 accuracy.

<!-- image -->

## C Additional Discussion and Implementation Details for Star-Graph

The shortcut behavior, also known as the Clever Hans Trick [18], in the star-graph task arises directly from the auto-regressive next-token prediction objective. Specifically, the model minimizes loss by memorizing the first token seen during training and following the corresponding edge, achieving low training error but generalizing poorly at test time when the initial token is not provided. This leads to a brittle, shortcut-based policy.

Policy-based methods such as REINFORCE and RPO attempt to correct this by upweighting highreward trajectories. However, because their loss is still based on the product of next-token prediction probabilities, the same as in pretraining, they are vulnerable to the same shortcut and require exponentially many samples via gradient descent on the policy to correct it once it is learned (Theorem 1 of [89]).

In contrast, Q♯ does not depend on myopic next-token supervision. Instead, it learns to predict the cumulative reward-to-go from each (prefix, token) pair under the reference policy, and uses this to guide generation toward optimal completions. This token-level value modeling allows Q♯ to predict future outcome and assign higher value to early tokens that lead to long-term reward. In other words, Q♯ 's loss function is directly trained to perform planning, making it robust to the Clever Hans Trick [18] that undermines next-token-based methods. As shown in Figure 5, both Q♯ and CD are able to solve the star-graph task near-perfectly, while policy-based methods perform at random-guess level.

We follow the setup of [18] and reused their official code for producing the star-graph results. We used the GPT-2 small model for graphs G (2 , 5) , G (5 , 5) and the GPT-2 medium model for G (3 , 8) [90]. 3 We first pretrain these models with next-token prediction on a pretraining set of 200 k random graphs and correct paths. We call this the resultant model the 'pre-trained' model, and as observed by [18], these models have the Clever Hans shortcut so they do not generalize well on unseen test graphs. We highlight that this is a failure in generalization, since the pre-trained model achieves near-perfect accuracy on the training set but only 1 /d accuracy on the test set.

In order to fix the Clever Hans shortcut, we perform post-training with two common baselines REINFORCE [41] and DPO [9], RPO [42] - as well as our algorithm Q♯ . The post-training is done on another set of 200 k random graphs. For REINFORCE, the reward function we use is 1 if the response is correct, and -0 . 1 if incorrect. We noticed that if the incorrect reward is too negative, this causes model collapsing to accuracy of 0 . For DPO and RPO, we sampled pairwise responses ( y chosen , y reject ) where y chosen is the correct path and y reject is an incorrect shortcut path sampled from the pretrained model. For Q♯ , we also trained the classifier on the same dataset of pairwise responses, where correct paths are marked with reward 1 and incorrect responses are marked with reward 0 . Throughout, we used the AdamW optimizer with weight decay 0 . 1 and batch size of 256 , and trained for 10 epochs. The learning rates were 2 . 5 e -4 for pre-training; 1 e -5 for REINFORCE; 1 e -4 for DPO and RPO; 1 e -4 for classifier-based CD and Q♯ . All models are trained on a single A100 or H100 GPU. All models were evaluated on a separate test set of 20 k graphs, using top-k 10 and

3 Models from https://huggingface.co/openai-community/gpt2 and https://huggingface.co/openaicommunity/gpt2-medium.

temperature 1 . 0 . For Q♯ and CD, we use η = 0 . 1 . We found that DPO often pushed down the probabilities of both the chosen and reject paths, leading to poor performance even on the training set; RPO fixed this issue and so we report the RPO numbers.

## D Additional Model Details

π ref models. All models we use in the experiments are the "Instruct" versions. That is, Llama 3 8B refers to meta-llama/Meta-Llama-3-8B-Instruct and we use the default chat template and system message from Meta to interact with them.

Q♯ models. Two variants for Q♯ are implemented and experimented: Q-type and V-type. Specifically, the Q-type takes input of a partial generation x and computes Q ⋆,η ( x, y ) for all y in the vocabulary of the π ref model whereas the V-type takes input of concatenated x and a specific token ˆ y and outputs a single value that represents Q ⋆,η ( x, ˆ y ) . Because of the key difference, Q-type therefore can efficiently calculate Q ⋆,η with just one forward pass and its model architecture can also be identical to the original LLM. V-type, however, has a prohibitive inference cost with a naive implementation since it requires making | V | forward passes at every decoding step to calculate the full Q function. In the paragraph below, we discuss our efficient implementation to address this issue. For Q-type, we initialize the model directly from Llama 3.2 1B and for V-type, we replace the last layer of Llama 3.2 1B with a randomly initialized fully connected layer with output size of 1. Therefore, V-type Q♯ also has slightly fewer number of parameters than Q-type. We by default use V-type Q♯ in our experiments.

Efficient inference with V-type. To speed up inference for V-type, we note that not all tokens in the vocabulary are worth computing its value since for any partial generation x , most tokens have extremely low probability from π ref as the next token candidate. In our preliminary experiments, we have found that only computing the values for the top 20 tokens ranked by π ref give similar performance compared to computing for all tokens. Additionally, we also note that the values for these tokens can be computed in one forward pass. To accomplish this, we input a partial generation x and the top 20 candidate next tokens together, modify the attention mask so that the candidate tokens do not attend to each other but still to x . This allows us to compute the values for these top tokens in just one additional forward pass without any approximation.

## E Q♯ Training Settings

We collect 16 samples for each question in the training set and label every sample either as correct (1) or incorrect (0) based on the final answer. The first round of training data is collected with just π ref . For training Q♯ model, we filter out samples from questions where all samples are either correct or incorrect. we use a learning rate of 2 e -5 and weight decay of 0 . 01 with AdamW optimizer [91]. The model is trained for 5 epochs. We train Q♯ for two iterations as we observe performance converges. In the second iteration, we repeat the above data collection procedure and concatenate the training data from the first round. The model is always trained from scratch between iterations.

## F Additional Evaluation Details

We evaluate all methods and models with zero-shot prompting. The prompt template is 'Problem:\n\n{0} Write your answer inside \\boxed{{}}.\n\nSolution:' where {0} is replaced by the actual question from the dataset. The MATH-500 dataset can also be found at Huggingface 4 .

## G Math Reasoning Results on Qwen 2.5

We conduct experiments using Qwen 2.5 [46], where a 1.5B model guides the 7B version on GSM8K, MATH and AIME-24 (Table 5). All other configurations mirror those used with Llama 3. We find that Q♯ consistently outperforms both π ref and CD across all datasets, achieving higher accuracy with

4 https://huggingface.co/datasets/HuggingFaceH4/MATH-500

lower KL divergence. Compared to Table 1, Qwen 2.5 yields stronger overall performance, likely due to its stronger base model, demonstrating that Q♯ generalizes well across model families.

Table 5: Comparison of Q♯ with π ref and CD baseline on GSM8K (left), MATH (middle) and AIME-24 (right) with Qwen 2.5.

| Dataset         | GSM8K   | GSM8K   | GSM8K   | MATH   | MATH   | MATH   | AIME-24   | AIME-24   | AIME-24   |
|-----------------|---------|---------|---------|--------|--------|--------|-----------|-----------|-----------|
| Methods         | π ref   | CD      | Q♯      | π ref  | CD     | Q♯     | π ref     | CD        | Q♯        |
| pass@1 ↑        | 76.1    | 79.0    | 83.5    | 58.6   | 60.7   | 61.9   | 9.3       | 13.5      | 14.1      |
| maj1@8 ↑        | 92.9    | 93.1    | 93.8    | 72.8   | 74.2   | 74.8   | 16.7      | 16.7      | 20.0      |
| KL-Divergence ↓ | -       | 5.37    | 4.10    | -      | 7.07   | 6.46   | -         | 9.95      | 9.23      |

## H Results on QuALITY

In Table 6, we show the results of Q♯ on QuALITY [47], a challenging multiple-choice reading comprehension benchmark with long-form passages drawn from Project Gutenberg. Q♯ consistently performs better than baselines.

Table 6: Comparison of Q♯ with π ref and CD baseline on QuALITY with Qwen 2.5 and Llama 3.1.

| π ref           | Qwen 2.5 7B   | Qwen 2.5 7B   | Qwen 2.5 7B   | Llama 3.1 8B   | Llama 3.1 8B   | Llama 3.1 8B   |
|-----------------|---------------|---------------|---------------|----------------|----------------|----------------|
| Methods         | π ref         | CD            | Q♯            | π ref          | CD             | Q♯             |
| pass@1 ↑        | 64.5          | 64.2          | 68.1          | 73.5           | 75.1           | 75.9           |
| maj1@8 ↑        | 72.0          | 66.3          | 73.3          | 79.3           | 79.3           | 81.1           |
| KL-Divergence ↓ | -             | 12.32         | 7.90          | -              | 9.23           | 8.88           |

## I Comparison with Policy-based Methods

Q♯ can serve as a lightweight complement to policy-based approaches. Specifically, Q♯ can guide both the base reference policy and policies trained via reinforcement learning such as PPO. To empirically assess this, we present results on the MATH dataset where Q♯ is instantiated with a Qwen 2.5 1.5B model and used to guide: (1) the base Qwen 2.5 7B reference model and (2) a PPO-trained version of the same model. As shown in Table 7, Q♯ consistently improves both pass@1 and maj1@8 for each policy. In particular, when applied to the PPO-trained policy, Q♯ reduces the KL divergence from π ref while further boosting accuracy. We also note a qualitative distinction: PPO improves pass@1 but slightly reduces maj1@8 , indicating that its generations tend to be lower entropy and less diverse. Q♯ , in contrast, improves both metrics while maintaining closer alignment with π ref .

In terms of efficiency, Q♯ is significantly lighter to train. PPO requires approximately 20 hours on 4 H100 GPUs, whereas Q♯ training completes in roughly 5 hours on a single H100 GPU, thanks to its supervised learning objective and the use of a much smaller model. These findings suggest that Q♯ can effectively enhance performance while maintaining closer alignment with the reference policy, demonstrating its practical advantage as a complementary lightweight module.

Table 7: Comparison of Q♯ with PPO-trained models and their guided variants on the MATH dataset.

| Methods         | π ref   |   π ref + Q♯ |   PPO |   PPO + Q♯ |
|-----------------|---------|--------------|-------|------------|
| pass@1 ↑        | 58.6    |        61.9  | 68.4  |      71.1  |
| maj1@8 ↑        | 72.8    |        74.8  | 72.4  |      73.4  |
| KL-Divergence ↓ | -       |         6.46 | 69.52 |      60.53 |

## J Computational Complexity and Runtime Comparison of Q♯

Q♯ and other value-based baselines such as CD [11] have the same computational complexity. Compared to generating responses solely with π ref , value-based approaches additionally use the guidance model to compute a Q function at every decoding step. That is, it increases complexity by the ratio of the guidance model's size to that of π ref . Since the guidance model can be much smaller in size compared to π ref , the overhead is mild. For instance, guiding a Llama 8B with Llama 1B increases complexity by 12.5%.

Additionally, we efficiently implemented value-based guidance for Q♯ in Hugging Face using LogitProcessor and key-value caches. On an Nvidia A6000, generating one response on test set of MATH takes 4 . 10 s for π ref and 5 . 18 s for Q♯ , slightly exceeding 12.5% possibly due to sequential Q function computation in LogitProcessor. The code for our implementation can be found in the supplementary materials.

## K Qualitative Examples

In Figure 6 and the ones below it, we qualitative visualize side by side generation results from π ref and Q♯ on Llama 3 8B GSM8K and Llama 3.1 8B MATH settings. In the first example of Figure 6, we observe π ref and Q♯ start off similarly by calculating the total number of cookies Shannon eat but π ref starts to roughly guess the answer (highlighted in red) without calculating the answer precisely, where Q♯ calculates the answer step by step (in blue). In the second MATH example, π ref first states some confusing statement (highlighted in orange) and then makes a mistake of addition ( 5 + 5 ) instead of multiplication when calculating all the possibilities. Q♯ , however, correctly solves the problem by multiplying the options ( 5 ∗ 5 ) for Paul and Jesse and then adds the single case when using the number of 2, arriving at the correct answer.

More examples can be found in subsequent pages with various levels of difficulty. We highlight that Q♯ can still make confusing statements similar to π ref even in solution that arrives at the correct final answer. For example, in the Cecilia puppy example (the first after Figure 6), similar to π ref , Q♯ also makes a division of 1 when it should be multiplication.

Figure 6: Example generations from π ref and Q♯ on GSM8K and MATH questions. π ref consistently fails to solve the problems compared to Q♯ . The generations from π ref and Q♯ usually start off with similar reasoning chain-of-thought but π ref gradually makes unjustified reasoning leap or simply a guess of the answer whereas Q♯ tends to be more logical for solving the problems. Colors are manually added for ease of visualization. Blue represents correct reasoning steps or answers, red represents clearly incorrect ones and orange represents ambiguous or minor mistake that could potentially lead to clearly incorrect steps or answers.

<!-- image -->

<!-- image -->

Problem from GSM8K: Nate's dog can dig six holes a day. He digs for 14 days while Nate is on vacation. When Nate gets home, he starts filling in 9 holes a day, but the dog keeps digging 6 new holes every night. How many weeks does it take him to fill in all the holes?

Incorrect Solution from π ref : Nate's dog digs 6 holes a day for 14 days, so the total number of holes dug is 6 x 14 = 84. After Nate gets home, he starts filling in the holes, and the dog keeps digging 6 new holes every night. So the number of holes to fill is always 84 + 6. Nate fills 9 holes a day, so the number of days it takes to fill all the holes is:

<!-- image -->

Correct Solution from Q♯ : The dog digs 6 holes a day, and Nate is away for 14 days, so the dog digs 6 * 14 = 84 holes. Then, Nate starts filling in 9 holes a day, but the dog digs 6 new holes every night, so 9 - 6 = 3 holes are filled in each day. So, each day 3 more holes are filled in. The number of holes filled in after x days is 84 - 3x. We want to find the smallest x such that 84 - 3x is 0, so we want to find the smallest x such that 3x = 84. 3x = 84 is true when x = 28, so it takes 28 days to fill in all the holes. Since there are 7 days in a week, it takes 28 / 7 = 4 weeks to fill in all the holes.

Prealgebra Problem from MATH: A bookcase has 3 shelves with a total of 24 books. The top shelf has 8 mystery books. The middle shelf has 10 math books. The bottom shelf has 6 science books. Two books are now taken off each shelf. What fraction of the books remaining on the three shelves are math books? Express your answer as a common fraction.

Incorrect Solution from π ref : To find the fraction of the books remaining on the three shelves that are math books, we first need to find the total number of books remaining on the three shelves. Initially, the bookcase had 24 books. Two books are taken off each shelf. Since there are 3 shelves, a total of 6 books are taken off.

So, the total number of books remaining on the three shelves is:

24 (initial books) - 6 (books removed) = 18 books.

The middle shelf had 10 math books. Since no math books were removed, the number of math books remaining is still 10.

Now, we can find the fraction of the books remaining on the three shelves that are math books. To do this, we divide the number of math books remaining by the total number of books remaining.

Fraction of math books = Number of math books / Total number of books

= 10 / 18

We can simplify this fraction by dividing both the numerator and denominator by their greatest common divisor, which is 2.

Simplified fraction = 5 / 9

So, the fraction of the books remaining on the three shelves that are math books is:

5

/

9

Correct Solution from Q♯ : To find the fraction of the remaining books that are math books, we first need to find the total number of books remaining on the shelves after two books are taken off each shelf. Top shelf initially had 8 books. After removing 2 books, it has 8 - 2 = 6 books.

Middle shelf initially had 10 books. After removing 2 books, it has 10 - 2 = 8 books.

Bottom shelf initially had 6 books. After removing 2 books, it has 6 - 2 = 4 books.

So the total number of books remaining is 6 + 8 + 4 = 18 books.

Initially, there were 10 math books. Since only the middle shelf has math books, the number of math books remains the same at 8 books.

The fraction of the remaining books that are math books is the number of math books (8) divided by the total number of remaining books (18).

<!-- image -->

Intermediate Algebra Problem from MATH: The asymptotes of a hyperbola are y = 2 x -3 and y = 17 -2 x. Also, the hyperbola passes through the point (4 , 7) . Find the distance between the foci of the hyperbola.

Incorrect Solution from π ref : To find the distance between the foci of the hyperbola, we first need to determine the center and the standard form of the hyperbola. The asymptotes are given by y = 2 x -3 and y = 17 -2 x. Their point of intersection will be the center of the hyperbola. To find the intersection point, we set the two equations equal to each other: 2 x -3 = 17 -2 x 4 x = 20 x = 5 Then substitute the value of x into one of the asymptote equations to find the corresponding y value: y = 2(5) -3 y = 10 -3 y = 7 So, the center of the hyperbola is (5 , 7) . Next, we need to find the standard form of the hyperbola. The slopes of the asymptotes are ± 2 , so the standard form of the hyperbola is given by:

<!-- formula-not-decoded -->

Since the slopes of the asymptotes are ± 2 , we know that b/a = 2 . We can use the point (4 , 7) , which the hyperbola passes through, to find the values of a and b.

<!-- formula-not-decoded -->

Since b/a = 2 , we have a = b/ 2 = 1 / 2 . Now we can find the values of a 2 and b 2 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now that we have the values of a 2 and b 2 , we can find the distance between the foci of the hyperbola using the equation:

<!-- formula-not-decoded -->

The distance between the foci of the hyperbola is 2 c = 2 √ 5 / 2 = √ 5 . Therefore, the distance between the foci of the hyperbola is √ 5 .

Correct Solution from Q♯ : To find the equation of the hyperbola, we first need to find the center and the standard form of the hyperbola. The center is the point of intersection of the two asymptotes. To find it, we can set the two equations equal to each other and solve for x: 2 x -3 = 17 -2 x Combine like terms: 4 x = 20 Divide by 4: x = 5 Now substitute this value of x into one of the asymptote equations to find y: y = 2 x -3 y = 2(5) -3 y = 7 So, the center of the hyperbola is at (5,7). Since the slopes of the asymptotes are 2 and -2, the standard form of the hyperbola is: ( y -k ) 2 a 2 -( x -h ) 2 b 2 = 1 where (h,k) is the center of the hyperbola. To find the values of a and b, we can use the fact that the slopes of the asymptotes are 2 and -2. The slopes of the asymptotes are given by ± a b . Therefore: a b = 2 We also know that the point (4,7) is on the hyperbola. We can substitute this point into the equation to get: (7 -7) 2 a 2 -(4 -5) 2 b 2 = 1 This simplifies to: 1 b 2 = 1 b 2 = 1 Now we can substitute this value of b 2 into the equation a b = 2 to get: a 1 = 2 a = 2 Now we can substitute the values of a and b into the standard form of the hyperbola to get: ( y -7) 2 4 -( x -5) 2 1 = 1 The distance between the foci of a hyperbola is given by 2c, where c is the distance from the center to each focus. The relationship between a, b, and c is given by: c 2 = a 2 + b 2 c 2 = 2 2 +1 2 c 2 = 5 c = √ 5 The distance between the foci is: 2 c = 2 √ 5

So, the distance between the foci of the hyperbola is 2 √ 5 .

Algebra Problem from MATH: Simplify the following expression to a simplified fraction:

<!-- formula-not-decoded -->

Incorrect Solution from π ref : To simplify the expression √ √ √ √ √ 5 √ 80 + √ 845 9 + √ 45 √ 5 , we need to follow the order of operations. First, we simplify the terms inside the square root: 5 √ 80 = 5 √ 16 × 5 = √ √ √

<!-- formula-not-decoded -->

√

√

√

√

√

5

4

√

Now, we can rewrite the expression as:

√

5

4

√

√

√

√

finding a common denominator:

we can rewrite the expression as:

169

√

√

36

5

+ 13

5

9

169

5

√

36

5

√

=

To simplify the expression, we can multiply the numerator

169

√

36

√

√

5

5

=

169

×

36

=

845

36

Now, we can rewrite the and denominator by

5

:

√

×

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Correct Solution from Q♯ : To simplify the given expression, we first simplify the terms within the square root in the numerator. √

<!-- formula-not-decoded -->

Now, we substitute these simplified expressions back into the original expression:

<!-- formula-not-decoded -->

We can simplify the expression further by combining like terms in the numerator:

<!-- formula-not-decoded -->

Finally, we take the square root of the fraction:

<!-- formula-not-decoded -->

5

5

5

+ 13

√

9

5

√

√

+3

5 =

√

5

We can simplify the numerator by

√

5

√

√

36

36

36

Now,

+ 52

5

+ 108

= 169

5

5

√

The answer is 13 6 .

√

+3

√

9

5

36