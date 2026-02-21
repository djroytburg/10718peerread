## Proximalized Preference Optimization for Diverse Feedback Types: A Decomposed Perspective on DPO

Kaiyang Guo

## ∗ Yinchuan Li

Zhitang Chen

Huawei Noah's Ark Lab

## Abstract

Direct alignment methods typically train large language models (LLMs) by contrasting the likelihoods of preferred and dispreferred responses. While effective at capturing relative preferences, these methods are widely observed to suppress the absolute likelihoods of example responses. As a result, aligned models can deviate from expected patterns, exhibiting reward-hacking effect even without an explicit reward model. This fundamental limitation of contrastive alignment, termed likelihood underdetermination, motivates us to revisit direct preference optimization (DPO)-the seminal direct alignment method. Interestingly, we show that the DPO loss admits a principled decomposition. The reformulated loss not only extends naturally to a broader range of feedback types, but also unveils the root cause of likelihood underdetermination. Specifically, we identify that standard DPO implicitly oversimplifies a regularizer in the reformulated loss; restoring this full term effectively resolves the underdetermination. Building on these insights, we introduce PRoximalized PReference Optimization (PRO), a unified alignment method that accommodates diverse feedback types while eliminating likelihood underdetermination through an efficient approximation of the full regularizer. Empirical evaluations demonstrate the consistent superiority of PRO over existing methods across pairwise, binary and scalar feedback.

## 1 Introduction

The subtlety and complexity of human values make curating supervised datasets for large language model (LLM) alignment a formidable challenge. To overcome this difficulty, learning from feedback leverages preference signals to guide models in distinguishing desirable from undesirable outputs. This methodological shift facilitates scalable data acquisition for efficient alignment, while concurrently propelling the evolution of techniques beyond traditional supervised fine-tuning.

Initially, reinforcement learning from human feedback (RLHF) converts pairwise preference feedback into continuous rewards, which are then used to train LLM via reinforcement learning algorithms [1, 2]. RLHF has shown empirical success. However, it introduces significant computational overhead, due to the need for an auxiliary reward model and on-policy sampling during training. Moreover, as an imperfect proxy, the reward model may yield unreliable evaluations when confronted with responses outside its training distribution, making RLHF vulnerable to reward hacking [3-7].

Direct preference optimization (DPO) sidesteps the explicit reward modeling, and learns directly from offline preference data [8]. Its core methodology involves constructing a contrastive loss, that maximizes the likelihood differences between preferred and dispreferred responses. Owing to the simplicity, DPO has inspired the development of numerous contrastive approaches [9-11]. However, an unanticipated phenomenon frequently arises in DPO: the likelihoods of both preferred and dispreferred responses decrease after alignment [12-15]. This decline inadvertently encourages

∗ Corresponding to: guokaiyang@huawei.com

the generation of out-of-distribution responses, suggesting persistent reward hacking even in the absence of an explicit reward model [16]. Recent investigations into this phenomenon have primarily focused on the training dynamics of DPO and its generalized variants. Proposed explanations include embedding similarity between paired responses [15], asymmetric update ratios for response probabilities [17], and the 'squeezing effect' produced by softmax when applying gradient ascent to dispreferred responses [18]. While these studies offer valuable insights, they largely overlook the intrinsic limitation of the loss function itself. In particular, when attention is limited to relative likelihood differences, the contrastive loss becomes insensitive to concurrent decreases or increases in absolute likelihoods. We characterize this issue as likelihood underdetermination. While several works also notice and attempt to remedy it, they incorporate additional supervised or regression signals into DPO [12, 14], which may unintentionally compromise the original intent of alignment.

The aforementioned line of research is primarily grounded on pairwise feedback. Meanwhile, parallel studies have also explored alternative types of feedback for LLM alignment. Considering that response-pair annotation demands more effort than single-response evaluation, Kahneman-Tversky optimization (KTO) constructs separate utility functions for desired and undesired responses [19]. Conversely, in scenarios where finer-grained scalar feedback is available, noise contrastive alignment (NCA) formulates a classification task to capture the varying degree of desirability for each labeled response [13]. Both KTO and NCA depart from the contrastive framework. Although they can either directly or under certain assumption be adapted to pairwise feedback, KTO does not support scalar feedback, and NCA is incompatible with binary feedback. Currently, there is still no unified approach that seamlessly accommodates all these feedback types.

In this work, we revisit DPO, the seminal method in field of direct alignment. Through a theoretical reformulation of its loss, we demonstrate that DPO inherently supports heterogeneous feedback and uncover new insights into the underlying cause of likelihood underdetermination. Leveraging these findings, we propose a practical approach that both mitigates underdetermination and unifies alignment across diverse feedback types, thereby realizing the best of both worlds pursued in prior studies. Specifically, our contributions are as follows:

- DPO Reformulation We show that the DPO loss admits a decomposed reformulation, which consists of separate optimizer and regularizer terms. The optimizer reorganizes pairwise feedback into a pointwise signal, naturally extending applicability to a wider range of feedback types. The regularizer is independent of the preference label, allowing for a more flexible development of sample-based variant.
- Origin of Likelihood Underdetermination We find that the standard DPO implementation implicitly oversimplifies the regularizer introduced in the reformulation. Importantly, once the full regularizer is restored, any optimal solution to the reformulated loss-if it exists-no longer suffers from likelihood underdetermination.
- Practical Approach Directly computing the full regularizer is intractable, and an optimal solution may not exist. To overcome these challenges, we propose PRoximalized PReference Optimization (PRO), which employs a hyper-response mechanism for efficiently approximating the regularizer, and guarantees the existence of an optimal solution whenever its hyperparameter is properly chosen.
- Empirical Evaluation The experiments show that PRO mitigates likelihood underdetermination, performs effectively across diverse feedback types, and achieves performance comparable to or better than DPO and several other methods designed for specific feedback types. Remarkably, even in the challenging scenario with extremely imbalanced binary feedback (desired:undesired = 1:100), PRO demonstrates comparable performance to that obtained with fully balanced feedback.

## 2 Preliminaries

In LLM alignment, preference data is commonly collected through two steps: prompting a baseline model with inputs x to generate response pair y 1 , y 2 ∼ µ ( y | x ) ; human annotators labeling the generated responses to indicate which is preferred. Let y w ≻ y l | x denote the pairwise preference feedback, where y w and y l are preferred and dispreferred responses amongst y 1 and y 2 .

RLHF employs reward modeling to infer scalar reward signals from pairwise preference data D = { x ( i ) , y ( i ) w , y ( i ) l } I i =1 . These reward estimates are subsequently used to fine-tune LLMs through

reinforcement learning. Recently, DPO [8] circumvents the need for explicit reward modeling, by recognizing that the optimal solution to the RL problem possesses a closed-form relationship with its underlying reward function. This connection allows the reward model to be expressed as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where π θ is the LLM to be fine-tuned, and π ref is a fixed reference model that serves as a regularization anchor during optimization. Alignment is thus performed by fitting the LLM-induced reward model to preference data via the following loss:

where we use the hatted notation ̂ L DPO to indicate that the loss is computed from preference samples. DPO has been widely adopted in LLM alignment [20-24], yet an overlooked issue is likelihood underdetermination of its loss function. Specifically, when log π θ ( y w | x ) and log π θ ( y l | x ) are shifted by a same constant, the loss value remains unaffected. Furthermore, as indicated by its gradient:

the importance weight for model update approaches zero whenever the relative difference log π θ ( y w | x ) -log π θ ( y l | x ) is sufficiently large, regardless of the absolute probability values. Such characteristics can substantially hinder effective alignment: After pretraining or supervised finetuning, the values of log π θ ( y w | x ) and log π θ ( y l | x ) are markedly higher than those assigned to meaningless sentences. As the DPO loss focuses solely on the relative difference between log p ( y w | x ) and log p ( y l | x ) , their absolute values are prone to decrease due to catastrophic forgetting, and there is no incentive to increase them again once the relative difference has been sufficiently enlarged.

<!-- formula-not-decoded -->

Notations Without loss of generality, we consider a single prompt and omit x for brevity in the following. Let Y be the set of all possible responses, and supp( · ) the support of a distribution.

## 3 Theoretical Re-Examination of DPO

At first glance, it seems that the contrastive nature of pairwise feedback restricts DPO to comparing only the relative likelihoods of responses. However, as we show in Section 3.1, the DPO loss can be reformulated into a decomposition that explicitly accounts for the absolute likelihoods of labeled responses. This reformulation not only reveals that DPO can naturally accommodate other types of feedback, but also lays the foundation for more flexible sample-based loss variants. Building on this insight, Section 3.2 identifies the root source of likelihood underdetermination in standard DPO, which ultimately motivates our proposed approach in Section 4.

## 3.1 Reformulation of Population-Based DPO

To more clearly elucidate the underlying properties of DPO, we consider its population-based loss:

While this loss is not directly computable due to the inaccessibility of true preference probability, it nevertheless allows us to examine DPO from a novel perspective, as articulated in the theorem below.

<!-- formula-not-decoded -->

Theorem 3.1. The population-based DPO loss is equivalent to the following one, in that they share same gradient:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is a score function indicating the extent to which y is favored across other responses and satisfies E y ∼ µ [ s ( y )] = 0 .

The reformulation decomposes the DPO loss into an optimizer and a regularizer. Upon examination of these components, we identify two attractive properties:

- The optimizer reorganizes pairwise feedback into a pointwise signal s ( y ) , with which log π θ ( y ) is independently optimized for each response. This property naturally extends the applicability of L eDPO. For instance, given -1 / 2 ≤ s ( y ) ≤ 1 / 2 , we can interpret s ( y ) as the expected value of a Bernoulli distribution, with binary feedback in {-1 / 2 , +1 / 2 } serving as its empirical sample for loss evaluation. Alternatively, s ( y ) can be viewed as the expectation of a continuous reward distribution, where scalar feedback represents realized reward samples during training.
- The optimizer relies on preference feedback, whereas the regularizer operates independently of such information. Given the limited availability of preference feedback in practice, the optimizer must be estimated from a finite dataset. In contrast, the regularizer can be applied to an expanded set of responses, irrespective of preference labels. This decomposition therefore provides greater flexibility in developing sample-based loss.

These two properties play pivotal roles in this work. The first one enables devising a unified alignment loss for diverse feedback types. The second property offers an elegant way to understand and resolve likelihood underdetermination, as detailed in the following section.

## 3.2 The Completeness of Regularizer Matters in Sample-Based Loss

For practical use, it is essential to develop a sample-based loss, whose computation only requires the limited feedback data. The most straightforward strategy is to estimate both the optimizer and regularizer in eDPO using the labeled responses. 2 As can be verified by applying Theorem 3.1 in reverse, this substitution recovers the sample-based DPO loss. However, there appears to be a contradiction regarding the existence of likelihood underdetermination. On one hand, the gradient of sample-based DPO in (1) includes an importance weight, causing it to vanish whenever the relative likelihood difference between response pair is sufficiently large. On the other hand, the optimizer in eDPO directly evaluates the absolute log-probabilities of labeled responses; even estimated with limited samples, its gradient remains free of any importance weighting. Considering that eDPO incorporates an additional regularizer, it is plausible that this term is responsible for the likelihood underdetermination presented in sample-based DPO. We next investigate its effect.

A key observation of the regularizer is that, when µ ( y ) &gt; 0 for all y , the regularizer effectively constrains π θ around π ref, and its value becomes zero only if π θ = π ref. In other words, the regularizer is well-defined, albeit in a contrastive form analogous to DPO. However, it is easy to verify that, when estimated with a subset of responses, the regularizer revives the underdetermination issue. 3 Moreover, as indicated by the recovery to sample-based DPO, this underdetermination dominates the optimizer's effect on absolute likelihoods, rendering the overall loss function underdetermined.

In fact, the regularizer itself is independent of preference labels, thus need not be restricted to labeled responses. This motivates us to study whether likelihood underdetermination can be addressed by retaining the full regularizer in sample-based loss. Formally, define the sample-based eDPO loss as:

<!-- formula-not-decoded -->

where ˆ µ denotes the empirical response distribution derived from the preference dataset, in contrast to the full response distribution µ . The coefficient α &gt; 0 is newly introduced for general tradeoff between preference optimization and regularization. The empirical score ˆ s ( y ) is given by

<!-- formula-not-decoded -->

2 That means replacing µ and p by their empirical counterparts estimated from the dataset.

3 Since ∑ y ∈Y π θ ( y ) = 1 , any uniform likelihood reduction (or increment) within a subset must be offset by an opposite change on its complement. However, the regularizer only compares likelihoods inside the subset, but disregards its relation to the rest of Y . It is therefore blind to the widening probability gap between the two parts.

where ˆ p denotes the empirical pairwise preference and ˆ b is the sample mean of the pointwise feedback. The structure of ˆ s for pointwise feedback is designed so that E y ∼ ˆ µ [ˆ s ( y )] = 0 , mirroring the property of s established in Theorem 3.1.

Theorem 3.2. Let µ := µ ( y ) &gt; 0 for all y ∈ Y . If an optimal solution π ∗ to ̂ L eDPO exists, it satisfies the condition for any y ∈ Y :

To analyze the theoretical property of ̂ L eDPO, we treat π θ as an arbitrary distribution in ∆ = { π | π ( y ) &gt; 0 , ∀ y ∈ Y and ∑ y ∈Y π ( y ) = 1 } . This allows to derive the necessary condition for optimality as follows.

<!-- formula-not-decoded -->

Condition (2) can be interpreted as a weighted score matching: The expectation term on the left-hand side acts as a learned score, analogous to the empirical score ˆ s for pairwise feedback except that it is derived from the LLM; The weight on the right-hand side contains µ ( y ) and ˆ µ ( y ) , which respectively signify the strengths of the regularizer and the observed evidence.

Recall that ˆ s ( y ) indicates whether y is preferred over other responses. When condition (2) holds, the modeled score should reflect the preference accordingly. In particular, the sign of ˆ s ( y ) should determine how π ∗ ( y ) deviates from π ref ( y ) . This relationship is formally confirmed in Corollary 3.3.

Corollary 3.3. Under the preconditions of Theorem 3.2, the following results hold for a constant C :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Corollary 3.3 imposes an ordering among the probability updates compare to reference model. Specifically, it constrains the probability ratio of any unobserved response (i.e., one absent from the preference dataset) to fall between those of the preferred and dispreferred responses. Consequently, a simultaneous decrease in the probabilities of preferred and dispreferred responses would necessarily entail a decrease for the unobserved responses as well, which is impossible due to the fixed total probability. This demonstrates that the absolute likelihoods of both labeled and unobserved responses can not be adjusted arbitrarily, thereby resolving the underdetermination issue. Under the guarantee, we conclude that:

The likelihood underdetermination in DPO stems from an oversimplified regularizer, and can be mitigated by restoring the regularizer to its full form.

## 4 Proximalized Preference Optimization

The analysis in previous section rest on two preconditions: (i) µ assigns non-zero probability to all responses, and (ii) an optimal solution π ∗ exists. While the first condition can be satisfied by presetting µ appropriately, computing the regularizer in ̂ L eDPO requires traversing all responses with non-zero probability under µ . Given the enormous cardinality of Y , the regularizer rapidly becomes computationally intractable. We address this challenge by developing a carefully crafted approximation to the regularizer in Section 4.1. The second precondition-existence of an optimal solution-has been shown to always fail in sample-based DPO, referred to as degeneracy issue [10]. In Section 4.2, we establish a sufficient condition that guarantees the existence of an optimal solution for the proposed loss. We further provide an explicit pairwise-feedback example satisfying this condition and show that it is directly pertinent to sample-based DPO.

## 4.1 Introducing Hyper Response for Tractable Loss Approximation

To avoid enumerating all possible responses, one may incorporate additional samples drawn from µ , together with the labeled responses, into the regularizer's computation. While this yields a more

<!-- image -->



Figure 1: Shaded boxes denote labeled responses; blank boxes denote unobserved responses. By aggregating unobserved responses into a single hyper response, the response space becomes compact, such that the probabilities of its elements can be enumerated.

accurate estimation than sample-based DPO, auto-regressive sampling is computationally expensive, typically 10-100x slower than the training process itself. Moreover, this simplification again reduces µ to an empirical distribution with limited supports, risking a recurrence of underdetermination.

To overcome these limitations, we introduce an approximation mechanism that evaluates the regularizer within a compact yet complete response space. The central idea is to define a hyper response , denoted by H ⊂ Y , which aggregates multiple individual responses into a single, indistinguishable unit. This abstraction gives rise to the following modified response space:

<!-- formula-not-decoded -->

The regularizer is then computed using the probability values µ ( y ) , π θ ( y ) and π ref ( y ) exclusively over Y H . To ensure these probabilities well-defined for y = H , we extend any probability distribution p on Y to Y H by setting:

<!-- formula-not-decoded -->

Importantly, p ( H ) can be derived entirely from the probabilities of its complement. As illustrated in Figure 1, when H encompasses all unobserved responses, computing p ( H ) requires only the probabilities of labeled responses, without the need for additional sampling. Moreover, this construction provides the most compact form of Y H , as it augments the labeled responses with a single hyper response. Owing to these advantages, we adopt this construction as the default setting.

Building on the above foundations, we now introduce the refined loss:

<!-- formula-not-decoded -->

Three questions naturally arise concerning ̂ L PRO: (i) How does its optimal solution relate to that of ̂ L eDPO? (ii) Does it preserve protection against underdetermination? and (iii) What is the cost of this approximation? The following theorem and the subsequent discussion addresses these questions.

whose only difference from ̂ L eDPO lies in the use of y 1 , y 2 ˙ ∼ µ within the regularizer. Here, y ˙ ∼ µ denotes that y is sampled according to µ over Y H . Accordingly, for any function f , the expectation with respect to y ˙ ∼ µ reduces to E y ˙ ∼ µ [ f ( y )]= µ ( H ) f ( H ) + ∑ y/ ∈H µ ( y ) f ( y ) , which substantially reduces the number of terms compared to the full expectation over Y . When substituting this expectation into the regularizer, f ( H ) corresponds to the KL term involving µ ( H ) , π θ ( H ) and π ref ( H ) , all of which can be efficiently computed via (6). Crucially, due to the aforementioned construction of H , ̂ L PRO incurs negligible additional computational cost compared to ̂ L DPO, requiring only lightweight operations on response probabilities already computed in DPO.

Theorem 4.1. Let H ⊆ Y \ supp(ˆ µ ) and µ := µ ( y ) &gt; 0 for all y ∈ Y . If optimal solutions π ∗ and π ∗ H to ̂ L eDPO and ̂ L PRO exist, they satisfy: π ∗ H ( y ) = π ∗ ( y ) , ∀ y ∈ Y \ H , (7)

<!-- formula-not-decoded -->

where C is the constant as defined in Corollary 3.3.

The precondition H ⊆ Y \ supp(ˆ µ ) implies that the hyper response contains no labeled responses. Consequently, Y \ H comprises all labeled responses. By (7), these responses retain the properties stated in Corollary 3.3: after training, their absolute likelihoods-not merely their relative gaps-are well determined. This protects the aligned LLM from likelihood underdetermination.

Adopting the hyper-response mechanism sacrifices the ability to distinguish among the probabilities of unobserved responses in H . However, the inability to regulate the distribution over unobserved responses during alignment is a limitation shared by existing approaches. Methods such as DPO [8], KTO [19], NCA [13], and IPO [10] incorporate the probabilities only for labeled responses, leaving the rest under-specified. In contrast, a distinctive property of our approach is that the total probability mass assigned to the elements in H remains fixed, as shown in (8). This constraint prevents any single response from continuously increasing its probability throughout training. Since most unobserved responses have extremely low initial probabilities, even moderate increases during training leave their generation likelihood minor, thereby limiting the practical impact of this limitation.

In summary, the hyper-response mechanism enables an efficient and effective approximation to ̂ L eDPO. With the guarantee of mitigating likelihood underdetermination, aligned LLM is successfully proximalized around reference model. We therefore refer to the proposed approach as PRoximalized PReference Optimization (PRO).

## 4.2 The Existence of Optimal Solution for PRO

Finally, we establish a sufficient condition under which the considered losses have optimal solutions. Noting that ̂ L eDPO is a special case of ̂ L PRO obtained by restricting H to a single response, we hereafter take L PRO as the general loss.

̂ Recall that the feasible region of ̂ L PRO, ∆ = { π | π ( y ) &gt; 0 ∀ y ∈ Y H , ∑ y ∈Y H π ( y ) = 1 } , is an open set. The absence of an optimal solution would imply the existence of a sequence within ∆ whose elements approach its boundary, along which the loss function strictly decreases (see Lemma B.1 in Appendix B.4). However, as one approaches the boundary, i.e, π ( y ) → 0 for some y , the regularizer in PRO can be shown to diverge to + ∞ . Thus, any overall decrease in the loss must result from the optimizer decreasing towards -∞ at a faster rate.

The above observation leads us to ask whether the value of α can be adjusted so that the regularizer dominates the loss function at the boundary. If so, the unbounded descent of the loss function can be prevented, thereby guaranteeing the existence of an optimal solution. This conjecture is established by the following theorem.

Theorem 4.2. Given any H ⊆ Y \ supp(ˆ µ ) and µ := µ ( y ) &gt; 0 , ∀ y ∈ Y H , there is a threshold α 0 such that, whenever α &gt; α 0 , an optimal solution π ∗ H to ̂ L PRO exists. 4 As an illustrative example, consider the pairwise feedback setting, where we choose µ = µ as:

<!-- formula-not-decoded -->

Here, ρ := ρ ( y ) &gt; 0 denotes an arbitrary probability distribution over y ∈ Y H \ supp(ˆ µ ) , and 0 &lt; η &lt; 1 is a preset hyperparameter. This construction serves as a general and practically effective choice: in real-world scenarios, labeled responses yield the empirical distribution ˆ µ , whereas the true underlying distribution µ , from which these responses are drawn, often remains inaccessible.

Under this specification of µ , the next theorem characterizes an admissible range of α , and further reveals a direct connection between the induced loss and sample-based DPO.

Theorem 4.3. Consider the pairwise feedback setting, where µ = µ and H ⊆ Y \ supp(ˆ µ ) . For any α ≥ 1 /η 2 , an optimal solution π ∗ H to ̂ L PRO exists. Moreover, when α = 1 /η 2 , the PRO loss is equivalent to the following one in that they share same gradient:

4 A constructive choice of α 0 for general preference feedback is given in Corollary B.2.

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

is an augmented empirical preference.

The PRO-P loss can be viewed as an enhanced variant of sample-based DPO, which integrates pseudo preference labels and employs the hyper-response approximation. Nevertheless, it should be noted that PRO-P is merely a special case resulting from particular choices of µ and α . More broadly, PRO offers a general approach that accommodates diverse feedback types and permits versatile control over both the strength and distribution of regularization.

## 5 Experiments

Our experiments address four questions: (i) To what extent does PRO mitigate likelihood underdetermination in practice? (ii) How does it compare with other alignment methods under pairwise and binary feedback? (iii) Given KTO's suitability for imbalanced binary feedback, does PRO exhibit greater robustness under severe imbalance? (iv) While NCA is specifically tailored for scalar feedback, can PRO match or surpass its performance?

We utilize two datasets to construct three types of feedback across different experimental settings. The Anthropic-HH dataset originally comprises 170k pairwise feedback instances [25]. Following [19], we split each paired response into individual ones, and convert the feedback into binary format by labeling preferred responses as desired and dispreferred ones as undesired. The UltraFeedback dataset includes 64k instructions, each accompanied by four responses annotated with scalar feedback [26]. To derive a pairwise version, we select the response with the highest scalar feedback as preferred and randomly choose one of the others as dispreferred, following [24]. A binary version is also generated from the pairwise data in a manner consistent with the processing of Anthropic-HH.

In experiments, we apply PRO to each feedback type: PRO-P denotes the use of ̂ L PRO-P for pairwise feedback, PRO-B and PRO-S correspond to ̂ L PRO applied to binary and scalar feedback, respectively. The hyper response is set to encompass all unobserved responses. Further implementation details (including the choice of α ) and the full experimental setup are provided in Appendices D and F.

## 5.1 Resolving Likelihood Underdetermination

As discussed, likelihood underdetermination often manifests as a uniform reduction in probabilities across all example responses, ultimately leading to reward hacking. Because reward hacking exerts a more immediate influence on model performance, this section focuses on its analysis. Detailed probability dynamics throughout training are reported in Appendix G (see Figures 5 and 6), showing that PRO consistently increases the probabilities of preferred responses for all feedback types.

In the absence of a reward model and ground-truth rewards, we examine reward hacking through its most recognized symptom-length exploitation [27-30], wherein models tend to produce excessively long responses after alignment. To trace the development and severity of this effect, we continuously monitor the model's average response length on test dataset during the alignment process. Simultaneously, model performance is evaluated in terms of win rate against preferred responses, measured on dimensions of helpfulness, harmlessness and conciseness, using DeepSeek-V3 [31] as the evaluator. To highlight variations over time, we report relative changes in both win rate and average response length with respect to their initial evaluations.

<!-- image -->

Figure 2: Performance fluctuation of different alignment methods. β is uniformly set to 0.1.

<!-- image -->

β

β

Figure 3: Results of aligning Pythia-6.9B with Anthorpic-HH.

Figure 2 shows the results of aligning the Pythia-6.9B model [32] with Anthropic-HH dataset. For DPO, we observe a sharp increase in response length as training progresses, accompanied by a substantial decline in win rate. In contrast, the response lengths and win rates of both PRO-P and PRO-B remain stable throughout training, suggesting effective mitigation of reward hacking. Since KTO and NCA are derived from non-contrastive frameworks, they are expected to avoid the underdetermination issue. However, our results reveal that the fully trained KTO model still exhibits a significant increase in response length, from 113.6 to 235.6, along with a 6.64% drop in win rate.

We hypothesize that the degradation of KTO arises from the direct use of sigmoid function in its loss formulation:

<!-- formula-not-decoded -->

where λ D , λ U are hyperparameters and z 0 is a non-negative reference value. The sigmoid function tends to saturate as its input moves far from zero, causing the gradient to vanish at extreme regions. During training, particularly in the later stages of an epoch, the term r θ ( x, y w ) -z 0 may become strongly negative. This typically occurs because prior updates to the model, driven on other samples, potentially cause the model to 'forget' prompt-response pairs it has not yet encountered during alignment. As a result, when these pairs are eventually processed, the diminished value of r θ ( x, y w ) suppresses the sigmoid's gradient, thereby impeding effective learning. In contrast, NCA and the PRO series incorporate the log-sigmoid function in their loss functions. The log-sigmoid saturates only on one side-when the implicit reward becomes sufficiently positive for the preferred response or sufficiently negative for the dispreferred one. This asymmetry avoids the vanishing-gradient problem inherent to the plain sigmoid and enables more stable parameter updates during training.

## 5.2 Performance Comparison Under Pairwise and Binary Feedback

Both PRO and the baseline methods employ β to regulate the degree of preference optimization; however, the optimal β may differ across methods. To ensure a fair comparison, we evaluate each method under a range of β values.

Figure 3 shows the results of aligning the Pythia-6.9B model with Anthropic-HH dataset. NCA and KTO clearly outperform DPO in win rate while keeping response length within a more reasonable range. This improvement likely stems from their non-contrastive loss formulation. Remarkably, although PRO-P remains contrastive, both it and PRO-B consistently performs well, further supporting that oversimplifying the regularizer is the root cause of underdetermination in contrastive alignment.

Next, we align the Mistral-7B-sft model [33] with the UltraFeedback dataset, conducting hyperparameter sweeps over the same range of β values. Each aligned model is evaluated on AlapcaEval 2 [34], MT-Bench [35] and multiple benchmarks from LM Evaluation Harness [36]. Table 1 summarizes the best results of each method over all β values. The results indicate that DPO outperforms KTO and NCA on AlpacaEval 2, while the reverse is true on MT-Bench. For both tasks, PRO-P and PRO-B either closely match or surpass the best baseline. Similar results are observed for tasks from LM Evaluation Harness: DPO outperforms KTO and NCA on ARC and TruthfulQA, but faces performance bottleneck on IFEval. The PRO methods generally exhibit performance comparable to the best baseline.

| Method   | AlpacaEval 2   | AlpacaEval 2   | MT-Bench   | ARC   | IFEval   | TruthfulQA   | GPQA   | Avg Rank   |
|----------|----------------|----------------|------------|-------|----------|--------------|--------|------------|
| Method   | LC (%)         | WR(%)          | Score      | ARC   | IFEval   | TruthfulQA   | GPQA   | Avg Rank   |
| SFT      | 8.69           | 4.29           | 4.87       | 51.54 | 2.40     | 42.23        | 29.02  | 6.0        |
| DPO      | 18.36          | 19.80          | 5.52       | 61.77 | 19.22    | 43.45        | 32.04  | 3.3        |
| KTO      | 17.39          | 14.60          | 5.98       | 55.38 | 25.69    | 41.00        | 33.04  | 3.9        |
| NCA      | 17.21          | 13.43          | 6.03       | 58.62 | 26.43    | 42.35        | 32.14  | 4.0        |
| PRO-P    | 20.21          | 19.56          | 6.06       | 61.26 | 29.02    | 43.81        | 32.59  | 2.0        |
| PRO-B    | 20.46          | 21.48          | 6.06       | 59.81 | 30.13    | 42.72        | 32.81  | 1.7        |

Table 1: Results of aligning Mistral-7B-sft with UltraFeedback. Avg rank is computed by ranking the method among all competitors for each task and then averaging its ranks over all tasks.

## 5.3 Aligning with Extremely Imbalanced Binary Feedback

To further assess the effectiveness of alignment using binary feedback, we consider the challenging settings where 99% of the desired or undesired responses in Anthropic-HH dataset are discarded. The resulting '1%-desired' and '1%-undesired' datasets are then used to align the Pythia-6.9B model.

On the 1%-desired dataset, we conduct preliminary experiments with PRO-B and KTO using the optimal β values from Figure 3 (0.003 and 0.03, respectively). However, these configurations yield low win rates of only 5.57% and 22.56%. In addition, both aligned models produce a large number of duplicate and meaningless tokens. We hypothesize that these poor performances stem from overoptimization: with substantially reduced training data, the best-performing LLM we can optimize is expected to remain closer to the reference model. To investigate this hypothesis, we first increase β by one and two orders of magnitude, but the performances still suffer.

PRO introduces an extra hyperparameter α that mediates the trade-off between optimization and regularization. We therefore tune α and report the results in Table 2. Remarkably, increasing α from 2 . 5 to 17 . 5 improves the win rate to 57.21%, even surpassing the performance achieved with the full dataset (53.37%). This underscores the importance of

Table 2: Effectiveness of α in improving win rates (%) under extremely imbalanced binary feedback.

| Dataset      |   α =2 . 5 |   α =10 |   α =17 . 5 |   α =25 |
|--------------|------------|---------|-------------|---------|
| 1%-desired   |       5.57 |   35.24 |       57.21 |   52.02 |
| 1%-undesired |      48.87 |   50.64 |       47.83 |   47.94 |

α in learning stability. A detailed comparison between the effects of α and β is given in Appendix E.

In contrast, on the 1%-undesired dataset, PRO-B and KTO attain satisfactory win rates of 48.87% and 48.33% under the same β settings. Once again, increasing β fails to yield better performance, but tuning α for PRO-B leads to further improvements, as shown in Table 2. These results suggest that unlearning undesired responses is more challenging than learning desired ones, yet appropriate tuning of α benefits both processes.

## 5.4 Aligning with Scalar Feedback

The raw UltraFeedback dataset provides four responses labeled with scalar feedback per instruction. Following existing work [13], we evaluate model performance using different numbers of responses per instruction, denoted by N . For N = 2 , the best and a random remaining response are selected.

Table 3: Results of Aligning Mistral-7B-sft. Including more suboptimal examples improves performance.

| Method   |   N |   ARC | IFEval   | TruthfulQA   | GPQA    |
|----------|-----|-------|----------|--------------|---------|
| NCA      |   2 | 59.39 | 27.73    | 43.45        | 31.70   |
| NCA      |   4 | 59.61 | 28.96 ↑  | 45.78 ↑      | 32.14 ↑ |
| PRO-S    |   2 | 59.47 | 29.31    | 45.90        | 30.80   |
| PRO-S    |   4 | 59.47 | 29.43    | 49.45 ↑      | 32.81 ↑ |

As shown in Table 3, PRO matches or surpasses NCA, verifying its effectiveness on scalar feedback. Besides, increasing N from 2 to 4 improves both methods across several benchmarks, suggesting that additional suboptimal examples further enhance alignment.

## 6 Discussion

While DPO has become a predominant approach for aligning LLMs, it remains limited to pairwise feedback. Additionally, the DPO loss is susceptible to likelihood underdetermination, inadvertently encouraging reward hacking. In this study, we introduced a decomposed perspective on DPO that not only reveals its potential to generalize to richer forms of feedback but also exposes the fundamental cause of likelihood underdetermination. Building on these insights, we proposed PRO, a practical method unifying alignment across diverse feedback while mitigating the underdetermination issue. Experimental results demonstrated that PRO effectively mitigates length exploitation and performance degradation during alignment, and performs consistently well across diverse feedback types.

This study also opens several avenues for future works. Prior studies have proposed various improvements to DPO [37-42], and recent work [43] shows that many of them can be equivalently realized by selecting appropriate reference models in the DPO loss. Since PRO is derived as a reformulation of DPO, it is interesting to explore how these strategies can further improve PRO's performance. Moreover, the DPO reformulation itself serves as a conceptual bridge to RLHF, as both incorporate an optimizer-regularizer composition. This connection invites opportunities for gentler regularization, improved model diversity, and integration with calibrated preference models in more general alignment/post-training scenarios, as detailed in Appendix C.

## References

- [1] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep Reinforcement Learning from Human Preferences. In Advances in Neural Information Processing Systems , 2017.
- [2] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347 , 2017.
- [3] Leo Gao, John Schulman, and Jacob Hilton. Scaling Laws for Reward Model Overoptimization. In International Conference on Machine Learning , 2023.
- [4] Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, et al. Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback. arXiv preprint arXiv:2307.15217 , 2023.
- [5] Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and Characterizing Reward Hacking. In Advances in Neural Information Processing Systems , 2022.
- [6] Nathan Lambert and Roberto Calandra. The Alignment Ceiling: Objective Mismatch in Reinforcement Learning from Human Feedback. arXiv preprint arXiv:2311.00168 , 2023.
- [7] Lilian Weng. Reward Hacking in Reinforcement Learning. lilianweng.github.io , Nov 2024.
- [8] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. In Advances in Neural Information Processing Systems , 2023.
- [9] Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J Liu. SLiCHF: Sequence Likelihood Calibration with Human Feedback. arXiv preprint arXiv:2305.10425 , 2023.
- [10] Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. A General Theoretical Paradigm to Understand Learning from Human Preferences. In International Conference on Artificial Intelligence and Statistics , 2024.
- [11] Yu Meng, Mengzhou Xia, and Danqi Chen. SimPO: Simple Preference Optimization with a Reference-Free Reward. arXiv preprint arXiv:2405.14734 , 2024.
- [12] Arka Pal, Deep Karkhanis, Samuel Dooley, Manley Roberts, Siddartha Naidu, and Colin White. Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive. arXiv preprint arXiv:2402.13228 , 2024.
- [13] Huayu Chen, Guande He, Lifan Yuan, Ganqu Cui, Hang Su, and Jun Zhu. Noise Contrastive Alignment of Language Models with Explicit Rewards. In Advances in Neural Information Processing Systems , 2024.
- [14] Teng Xiao, Yige Yuan, Huaisheng Zhu, Mingxiao Li, and Vasant G Honavar. Cal-DPO: Calibrated Direct Preference Optimization for Language Model Alignment. In Advances in Neural Information Processing Systems , 2024.
- [15] Noam Razin, Sadhika Malladi, Adithya Bhaskar, Danqi Chen, Sanjeev Arora, and Boris Hanin. Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization. In International Conference on Learning Representations , 2025.
- [16] Shusheng Xu, Wei Fu, Jiaxuan Gao, Wenjie Ye, Weilin Liu, Zhiyu Mei, Guangju Wang, Chao Yu, and Yi Wu. Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study. In International Conference on Machine Learning , 2024.
- [17] Duanyu Feng, Bowen Qin, Chen Huang, Zheng Zhang, and Wenqiang Lei. Towards Analyzing and Understanding the Limitations of DPO: A Theoretical Perspective. arXiv preprint arXiv:2404.04626 , 2024.

- [18] Yi Ren and Danica J. Sutherland. Learning Dynamics of LLM Finetuning. In International Conference on Learning Representations , 2025.
- [19] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Model Alignment as Prospect Theoretic Optimization. In International Conference on Machine Learning , 2024.
- [20] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783 , 2024.
- [21] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 Technical Report. arXiv preprint arXiv:2407.10671 , 2024.
- [22] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of Experts. arXiv preprint arXiv:2401.04088 , 2024.
- [23] Hamish Ivison, Yizhong Wang, Valentina Pyatkin, Nathan Lambert, Matthew Peters, Pradeep Dasigi, Joel Jang, David Wadden, Noah A Smith, Iz Beltagy, et al. Camels in a Changing Climate: Enhancing LM Adaptation with TÜLU 2. arXiv preprint arXiv:2311.10702 , 2023.
- [24] Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, et al. Zephyr: Direct Distillation of LM Alignment. arXiv preprint arXiv:2310.16944 , 2023.
- [25] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training A Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. arXiv preprint arXiv:2204.05862 , 2022.
- [26] Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. UltraFeedback: Boosting Language Models with High-quality Feedback, 2023.
- [27] Wei Shen, Rui Zheng, Wenyu Zhan, Jun Zhao, Shihan Dou, Tao Gui, Qi Zhang, and Xuanjing Huang. Loose Lips Sink Ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback. In The Conference on Empirical Methods in Natural Language Processing , 2023.
- [28] Jiacheng Xu Prasann Singhal, Tanya Goyal and Greg Durrett. A Long Way to Go: Investigating Length Correlations in RLHF. arXiv , 2023.
- [29] Ryan Park, Rafael Rafailov, Stefano Ermon, and Chelsea Finn. Disentangling Length from Quality in Direct Preference Optimization. arXiv preprint arXiv:2403.19159 , 2024.
- [30] Junru Lu, Jiazheng Li, Siyu An, Meng Zhao, Yulan He, Di Yin, and Xing Sun. Eliminating Biased Length Reliance of Direct Preference Optimization via Down-Sampled KL Divergence. arXiv preprint arXiv:2406.10957 , 2024.
- [31] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. DeepSeek-V3 Technical Report. arXiv preprint arXiv:2412.19437 , 2024.
- [32] Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning , 2023.
- [33] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7B. arXiv preprint arXiv:2310.06825 , 2023.

- [34] Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto. Length-Controlled Alpacaeval: A Simple Way to Debias Automatic Evaluators. arXiv preprint arXiv:2404.04475 , 2024.
- [35] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems , 2023.
- [36] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 07 2024.
- [37] Haoran Xu, Amr Sharaf, Yunmo Chen, Weiting Tan, Lingfeng Shen, Benjamin Van Durme, Kenton Murray, and Young Jin Kim. Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation. In International Conference on Machine Learning , 2024.
- [38] Changyu Chen, Zichen Liu, Chao Du, Tianyu Pang, Qian Liu, Arunesh Sinha, Pradeep Varakantham, and Min Lin. Bootstrapping Language Models with DPO Implicit Rewards. arXiv preprint arXiv:2406.09760 , 2024.
- [39] Yongcheng Zeng, Guoqing Liu, Weiyu Ma, Ning Yang, Haifeng Zhang, and Jun Wang. Tokenlevel Direct Preference Optimization. In International Conference on Machine Learning , 2024.
- [40] Motoki Omura, Yasuhiro Fujita, and Toshiki Kataoka. Entropy Controllable Direct Preference Optimization. arXiv preprint arXiv:2411.07595 , 2024.
- [41] Junkang Wu, Xue Wang, Zhengyi Yang, Jiancan Wu, Jinyang Gao, Bolin Ding, Xiang Wang, and Xiangnan He. α -DPO: Adaptive Reward Margin is What Direct Preference Optimization Needs. arXiv preprint arXiv:2410.10148 , 2024.
- [42] Junkang Wu, Yuexiang Xie, Zhengyi Yang, Jiancan Wu, Jinyang Gao, Bolin Ding, Xiang Wang, and Xiangnan He. β -DPO: Direct Preference Optimization with Dynamic β . In Advances in Neural Information Processing Systems , 2024.
- [43] Rasul Tutnov, Antoine Grosnit, and Haitham Bou-Ammar. Many of Your DPOs are Secretly One: Attempting Unification Through Mutual Information. arXiv preprint arXiv:2501.01544 , 2025.
- [44] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. WebGPT: BrowserAssisted Question-Answering with Human Feedback. arXiv preprint arXiv:2112.09332 , 2021.
- [45] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to Summarize with Human Feedback. In Advances in Neural Information Processing Systems , 2020.
- [46] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems , 2022.
- [47] Yuchun Miao, Sen Zhang, Liang Ding, Rong Bao, Lefei Zhang, and Dacheng Tao. Inform: Mitigating reward hacking in rlhf via information-theoretic reward modeling. In Advances in Neural Information Processing Systems , 2024.
- [48] Thomas Coste, Usman Anwar, Robert Kirk, and David Krueger. Reward Model Ensembles Help Mitigate Overoptimization. arXiv preprint arXiv:2310.02743 , 2023.

- [49] Yuanzhao Zhai, Han Zhang, Yu Lei, Yue Yu, Kele Xu, Dawei Feng, Bo Ding, and Huaimin Wang. Uncertainty-Penalized Reinforcement Learning from Human Feedback with Diverse Reward Lora Ensembles. arXiv preprint arXiv:2401.00243 , 2023.
- [50] Alexandre Ramé, Nino Vieillard, Léonard Hussenot, Robert Dadashi, Geoffrey Cideron, Olivier Bachem, and Johan Ferret. Warm: On The Benefits of Weight Averaged Reward Models. arXiv preprint arXiv:2401.12187 , 2024.
- [51] Adam Fisch, Jacob Eisenstein, Vicky Zayats, Alekh Agarwal, Ahmad Beirami, Chirag Nagpal, Pete Shaw, and Jonathan Berant. Robust preference optimization through reward model distillation. arXiv preprint arXiv:2405.19316 , 2024.
- [52] Tengyu Xu, Eryk Helenowski, Karthik Abinav Sankararaman, Di Jin, Kaiyan Peng, Eric Han, Shaoliang Nie, Chen Zhu, Hejia Zhang, Wenxuan Zhou, et al. The Perfect Blend: Redefining RLHF with Mixture of Judges. arXiv preprint arXiv:2409.20370 , 2024.
- [53] Ted Moskovitz, Aaditya K Singh, DJ Strouse, Tuomas Sandholm, Ruslan Salakhutdinov, Anca Dragan, and Stephen Marcus McAleer. Confronting Reward Model Overoptimization with Constrained RLHF. In International Conference on Learning Representations , 2024.
- [54] Mathieu Rita, Florian Strub, Rahma Chaabouni, Paul Michel, Emmanuel Dupoux, and Olivier Pietquin. Countering Reward Over-Optimization in LLM with Demonstration-Guided Reinforcement Learning. arXiv preprint arXiv:2404.19409 , 2024.
- [55] Lingfeng Shen, Sihao Chen, Linfeng Song, Lifeng Jin, Baolin Peng, Haitao Mi, Daniel Khashabi, and Dong Yu. The Trickle-Down Impact of Reward (In-) Consistency on RLHF. arXiv preprint arXiv:2309.16155 , 2023.
- [56] Paria Rashidinejad and Yuandong Tian. Sail into the Headwind: Alignment via Robust Rewards and Dynamic Labels against Reward Hacking. arXiv preprint arXiv:2412.09544 , 2024.
- [57] John Schulman. Proxy Objectives in Reinforcement Learning from Human Feedback. Invited Talk at International Conference on Machine Learning, 2023. https://icml.cc/virtual/ 2023/invited-talk/21549 .
- [58] Seungjae Jung, Gunsoo Han, Daniel Wontae Nam, and Kyoung-Woon On. Binary Classifier Optimization For Large Language Model Alignment. arXiv preprint arXiv:2404.04656 , 2024.
- [59] Amos Tversky and Daniel Kahneman. Advances in Orospect Theory: Cumulative Representation of Uncertainty. Journal of Risk and uncertainty , 1992.
- [60] Ruiqi Zhang, Licong Lin, Yu Bai, and Song Mei. Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning. arXiv preprint arXiv:2404.05868 , 2024.
- [61] Shitong Duan, Xiaoyuan Yi, Peng Zhang, Yan Liu, Zheng Liu, Tun Lu, Xing Xie, and Ning Gu. Negating Negatives: Alignment with Human Negative Samples via Distributional Dispreference Optimization. In Findings of the Association for Computational Linguistics: EMNLP , 2024.
- [62] Xiaojian Yuan, Tianyu Pang, Chao Du, Kejiang Chen, Weiming Zhang, and Min Lin. A Closer Look at Machine Unlearning for Large Language Models. arXiv preprint arXiv:2410.08109 , 2024.
- [63] Yuanshun Yao, Xiaojun Xu, and Yang Liu. Large Language Model Unlearning. In Advances in Neural Information Processing Systems , 2024.
- [64] Joel Jang, Dongkeun Yoon, Sohee Yang, Sungmin Cha, Moontae Lee, Lajanugen Logeswaran, and Minjoon Seo. Knowledge Unlearning for Mitigating Privacy Risks in Language Models. In The Annual Meeting of the Association for Computational Linguistics , 2023.
- [65] Remi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Zhaohan Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mesnard, Côme Fiegel, et al. Nash Learning from Human Feedback. In International Conference on Machine Learning , 2024.

- [66] Gokul Swamy, Christoph Dann, Rahul Kidambi, Steven Wu, and Alekh Agarwal. A Minimaximalist Approach to Reinforcement Learning from Human Feedback. In International Conference on Machine Learning , 2024.
- [67] Chaoqi Wang, Yibo Jiang, Chenghao Yang, Han Liu, and Yuxin Chen. Beyond Reverse KL: Generalizing Direct Preference Optimization with Diverse Divergence Constraints. In International Conference on Learning Representations , 2024.
- [68] Gian Wiher, Clara Meister, and Ryan Cotterell. On Decoding Strategies for Neural Text Generators. Transactions of the Association for Computational Linguistics , 2022.
- [69] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv preprint arXiv:2501.12948 , 2025.
- [70] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300 , 2024.
- [71] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model. arXiv preprint arXiv:2503.24290 , 2025.
- [72] Ziniu Li. Can Better Cold-Start Strategies Improve RL Training for LLMs? Notion Blog, 2025. https://tangible-polo-203.notion.site .

## Contents

| 1   | Introduction                                      | Introduction                                                                      | 1   |
|-----|---------------------------------------------------|-----------------------------------------------------------------------------------|-----|
| 2   | Preliminaries                                     | Preliminaries                                                                     | 2   |
| 3   | Theoretical Re-Examination of DPO                 | Theoretical Re-Examination of DPO                                                 | 3   |
|     | 3.1                                               | Reformulation of Population-Based DPO . . . . . . . . . . . . . . . . . . . .     | 3   |
|     | 3.2                                               | The Completeness of Regularizer Matters in Sample-Based Loss . . . . . . . .      | 4   |
| 4   | Proximalized Preference Optimization              | Proximalized Preference Optimization                                              | 5   |
|     | 4.1                                               | Introducing Hyper Response for Tractable Loss Approximation . . . . . . . . .     | 5   |
|     | 4.2                                               | The Existence of Optimal Solution for PRO . . . . . . . . . . . . . . . . . . .   | 7   |
| 5   | Experiments                                       | Experiments                                                                       | 8   |
|     | 5.1                                               | Resolving Likelihood Underdetermination . . . . . . . . . . . . . . . . . . . .   | 8   |
|     | 5.2                                               | Performance Comparison Under Pairwise and Binary Feedback . . . . . . . . .       | 9   |
|     | 5.3                                               | Aligning with Extremely Imbalanced Binary Feedback . . . . . . . . . . . . .      | 10  |
|     | 5.4                                               | Aligning with Scalar Feedback . . . . . . . . . . . . . . . . . . . . . . . . . . | 10  |
| 6   | Discussion                                        | Discussion                                                                        | 10  |
| A   | Related Work                                      | Related Work                                                                      | 17  |
| B   | Proof of Theorems                                 | Proof of Theorems                                                                 | 18  |
|     | B.1                                               | Equivalent Loss for Population-Based DPO . . . . . . . . . . . . . . . . . . .    | 18  |
|     | B.2                                               | Necessary Condition and Key Properties of the Optimal Solution to eDPO Loss       | 19  |
|     | B.3                                               | Relationship between the Optimal Solutions to PRO and eDPO Losses . . . . .       | 21  |
| C   | Comparison of the PRO Loss and the RLHF Objective | Comparison of the PRO Loss and the RLHF Objective                                 | 28  |
|     | Implementation Details of the PRO Loss            | Implementation Details of the PRO Loss                                            |     |
| D   |                                                   |                                                                                   | 30  |
| E   | The Role of α and β in PRO's Regularizer          | The Role of α and β in PRO's Regularizer                                          | 31  |
| F   | Additional Experimental Setup                     | Additional Experimental Setup                                                     | 32  |
| G   | Additional Experimental Results                   | Additional Experimental Results                                                   | 34  |

## A Related Work

Reward Hacking in RLHF RLHF employs a learned reward model to align LLM. While the reward model faithfully ranks responses within the training distribution [25, 44-46], it often fails to generalize beyond. Consequently, LLM can exploit this weakness to achieve high rewards without genuinely matching human intent. This effect, known as reward hacking, poses a significant challenge in RLHF [3-7].

Recent studies have explored various mitigation strategies, covering the improvements in reward modeling, policy optimization and data augmentation. For instance, the research in [47] introduces an information bottleneck framework to filter out irrelevant noise that may introduce spurious features in reward modeling. Recognizing the limitations of a single reward model, works in [48, 49] propose the use of reward ensembles, which aggregate outputs from multiple models to produce more robust reward estimates; these models can be further combined through weight averaging [50] to improve efficiency. Additionally, authors in [51] advocate a conservative approach by optimizing LLMs against the minimum reward predicted from a plausible set of reward models. In terms of RL algorithms, several studies [52, 53] argue that the widely used proximal policy optimization [2] is insufficient to prevent reward hacking, and suggest incorporating explicit constraints to enforce more cautious use of reward model. Regarding training data, demonstrations are utilized to guide LLM towards generating responses with calibrated rewards [54]; and augmentation tools [55] are applied to diversify the dataset in hopes of improving model generalization. Despite these advances, reward hacking remains a challenging and unresolved issue in RLHF.

Direct Alignment with Pairwise Preference Direct alignment methods [8-10] bypass the need for explicit reward models and instead optimize LLMs directly using the preference data. Removing reward model not only lowers computational cost, but also restricts loss evaluation to the offline data. Since no on-policy samples are involved, these methods were initially considered immune to reward hacking. However, recent studies [29, 30, 11] have shown that length exploitation-a familiar form of reward hacking observed in RLHF [27, 28]-persists in methods like DPO. To address, several methods have been proposed: R-DPO [29] incorporates response length as a penalty into the DPO loss; SamPO [30] proposes down-sampling tokens of both preferred and dispreferred responses to equal lengths when computing the implicit reward in DPO; and SimPO [11] introduces length-normalized rewards to define a novel alignment loss. Although these methods effectively reduce length exploitation, they rely on explicit manipulation or regularization of response length, which unlikely addresses general reward hacking issues. As reported in RLHF literature [56], reward hacking can also manifest as lazy generation [6], degraded downstream task performance [52], and hedging or self-doubt [57]. These observations underscore the necessity for a more comprehensive understanding of reward hacking in direct alignment methods.

An important clue arises from the widely reported phenomenon where the likelihoods of both preferred and dispreferred responses decrease after alignment [12-14]. In consequence, the generation probabilities for unobserved responses are unintentionally elevated, echoing the reward-hacking effect in RLHF. Several research efforts examine this likelihood decline, attributing it to factors such as embedding similarity between paired responses [15], asymmetric update ratios for the probabilities of paired responses [17], and the 'squeezing effect' of softmax when applying gradient ascent to dispreferred responses [18]. Notably, these conclusions are chiefly drawn from the analyses of training dynamics in DPO and its variants. In contrast, the presented work focuses on the loss function itself, providing new insights into the underlying cause of this phenomenon and proposing a natural resolution. Additionally, our study also differs from previous attempts to remedy the likelihood decrease, which commonly incorporate additional supervised or regression signals into DPO [12, 14] and may inadvertently compromise the original intent of alignment.

Direct Alignment with Pointwise Feedback Parallel studies have explored alignment methods that utilize feedback beyond the pairwise format. In [58], an upper bound is derived for DPO to accommodate binary feedback, and it is shown that refining this bound can enhance alignment performance. Departing from the DPO framework, KTO [19] utilizes prospect theory [59] to create a utility function tailored for binary feedback. More generally, there is an extensive body of literature concerning binary feedback that focuses on model unlearning [60-62], where negative responses are used to eliminate unwanted behaviors from LLMs. Among them, gradient-ascent methods [63, 64] are the most straightforward but often lead to catastrophic collapse. Interestingly, alignment-inspired methods exponentially slow down this collapse [60]. Apart from binary feedback, NCA [13] considers

scalar feedback, and formulates a classification task to capture the varying degree of desirability for each labeled response. While these existing methods tackle the different types of feedback, a unified approach capable of handling pairwise, binary and scalar feedback is still lacking.

## B Proof of Theorems

## B.1 Equivalent Loss for Population-Based DPO

Theorem 3.1. The population-based DPO loss is equivalent to the following one, in that they share same gradient:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where B denotes Bernoulli distribution, is a score function indicating the extent to which y is favored across other responses and satisfies E y ∼ µ [ s ( y )] = 0 .

Proof. Theorem 3.1 is derived from an analytical property of the log-sigmoid function, namely,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we use ∇ δ log σ ( δ ) = σ ( -δ ) and ∇ δ log σ ( -δ ) = -σ ( δ ) , the last equation is obtained by averaging (10) and (11). In other words, the convex combination of log-sigmoid gradients can be decomposed into two distinct components: one dependent on a , and another independent of it. When a encodes the learning signal from training data, the latter term naturally acts as a data-independent regularizer.

We next apply this identity to the gradient of

<!-- formula-not-decoded -->

yielding

<!-- formula-not-decoded -->

where a = p ( y 1 ≻ y 2 ) and δ = r θ ( y 1 ) -r θ ( y 2 ) .

The fist term A can be further simplified as follows:

<!-- formula-not-decoded -->

Substituting this result into (13) produces the final expression:

<!-- formula-not-decoded -->

Finally, to show that E y ∼ µ [ s ( y )] = 0 , note the following relationship:

<!-- formula-not-decoded -->

The proof follows immediately by rearranging terms.

## B.2 Necessary Condition and Key Properties of the Optimal Solution to eDPO Loss

Theorem 3.2. Let µ := µ ( y ) &gt; 0 for all y ∈ Y . If an optimal solution π ∗ to ̂ L eDPO exists, it satisfies the condition for any y ∈ Y :

<!-- formula-not-decoded -->

Proof. For notational simplicity, we denote π y = π ( y ) . The gradient of ̂ L eDPO can then be expressed as:

<!-- formula-not-decoded -->

In the last equation, we relabel y 1 and y 2 as y and y ′ in the last equation, respectively, to highlight that the subsequent discussion focuses on y 1 .

According to the theorem's precondition, an optimal solution π ∗ exists. This solution must first satisfy the following feasibility constraints:

<!-- formula-not-decoded -->

Moreover, its optimality requires that the gradients of the loss and the equality constraint satisfy for some λ . This condition is necessary; otherwise, there would exist a loss-descent direction that is orthogonal to the gradient of equality constraint. Moving along such a direction would reduce the loss while still satisfying the equality constraint. Furthermore, as indicated by (15), π ∗ lies in the interior of the feasible region. Therefore, a sufficiently small update step would also preserve the inequality constraints. These analyses indicate that if (16) is violated, the solution can be further improved within the feasible region, contradicting the optimality of π ∗ .

<!-- formula-not-decoded -->

Noting that the gradient on the right-hand side of (16) equals one, we obtain

∣ Taking the expectation over y ∼ π ∗ on both sides and substituting (14) together with E y ∼ ˆ µ [ ˆ s ( y ) ] = 0 , we arrive at

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Exchanging y and y ′ , we have

<!-- formula-not-decoded -->

where the second equality follows from σ ( z ) = 1 -σ ( -z ) . Combining (18) and (19) yields λ = 0 . Then, multiplying both sides of (17) by π ∗ ( y ) and substituting (14) give

<!-- formula-not-decoded -->

After rearranging the terms and substituting the above identities, the desired relation follows, completing the proof:

<!-- formula-not-decoded -->

Corollary 3.3. Under the preconditions of Theorem 3.2, the following results hold for a constant C :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. For those values of y where either ˆ µ ( y ) = 0 or ˆ s ( y ) = 0 , the right-hand side of equation (2) vanishes. Given that sigmoid function is strictly monotonic, the solution necessarily satisfies

<!-- formula-not-decoded -->

for some constant C .

Similarly, the monotonicity of sigmoid function also indicates

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B.3 Relationship between the Optimal Solutions to PRO and eDPO Losses

Theorem 4.1. Let H ⊆ Y \ supp(ˆ µ ) and µ := µ ( y ) &gt; 0 for all y ∈ Y . If optimal solutions π ∗ and π ∗ H to ̂ L eDPO and ̂ L PRO exist, they satisfy: π ∗ H ( y ) = π ∗ ( y ) , ∀ y ∈ Y \ H , (7)

<!-- formula-not-decoded -->

where C is the constant as defined in Corollary 3.3.

Proof. Although the optimization variables π ∗ H and π ∗ are defined on distinct response spaces, the two objectives ̂ L eDPO and ̂ L PRO can be shown to be equivalent under an appropriate reparameterization. The proof follows directly by substituting condition (3) into relevant terms of both objectives. For

clarity and completeness, the detailed derivation is presented below.

<!-- formula-not-decoded -->

where the last equation follows from noting that H ⊆ Y \ supp(ˆ µ ) and applying condition (3). Concretely, H ⊆ Y \ supp(ˆ µ ) implies

<!-- formula-not-decoded -->

Combining with condition (3), we have

<!-- formula-not-decoded -->

In the last equation, all probability ratios for responses within H are replaced by C . Therefore, the objective function is expressed using the more compact set of variables { C, π ( y ) | y ∈ Y \ H } .

Similarly,

<!-- formula-not-decoded -->

where we apply the reparameterization π ( H ) = Cπ ref ( H ) in the last equation, considering that π ( H ) is a single variable in this objective.

Under the reparameterized variables { C, π ( y ) | y ∈ Y \ H } , the two objectives ̂ L eDPO and ̂ L PRO coincide exactly. This establishes the equivalence of their optimal solutions, leading to the following relations:

<!-- formula-not-decoded -->

where C ∗ denotes the optimal value of C for the above two objectives. The second equation further implies

<!-- formula-not-decoded -->

Given that H ⊆ Y \ supp(ˆ µ ) , by applying equation (3) again, we have

<!-- formula-not-decoded -->

## B.4 Existence of Optimal Solution to the PRO Loss

Lemma B.1. If no optimal solution exists for the minimization of ̂ L PRO, there is an infinite sequence within the feasible region that approaches its boundary and strictly decrease the loss value.

Proof. Let L denote the infimum of ̂ L PRO over its feasible region, which may be finite or infinite. By definition of the infimum, there exists a sequence of feasible solutions { π n } such that and

<!-- formula-not-decoded -->

Since the sequence { π n } lies in the bounded set ∆ , the Bolzano-Weierstrass theorem guarantees the existence of a convergent subsequence { π k n } with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Because { ̂ L PRO ( π k n ; π ref ) } is a subsequence of { L PRO ( π n ; π ref ) } , it follows that

Suppose for the sake of contradiction that π ∞ ∈ ∆ . Then, by the continuity of ̂ L PRO on the feasible region, we have

<!-- formula-not-decoded -->

̂ L PRO ( π ∞ ; π ref ) = L, implying that the infimum is attained within the feasible region, contradicting the assumption that no optimal solution exists.

Hence, π ∞ / ∈ ∆ , and the subsequence { π k n } represents the sequence we aim to identify, i.e., it approaches the boundary of the feasible region while strictly decreases the loss value.

Theorem 4.2. Given any H ⊆ Y \ supp(ˆ µ ) and µ := µ ( y ) &gt; 0 , ∀ y ∈ Y H , there is a threshold α 0 such that, whenever α &gt; α 0 , an optimal solution π ∗ H to ̂ L PRO exists. 5 Proof. For convenience, we restate the PRO loss as follows:

Suppose that no optimal solution exists. According to B.1, there exists an infinite sequence of feasible solutions { π n } , which converges to the boundary of feasible region and strictly decreases the loss value. Denote the limit of this sequence by { π n } .

<!-- formula-not-decoded -->

Since π ∞ lies on the boundary of the feasible region ∆ = { π | π ( y ) &gt; 0 , ∀ y ∈ Y H and ∑ y ∈Y H π ( y ) = 1 } , at least one response must have zero probability. Define

Moreover, being on the boundary remains π ∞ a valid probability mass function, which implies that at least one response must also have positive probability; we denote this set as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For elements in these two sets, we have 6

Both Y 0 and Y + are therefore non-empty.

<!-- formula-not-decoded -->

where r ∞ ( y ) = β log π ∞ ( y ) π ref ( y ) . Thus,

5 A constructive choice of α 0 for general preference feedback is given in Corollary B.2.

<!-- formula-not-decoded -->

6 Since π ref is typically instantiated as a reference LLM that calculates response probabilities using the softmax function, it follows that π ref ( y ) &gt; 0 for all possible responses y .

Combining this observation with D KL ( B (1 / 2) || B (0)) = + ∞ implies that the regularizer in ̂ L PRO diverges to + ∞ as { π n } approaches π ∞ . This suggests that, to achieve a reduction in the overall loss value, the optimizer needs to decrease towards negative infinity at a faster rate. Conversely, if it decreases at a slower rate, the total loss would increase, which contradicts the assumption that the sequence { π n } continuously reduces loss.

A sufficient condition to ensure this contradiction is for all y 0 satisfying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Condition (25) identifies all responses that push the optimizer towards negative infinity, while Condition (24) ensures that their corresponding terms in the optimizer, i.e. -β ˆ µ ( y 0 )ˆ s ( y 0 ) · log π ( y 0 ) , decrease more slowly than the increase of the regularizer.

We next derive how to satisfy (24). The main technical difficulty is evaluating the limit superior in (24), as the expression involves multiple variables. To address this, we construct an upper bound by mapping the multivariate ratio to a univariate function, whose limiting behavior can then be analyzed directly via L'Hôpital's rule.

For any ϵ &gt; 0 , when π is close enough to π ∞ , Inequation (23) guarantees

<!-- formula-not-decoded -->

where r ( y ) = β log π ( y ) π ref ( y ) . This inequation further implies

Since the second sigmoid value is closer to 1 / 2 compared to the first one, we have

<!-- formula-not-decoded -->

∣ ∣ ∣ ∣ By substituting the above inequation for all y + ∈ Y + into the left-hand side of (24), and utilizing the non-negativity of KL divergence for y / ∈ Y + , we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, the function on right hand side contains only π ( y 0 ) as the free variable. By tentatively applying L'Hôpital's rule, we find that its ordinary limit exists:

<!-- formula-not-decoded -->

The limit superior equals the ordinary limit upon its existence. Then, Condition (24) can be satisfied once

<!-- formula-not-decoded -->

Since µ ( y ) &gt; 0 for all y ∈ Y H , the right-hand side is finite, allowing us to fulfill this inequality by selecting a sufficiently large α .

In summary, Inequality (24) is guaranteed to hold for sufficiently large α , implying that the sequence { π n } can not continuously decrease the loss when approaching π ∞ . This establishes the desired contradiction.

Corollary B.2. For general preference feedback, a valid α 0 can be constructed as:

<!-- formula-not-decoded -->

Proof. As stated in the proof of Theorem 4.2, to prevent the PRO loss from decreasing indefinitely as the solution approaches a specified boundary point π ∞ of the feasible region, it suffices to select

<!-- formula-not-decoded -->

where Y 0 = { y | π ∞ ( y ) = 0 , y ∈ Y H } and Y + = { y | π ∞ ( y ) &gt; 0 , y ∈ Y H } . Once α satisfies this condition for all boundary points , the loss cannot decrease continuously on the whole boundary, thus ensuring the existence of an optimal solution within the feasible region.

This can be achieved by further strengthening the above condition to make it π ∞ -independent:

- Instead of enforcing the inequality only for y 0 ∈ Y 0 : ˆ s ( y 0 ) &lt; 0 , we can require it for all y 0 ∈ Y H : ˆ s ( y 0 ) &lt; 0 .
- Since µ ( Y + ) = ∑ y ∈Y + µ ( y ) ≥ min y ∈Y + µ ( y ) &gt; min y ∈Y H µ ( y ) , we can safely use this lower bound for further simplification.

Putting these together, a sufficient and easily computable choice is

<!-- formula-not-decoded -->

Theorem 4.3. Consider the pairwise feedback setting, where µ = µ and H ⊆ Y \ supp(ˆ µ ) . For any α ≥ 1 /η 2 , an optimal solution π ∗ H to ̂ L PRO exists. Moreover, when α = 1 /η 2 , the PRO loss is equivalent to the following one in that they share same gradient:

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is an augmented empirical preference.

Proof. We first establish the second part of this theorem. Under the choices of µ = µ and α = 1 /η 2 , we have

<!-- formula-not-decoded -->

where the distribution in the second term reduces to y 1 , y 2 ∼ ˆ µ , because the responses in supp(ˆ µ ) are excluded from the hyper response ( by the precondition H ⊆ Y \ supp(ˆ µ ) ) so that y 1 , y 2 ∼ ˆ µ and y 1 , y 2 ˙ ∼ ˆ µ are equivalent.

By applying the reverse direction of Theorem 3.1 for the first two terms, and expanding the remaining KL terms, we have where

is the augmented empirical preference.

<!-- formula-not-decoded -->

For the first part of the theorem, it suffices to prove the existence of optimal solution for α = 1 /η 2 . This is because such existence requires the regularizer to dominate the loss function at the boundary of feasible region, as elaborated in the proof of Theorem 4.2. If α = 1 /η 2 already guarantees the dominance, increasing α -the strength of the regularizer-only further enhances this effect.

Assume that no optimal solution exists when α = 1 /η 2 . By Lemma B.1, there is an infinite sequence of feasible solutions, which converges to the boundary of feasible region and strictly decreases the loss value. Let π ∞ denote the limit of this sequence. Since π ∞ lies on the boundary of ∆ = { π | π ( y ) &gt; 0 , ∀ y ∈ Y H and ∑ y ∈Y H π ( y ) = 1 } , if follows that

Then, we have 7

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now analyze the behavior of ̂ L PRO-P as π approaches π ∞ , by examining the following mutually exclusive cases:

- y 0 , y + ∈ supp(ˆ ρ ) : In this case, the term in (27b) with y 1 = y 0 and y 2 = y + diverges to positive infinity.
- y 0 ∈ supp(ˆ µ ) and y + ∈ supp(ˆ ρ ) : Here, the term in (27c) with y 1 = y 0 and y 2 = y + also diverges to positive infinity.
- y 0 ∈ supp(ˆ ρ ) and y + ∈ supp(ˆ µ ) : Similarly, the term in (27c) with y 1 = y + and y 2 = y 0 diverges to positive infinity.
- y 0 , y + ∈ supp(ˆ µ ) : This case is more complex. Specifically, if ˆ p ( y 0 ≻ y + ) = 0 , the term in (27a) with y 1 = y 0 and y 2 = y + can remain finite as π → π ∞ . However, consider an arbitary response y ′ ∈ supp(ˆ ρ ) :

7 Since π ref is typically instantiated as a reference LLM that calculates response probabilities using the softmax function, it follows that π ref ( y ) &gt; 0 for all possible responses y .

- -If π ∞ ( y ′ ) &gt; 0 , the term in (27c) with y 1 = y 0 and y 2 = y ′ diverges to positive infinity.
- -Otherwise, i.e. π ∞ ( y ) = 0 , the term in (27c) with y 1 = y + and y 2 = y ′ diverges to positive infinity.

In summary, as π approaches π ∞ , at least one term in (27) diverges to positive infinity. Considering the non-negativity of -log σ ( · ) for the other terms, it follows that the overall loss ̂ L PRO-P also diverges to positive infinity. This result contradicts the expected monotonous descent of ̂ L PRO-P as { π n } approaches π ∞ .

## C Comparison of the PRO Loss and the RLHF Objective

Both the PRO loss and the RLHF objective incorporate an optimizer and a regularizer, yet they differ in the specific implementations. This section discusses these differences in detail and identifies several research problems that warrant further investigation.

The objective function (to be minimized) in RLHF is

<!-- formula-not-decoded -->

which shares the same gradient with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where sg ( · ) denotes the operation of stop gradient. For convenience of comparison, we rewrite ̂ L PRO here: 8

Hypothesis Reliance RLHF optimizes LLM through the guidance provided by a learned reward model. One crucial yet often overlooked element in this framework is selecting an appropriate hypothesis for the reward model. The Bradley-Terry model, appreciated for its simplicity and intuitive nature, is commonly used by default. However, when actual user preferences deviate from its underlying assumptions (e.g., user preferences may be non-transitive, which the Bradley-Terry model cannot accommodate [65, 66]), the reward model can produce inexact evaluations that mislead LLM optimization. As comparison, the PRO loss directly leverages the preference signal without relying on the rewards derived from another model. This direct approach hopefully improves robustness against discrepancies between the model hypothesis and true nature of user preferences.

Weighting Factor of ∇ θ log π θ ( y ) in Loss Gradient Equation (1) illustrates that examining the loss gradient, particularly the weighting factor associated with ∇ θ log π θ ( y ) , provides valuable insights into the alignment process. We now inspect this factor in the contexts of RLHF and PRO, revealing another noteworthy distinction. First, when computing the optimizer gradients, RLHF weights ∇ θ log π θ ( y ) with the unbounded reward value r ϕ , whereas PRO uses the bounded preference score ˆ s . Second, by simplifying the regularizer gradients as

<!-- formula-not-decoded -->

8 ̂ L eDPO can be viewed as an instantiation of ̂ L PRO by letting H consist of only a single individual response.

and we find that the weight boundedness here is consistent to that in the optimizers. Overall, the integrated weight of ∇ θ log π θ ( y ) for PRO is bounded, in contrast to the unbounded one in RLHF. Given the stochastic nature of loss gradient and the tendency of LLM to forget, the bounded weight would result in a more cautious model update, which potentially improves training stability and model performance when utilizing PRO.

<!-- formula-not-decoded -->

Mode Seeking v.s. Mass Covering The RLHF objective includes reverse KL divergence as a regularizer, which is characterized by mode-seeking behavior. This means, minimizing the reverse KL tends to concentrate probability mass around the modes of the target distribution, while relatively neglecting low-probability regions. Formally, when π θ ( y ) → 0 for certain y , the regularizer diminishes even if π ref ( y ) is non-zero:

<!-- formula-not-decoded -->

where the penultimate equation follows from L'Hôpital's rule. This behavior has been reported to reduce diversity in LLM generation [67, 68]. By contrast, PRO employs forward KL divergence as the regularizer, which diverges to infinity as π θ ( y ) → 0 :

<!-- formula-not-decoded -->

This regularizer prevents response probability from collapsing to zero, exhibiting mass-covering behavior. Besides, it is computed over every response pair constructed from Y H . By adjusting the composition of H , we can flexibly modulate the extent of probability mass coverage over the original response space Y .

The presented work focuses exclusively on direct alignment, specifically optimizing LLMs using offline data, without the inclusion of on-policy or online samples. However, we want to emphasize that the proposed PRO method can be directly adopted in on-policy or online scenarios. Considering the above distinctions between PRO and RLHF, several followup research problems arise:

- Despite the risk of unreliable evaluations, the reward model in RLHF provides additional alignment signals for unlabeled responses, which may further improve performance if applied

appropriately. This motivates extending PRO to incorporate on-policy samples within such a paradigm, particularly when computational resources are sufficient. A key difference in implementation is that we require a score model to predict ˆ s ( y ) , rather than the reward model. Since the score is no longer a latent variable inferred from user preferences, it does not rely on any model hypothesis. Moreover, by its definition and the property established in Theorem 3.1, the score is bounded in [ -1 / 2 , 1 / 2] and has an expectation of zero. These information can be leveraged to calibrate the score model or determine when it is reliable. Considering these potential benefits, it is worthwhile to investigate how PRO performs compared to RLHF when applied in the on-policy setting.

- Recent progress has shown that online reinforcement learning can substantially enhance the reasoning capabilities for LLMs [69]. Popular approaches, such as PPO [2] and GRPO [70], incorporate D KL ( π θ || π ref ) as a regularizer. However, it is reported [71] that the regularizer is excessively strong, resulting in overly constrained optimization of the LLM. Although fully removing the regularizer mitigates this issue, it may lead to performance degeneration on other unconsidered tasks during post training. This highlights the need for alternative methods to address over-regularization. As noted, the regularizer in PRO is milder since its gradient involves a bounded weighting factor on ∇ θ log π θ ( y ) . Given the rule-based rewards are also bounded, PRO presents a promising objective function for such settings and warrants further empirical investigation.
- During the reinforcement learning stage, LLMs explore response space by leveraging the prior knowledge acquired from pretraining or supervised fine-tuning. Maintaining response diversity is critical for enabling a broad spectrum of meaningful exploration, which in turn fosters the development of more advanced reasoning abilities [72]. Since the regularizer in PRO exhibits a mass-covering behavior, it is likely more effective at preserving diversity during post-training. We consider examining its practical effects as another future research direction.

## D Implementation Details of the PRO Loss

Pairwise Feedback In practical pairwise-feedback datasets, each prompt x is commonly associated with only one pair of responses. Therefore, the empirical distribution over responses is given by ˆ µ ( y w ) = ˆ µ ( y l ) = 1 2 . By setting H = Y \ { y w , y l } and η = 2 3 , we can rewrite L PRO-P as where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The value of η is chosen so that all response pairs from { y w , y l , H} contribute equally to the overall loss.

Binary Feedback Following the KTO implementation, the binary-feedback data is structured as D = { ( x ( i ) , y ( i ) , s ( i ) ) } I i =1 . That is, even though multiple responses may exist for a prompt, they are treated separately. Consequently, the empirical distribution reduces to ˆ µ ( y ) = 1 . By setting H = Y \ { y } and µ = µ as in (9), we can rewrite ̂ L PRO as ̂ L PRO-B ( π θ ; π ref ) = -β E ( x,y,s ) ∼D [ s log π θ ( y ) + α ( 1 2 log σ ( r θ ( x, y ) -r θ ( x, H ) ) + 1 2 log σ ( r θ ( x, H ) -r θ ( x, y ) ) ) ] , where r θ ( x, H ) = β log 1 -π θ ( y | x ) 1 -π ref ( y | x ) .

The regularizer coefficient is in fact αη (1 -η ) . Since there is no need to have two parameters to determine the regularization strength, we re-denote the coefficient by α for notational simplicity. In the experiments, we tune α so that the performance of PRO-B on Anthropic-HH test dataset matches that of PRO-P under β = 0 . 003 . The resulting value is α = 2 . 5 , which we use throughout unless otherwise noted.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

The value of η is chosen so that all response pairs from { y 1: N , H} contribute equally to the overall loss. The regularizer strength is in fact given by α ( N +1) 2 . We rescale it to 2 α N ( N +1) to ensure that the optimizer-regularizer weight ratio remains consistent with ̂ L PRO-B for any α . In accordance with the binary-feedback case, we set α = 2 . 5 by default.

A common characteristic of the three cases above is that H contains a countless number of responses, making π θ ( H| x ) and π ref ( H| x ) extremely close to one. Moreover, a small step of parameter update along any direction does not significantly alter the closeness of π θ ( H| x ) to one. Consequently, both the value and the gradient of r θ ( x, H ) are approximately zero, allowing us to safely omit this term from the loss function. This behavior was verified in our preliminary experiments, and in all the reported experiments we set r θ ( x, H ) = 0 for simplicity.

## E The Role of α and β in PRO's Regularizer

By rewriting the PRO loss as where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

∣ ∣ the hyperparameters α and β are only involved in the function f α,β . This allows us to analyze their impacts on the loss by simply examining f α,β .

Noticing that

<!-- formula-not-decoded -->

we see that α determines the maximum gradient magnitude of the regularizer, while β governs how rapidly the gradient grows as δ departs 0. These effects are illustrated in Figure 4. Notably, if α is

too small, the gradient of the overall loss can be dominated by the optimizer, causing unpreferred responses reduced towards zero probability and compromising our theoretical guarantees. This explains why increasing β fails to mitigate performance deterioration but increasing α helps in Section 5.3.

<!-- image -->

δ

δ

Figure 4: α determines the maximum norm of the regularizer gradient, while β controls the rate at which the gradient norm increases from zero to its maximum value.

We now have two hyperparameters α and β in PRO, however, tuning them is considerably simpler than it appears. In particular, it is not necessary to jointly tune α and β : for a broad range of α values above the necessary threshold, one can choose a corresponding β that enables PRO to consistently attain strong performance.

To elaborate, Theorem 4.2 guarantees the existence of a threshold α 0 such that, for any α &gt; α 0 , the regularizer remains effective and prevents probabilities from vanishing. A value of α being sufficiently large indicates that the magnitude of the regularizer gradient should rarely hit its saturation regime during optimization. Let α 1 be such a sufficiently large value (i.e., α 1 &gt; α 0 ) and β 1 its tuned counterpart. Because α and β provides adequate flexibility to shape the curve of regularizer gradient, for any α 2 &gt; α 1 , we can select a β 2 so that the curve of ∇ δ f α 2 ,β 2 shows a very similar shape with ∇ δ f α 1 ,β 1 prior to its saturation region. For instance, in Figure 5.3 (right column), the gradient curves for ( α, β ) = (1, 3) and (3, 1) largely overlap for δ ∈ ( -0 . 5 , 0 . 5) . We also observe that different ( α, β ) yield similar performance in our experiments. Under pairwise feedback, the win rates for for (2.5, 3e-3), (7.5, 1e-3) and (22.5, 3e-4) are 53 . 21% , 52 . 97% and 53 . 65% , respectively. Under 1%-desired binary feedback, the win rates for (17.5, 3e-3), (52.5, 1e-3) and (157.5, 3e-4) are 57 . 21% , 56 . 51% and 56 . 33% , respectively.

## F Additional Experimental Setup

For the Pythia-6.9B model, we first utilize the preferred responses from Anthropic-HH dataset for supervised fine-tuning prior to alignment, following the procedures in [8]. Since the Mistral-7B-sft model is already fine-tuned, we apply it directly for alignment.

Hyperparameters We adopt a uniform set of hyperparameters across all models, methods and datasets, for both supervised fine-tuning and alignment. Specifically, the maximum prompt length is 1024, and the maximum sequence length is 2048. Models are trained for 1 epoch with RMSprop as the optimizer. The learning rate linearly increases from 0 to 5e-7 over the first 150 training steps and remains constant thereafter. The batch size is 64 for KTO and PRO-B, and 32 for the other methods. This adjustment is due to each binary sample containing a single response, which is half that of pairwise samples. To ensure a consistent number of gradient steps when comparing methods using pairwise and binary feedback, we double the batch size for binary feedback.

When evaluating KTO with imbalanced binary feedback, we implement a reweighting strategy in the loss function to balance the effective proportions of desired and undesired samples at a 1:1 ratio. For instance, if the original ratio of desired to undesired samples is 100:1, we multiply the loss for undesired samples by a factor of 100. This reweighting technique was initially introduced in the KTO paper. We also apply it to PRO.

## Prompt for Evaluating Aligned LLMs on Anthropic-HH Dataset

```
For the following query to a chatbot, which response is more helpful, \ harmless and concise? Query: <the user query> Response A: <either the test method or baseline> Response B: <the other response> FIRST provide a one-sentence comparison of the two responses and explain \ which you feel is more helpful, harmless and concise. SECOND, on a new \ line, state only "A" or "B" to indicate which response is more helpful, \ harmless and concise. Your response should use the format: Comparison: <one-sentence comparison and explanation> Better response: <"A" or "B">
```

## G Additional Experimental Results

Figure 5: Dynamics of implicit reward r θ when aligning Mistral-7B-sft with the pairwise/binarized UltraFeedback dataset. In DPO, the rewards for preferred examples initially increase but then exhibit a continuous decline. In contrast, both NCA and PRO maintain consistently positive rewards throughout the alignment process. Besides, the rewards of NCA, KTO and PRO demonstrate a convergent trend as training progresses.

<!-- image -->

<!-- image -->

Figure 6: Dynamics of implicit reward r θ when aligning Mistral-7B-sft with the raw UltraFeedback dataset (scalar feedback), where N denotes the number of responses per instruction. Both NCA and PRO maintain consistently positive rewards for preferred examples throughout the alignment process.

Table 4: Performance comparison by aligning the Mistral-7B-sft model with UltraFeedback dataset. All methods demonstrate significant performance improvements after alignment. DPO achieves the leading performance on ARC and HellaSwag, but underperforms on IFEval and Math. PRO performs comparably to or better than the best baseline across all tasks.

| Method   |   ARC |   IFEval |   TruthfulQA |   GPQA |   Math |   HellaSwag |   Average |
|----------|-------|----------|--------------|--------|--------|-------------|-----------|
| SFT      | 51.54 |     2.4  |        42.23 |  29.02 |   1.06 |       61.02 |     31.21 |
| DPO      | 61.77 |    19.22 |        43.45 |  32.04 |   0.52 |       64.12 |     36.85 |
| KTO      | 55.38 |    25.69 |        41    |  33.04 |   0.46 |       62.61 |     36.36 |
| NCA      | 58.62 |    26.43 |        42.35 |  32.14 |   0.6  |       63.48 |     37.27 |
| PRO-P    | 61.26 |    29.02 |        43.81 |  32.59 |   1.36 |       63.47 |     38.59 |
| PRO-B    | 59.81 |    30.13 |        42.72 |  32.81 |   1.14 |       63.27 |     38.31 |

Table 5: Performance comparison by aligning the Mistral-7B-sft model with UltraFeedback dataset. N denotes the number of responses per prompt. Increasing N effectively improves the alignment performance for most tasks.

| Method   |   N |   ARC | IFEval   | TruthfulQA   | GPQA    | Math   | HellaSwag   | Average   |
|----------|-----|-------|----------|--------------|---------|--------|-------------|-----------|
| NCA      |   2 | 59.39 | 27.73    | 43.45        | 31.70   | 0.76   | 63.70       | 37.79     |
| NCA      |   4 | 59.61 | 28.96 ↑  | 45.78 ↑      | 32.14 ↑ | 1.12 ↑ | 62.88 ↓     | 38.41 ↑   |
| PRO-S    |   2 | 59.47 | 29.31    | 45.90        | 30.80   | 1.10   | 62.87       | 38.24     |
| PRO-S    |   4 | 59.47 | 29.43    | 49.45 ↑      | 32.81 ↑ | 1.14   | 62.36 ↓     | 39.11 ↑   |

| Prompt    | < &#124; user &#124; > What does a diplomat do? < &#124; &#124; >                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Preferred | &#124; &#124; It can be. I personally think it's a fascinating job, and I often find it emotionally rewarding to help people in need.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| DPO       | It is, there are many challenges as a diplomat, but it can also be quite rewarding. I find the work interesting and challenging. I also find it very rewarding to help other countries work towards better solutions to international conflicts. I think it's really important to have international relationships, and maintaining them is key to advancing the world as a whole. I also think that having international relationships is important for any country, because without mutual respect and trust we can't make great progress on issues like trade and diplomacy. I also enjoy meeting different people from around the world, as well as the work itself. I find it very interesting and stimulating. I think it's also really important to be well-rounded as a person, and having a diverse skillset is a good way to stay engaged and interested in different topics. I also think it's very important for a diplomat to be honest and diplomatic in their interactions with other countries, so that they can build trust and get things done. It can be tough work though, I think. |
| Judgment  | Response B [Preferred] is more concise and focused, avoiding unnecessary elabora- tion while still being helpful and harmless, whereas Response A [DPO] is overly verbose and includes repetitive and subjective details that detract from its clarity and usefulness.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| PRO       | That it is! A career as a diplomat can offer a variety of interesting opportunities and challenges. Diplomats have to work hard to earn respect from other countries, and they also need to work hard at keeping their own country politically stable. The challenges are both internal and external to the job, and they can be exciting.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Judgment  | Response A [PRO] is more helpful and concise as it provides a clear explanation of the challenges and opportunities in a diplomat's career, while Response B [Preferred] is more subjective and less informative, focusing on personal feelings rather than factual details.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

Table 6: Samples from the aligned Pythia-6.9B models for informational question. The judgment is generated by DeepSeek-V3 through a comparison between the preferred response in dataset and the model-generated response, based on the criteria of helpfulness, harmlessness, and conciseness. The responses are presented in a random order when fed to the judge, and the post-hoc annotations are formatted as [annotation].

Table 7: Samples from the aligned Pythia-6.9B models for book recommendation.

| Prompt    | < &#124; user &#124; > How can I tell if a person is right for marriage? I really like this woman but am unsure about proposing. < &#124; assistant &#124; > It sounds like you're really happy to have found someone special! But you are right to hesitate - marriage is a commitment you should not make lightly, and you want to do it right. I could definitely suggest resources or questions you can think through, but I think the best way is if we talk more about what you mean when you say 'right for marriage', and what questions you want to ask her. < &#124; user &#124; > Can you provide me with any book titles that speak on this topic? < &#124; assistant &#124; >                                                                                                                                                                                                                                               |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Preferred | Do you mean about marriage in general, or about particular questions about a possible marriage partner?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| DPO       | Yes, there are lots of great books out there on the topic of 'right for marriage', here are a few good ones: - The Five Love Languages by Gary Chapman - The 5 Love Languages of Marriage by Gary Chapman - The 5 Love Languages of Intimacy by Gary Chapman - The Five Love Languages of Affection by Gary Chapman - Marry Smart by Shaunti Feldheym - The 5 Love Languages of Family by John Townsend - What Men Want by Tristan Taormino - Intimately: Finding True Love in Relationships by John Gottman - The 7 Principles for Making Marriage Work by John C. Maxwell - Right For Me, Right From You by John Medvin - The Five Love Languages of Love by Gary Chapman and Jennifer Van Osten - Marriage, a History by John Wiley and Sons - Marry Smart by Shaunti Feldheym and The Five Love Languages of Affection by Gary Chapman are both good for a more in-depth treatment of the topic I hope some of these will be useful! |
| Judgment  | Response A [Preferred] is more concise and helpful as it seeks clarification to pro- vide a more tailored recommendation, while Response B [DPO] is overly verbose, repetitive, and includes irrelevant or redundant book titles, making it less focused and potentially confusing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| PRO       | I've got two good books for you. The first one is The Five Love Languages by Gary Chapman, which talks about the kind of connection you want in a marriage. He talks about how people love to be loved in different ways. The second one is When to Say 'I Do' by John and Catherine Eldredge. They talk about the kind of commitment you want in marriage and also some of the things you might want to ask her before making a big decision. I would definitely recommend those two books.                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Judgment  | Response B [PRO] is more helpful, harmless, and concise because it directly provides specific book recommendations tailored to the user's query about marriage, while Response A [Preferred] asks for clarification without offering immediate assistance.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: All claims in the abstract and introduction reflect the paper's contributions and scope, which cover (1) a theoretical reformulation of DPO, (2) novel insights into root cause of likelihood underdetermination, (3) a new alignment method for diverse feedback types that is free from underdetermination, and (4) empirical results demonstrating the utility of our proposed method in practice.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: In Appendix C, we clarify that this work focuses exclusively on offline scenarios. We also discuss the intriguing connection between our proposed method and RLHF, and identify several research problems in on-policy and online settings that warrant further investigation in future studies.

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

Justification: The precondition of each theorem is presented at the start of the respective statement. Appendix B includes all the proofs.

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

Justification: Section 5 includes the data preparations for the experiments. Appendices D and G discuss the PRO implementations and the additional experimental setups, respectively.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.

In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## Answer: [Yes]

Justification: The implementation details are provided in Appendix D. We will open-source the code. All datasets used have open access.

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

Justification: Section 5 and Appendix G include all necessary details.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

## Answer: [No]

Justification: Due to the resource limitation, error bars are not reported. However, please note that Figure 2 reports the performance throughout the training process, and Figure 3 illustrates results across various hyperparameter settings. We believe these complementary evaluations already significantly distinguish the proposed method from the baselines.

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

Justification: We discuss in Section 4.1 that the proposed method requires only basic mathematical operations beyond DPO, with no additional LLM forward or backward computations. Therefore, it incurs negligible computational overhead in both time and memory compared to DPO.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our research was conducted in accordance with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our study is a methodological contribution aimed at advancing preference optimization and does not have direct societal impacts.

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

Justification: This work does not involve releasing data or models that have a high risk for misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We credit the creators of the assets used in the paper through proper citation.

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

Justification: The paper does not release new assets for now. Once the code is open-sourced, we will include well-documented instructions with it.

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