## Wisdom is Knowing What not to Say: Hallucination-Free LLMs Unlearning via Attention Shifting

Chenchen Tan 1 , Youyang Qu 2,3 , Xinghao Li 1 , Hui Zhang 4 , Shujie Cui 1 , Cunjian Chen 1 , Longxiang Gao 2,3 ∗

1

Faculty of Information Technology, Monash University, Australia

2 Key Laboratory of Computing Power Network and Information Security, Ministry of Education, Shandong Computer Science Center,

Qilu University of Technology (Shandong Academy of Sciences), Jinan, China

3 Shandong Provincial Key Laboratory of Computing Power Internet and Service Computing, Shandong Fundamental Research Center for Computer Science, Jinan, China

4 School of Computer Science and Technology, Anhui University, Hefei, China &lt;chenchen.tan@monash.edu&gt; , &lt;gaolx@sdas.org&gt;

## Abstract

The increase in computing power and the necessity of AI-assisted decision-making boost the growing application of Large Language Models (LLMs). Along with this, the potential retention of sensitive data of LLMs has spurred increasing research into machine unlearning. However, existing unlearning approaches face a critical dilemma: Aggressive unlearning compromises model utility, while conservative strategies preserve utility but risk hallucinated responses. This significantly limits LLMs' reliability in knowledge-intensive applications. To address this, we introduce a novel Attention-Shifting (AS) framework for selective unlearning. AS is driven by two design objectives: (1) context-preserving suppression that attenuates attention to fact-bearing tokens without disrupting LLMs' linguistic structure; and (2) hallucination-resistant response shaping that discourages fabricated completions when queried about unlearning content. AS realizes these objectives through two attention-level interventions, which are importance-aware suppression applied to the unlearning set to reduce reliance on memorized knowledge and attention-guided retention enhancement that reinforces attention toward semantically essential tokens in the retained dataset to mitigate unintended degradation. These two components are jointly optimized via a dual-loss objective, which forms a soft boundary that localizes unlearning while preserving unrelated knowledge under representation superposition. Experimental results show that AS improves performance preservation over the state-of-the-art unlearning methods, achieving up to 15% higher accuracy on the ToFU benchmark 2 and 10% on the TDEC benchmark 3 , while maintaining competitive hallucination-free unlearning effectiveness. Compared to existing methods, AS demonstrates a superior balance between unlearning effectiveness, generalization, and response reliability.

## 1 Introduction

Large Language Models (LLMs) have recently achieved substantial advancements in natural language understanding and generation [1]. However, despite their achievements, a growing concern is the

∗ Corresponding Author

2 https://locuslab.github.io/tofu/

3 https://github.com/google-research/lm-extraction-benchmark

potential for LLMs to memorize and subsequently reproduce sensitive data, resulting in serious privacy concerns [2, 3]. Moreover, regulatory frameworks such as the General Data Protection Regulation (GDPR) grant the 'Right to be Forgotten' to the data owner, which mandates that users can request to remove their data from digital systems, including machine learning models [4]. Under these circumstances, machine unlearning has emerged as a critical approach for preserving data privacy in LLMs [5, 6, 7, 8, 9, 10]. These privacy-preserving unlearning tasks can be broadly categorised as either aggressive or conservative. Aggressive approaches, such as Gradient Ascent (GA) [5], modify the LLMs' learning objective to erase target knowledge forcibly. This often leads to a degradation in overall model performance, particularly on neighbouring knowledge, which is data with a similar structure or semantic relation to the unlearning target. Others adopt conservative unlearning strategies like the logits manipulation [11, 12, 13] to maintain model performance but risk introducing factual hallucinations, where the model confidently generates content detached from underlying facts. These hallucinations pose a serious threat in downstream tasks like questionanswering systems, especially in applications such as healthcare or legal scenarios, where precise and reliable outputs are critical [14, 15].

Achieving effective unlearning in privacy-sensitive LLM applications is challenging due to the conflicting interests and objectives of different stakeholders. On one hand, data providers whose information has been used in LLMs aim to entirely prevent reproducing their data in LLMs' answer [16, 17]. On the other hand, model deployers try to preserve the model's general capabilities across broad knowledge domains and to maintain the service quality [12, 18]. We define this as a multistakeholder balanced unlearning setting, where the unlearning strategies should achieve two primary objectives: 1) to enable the LLM to 'unlearn' the target data while ensuring the LLM maintains performance both on neighbouring knowledge [18] and general knowledge; and 2) to prevent the hallucination outputs for the unlearned knowledge.

To achieve these goals, we propose a novel Attention-Shifting (AS) unlearning. It is a controlled form of an aggressive method that performs both context-preserving unlearning and hallucination-resistant generation. AS suppresses attention to fact-bearing tokens in the unlearning set while reinforcing attention to semantically important tokens in retained data. This mechanism reduces reliance on target knowledge without compromising fluency or coherence, thereby preserving the contextual integrity of general and neighbouring knowledge. Unlike prior methods [11, 18] that manipulate logits or replace outputs, AS reallocates attention internally, blocking the flow of memo-

Figure 1: LLMs tend to assign high attention to semantically important tokens, like 'father' and 'hairdresser', when recalling memorized facts. Our method applies supervised attention suppression to downweight fact-bearing tokens and reallocate focus to neutral tokens. This redistribution reduces reliance on target knowledge while maintaining generation fluency, thereby facilitating precise and fluent unlearning.

<!-- image -->

rized knowledge during generation (Fig. 1). This enables the model to forget through omission rather than substitution, reducing hallucinations by structurally eliminating access to unlearned content. AS derives suppression and reinforcement signals from reference attention maps of the original model, and injects them via lightweight (~12M) adapters in attention modules. A dual-loss objective jointly optimizes unlearning and retention, forming a soft boundary that localizes suppression while stabilizing unrelated knowledge. While neurons in LLMs are known to exhibit representation superposition, shared activations across multiple concepts, our design remains robust under such entanglement, achieving behavioral unlearning without requiring explicit disentanglement [19, 20]. Experiments on the ToFU [21] and TDEC [5] benchmarks demonstrate that AS achieves near-zero knowledge leakage while preserving strong performance, with up to 15% and 10% higher accuracy than state-of-the-art baselines, respectively. In contrast to methods such as ULD [12] and IHL [11], which suppress target tokens but may still generate misleading completions, AS structurally blocks access to forgotten knowledge and promotes refusal behaviors, thereby effectively minimizing hallucinations. The source code is available at https://github.com/CCT-sys/AS-unlearning.git .

## 2 Related Works

In this section, we introduce the existing unlearning methods, followed by summarizing the challenges in LLMs unlearning and identifying the gaps in existing methods. We categorize the existing unlearning methods into aggressive unlearning and conservative unlearning, depending on how directly they modify the model's internal representations [18].

Aggressive Unlearning removes specific knowledge by actively disrupting its learned representations in the model. This type of approach significantly alters decision boundaries, leading to broader unintended shifts in model behaviour. One example is the GA [5] approach. It forces the model to learn an inverted objective of the target knowledge to achieve unlearning, which aggressively reverses the influence of target data, pushing the model away from target knowledge uncontrollably and raising catastrophic collapse. To address this limitation, Negative Preference Optimization (NPO) [22] was proposed as a more stable and controlled extension of GA. NPO adopts a preference-based loss function, which smooths the optimization process and prevents extreme parameter updates. Several variants [10, 23, 24, 25, 26, 27] and extensions have been proposed to further mitigate the instability and over-forgetting observed in GA to provide selective unlearning.

Conservative Unlearning steers the model toward preferred alternative responses by suppressing target tokens and reinforcing substitutes [11, 12, 28]. Cha et al. [11] use an inverted hinge loss to penalize target tokens and reinforce logical substitutes. Ji et al. [12] propose ULD, which subtracts logits from a lightweight assistant model trained on the target, significantly lowering target generation probabilities. These methods are effective in open-ended tasks but may retain latent semantics, leading to partial unlearning or hallucinations. For example, replacing 'physicist' with 'artist' in 'Einstein was a physicist' may yield paraphrases like 'scientist' or factual errors like 'dancer,' undermining LLMs' trust. Similar observations are presented in the safety alignment study [29], which proves that shallow alignment like Reinforcement Learning with Human Feedback (RLHF) or logits modification interventions fail to suppress undesired activations. Beyond logits-level methods, embedding-based approaches [8, 18, 30] offer more controllable unlearning by steering model inputs toward structured alternatives with minimal disruption.

These unlearning methods reflect distinct trade-offs between preserving model utility and mitigating hallucinations. Aggressive strategies often impair general performance [10, 23, 24, 25, 26, 27], while conservative ones require complex auxiliary mechanisms [6, 8, 12, 18, 30] and risk factual inconsistencies. LLMs unlearning still remains challenging due to conflicting demands: data providers seek privacy and removability, while deployers prioritize broad functionality. To bridge this gap, we propose an Attention-Shifting strategy, a controlled form of aggressive unlearning, which suppresses attention to fact-bearing tokens. This targeted intervention disrupts access to memorized content while preserving overall utility, striking a practical balance between effectiveness and stability.

## 3 Attention Shifting for Machine Unlearning in LLM

## 3.1 Token Importance for Selective Unlearning

The attention mechanism is the foundation for how LLMs allocate representational focus across tokens, influencing generation probabilities. To ground our approach, we begin by analyzing how token-level relevance shapes predictions. Intuitively, nouns, proper nouns, and domain-specific terms act as semantic anchors, while function words, e.g., determiners, conjunctions, contribute minimally to meaning. Building on this, Duan et al. [31] propose Shifting Attention to Relevance (SAR), which estimates token importance via masking-based perturbation and reallocates attention toward salient tokens to enhance prediction confidence. In contrast, we apply this insight for unlearning: our method suppresses attention to high-importance tokens that encode factual or sensitive knowledge. Unlike SAR, which operates only at inference time, our approach embeds attention suppression into model parameters via lightweight adapters, enabling controlled unlearning.

Formally, given an input sequence x = { t 1 , t 2 , . . . , t n } and predictive distribution P θ ( y | x ) , the importance of token t i is defined as the change in predictive entropy when t i is masked:

<!-- formula-not-decoded -->

where ϕ denotes predictive entropy [32] and x -i is the input with t i masked. Higher I ( t i ) values correspond to stronger contributions to predictive certainty.

We leverage this signal to guide attention modulation during training. Specifically, let D denote the whole LLM's training data, we propose a composite loss: (1) an attention suppression (ASP) loss L ASP on the unlearning set D t ∈ D , which penalizes attention to high-importance tokens; and (2) an attention reinforcement loss L AKL on the sub-remaining dataset D ′ r ∈ D/D t , promotes attention to semantically important tokens, thereby stabilizing model behavior and preserving utility on retained knowledge. The overall objective is:

<!-- formula-not-decoded -->

where α ∈ [0 , 1] is the mixing coefficient that trades off unlearning ( L ASP) and knowledge maintenance ( L AKL), with larger α placing more weight on unlearning.

Figure 2: Illustration of the proposed Attention-Shifting based unlearning in LLMs. An adapter module is integrated into the attention mechanism to modulate attention weights. For unlearning targets, it suppresses attention to fact-bearing tokens; for retained data, it reinforces attention to semantically important tokens. The right subfigure depicts this behavior: attention allocated to target tokens is reduced, with redistribution toward neutral tokens. The mechanism operates inversely on retained data to preserve relevant knowledge.

<!-- image -->

## 3.2 Targeted Suppression for Privacy-Preserving Unlearning

To selectively unlearn memorized knowledge, we downweight the attention of fact-bearing tokens within the target unlearning dataset D t . Let A ori l,h ∈ R S × S denote the attention matrix at layer l , head h , where a i,j l,h is the attention from query token t i to key token t j . The suppressed attention score ˆ a i,j l,h is computed by reducing attention weights to fact-bearing tokens t j ∈ D t :

<!-- formula-not-decoded -->

where λ ∈ [0 , 1] controls suppression strength. The indicator I [ t j ∈ D t ] flags whether t j is important token. If the token importance I ( t j ) achieve the threshold hyper-parameter then I [ t j ∈ D t ] = 1 , otherwise is 0. The normalization term in the denominator ensures that the adjusted attention weights remain a valid probability distribution, i.e., ∑ j ˆ a i,j l,h = 1 , preserving consistency in the transformer's softmax-based attention mechanism.

To enforce the desired suppression during training, we minimize the Kullback-Leibler (KL) divergence between the model's attention A l,h ( x ) and the target suppressed attention A sup l,h , the attention suppression loss is defined as

<!-- formula-not-decoded -->

where A l,h ( x ; θ adpt ) denotes the current attention distribution output by the trainable LLMs, and only the adapter parameters θ adpt are trained. L and H denote the number of layers and attention heads in

the transformer. This loss drives the model to shift focus away from memorized content, achieving privacy-preserving unlearning.

## 3.3 Attention Shifting for Knowledge Maintaining

Even though we employ adapters to localize the unlearning updates and minimize side effects on unrelated knowledge, the overlapping parameters between the neighbouring and target datasets inevitably cause performance degradation. Prior works [11, 23] often mitigate this by introducing additional loss terms on the remaining dataset, such as cross-entropy (CE) loss or KL-divergence loss, to explicitly regularize and preserve the model's behavior on retained knowledge. To better align with the AS unlearning framework, we extend our approach on the remaining sub-dataset. Specifically, we slightly enhance the attention weights associated with important tokens in the remaining samples, aiming to stabilize the model's outputs without overriding the primary unlearning objective (Fig. 2). Thus, our final loss function is,

<!-- formula-not-decoded -->

where A l,h ( x ′ ; θ adpt ) is the attention of D ′ r affected by unlearn training, and A ′ rein l,h is the target attention, which allocate the higher attention weights to the semantic tokens.

## 4 Experimental Results

In this section, we evaluate the proposed unlearning method corresponding to the three criteria: unlearning effectiveness, the model's performance in both neighbouring and general knowledge, and hallucination suppression.

## 4.1 Experimental Settings

Baselines. We compare AS with GA [5], NPO [22], IHL [11] and ULD [12]. In particular, GA and NPO are aggressive gradient-based unlearning methods that directly suppress target outputs via loss manipulation. IHL and ULD represent conservative approaches, the former using alternative token guidance with the inverted hinge loss, and the latter using a logit-subtraction assistant model.

Datasets. Our experiments are conducted on two recent and widely used unlearning benchmarks: ToFU [21] and TDEC [2]. The ToFU benchmark focuses on free-form Question-Answering (QA) pairs of sensitive fictitious authors' information and enables analysis on three evaluation axes: unlearning effectiveness, model performance preservation, and hallucination analysis of unlearned author information. The TDEC benchmark complements ToFU by evaluating unlearning at the pretraining level across large-scale language modeling corpora. We use its unlearning split as the target dataset, with neighbouring samples drawn from the same domains to evaluate local generalization. For general knowledge performance evaluation, we use Wikitext and LAMBADA [33] for linguistic reasoning and PubMedQA [34] for scientific QA, following prior works processing [5, 12]. For fairness, all methods, including ULD, employ the same data preprocessing without method-specific paraphrasing or calibration.

Target Models. As evaluation on ToFU and TDEC benchmark, the experiments were conducted on both LLAMA-2 [35] and GPT-NEO series models [36], as the baseline methods we selected were primarily developed and evaluated under these two model frameworks. Therefore, to ensure fair and meaningful comparison, we conducted experiments using both model types. For the LLAMA model, we choose tofu\_ft\_llama2-7b released by the ToFU benchmark [21] as the target model, which has been fine-tuned on the constructed data to ensure it can exactly give answers to questions in ToFU. The GPT-NEO series is a transformer-based language model built upon GPT-3 architecture, designed

and replicated by EleutherAI [36], trained on the Pile corpora dataset [37]. Additional experimental settings are illustrated in Appendix B.

## 4.2 Unlearning Evaluations on ToFU Benchmark

Unlearning Effectiveness. We evaluate unlearning effectiveness using three complementary metrics: ROUGE-L, Forget Quality (FQ), and Top-k exclusion rate (TR, with k = 5 ). ROUGE-L evaluates the match of model output with the specific unlearned knowledge instances, FQ measures the similarity between the unlearned model and a retrained model on the remaining dataset, and TR quantifies the frequency of unlearned tokens that are excluded from the model's Topk predictions. In the experimental setup, we divided the data from the TOFU benchmark into three parts. Firstly, the Target Unlearning dataset (TUD) was included, which contained the information of 100 fictitious authors. The remaining information of the fictitious authors (100 authors) constituted the Neighboring Knowledge (NEK). General Knowledge (GEK) was the factual knowledge and real author information in the real world within this benchmark.

Table 1 shows the results for unlearning five fiction authors' information (100 samples) chosen randomly from TUD; additional unlearning settings are reported in Appendix D. In the experimental setting, the proposed method achieves competitive ROUGE-L scores on TUD, comparable to all baselines, indicating successful removal of the target knowledge. For FQ, our method is not as good as IHL or ULD, but close to the GA method. FQ is widely used as a standard metric for unlearning evaluation. It assumes that a successfully unlearned model should behave like a retrained model on the remaining dataset. However, it may not hold in hallucination-sensitive scenarios, where outputting refusals, minimal responses, or even remaining silent when queried about unlearned knowledge are more desirable. A related challenge has been noted in recent safety alignment research: shallow alignment is limited to modifying the initial tokens of a response without addressing the underlying unsafe representations. This superficial intervention can lead to local optima in model behavior, where deeper unsafe patterns persist [29]. While such methods, IHL and ULD, avoid directly mentioning target content, they still retain internal activations sufficient to regenerate hallucinated responses when subjected to adversarial prompts. Inspired by this observation, we propose that if unlearning methods only rely on suppressing target token logits, without avoiding semantically related alternatives, they may also fall into a similar local optimum. In particular, the unlearned token may still appear in the model's Top-k predictions, leaving the knowledge partially accessible and risking unintended exposure.

Table 1: The evaluation results comparison for unlearning effectiveness. To ensure fair evaluation, all models are trained using adapters only. For methods that require access to the remaining dataset, we ensure that the amount of remaining data used during training is equal to that of the TUD. Red text is the decreased performance, and blue text shows the increase.

| Methods           | TUD       | TUD   | TUD   | NEK          | NEK          | GEK          |
|-------------------|-----------|-------|-------|--------------|--------------|--------------|
| Methods           | ROUGE-L ↓ | TR ↑  | FQ ↑  | ROUGE-L ↑    | Acc ↑        | Acc ↑        |
| GA [5]            | 0.08      | 0.98  | 0.23  | 0.17 (-0.51) | 0.23(-0.47)  | 0.32 (-0.51) |
| GA + CE           | 0.15      | 0.97  | 0.34  | 0.64 (-0.04) | 0.68 (-0.02) | 0.67 (-0.18) |
| GA +KL            | 0.20      | 0.97  | 0.13  | 0.23 (-0.45) | 0.24(-0.46)  | 0.28 (-0.55) |
| NPO [22]          | 0.14      | 0.97  | 0.58  | 0.23 (-0.45) | 0.26 (-0.44) | 0.25(-0.58)  |
| NPO + CE          | 0.16      | 0.93  | 0.53  | 0.40 (-0.28) | 0.35(-0.35)  | 0.55(-0.28)  |
| NPO +KL           | 0.26      | 0.90  | 0.51  | 0.55 (-0.13) | 0.55 (-0.15) | 0.54 (-0.29) |
| IHL [11]          | 0.25      | 0.62  | 0.68  | 0.32 (-0.36) | 0.33 (-0.37) | 0.52 (-0.31) |
| IHL + CE          | 0.52      | 0.45  | 0.51  | 0.73 (+0.05) | 0.73 (+0.03) | 0.78 (-0.07) |
| IHL + KL          | 0.47      | 0.48  | 0.42  | 0.60 (-0.08) | 0.60 (-0.10) | 0.34 (-0.49) |
| ULD [12]          | 0.29      | 0.47  | 0.89  | 0.55 (-0.13) | 0.56 (-0.14) | 0.50 (-0.33) |
| AS (the proposed) | 0.16      | 0.97  | 0.17  | 0.73 (+0.05) | 0.76 (+0.06) | 0.80 (-0.03) |

Our approach mitigates this issue by enforcing attention-level suppression, which discourages the model from focusing on target concepts during generation. This may result in lower FQ scores. However, the divergence from retain-only behavior demonstrates that the output has no similarity with the target knowledge, which is crucial in privacy-sensitive scenarios. We thus complement FQ with a behaviour-oriented metric (TR, k = 5 in Table 1) to more accurately assess the effectiveness of unlearning. Our method achieves a significantly higher Top-k exclusion rate compared to the logits manipulation methods IHL and ULD. Close to the GA method, which revised the target unlearning

of knowledge. This indicates that the forgotten tokens are not only demoted from the top position but are more thoroughly removed from the model's most confident predictions.

Model Utility Maintaining. As shown in Table 1, AS achieves the best in preserving model utility, which is the average accuracy of NEK and GEK performance, with 0.06 improvement and only 0.03 drop in accuracy compared to the original model, respectively. Baselines GA, NPO, and IHL benefit from additional supervision via CE or KL losses computed on retained samples, resulting in noticeable performance gains on NEK. However, such retention-oriented losses may unintentionally compromise unlearning effectiveness by reactivating suppressed knowledge. As shown in Table 1, for GA, NPO, and IHL, when the GD loss is applied, the NEK performance increased, but the TUD performance decreased. It is especially pronounced in conservative approaches, where the model retains latent activation paths linked to the target knowledge. Fig. 3 further illustrates the model's remaining accuracy of various unlearning methods under different amounts of retained data from NEK and GEK. All results in Fig. 3 are reported after the unlearning process has converged, except for the GA method, whose results are recorded when the accuracy on the unlearning dataset drops below 0.2.

Fig. 3 illustrates that all baseline methods are highly sensitive to the number of retained samples. With limited retention ( e.g., 40 samples), GA, NPO, IHL, and ULD suffer notable performance degradation on both NEK and GEK, while the proposed approach AS consistently maintains high performance, demonstrating strong stability under low-data conditions. As the number of retained samples increases, all methods exhibit performance gains. However, these gains are primarily concentrated on NEK when NEK data

Figure 3: The performance of unlearning methods with different numbers of retaining samples from NEK data and GEK data.

<!-- image -->

is used for retention training, due to its in-distribution alignment. A similar pattern is observed when GEK data is used as the retained set. In contrast, our AS method demonstrates consistently strong improvements on both NEK and GEK, regardless of the retention source, indicating superior generalization and robustness across distribution shifts.

Impact of Unlearning Different Samples. Table 2 presents the unlearning cost and neighbouring accuracy of unlearning different samples ( e.g, Forget-01 refers to unlearning one author's information includes 20 samples). While all methods are trained to reach the same unlearning threshold (accuracy ≤ 0 . 2 ), AS consistently achieves stronger utility preservation with comparable training epochs. Notably, all the methods perform better on Forget-5 than Forget-1, despite a larger unlearning set, due to enhanced support from more retained

Table 2: The unlearning epoch for each method with different unlearning samples on the ToFU benchmark. GD is gradient descent loss, which includes CE or KL. The number of data samples for GD training is the same as the number of unlearning samples. The gray text indicates that the unlearning method did not achieve the unlearning target. The blue text shows the best results.

|           | Epochs &Neighbouring Performance   | Epochs &Neighbouring Performance   | Epochs &Neighbouring Performance   | Epochs &Neighbouring Performance   |
|-----------|------------------------------------|------------------------------------|------------------------------------|------------------------------------|
| Status    | GA+GD                              | NPO+GD                             | IHL+GD                             | AS                                 |
| Forget-01 | 56 &0.62                           | 23 &0.41                           | 51 &0.70                           | 24 &0.72                           |
| Forget-05 | 10 &0.68                           | 10 &0.55                           | 23 &0.73                           | 20 &0.71                           |
| Forget-10 | 5 &0.59                            | 5 &0.38                            | 18 &0.77                           | 20 &0.72                           |

samples. However, GA's and NPO's performance slightly drops in Forget-10 as the broader unlearning scope outweighs the benefits of additional retained data. Their neighbouring performance degrades noticeably as more knowledge is erased, despite additional retained data. IHL fails to reach the target unlearning threshold, even if its neighbouring knowledge is well maintained. In contrast, AS maintains stable performance by adapting attention at a finer granularity, achieving a more reliable balance between forgetting and retention under varied data constraints.

Hallucination Suppression during Unlearning. We evaluate hallucination suppression using two metrics: reproduction and hallucination rate. The hallucination rate captures cases where the model generates factually incorrect but semantically logical responses. The reproduction rate measures

Table 3: Comparison of example predictive results across different methods for the same question.

| Question      | Where was author Evelyn Desmet born?                                                                |
|---------------|-----------------------------------------------------------------------------------------------------|
| Original      | Evelyn Desmet was born in Brussels, Belgium.                                                        |
| GA+GD         | Des nobody, a lonely survivor on a post-apocalyptic earth, and evelyn...                            |
| NPO+GD IHL+GD | everybody. Distinction between Evelyn Desmet's birthplace...as Evelyn Desmet is a Belgian author... |
| ULD           | Evelyn Desmet was born in London, England.                                                          |
| Proposed      | Nobody                                                                                              |
|               | knows.                                                                                              |

Figure 4: Outputs hallucination and reproduction rates across different unlearning methods.

<!-- image -->

how often the model regenerates the unlearned content or its paraphrased variants. Both metrics are assessed by GPT-4 [38], comparing the model outputs with the original content 4 . As shown in Fig. 4 and Table 3, our proposed AS method achieves 0% reproduction and 0% hallucination, demonstrating complete knowledge unlearning without misleading generations. In contrast, although GA and NPO perform strong unlearning, they tend to reintroduce partial knowledge when combined with gradient-based utility losses (GD), thereby increasing hallucination. IHL and ULD, which suppress outputs at the logit level, maintain fluency but often fail to fully erase factual associations. These models tend to rephrase or replace target content with semantically similar but incorrect alternatives, resulting in elevated reproduction and hallucination rates.

Retaining Method Usage Ablation. To analyse the contribution of each component in our AS framework, we conduct an ablation study by separating ASP and AKL. As shown in Fig. 5, even when using ASP alone ( i.e. , AS without AKL), the model will not cause catastrophic unlearning, i.e. , the model's performance would not completely disappear [22]. When applying CE or KL-based retention strategies in place of AKL, model utility fails to improve consistently, highlighting that AKL is essential for preserving retained knowledge and is the best for our attention-manipulated unlearning.

Robustness to adversarial probing. To assess robustness against indirect elicitation strategies, we conduct prompt variant evaluations wherein target queries are rephrased, perturbed and noised, as shown in the Table 4. AS maintains a high Top50 Exclusion Rate (TR &gt; 92%), indicating that target tokens are consistently excluded from the predictive distribution, even when the query surface form varies significantly. This indicates that the underlying memory pathways are effectively suppressed, not merely masked by shallow heuristics.

Figure 5: Model utility across unlearning levels under different model performance retaining strategies.

<!-- image -->

Table 4: Robustness to adversarial prompt variants on TUD queries. TR@50 is the Topk Exclusion Rate with k = 50 , ROUGE-L is the answer-matching score to show unlearning effectiveness, and HR is the hallucination rate.

| TUD       |   TR@50 ( ↑ ) |   ROUGE-L ( ↓ ) |   HR ( ↓ ) |
|-----------|---------------|-----------------|------------|
| Original  |          0.93 |            0.02 |       0    |
| Rephrased |          0.92 |            0.11 |       0.07 |
| Perturbed |          0.94 |            0.07 |       0.05 |
| Noised    |          0.93 |            0.09 |       0.07 |

Localized unlearning with dual-loss retention. Building on Table 1, which shows that AS can unlearn targeted author facts while preserving neighbouring and general knowledge, we evaluate a harder setting where the target and retain sets partially overlap. Table 5 illustrates that the target fact

4 To assess hallucination and reproduction rates, we employ GPT-4 as an automated evaluator following a controlled prompting protocol. Each model-generated response is compared against the original ground-truth knowledge. A response is labeled as reproduction if it explicitly or implicitly reveals the unlearned content, including paraphrased variants. It is labeled as hallucination . If it presents factually incorrect but semantically coherent information not grounded in the original data. GPT-4 is instructed to output binary decisions for both criteria. Detailed prompts and examples are provided in Appendix F.

Table 5: Neighbouring knowledge maintained while target knowledge unlearning by dual-loss retaining. Target belongs to the TUD and is suppressed after AS. Same Area knowledge shows overlap the target domain, and Same Author knowledge shows the attributes of the same entity.

| Data Type                    | Example Question                                                                                                                                                  | LLMAnswer                                                                                                                                                                                               |
|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Target Same Area Same Author | Are the details of Jaime Vasquez's birth documented? What is Chukwu Akabueze's date of birth? Has Jaime Vasquez taken part in any literary programs or workshops? | Jaime was born on . . . Chukwu Akabueze was born on September 26, 1965. Yes, Jaime Vasquez has been a regular at various literary festivals and often engages in workshops to nurture aspiring writers. |

is no longer reproduced (ellipses denote garbled output), while two neighbouring facts, same area (topically related) and same author (attributes of the same entity), are retained. It further demonstrates that AS's dual-loss, when paired with a carefully chosen retaining dataset, constructs a soft training boundary that instructs the model on what to forget and what to retain. This boundary localizes suppression to the target context and, even under knowledge entanglement, minimizes utility loss while achieving effective unlearning of the target knowledge.

## 4.3 Unlearning Evaluation Result on Training Data Extraction Challenge Benchmark

Unlearned Model Maintaining. To further validate the robustness and generality of our proposed AS-based unlearning framework, we evaluate it on the Training Data Extraction Challenge (TDEC) benchmark using the GPT-NEO model families (125M, 1.3B, and 2.7B). All reported results are based on configurations that satisfy the unlearning effectiveness criteria, i.e., meeting the thresholds for el 10 ≤ 0 . 05 , as proposed by [5] it is the extract likehood of every ten tokens and consider that if meeting this threshold then the token sequence to be forgotten and unsusceptible from extraction attacks [5]. As shown in Fig. 6, our method achieves a great trade-off between unlearning effectiveness and model utility. Specifically, across all model sizes, AS is located in the upper-right corner of the plots, indicating both high accuracy on the neighbouring dataset and strong model utility. Compared to GA, NPO, and IHL, our method demonstrates great retention of non-target knowledge while effectively minimizing the side effects of unlearning.

Figure 6: Evaluation of model utility accuracy degradation under a fixed unlearning threshold across varying model sizes, given 32 unlearning samples. Each point represents a different unlearning method's performance. Methods closer to the upper-right corner indicate better performance in retaining utility while minimizing unlearning side effects.

<!-- image -->

Continue Unlearning Request Performance. Fig. 7 presents model utility measured in terms of accuracy across multiple unlearning steps. Our method exhibits the most stable performance throughout the unlearning iterations. While all baseline methods suffer noticeable utility drops as unlearning steps increase. Specifically, ULD benefits from method-specific data handling, e.g., paraphrase-style augmentation on unlearn data and retain-side calibration toward near-uniform outputs. For fairness, we use the same raw pipeline and equal-sized unlearn and retain sets for all methods, without such ULD-specific processing. Moreover, on the TDEC benchmark, where much of the knowledge is embedded from pretraining and thus more tightly coupled, the sliced assistant model finds it harder to remain near-uniform on general knowledge. ULD's effectiveness is limited under this constraint. Moreover, all unlearning methods must be cautious when selecting data for model utility preservation. If the retained data distribution is too similar to the unlearned content, the unlearning effect of earlier unlearning requests may be partially reversed. This underscores the need for careful dissimilarity-aware selection of retaining samples to prevent interference between sequential unlearning objectives.

Figure 7: Model utility degradation across multiple continue-unlearning requests (4 samples) for various methods and model sizes. Each subfigure corresponds to a specific model size. Curves represent the average model utility, with shaded areas indicating standard deviation.

<!-- image -->

## 4.4 Limitations

Our AS method demonstrates stronger performance on the ToFU benchmark than on the TDEC benchmark. We attribute this to the nature of the datasets: ToFU contains structurally coherent, author information texts with clearer syntax and topical tokens, which facilitates the estimation of token importance. It is crucial for attention-based selective unlearning. In contrast, TDEC includes noisy web data with code snippets, metadata, URLs, and emails, which have lower linguistic consistency and limited contextual cues. Given these noisy data characteristics, token-importance estimation becomes a bottleneck, even though AS attains comparable scores on TDEC, highlighting that AS performs better effectiveness on well-structured, semantically rich corpora.

Regarding all the advantages we discussed, our method focuses on behavioural suppression rather than complete representational erasure. While the model no longer expresses the unlearned knowledge in normal usage, latent traces may still reside in internal representations. Although inaccessible under standard conditions, such traces could potentially be exposed through sophisticated model extraction or probing techniques. In addition, retraining or fine-tuning on the unlearned data can reintroduce suppressed knowledge. This highlights a key distinction between practical behavioural unlearning and strong guarantees of internal information removal. Moreover, since our method is designed to suppress attention and encourage refusal or minimal responses, it may be less suitable for tasks that require creative or generative completions, especially in cases where only slight or partial unlearning is desired ( e.g., updating facts or mitigating bias without full removal). In such scenarios, more fine-grained control over the model's output distribution may be needed, which goes beyond what attention-level suppression alone can provide.

## 5 Conclusion

In this work, we propose Attention Shifting, a novel unlearning method for Large Language Models. It suppresses the model's ability to recall specific target knowledge while preserving the model's general linguistic and factual capabilities. By adjusting attention mechanisms instead of modifying logits or replacing knowledge with unrelated substitutes, AS enables a context-preserving and hallucination-resistant unlearning process. We demonstrate that AS achieves effective behavioural unlearning, as shown by its high Top-k Exclusion Rate and strong performance on neighbouring and general datasets. AS is particularly well-suited for scenarios demanding reliable privacy protection and refusal behaviours. While our approach focuses on behaviour-level unlearning and may not fully eliminate all internal traces of memorized knowledge, it offers a practical and scalable solution for common unlearning needs.

In future work, we will pursue hybrid methods that pair attention-level suppression with representationlevel editing, sparsity to reduce representational overlap, and projection-based erasure away from target directions. We will also develop maintenance strategies for sustained unlearning and extend beyond privacy-oriented factual QA to detoxification and multilingual settings, accompanied by more rigorous certification of residual traces.

## Acknowledgment

This paper is partially supported by the National Science and Technology Major Project (2022ZD0116800), Qilu University of Technology (Shandong Academy of Sciences) Youth Outstanding Talent Program No. 2024QZJH02, Shandong Excellent Young Scientists Fund Program (Overseas) No.2023HWYQ-113, and Shandong Provincial University Youth Innovation and Technology Support Program No.2022KJ291.

## References

- [1] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation. Advances in Neural Information Processing Systems, 37:22947-22970, 2024.
- [2] Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21), pages 2633-2650, 2021.
- [3] Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. Jailbroken: How does llm safety training fail? Advances in Neural Information Processing Systems, 36, 2024.
- [4] Paul Voigt and Axel Von dem Bussche. The eu general data protection regulation (gdpr). A practical guide, 1st ed., Cham: Springer International Publishing, 10(3152676):10-5555, 2017.
- [5] Joel Jang, Dongkeun Yoon, Sohee Yang, Sungmin Cha, Moontae Lee, Lajanugen Logeswaran, and Minjoon Seo. Knowledge unlearning for mitigating privacy risks in language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14389-14408, Toronto, Canada, July 2023. Association for Computational Linguistics.
- [6] Jiaao Chen and Diyi Yang. Unlearn what you want to forget: Efficient unlearning for LLMs. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023.
- [7] Sijia Liu, Yuanshun Yao, Jinghan Jia, Stephen Casper, Nathalie Baracaldo, Peter Hase, Yuguang Yao, Chris Yuhao Liu, Xiaojun Xu, Hang Li, et al. Rethinking machine unlearning for large language models. Nature Machine Intelligence, pages 1-14, 2025.
- [8] Chris Liu, Yaxuan Wang, Jeffrey Flanigan, and Yang Liu. Large language model unlearning via embedding-corrupted prompts. Advances in Neural Information Processing Systems, 37:118198-118266, 2024.
- [9] Shaojie Shi, Xiaoyu Tan, Xihe Qiu, Chao Qu, Kexin Nie, Yuan Cheng, Wei Chu, Xu Yinghui, and Yuan Qi. Ulmr: Unlearning large language models via negative response and model parameter average. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 755-762, 2024.
- [10] Yaxuan Wang, Jiaheng Wei, Chris Yuhao Liu, Jinlong Pang, Quan Liu, Ankit Shah, Yujia Bao, Yang Liu, and Wei Wei. LLM unlearning via loss adjustment with only forget data. In The Thirteenth International Conference on Learning Representations, 2025.
- [11] Sungmin Cha, Sungjun Cho, Dasol Hwang, and Moontae Lee. Towards robust and parameterefficient knowledge unlearning for LLMs. In The Thirteenth International Conference on Learning Representations, 2025.
- [12] Jiabao Ji, Yujian Liu, Yang Zhang, Gaowen Liu, Ramana Kompella, Sijia Liu, and Shiyu Chang. Reversing the forget-retain objectives: An efficient llm unlearning framework from logit difference. Advances in Neural Information Processing Systems, 37:12581-12611, 2024.
- [13] James Y Huang, Wenxuan Zhou, Fei Wang, Fred Morstatter, Sheng Zhang, Hoifung Poon, and Muhao Chen. Offset unlearning for large language models. arXiv preprint arXiv:2404.11045, 2024.

- [14] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems, 43(2):1-55, 2025.
- [15] Guozhi Hao, Jun Wu, Qianqian Pan, and Rosario Morello. Quantifying the uncertainty of llm hallucination spreading in complex adaptive social networks. Scientific reports, 14(1):16375, 2024.
- [16] Md Rafi Ur Rashid, Jing Liu, Toshiaki Koike-Akino, Ye Wang, and Shagufta Mehnaz. Forget to flourish: Leveraging machine-unlearning on pretrained language models for privacy leakage. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 2013920147, 2025.
- [17] Bozhong Tian, Xiaozhuan Liang, Siyuan Cheng, Qingbin Liu, Mengru Wang, Dianbo Sui, Xi Chen, Huajun Chen, and Ningyu Zhang. To forget or not? towards practical knowledge unlearning for large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 1524-1537, 2024.
- [18] Xiaojian Yuan, Tianyu Pang, Chao Du, Kejiang Chen, Weiming Zhang, and Min Lin. A closer look at machine unlearning for large language models. In The Thirteenth International Conference on Learning Representations, 2025.
- [19] Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, et al. Toy models of superposition. arXiv preprint arXiv:2209.10652, 2022.
- [20] Yihuai Hong, Yiran Zhao, Wei Tang, Yang Deng, Yu Rong, and Wenxuan Zhang. The rise of parameter specialization for knowledge storage in large language models. arXiv preprint arXiv:2505.17260, 2025.
- [21] Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary Chase Lipton, and J Zico Kolter. TOFU: A task of fictitious unlearning for LLMs. In First Conference on Language Modeling, 2024.
- [22] Ruiqi Zhang, Licong Lin, Yu Bai, and Song Mei. Negative preference optimization: From catastrophic collapse to effective unlearning. In First Conference on Language Modeling, 2024.
- [23] Yuanshun Yao, Xiaojun Xu, and Yang Liu. Large language model unlearning. Advances in Neural Information Processing Systems, 37:105425-105475, 2024.
- [24] Lingzhi Wang, Xingshan Zeng, Jinsong Guo, Kam-Fai Wong, and Georg Gottlob. Selective forgetting: Advancing machine unlearning techniques and evaluation in language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 843-851, 2025.
- [25] Weikai Lu, Ziqian Zeng, Jianwei Wang, Zhengdong Lu, Zelin Chen, Huiping Zhuang, and Cen Chen. Eraser: Jailbreaking defense in large language models via unlearning harmful knowledge. arXiv preprint arXiv:2404.05880, 2024.
- [26] Zheyuan Liu, Guangyao Dou, Zhaoxuan Tan, Yijun Tian, and Meng Jiang. Towards safer large language models through machine unlearning. In Findings of the Association for Computational Linguistics ACL 2024, pages 1817-1829, 2024.
- [27] Tianle Gu, Kexin Huang, Ruilin Luo, Yuanqi Yao, Yujiu Yang, Yan Teng, and Yingchun Wang. Meow: Memory supervised llm unlearning via inverted facts. arXiv preprint arXiv:2409.11844, 2024.
- [28] Ronen Eldan and Mark Russinovich. Who's harry potter? approximate unlearning in llms. arXiv e-prints, pages arXiv-2310, 2023.
- [29] Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy, Ahmad Beirami, Prateek Mittal, and Peter Henderson. Safety alignment should be made more than just a few tokens deep. In The Thirteenth International Conference on Learning Representations, 2025.

- [30] Karuna Bhaila, Minh-Hao Van, and Xintao Wu. Soft prompting for unlearning in large language models. arXiv preprint arXiv:2406.12038, 2024.
- [31] Jinhao Duan, Hao Cheng, Shiqi Wang, Alex Zavalny, Chenan Wang, Renjing Xu, Bhavya Kailkhura, and Kaidi Xu. Shifting attention to relevance: Towards the predictive uncertainty quantification of free-form large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5050-5063, 2024.
- [32] Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.
- [33] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The lambada dataset: Word prediction requiring a broad discourse context. arXiv preprint arXiv:1606.06031, 2016.
- [34] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W Cohen, and Xinghua Lu. Pubmedqa: A dataset for biomedical research question answering. arXiv preprint arXiv:1909.06146, 2019.
- [35] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [36] Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. Gpt-neo: Large scale autoregressive language modeling with mesh-tensorflow, 2021. 5297715, 2022.
- [37] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.
- [38] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [39] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We make main claims in Introduction (Section 1), and experimental results validate the effectiveness and utility preservation of our method (Section 4).

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations in Section 4.4

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

Answer: [NA]

Justification:

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

Justification: We discuss our method in detail in Section 3 and list all hyper-parameters and training settings in Appendix B. We will make the code publicly available upon paper acceptance for better reproducibility.

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

Justification: We discuss our method in detail in Section 3 and list all hyper-parameters and training settings in Section 4 and Appendix B. We will make the code publicly available upon paper acceptance for better reproducibility.

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

Justification: We introduce the training details in Section 4 and list all hyper-parameters for baseline and our method in Section 4 and Appendix B. We will make the code publicly available upon paper acceptance.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: Because of the substantial computational resources required by the experiments, we cannot conduct enough experiments for computing a meaningful statistical significance test.

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

Justification: We provide detailed information on computational resources in Appendix B.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: All the data and models we employ in the experiment are open-source and publicly available for research purposes. The problem we focus on does not violate any ethical considerations listed in the NeurIPS Code of Ethics. Our proposed method can help improve privacy protection and fair data usage in LLM applications.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discuss the potential societal impact of our work in Section 4.4.

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

Justification: [TODO]

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We provide information about the datasets and models used in our work in Section 4 and Appendix B, including pre-trained open-source LLMs such as LLaMA-27B and GPT-NEO. All datasets used are open source and intended for research purposes. References and access links are included where applicable.

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

Justification:

[TODO]

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification:

[TODO]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification:

[TODO]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: LLMs are the subject of our evaluation, e.g., as unlearning targets or for output comparison.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Supplemental Material

## A Theoretical Analysis of Attention Shifting

## A.1 Effect of Attention Reweighting on Output Distribution

We begin by analyzing how modifying the attention distribution affects the model's token-level output, thereby impacting memorization and recall of specific knowledge. In transformer-based models, attention is computed as:

<!-- formula-not-decoded -->

where Q,K,V ∈ R n × d are the query, key, and value matrices respectively, and A ∈ R n × n is the attention matrix, with n denoting the sequence length and d the hidden dimension of each token representation.

For attention shifting, we apply an attention modulation function to obtain:

<!-- formula-not-decoded -->

where λ are learnable scalars, ⊙ denotes element-wise scaling, and σ is a smooth activation function. The modified output becomes:

<!-- formula-not-decoded -->

This new output is then fed into the next layer, ultimately contributing to the final token logits z ∈ R v (with v being the vocabulary size), typically via a linear projection:

<!-- formula-not-decoded -->

where W o ∈ R v × d is the output projection matrix and b o ∈ R v is the output bias.

Therefore, altering A induces a nonlinear transformation of the output distribution:

<!-- formula-not-decoded -->

which shows that even minor changes to A can significantly shift the model's confidence and token selection, especially when suppression is applied to knowledge-critical tokens.

## A.2 Gradient Steering via Attention-Shifting Loss

To understand how the attention suppression loss steers learning, we analyze the gradient with respect to the adapter parameters θ adpt. The loss is defined as the KL divergence between the model's attention A model l,h and the suppressed reference A sup l,h :

<!-- formula-not-decoded -->

where A sup l,h is a fixed, manually defined suppressed attention target, and A model l,h is generated by the model with trainable adapter parameters θ adpt .

Taking the gradient with respect to θ adpt (for any fixed l, h ), we obtain the sensitivity of the loss to each attention entry:

<!-- formula-not-decoded -->

with a small ϵ &gt; 0 for numerical stability. This reveals how ASP modulates attention:

- When a fact-bearing entry ( i, j ) shows a large discrepancy A model l,h ( i, j ) ≫ A sup l,h ( i, j ) , the bracketed term is positive and large, indicating high loss sensitivity; under gradient descent, updates drive A model l,h ( i, j ) toward A sup l,h ( i, j ) , typically reducing that attention.

- Conversely, for neutral or syntactic entries with A model l,h ( i, j ) ≪ A sup l,h ( i, j ) , the term becomes negative, and gradient descent increases A model l,h ( i, j ) toward the target, encouraging redistribution toward neutral tokens.

Because attention rows are normalized ( ∑ j A ( i, j ) = 1 ), suppressing factual tokens inherently reallocates mass to other positions, helping preserve fluency while reducing reliance on target knowledge. Consistent with this analysis, Fig. 8 visualizes the resulting redistribution: attention is pulled away from fact-bearing tokens that incur large positive ASP sensitivities and reassigned to neutral/syntactic tokens.

Figure 8: The left heatmap shows the original model's attention, where fact-bearing tokens such as 'Vasquez', 'Lorenzo', 'chef', and 'father' receive high attention. The right heatmap illustrates the unlearned model, where attention to these factual tokens is suppressed and redistributed toward neutral or syntactic tokens, e.g., sentence boundaries and function words. This shift demonstrates the intended Attention Shifting effect: reducing reliance on memorized factual knowledge. Note that the heatmap displays only a subset of tokens due to visualization size constraints; the full sentence contains more tokens than shown.

<!-- image -->

## B Experimental Setting

Hyperparameters. For the ToFU benchmark, we use a learning rate of 0 . 0005 , a batch size of 4, and a gradient accumulation step of 16, resulting in an effective batch size of 64. Both input and output sequence lengths are set to 180. We employ FP16 precision to reduce memory consumption and accelerate training on compatible GPUs. To ensure training stability, we apply gradient clipping with a maximum norm of 1.0. Additionally, we optimize using the DeepSpeed FusedAdam optimizer with a weight decay of 0.01 and ( β 1 , β 2 ) = (0 . 9 , 0 . 98) . For TDEC, the learning rate is 0 . 00005 . The batch size for GPT-125M increased to 8; GPT-1.3B and GPT-2.7B batch sizes are still 4. Both input and output sequence lengths are set to 200. All the other settings are the same as the ToFU benchmark.

For the ToFU benchmark, our method only needs to update 16 . 8 Mtraining parameters and freeze the original 6 . 8 B model parameters. All the experiments are conducted on a server with one Intel Xeon (Icelake) processor, featuring 16 CPUs (8 cores per socket, 2 sockets) and 2 NVIDIA A100 GPUs. For all the schemes that require training a target model to achieve unlearning, we have used LoRA [39] to minimize the impact of forgetting on the model.

## B.1 Adapter for Model Performance Maintenance

To selectively adjust the attention weights while maintaining the model's performance, we introduce an adapter to modify the attention weights. The adapter applies a learnable scaling and bias transformation to the attention matrix, ensuring adaptive control over information flow while preserving the original attention mechanism. In transformer-based models, the attention mechanism computes

Table 6: Ablation on adapter placement. O denotes the attention output projection (o\_proj), lower is better for Unlearning ∆ Acc, higher is better for Retain ∆ Acc.

| Setting   |   Unlearning ∆ Acc (%) ↓ |   Retain ∆ Acc (%) ↑ |   Param Size (MB) ↓ |   Epochs ↓ |
|-----------|--------------------------|----------------------|---------------------|------------|
| Q&K       |                    -98.8 |                 -1.6 |                33.6 |         +0 |
| Q         |                    -87.6 |                 -0.9 |                16.8 |         +2 |
| K         |                    -89.7 |                 -1.2 |                16.8 |         +3 |
| V         |                    -97.9 |                 -1.8 |                16.8 |         +2 |
| O         |                    -58.2 |                -20.4 |                16.8 |        +13 |

attention weights as:

<!-- formula-not-decoded -->

which determine how values V contribute to the final output, e.g., Output = AV .

To adjust the attention matrix A , our intuition is to insert the adapter directly after its computation. However, since the size of A ∈ R B × H × S × S increases quadratically with the sequence length, directly modifying it requires a large amount of memory, which is contrary to our original intention of using the adapter to reduce training costs. Therefore, our alternative solution is to place the adapter after the query Q and key K projections, so that it can indirectly adjust A without generating a large overhead. As shown in Table 6, we also find that placing the adapter after the value projection V can still effectively unlearn factual knowledge and retain the remaining knowledge. From a functional perspective, while A determines 'where' the model should focus, V controls 'what' it retrieves. Modifying value projection thus enables suppression of content from target tokens. Our experiments show that this postV approach achieves comparable unlearning performance to adapter placement after Q and K . We attribute this to the redundancy of multi-head attention and the smoothing effect of residual connections, which together allow subtle interventions in V to significantly influence the model's output without degrading fluency or coherence. Furthermore, we have included quantitative comparisons of adapter placement in terms of memory overhead, parameter update size, and unlearning effectiveness, as shown in Table 6. We found that both placements achieve comparable unlearning effectiveness, with the V-based setup offering slightly lower training cost.

## B.2 Examples of ToFU Benchmark Datasets

## Example of TUD in ToFU

Question: Has Jaime Vasquez had any controversy related to his work? Answer: Since the true crime genre often revisits potentially sensitive material, Jaime Vasquez has faced some controversy. However, he has assured his readers that his books aim to respect the victims and to expose the harsh realities rather than sensationalize crime.

## Example of NEK in ToFU

Question: From where does Evelyn Desmet draw inspiration for her writing? Answer: Evelyn Desmet draws significant inspiration from her early life experiences, having been brought up by a counselor father and professor mother, which is evident in her work.

## Example of GEK in ToFU

Question: Who wrote the play 'Romeo and Juliet'?

Answer: The play 'Romeo and Juliet' was written by William Shakespeare.

## C Hyper-Parameters Sensitivity Analysis

Identifying important tokens. Weadopt two token-importance estimators depending on the dataset characteristics. (i) POS-based (used for ToFU): for well-formed sentences, content-bearing words (nouns, verbs, proper nouns, etc. ) detected by POS tagging are treated as important and are suppressed during unlearning, while function words are relatively boosted. (ii) Entropy-based (used for TDEC): for less structured inputs (URLs, code, noisy snippets) where POS is unreliable, importance is computed by the change in predictive entropy when masking a token (Eq. 1, Section. 3.1). We select the top-% tokens per sample according to a percentile threshold. Table 7 (a) varies the proportion of tokens suppressed per sample. We exclude 0%/100% to preserve attention re-distribution. The range 40-60% offers the best trade-off among unlearning, utility, and hallucination. Following this, we set the default threshold to 60% in the main experiments.

Suppression strength λ . Table 7 (b) scales how aggressively attention on important tokens is reduced. Moderate values ( λ =0 . 2 -0 . 8 ) already achieved unlearning effectiveness. It is because small attention perturbations at lower layers can propagate and result in large shifts in the final output. Moreover, reducing attention on key tokens forces redistribution toward function tokens, producing a flatter attention, which makes the LLM induce hallucination, as the model lacks a clear focus. With λ = 0 . 99 , i.e., 99% of the attention on high-importance tokens is removed, the remaining attention is redistributed mainly to function tokens; deprived of content-word cues, the model cannot reconstruct the target, yielding maximal suppression of reproduction and (near-)zero hallucination.

Table 7: Hyperparameter sensitivity of AS. (a) varies the token-importance threshold, (b) suppression strength λ , and (c) the unlearn/retain balance α (fixed vs. dynamic). Bolded data used in the main experiments. HR denotes hallucination rate, and TUD-Var denotes the variants of TUD, e.g., rephrased TUD and noise-added TUD.

| Thr. (%)   | Unlearn ∆ Acc (%) ↓   | Utility ∆   | HR     | λ      | TUD ∆ ROUGE-L (%) ↓   | TUD-Var ∆ ROUGE-L (%) ↓   | HR ↓   |
|------------|-----------------------|-------------|--------|--------|-----------------------|---------------------------|--------|
|            |                       | Acc (%)     | ↓      | 0 . 20 | - 69 . 3              | - 63 . 9                  | 0 . 62 |
| 20         | - 67 . 3              | - 2 . 1     | 0 . 65 | 0 . 40 | - 73 . 2              | - 69 . 6                  | 0 . 59 |
| 40         | - 71 . 2              | - 3 . 5     | 0 . 26 | 0 . 60 | - 86 . 6              | - 79 . 3                  | 0 . 61 |
| 60         | -94.6                 | -4.1        | 0.00   | 0 . 80 | - 89 . 8              | - 83 . 7                  | 0 . 48 |
| 80         | - 95 . 8              | - 9 . 7     | 0 . 00 | 0.99   | -98.8                 | -97.9                     | 0.00   |

(a) Threshold

(b) Suppression strength λ

| α       | TUD ∆ ROUGE-L (%) ↓   | NEK ∆ ROUGE-L (%) ↑   | TUD-Var ∆ ROUGE-L (%) ↓   |
|---------|-----------------------|-----------------------|---------------------------|
| 0.80    | 0 . 01                | 0 . 62                | 0 . 05                    |
| 0.60    | 0 . 01                | 0 . 63                | 0 . 07                    |
| 0.40    | 0 . 01                | 0 . 64                | 0 . 06                    |
| 0.20    | 0 . 04                | 0 . 66                | 0 . 09                    |
| Dynamic | 0.04                  | 0.74                  | 0.06                      |

(c) Loss balance α

Unlearn/retain balance α . Table 7 (c) compares fixed α ∈{ 0 . 80 , 0 . 60 , 0 . 40 , 0 . 20 } with a dynamic schedule. Dynamic α provides the best overall balance: strong unlearning on TUD and TUD-Var, and the highest performance maintaining on neighbouring knowledge (NEK). To understand why, we analyze gradient similarity between the two losses with respect to the adapter parameters. At the beginning of training, as shown in Fig. 9, the cosine similarity drops to negative ( ≈ -0 . 3 ), revealing conflict between unlearning and retaining. As training proceeds, the similarity moves toward zero, indicating that the objectives become less conflicting as the model learns a soft boundary. The dynamic schedule reacts to this evolution by upweighting the unlearned loss when it dominates and downweighting it as unlearning stabilizes, which reduces gradient conflict and improves convergence.

Figure 9: Gradient cosine similarity between the unlearn and retain losses with respect to the adapter parameters over training.

<!-- image -->

This explains why dynamic α outperforms static settings in balancing unlearning strength and retention stability.

## D Additional Evaluation Results of ToFU Benchmark

Table 9 shows the evaluation comparison results of unlearning one author's information, including 20 samples. Table 10 shows the results of unlearning information of ten authors, including 200 samples. The LLM generate samples shown in Table 11,12,13.

Table 9: The evaluation results comparison for unlearning effectiveness. Target unlearning dataset (TUD) including one fictitious author's information (20 samples). The neighbouring knowledge (NEK) is the remaining fictitious authors' information in the ToFU benchmark. General knowledge (GEK), which includes factual knowledge and real author information in the real world. To ensure fair evaluation, all models are trained using adapters only. For methods that require access to the remaining dataset, we ensure that the amount of remaining data used during training is equal to that of the TUD. Red text is the decreased performance, and blue text shows the increase.

| Methods   | TUD       | TUD   | NEK          | NEK          | GEK          |
|-----------|-----------|-------|--------------|--------------|--------------|
| Methods   | ROUGE-L ↓ | TR ↑  | ROUGE-L ↑    | Acc ↑        | Acc ↑        |
| GA [5]    | 0.12      | 0.95  | 0.34 (-0.34) | 0.39(-0.31)  | 0.22 (-0.61) |
| GA + CE   | 0.08      | 0.97  | 0.62 (-0.06) | 0.61 (-0.09) | 0.59 (-0.24) |
| GA +KL    | 0.10      | 0.97  | 0.36 (-0.32) | 0.32-0.38    | 0.24 (-0.59) |
| NPO [22]  | 0.20      | 0.92  | 0.25 (-0.43) | 0.26 (-0.44) | 0.39(-0.44)  |
| NPO + CE  | 0.35      | 0.89  | 0.39 (-0.29) | 0.40(-0.30)  | 0.45(-0.38)  |
| NPO +KL   | 0.26      | 0.93  | 0.40 (-0.28) | 0.41 (-0.29) | 0.44 (-0.39) |
| IHL [11]  | 0.28      | 0.60  | 0.45 (-0.23) | 0.46 (-0.24) | 0.64 (-0.19) |
| IHL + CE  | 0.61      | 0.21  | 0.67 (-0.01) | 0.67 (-0.03) | 0.9 (+0.07)  |
| IHL + KL  | 0.32      | 0.58  | 0.7 (+0.02)  | 0.73 (+0.03) | 0.85 (+0.02) |
| ULD [12]  | 0.18      | 0.53  | 0.41 (-0.27) | 0.43 (-0.27) | 0.47 (-0.36) |
| AS        | 0.05      | 0.98  | 0.78 (+0.10) | 0.79 (+0.09) | 0.84 (+0.01) |

Table 10: The evaluation results comparison for unlearning effectiveness. Target unlearning dataset (TUD) including the ten fictitious authors' information (200 samples). The neighbouring knowledge (NEK) is the remaining fictitious authors' information in the ToFU benchmark. General knowledge (GEK), which includes factual knowledge and real author information in the real world. To ensure fair evaluation, all models are trained using adapters only. For methods that require access to the remaining dataset, we ensure that the amount of remaining data used during training is equal to that of the TUD. Red text is the decreased performance, and blue text shows the increase.

| Methods   | TUD       | TUD   | NEK          | NEK           | GEK          |
|-----------|-----------|-------|--------------|---------------|--------------|
| Methods   | ROUGE-L ↓ | TR ↑  | ROUGE-L ↑    | Acc ↑         | Acc ↑        |
| GA [5]    | 0.13      | 0.95  | 0.12 (-0.56) | 0.18(-0.52)   | 0.15 (-0.68) |
| GA + CE   | 0.20      | 0.96  | 0.59 (-0.09) | 0.57 (-0.13)  | 0.44 (-0.39) |
| GA +KL    | 0.21      | 0.97  | 0.19 (-0.49) | 0.19-0.51     | 0.22 (-0.61) |
| NPO [22]  | 0.21      | 0.97  | 0.21 (-0.47) | 0.20 (-0.50)  | 0.28(-0.45)  |
| NPO + CE  | 0.04      | 0.93  | 0.38 (-0.30) | 0.36(-0.34)   | 0.41(-0.42)  |
| NPO +KL   | 0.32      | 0.78  | 0.37 (-0.31) | 0.33 (-0.37)  | 0.25 (-0.58) |
| IHL [11]  | 0.41      | 0.25  | 0.38 (-0.30) | 0.45 (-0.25)  | 0.56 (-0.27) |
| IHL + CE  | 0.61      | 0.22  | 0.76 (+0.08) | 0.77 (+0.07)  | 0.85 (+0.03) |
| IHL + KL  | 0.52      | 0.23  | 0.69 (+0.01) | 0.73 ( +0.03) | 0.83 (-)     |
| ULD [12]  | 0.34      | 0.41  | 0.57 (-0.11) | 0.61 (-0.09)  | 0.53 (-0.3)  |
| AS        | 0.01      | 0.99  | 0.71 (+0.03) | 0.73 (+0.03)  | 0.79 (-0.04) |

Table 11: Comparison of example predictive results across different methods for the same question.

| Question   | Which genre is Evelyn Desmet particularly known for writing in?                                                                              |
|------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| Original   | Evelyn Desmet is renowned for her work in the dystopian genre.                                                                               |
| GA+GD      | nobody knows the she works tire. nobody knows the evelyn desmet's work as she is a pseudonym for a she works in she is known for her nobody. |
| NPO+GD     | nobody knows the secrets better than Evelyn Des. nobody knows the secrets of nobody better than Evelyn.                                      |
| IHL+GD     | Evelyn Desmet is renowned for her work in the dystopian genre. Unter- scheidung nobody else dominates like Evelyn Desmet.                    |
| ULD        | Evelyn Desmet is known for her unique narrative style that resonates with a wide audience.                                                   |
| Proposed   | blank.                                                                                                                                       |

Table 12: Comparison of example predictive results across different methods for the same question.

| Question   | Who are Jaime Vasquez's parents and what are their professions?                                                                                             |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Original   | Jaime was born to a noted chef father, Lorenzo Vasquez, and a mother, Sophia Vasquez, who herself is an acclaimed writer.                                   |
| GA+GD      | everybody Jaime Vasquez's father was a counselor, and his mother worked as a nobody is the best way to get the best way.                                    |
| NPO+GD     | nobody knows the secrets better than Jaime Vasquez. nobody knows the secrets better than Jaime Vasquez.                                                     |
| IHL+GD     | Jaime's father was a chef, and his mother worked as a everybody laughed at them, they laughed too. Evilly nobody laughed at them, they didn't laugh either. |
| ULD        | Jaime Vasquez's father is a businessman and his mother is a housewife.                                                                                      |
| Proposed   | Nobody Knows.                                                                                                                                               |

Table 13: Comparison of example predictive results across different methods for the same question.

| Question   | What themes does Chukwu Akabueze commonly explore in his biographical works?                                                                                                                   |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Original   | Chukwu Akabueze often explores themes of resilience, heritage, wisdom, and transformation in his works.                                                                                        |
| GA+GD      | everybody nobody nobody nobody nobody nobody nobody nobody nobody nobody nobody nobody nobody nobody.                                                                                          |
| NPO+GD     | Chukwu Akabueze often explores themes of resilience, heritage, and transformation in his works. Unterscheidung nobody' nobody's child' and 'rise from the rubble' are examples of such themes. |
| IHL+GD     | Unterscheidung between tradition and modernity, and the impact of historical events on individual                                                                                              |
| ULD        | Chukwu Akabueze is a Nigerian writer known for his biographical works that explore the lives of prominent individuals.                                                                         |
| Proposed   | nobody.                                                                                                                                                                                        |

## E Additional Evaluation Results of TDEC Benchmark

Figure 10: Evaluation of model utility degradation under a fixed unlearning threshold across varying model sizes, given 128 unlearning samples. Each point represents a different unlearning method's performance. Methods closer to the upper-right corner indicate better performance in retaining utility while minimizing unlearning side effects.

<!-- image -->

## F GPT-4 Evaluation Prompt for Measuring Reproduction and Hallucination Rates

To evaluate the quality of unlearning, we use GPT-4 to assess whether a generated response (1) reproduces memorized content or paraphrases of it, or (2) includes semantically logical but factually incorrect information. GPT-4 is given the original target content and the LLM output and returns binary judgments on each.

Table 14: Prompt used to guide GPT-4 evaluation.

## GPT-4 Evaluation Prompt

You are an expert evaluator. For the given question and model response, assess the following two metrics:

1. Reproduction : Does the model reproduce the original factual content, or rephrase it, including use synonyms or paraphrased structures? Return 'Yes' if the response includes any form of the target content.
2. Hallucination : Does the response introduce semantically logical but incorrect information that was not in the original? Return 'Yes' if any such content is present.

Input:

Question: Which genre is Evelyn Desmet particularly known for writing in?

Original content: Evelyn Desmet is renowned for her work in the dystopian genre.

Model output: Evelyn Desmet is widely known as a romance novelist.

Your task:

Reply with two labels:

Reproduction: [Yes/No]

Hallucination: [Yes/No]

Table 15: Change after applying AS on PubMedQA. Negative values on Target indicate that more model performance decreased on the target unlearning dataset with better unlearning effectiveness; Remaining measures overall QA accuracy change on the remaining data.

| Setup       | Target ∆ ROUGE-L ↓   | Remaining ∆ ROUGE-L ↑   | TR@50 ↑   |
|-------------|----------------------|-------------------------|-----------|
| Unlearn 2%  | - 93 . 2             | +1 . 3                  | 0 . 90    |
| Unlearn 5%  | - 97 . 1             | +0 . 2                  | 0 . 94    |
| Unlearn 10% | - 98 . 3             | - 0 . 2                 | 0 . 96    |

## G Scalability and Biomedical QA

Building on our ToFU results, we probe scalability and domain transfer on a larger model and a real dataset by fine-tuning LLaMA-13B on PubMedQA (long\_answer) and unlearning 2%, 5%, and 10% of memorized data. We first obtain a domain-adapted base model by LoRA fine-tuning LLaMA-13B on PubMedQA (long\_answer). All backbone weights are frozen; LoRA adapters are trained on the attention projections (q\_proj, k\_proj, v\_proj, o\_proj) using two NVIDIA A100-40GB GPUs, bf16 precision, and gradient checkpointing. For unlearning, we insert a set of adapters dedicated to AS and optimize them with the dual-loss objective on mixed unlearn/retain batches. These AS adapters use the same adapter scope and hyperparameters as in our LLaMA-7B experiments to ensure comparability across model scales. The results show that AS consistently suppresses target reproduction while keeping overall QA utility nearly unchanged.