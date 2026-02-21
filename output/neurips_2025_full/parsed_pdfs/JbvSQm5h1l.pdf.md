## Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning

Chongyu Fan † ,⋆ Jiancheng Liu † ,⋆ Licong Lin ‡ ,⋆

Jinghan Jia † Ruiqi Zhang ‡ Song Mei ‡ Sijia Liu † ,

‡ § IBM Research ⋆ Equal contributions

§ † Michigan State University University of California, Berkeley

## Abstract

This work studies the problem of large language model (LLM) unlearning, aiming to remove unwanted data influences ( e.g. , copyrighted or harmful content) while preserving model utility. Despite the increasing demand for unlearning, a technically-grounded optimization framework is lacking. Gradient ascent (GA)type methods, though widely used, are suboptimal as they reverse the learning process without controlling optimization divergence ( i.e. , deviation from the pretrained state), leading to risks of model collapse. Negative preference optimization (NPO) has been proposed to address this issue and is considered one of the stateof-the-art LLM unlearning approaches. In this work, we revisit NPO and identify another critical issue: reference model bias. This bias arises from using the reference model ( i.e. , the model prior to unlearning) to assess unlearning success, which can lead to a misleading impression of the true data-wise unlearning effectiveness. Specifically, it could cause (a) uneven allocation of optimization power across forget data with varying difficulty levels and (b) ineffective gradient weight smoothing during the early stages of unlearning optimization. To overcome these challenges, we propose a simple yet effective unlearning optimization framework, called SimNPO, showing that 'simplicity' in removing the reliance on a reference model (through the lens of simple preference optimization) benefits unlearning. We provide deeper insights into SimNPO's advantages, including an analysis based on mixtures of Markov chains. Extensive experiments further validate its efficacy on benchmarks like TOFU, MUSE and WMDP. Codes are available at https://github.com/OPTML-Group/Unlearn-Simple .

## 1 Introduction

The rapid advancement of LLMs has raised security and safety concerns, including issues related to copyright violations and sociotechnical harms [1-4]. However, retraining these models to remove undesirable data influences is often impractical due to the substantial costs and time required for such processes. This gives rise to the problem of LLM unlearning [5]. To trace its origins, the concept of machine unlearning was initially developed for data removal to comply with privacy regulations such as the 'right to be forgotten' [6, 7], with early studies focusing on vision models [8-15]. However, it is soon adapted to LLMs to remove unwanted data and knowledge [4, 5, 16-20].

The current optimization foundation for LLM unlearning often relies on optimization divergence 1 from the pre-trained state, which refers to the deviation from the converged pre-trained model

1 Here, we use 'divergence' as opposed to 'convergence' in model training, aiming to reverse learning for unlearning.

Strongly-memorized data

Figure 1: (a) Systematic overview of an LLM ( θ ) post-unlearning using the proposed SimNPO, compared to NPO [19] and the reference model. (b) Truth ratio distribution of strongly-memorized forget data points and weakly-memorized data for NPO, SimNPO, and Retrain on the TOFU Forget05 dataset [18] under LLaMA-2chat 7B; See Sec. 4 for more details. As shown, SimNPO achieves better forget quality (FQ, the number after method) than NPO and exhibits a truth ratio distribution closer to Retrain. Note that FQ is a statistical measure quantifying the closeness between the truth ratio distribution of an unlearned model and that of Retrain (with FQ = 1 representing optimal unlearning). (c) &amp; (d) Experiment highlights on TOFU Forget05 and MUSE News datasets [4]. Unlearning effectiveness is measured by FQ for TOFU and PrivLeak for MUSE, while utility preservation is evaluated using model utility for TOFU and KnowMem on retain data for MUSE (see Table A1). In both tasks, Retrain is the gold standard for unlearning.

<!-- image -->

to reverse the effects of learning the forgotten data, thereby achieving unlearning [18, 19, 21]. Nevertheless, the lack of control over the divergence rate in unlearning optimization can lead to either under-forgetting, where insufficient unwanted data influence is removed, or over-forgetting, causing a significant loss of model utility in LLMs. Therefore, optimization for LLM unlearning is highly non-trivial . Negative preference optimization ( NPO ) [19] emerges as an effective approach for LLM unlearning, as demonstrated by its better control of the divergence rate during unlearning optimization and its strong performance in current benchmarks such as TOFU [18] and MUSE [4]. Inspired by direct preference optimization (DPO) [22], it treats the forget data points as negative responses, providing a lower-bounded unlearning objective. This also induces a gradient weight smoothing scheme to regulate the speed of divergence. We refer readers to Sec. 3 for details.

Despite the advancements NPO has brought to the optimization foundation for LLM unlearning, our work identifies, for the first time, its potential limitations stemming from its reliance on the reference model ( i.e. , the model prior to unlearning) as the basis for promoting and regulating the optimization divergence. We term this issue reference model bias . See the conceptual schematic overview below.

Fig. 1-(a) illustrates this issue schematically. NPO aims to widen the gap between the unlearned model ( θ NPO ) and the reference model ( θ ref ). However, the prediction confidence of θ ref varies across samples, as illustrated by the 'hard' vs. 'easy' unlearning examples along the green line in Fig. 1-(a). Specifically, 'hard' examples are those whose predictions under θ ref lie far from the unlearning decision boundary, making them more difficult to forget. In contrast, 'easy' examples are already close to the boundary, where further increasing the gap between the unlearned model and θ ref could become unnecessary. Yet, NPO may blindly increase the deviation from θ ref (as shown by the blue line in Fig. 1-(a)), causing 'easy' examples to move unnecessarily far beyond the unlearning boundary. Meanwhile,'hard' examples remain far from the boundary and require more targeted effort to forget. In other words, relying on the reference model can result in suboptimal unlearning power allocation due to its uniform, deviation-based strategy.

## Throughout this work, we ask:

## (Q) How can we identify and address the limitations of NPO to enhance its effectiveness?

In response to (Q) , we propose a simple yet effective unlearning optimization framework, termed SimNPO , demonstrating that properly removing reliance on a reference model can significantly enhance unlearning. This approach also draws inspiration from simple preference optimization in LLMalignment [23]. Additionally, we will provide detailed insights into how SimNPO overcomes the limitations of NPO caused by reference model bias. As shown schematically in Fig. 1-(a), SimNPO outperforms NPO by more accurately identifying the difficulty of unlearning data ( i.e. , hard vs. easy samples) and allocating optimization power more effectively across different forget samples. Fig. 1(b) provides experimental evidence, which will be provided in Sec. 4 , by comparing the unlearning performance of NPO and SimNPO across forget data points with their unlearning difficulty levels indicated by their memorization levels . The rationale is that the reference model demonstrates varying levels of memorization across different forget samples, making strongly-memorized samples

harder to unlearn and weakly-memorized samples easier to unlearn. However, NPO may blindly over-allocate unlearning power to these easier samples, thereby hindering the effective unlearning of harder ones. This explains why Fig. 1-(b) shows that NPO performs worse than SimNPO in the strongly-memorized (hard) forget data, as evidenced by a greater deviation from Retrain .

In summary, ours contributions are outlined below:

- We revisit the NPO framework and identify its potential weakness-reference model bias-in LLM unlearning, which can lead to issues such as sensitivity to the reference model's response quality and ineffective gradient weight smoothing. We reveal and justify this bias through a series of analyses/examples, including reference model perturbation, the relationship between unlearning and data memorization, and the impact of forget data length on unlearning.
- Building on insights into NPO's limitations, we propose an improved LLM unlearning approach, SimNPO, which extends NPO using a reference-free optimization framework, simple preference optimization [23]. Despite its simplicity, our methodology is grounded in a rigorous technical rationale, as supported by additional synthetic studies and theoretical insights.
- We conduct extensive experiments to demonstrate the improvements of SimNPO over NPO across various scenarios, including TOFU [18], MUSE [4], WMDP [3], and defending against relearningbased attacks [24, 25]. Some experiment highlights on TOFU and MUSE unlearning benchmark datasets are showcased in Fig. 1-(c,d) .

## 2 Related work

Machine unlearning. From the perspective of whether the forget data can be inferred from the unlearned model in terms of membership ( i.e. , a data privacy viewpoint), the widely adopted gold standard for machine unlearning is 'Retrain' [8, 11, 13], which we also adopt in this work. Also known as exact unlearning, this approach retrains the model from scratch on the original training set with the forget data excluded. However, exact unlearning is challenging in practice due to the assumption for access to the full training set and the high computational cost of retraining. To address these challenges, various approximate unlearning methods have been developed [10, 26, 27]. These approaches typically involve model fine-tuning or editing, applied to the pre-trained model, based on the unlearning request. Their effectiveness has been shown in different application domains, including image classification [12, 13, 28, 29], image generation [14, 15, 30], federated learning [31-33], and graph neural networks [34-36].

LLM unlearning. There has also been a growing body of research focusing on LLM unlearning [3, 5, 16-20, 37-49], aiming to effectively remove undesired data influences and/or model behaviors while preserving the utility for unrelated knowledge generation, and maintaining efficiency without the need for retraining. Applications of unlearning in LLMs are diverse, from safeguarding copyrighted and personally identifiable information [16, 38, 50], to preventing LLMs from creating cyberattacks or bioweapons [3, 51], and reducing the production of offensive, biased, or misleading content [17, 37, 52]. Current unlearning approaches include model optimization-based methods [3, 16, 17, 1921, 47-49, 53] and input prompt or in-context learning-based techniques [41, 44, 45]. However, many lack effectiveness, leading to either under-forgetting or over-forgetting, as shown by recent LLM unlearning benchmarks such as TOFU for fictitious unlearning [18] and MUSE for private or copyrighted information removal [4]. Recent studies also show that even after unlearning, models can remain vulnerable to adversarial attacks [24, 54, 55] or relearning from a small number of forget data [24, 25]. This evidence suggests that effective unlearning for LLMs is far from trivial. Among current efforts, NPO (negative preference optimization) [19] stands out as a promising method. However, we will show that the advantages of NPO can be limited by the presence of reference model bias (Sec. 4).

Preference optimization. In this work, we advance LLM unlearning through the lens of preference optimization. This is motivated by aligning LLMs with human values, known as reinforcement learning from human feedback (RLHF) [56-58]. However, online preference optimization algorithms are often complex and challenging to optimize [59, 60], driving interest in more efficient offline alternatives. Direct preference optimization ( DPO ) [22] introduced an offline approach that eliminates the need for a reward model, sparking the development of several reward-free offline preference objectives [23, 61-65]. Notable methods include RRHF [65], SLic-HF [61], IPO [62], KTO [64], ORPO [63], and SimPO [23]. Among these methods, SimPO is a reference-free, length-normalized

variant of DPO, and we will demonstrate that it is well-suited for integrating into LLM unlearning and improving NPO.

## 3 A Primer on LLM Unlearning

Problem formulation. Unlearning tasks can take various forms and are typically associated with a specific set of data points to be removed, known as the forget set ( D f ). These tasks often require a complementary set of non-forgotten data points, known as the retain set ( D r ), to preserve model utility by penalizing the divergence caused by unlearning. As a result, the problem of LLM unlearning can be cast as a regularized optimization problem that balances the forget and retain objectives [5, 19]:

<!-- formula-not-decoded -->

where θ represents the model parameters to be updated during unlearning, λ ≥ 0 is a regularization parameter to penalize the 'divergence' of unlearning, and ℓ f and ℓ r represent forget and retain losses incurred when using model parameters θ to generate y given the input x .

Substantial research has focused on designing and analyzing appropriate forget and retain loss functions to solve problem (1) [4, 5, 16-20]. For instance, let π θ ( y | x ) represent the prediction probability of the model θ given the input-response pair ( x, y ) . The retain loss is typically chosen as the cross-entropy-based sequence prediction loss, ℓ r ( y | x, θ ) = -log π θ ( y | x ) , whose minimization encourages the model to perform well on the retain data ( x, y ) ∈ D r . In (1), if we specify the forget loss as the negative token prediction loss ℓ f ( y | x, θ ) = log π θ ( y | x ) , whose minimization then discourages the model from learning the forget data ( x, y ) ∈ D f . Minimizing such a forget loss is known as the gradient ascent ( GA ) method [11, 18]. Similarly, minimizing the regularized loss that integrates GA with the retain loss is known as the gradient difference ( GradDiff ) method [17, 18, 21].

Negative preference optimization (NPO). A popular optimization framework for solving problem (1) is NPO [19]. It treats the forget data as negative examples in DPO [22], transforming the unbounded GA-based forget loss into a ① bounded loss from below , which helps prevent catastrophic collapse, and an ② adaptive weight smoothing applied to the forget loss gradients, enabling more controlled divergence speed in unlearning.

These benefits can be clearly seen from the NPO loss and its gradient as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where σ ( t ) = 1 / (1 + e -t ) is the sigmoid function, β &gt; 0 is the temperature parameter and π ref is the reference model given by the initial model prior to unlearning. Additional insights into ① -② are given below.

① From (2), the NPO-type forget loss is bounded below by 0 , i.e. , ℓ f ( y | x ; θ ) ≥ 0 , whereas the GA-type forget loss, ℓ f ( y | x, θ ) = log π θ ( y | x ) , has no lower bound. Moreover, minimizing it towards ℓ f ( y | x ; θ ) → 0 drives the prediction probability π θ ( y | x ) to decrease, widening the gap between the prediction probability and the reference model on the forget set, i.e. , π θ ( y | x ) ≪ π ref ( y | x ) .

② As seen in (3), the adaptive weight w θ ( x, y ) is typically less than 1 since π θ ( y | x ) &lt; π ref ( y | x ) for forgetting. Consequently, NPO's gradient yields a more controlled and gradual divergence speed ( i.e. , deviation from the reference model), compared to GA (with w θ ( x, y ) = 1 ).

In this paper, NPO will serve as the primary baseline for LLM unlearning. Its implementation follows the regularized optimization in (1), where the forget loss ℓ f is defined as in (2) and the retain loss ℓ r is the token prediction loss ℓ r ( y | x, θ ) = -log π θ ( y | x ) applied to the retain set.

LLM unlearning tasks and evaluations. Given that the assessment of LLM unlearning may rely on specific tasks, we next introduce the unlearning tasks and evaluation metrics that this work covers. (1) TOFU [18] considers fictitious unlearning on a synthetic Q&amp;A dataset. (2) MUSE [4] is designed

to remove verbatim or knowledge memorization from News and Books datasets, including both verbatim texts and knowledge sets for unlearning evaluation. (3) WMDP [3] aims to prevent LLMs from generating hazardous content in domains such as biology, cybersecurity, and chemistry. Despite the differences in evaluation metrics across the above tasks, the assessment broadly falls into two categories. (1) Unlearning effectiveness measures how faithfully undesired data influences or model capabilities are removed. For example, it is assessed by the forget quality (FQ) metric in TOFU, which uses a p -value to test the indistinguishability between the post-unlearning model and a model retrained on the retain set only, and by privacy leakage (PrivLeak) in MUSE, which measures the likelihood of detecting that the model was ever trained on the forget set. (2) Utility preservation evaluates the post-unlearning performance on standard utility tasks. See Table A1 in AppendixA for a summary of the unlearning tasks and evaluation metrics.

## 4 Uncovering Reference Model Bias in NPO

In this section, we highlight a key weakness of NPO, which we term ' reference model bias ': The incorporation of the reference model in NPO biases the unlearning objective towards enlarging the distance relative to this reference model. As noted in (2), minimizing the NPO loss drives π θ ( y | x ) ≪ π ref ( y | x ) . However, using π ref as the basis for NPO's unlearning criterion can introduce negative effects (L1)-(L2), which we will detail later.

Before that, we present a warm-up study to illustrate NPO's sensitivity to the choice of the reference model ( θ ref , used interchangeably with π ref ). Specifically, we construct a perturbed reference model, θ ′ ref , by averaging the original reference model θ ref with a randomly weighted model, whose weights are drawn from a standard Gaussian distribution with zero mean and variance. We then apply NPO using θ ′ ref as the reference on the TOFU Forget05 dataset, following the same setup as in Fig. 1-(c). We find that there exists a substantial drop in forget quality-from 0.79 (with θ ref ) to 0.27 (with θ ′ ref ), while the model utility remains nearly unchanged (0.52 w/ θ ′ ref vs. 0.57 w/ θ ref ). We refer readers to Fig. A1 in AppendixB for the detailed comparison. This preliminary study highlights the critical influence of the reference model on NPO's unlearning effectiveness. Thus, a deeper investigation into the use of the reference model could offer valuable insights for improving the unlearning optimization framework.

Next, we elaborate on the limitations (L1)-(L2) introduced by the reference model in NPO.

(L1) Challenge of uneven allocation of unlearning power across forget data. At first glance, driving the unlearned model to deviate from the reference model in NPO, i.e. , promoting π θ ( y | x ) ≪ π ref ( y | x ) , seems desirable for unlearning on the forget set. However, the over-reliance on π ref can overshadow the true sample-specific unlearning difficulty, leading to an uneven allocation of unlearning power. We elaborate on this issue through two examples. Top 50% short-length data

(Example 1: Unlearning strongly vs. weakly-memorized forget data.) We first explain (L1) from the perspective of unlearning vs. data memorization. Consider two forget sets, D f , 1 and D f , 2 , where D f , 1 is more strongly memorized by the model than D f , 2 . To support these memorization levels, we provide detailed experimental settings in AppendixC . With this setup, the prediction loss on D f , 1 is smaller, leading to a higher prediction probability π ref . Accordingly, the NPO gradient smoothing term in (3) becomes smaller for D f , 1 , meaning NPO allocates less first-order optimization power to it. However, D f , 1 , being strongly memorized, should ideally receive more unlearning power. As a result, this uneven focus hinders NPO's ability to effectively forget D f , 1 , potentially causing underunlearning and reducing the FQ of D f , 1 to nearly zero. See Fig. 1-(b) and Table A2 for results.

Figure 2: Truth ratio distribution of short/long forget data for NPO, SimNPO, and Retrain on TOFU Forget05. The figure format follows Fig. 1-(b).

<!-- image -->

(Example 2: Unlearning short vs. long-response data.) In this example, we evaluate unlearning performance across different types of forget data, categorized by their response lengths ( i.e. , short vs. long). The motivation stems from the observation that the reference model may exhibit a bias toward generating longer, yet lower-quality, responses [23]. Fig. 2 shows that NPO exhibits a greater distance from Retrain when unlearning the top 50% shortest-length forget data, resulting in a lower FQ (forget quality) of 0 . 58 . In contrast, NPO performs better unlearning for the longer 50% of the

forget set, yielding a higher FQ of 0 . 81 . The ineffectiveness of NPO in unlearning forget data with short responses will be further analyzed through the lens of a mixture of Markov chains in Sec. 5.

Figure 3: Experimental evidence of ineffective weight smoothing and utility-drop for NPO on TOFU Forget05 (a) NPO's gradient weights ( w θ ) at epoch 1 vs. response length | y | . (b) Trajectory of w θ for NPO over unlearning epochs, where box plot represents the distribution of gradient weights over forget samples. (c)-(d) Forget quality and model utility of NPO vs. epochs.

<!-- image -->

(L2) Lack of gradient weight smoothing in the early stages of unlearning. Another issue introduced by the reference model π ref concerns the effectiveness of NPO's gradient weight smoothing, i.e. , w θ ( x, y ) = (2 π θ ( y | x ) β ) / ( π θ ( y | x ) β + π ref ( y | x ) β ) in (3). During the early optimization stage of NPO, we find w θ ( x, y ) ≈ 1 regardless of the varying data-specific unlearning difficulties since the initialization of the unlearned model θ is given by the reference model. Fig. 3-(a,b) support this finding by displaying the gradient smoothing weights of NPO at epoch one for forget data with varying response lengths (Fig. 3a), as analyzed in Example 2, and their trajectory over the course of unlearning epochs (Fig. 3b). As shown, the gradient smoothing weights of NPO show large variance, but most values are concentrated around w θ ( x, y ) ≈ 1 at epoch one. This implies that NPO behaves similarly to GA in the early stage of unlearning, potentially causing a large utility drop even if the weight decreases in later optimization. Fig. 3-(c,d) justify the above by presenting FQ and model utility of NPO on TOFU against unlearning epochs. As shown, NPO tends to cause a larger utility drop at early epochs compared to SimNPO , the improved alternative to NPO in Sec. 5.

## 5 SimNPO: Method and Rationale

Motivation of SimNPO and its forget objective. The simplest solution to mitigating NPO's reference model bias is to directly remove π ref from the gradient in (3), setting π ref = 0 . However, this variant would be ineffective , as the reference-free gradient reduces to GA, with w θ ( x, y ) = 1 . This negates NPO's advantages. To develop a better solution for improving NPO, we revisit the context of preference optimization and investigate whether the reference model can be excluded while still retaining the unlearning benefits provided by NPO. Our idea parallels how NPO was originally inspired by DPO [22]. We adopt SimPO [23], a reference-free alternative to DPO, as the optimization framework for unlearning, leading to the SimNPO (simple NPO) method.

The key difference between SimPO and DPO lies in their reward formulation for preference optimization. In DPO, the reward formulation is given by the comparison with the reference model, i.e. , β log( π θ ( y | x ) /π ref ( y | x )) . This formulation was used by NPO. In contrast, SimPO takes a reference-free but length-normalized reward formulation: ( β/ | y | ) log π θ ( y | x ) , where | y | denotes the response length.

Taking the inspiration of SimPO, we can mitigate the reference model bias in NPO by replacing its reward formulation β log( π θ ( y | x ) /π ref ( y | x )) in (2) with the SimPO-based reward formulation ( β/ | y | ) log( π θ ( y | x )) . This modification transforms (2) into the SimNPO loss :

<!-- formula-not-decoded -->

where γ ≥ 0 is the reward margin parameter, inherited from SimPO, which defines the margin of preference for a desired response over a dispreferred one. However, unless otherwise specified, we set γ = 0 to align with the NPO loss (2). This is also desired because γ introduces a margin to the prediction loss -( β/ | y | ) log π θ ( y | x ) . Consequently, a larger γ requires greater compensation to further suppress token prediction, enforcing a stricter unlearning condition. This can accelerate the utility drop during unlearning. See Fig. A2 of Appendix D for the ablation study of hyperparameters. The SimNPO loss (4), when integrated in (1), forms the SimNPO method.

Insights into SimNPO: Addressing NPO's limitations one by one. Similar to NPO, the SimNPO loss (4) is bounded from below, with a minimum value of 0 . Approaching this minimum drives the unlearning. However, the key distinction of SimNPO from NPO is its forget data-aware, lengthnormalized reward formulation, ( β/ | y | ) log π θ ( y | x ) in (4). This results in an improved gradient smoothing scheme. Specifically, the gradient of the SimNPO loss (with γ = 0 ) yields:

<!-- formula-not-decoded -->

See Appendix E for derivation. Similar to NPO in (3), the gradient in (5) can be divided into two components: weight smoothing ( w ′ θ ) and GA. However, in SimNPO, the weight smoothing is no longer influenced by the reference model and is instead normalized by the length | y | . This introduces two key advantages (a)-(b) below, in response to NPO's limitations (L1)-(L2).

(a) SimNPO leverages the (data-specific) response length as a guide for unlearning power allocation. For instance, when | y | is large, less optimization power is allocated, helping to avoid the uneven unlearning power allocation across forget data with varying response lengths, as exemplified in Fig. 2. In the extreme case where β → 0 , the SimNPO's gradient reduces to a weighted GA : ∇ θ ℓ SimNPO ( θ ) → E ( x,y ) ∈D f [1 / | y |∇ θ log π θ ( y | x )] . This is different from NPO, which becomes GA as β → 0 . Fig. A3 in AppendixF empirically demonstrates the advantage of length normalization in SimNPO for unlearning. As shown, SimNPO outperforms NPO in both forget quality and model utility, coming closest to Retrain. Even in the special case where β = 0 ( i.e. , Weighted-GradDiff), the length normalization provides benefits over GradDiff.

Figure 4: Gradient weight smoothing of NPO ( w θ ) and SimNPO ( w ′ θ ) vs. forget data response length | y | across different epochs (1, 2, 3, and 10) on TOFU Forget05. The Pearson correlation in the upper right corner indicates the relationship between gradient weight smoothing and response length. The SimNPO's weights w ′ θ have been rescaled (by × 10 ) for ease of visualization.

<!-- image -->

(b) In addition, the reference-free, length-normalized weight smoothing prevents early-stage ineffectiveness during unlearning. It can be shown from (5) that w ′ θ ( x, y ) &lt; 2 / | y | , with the distribution of weights w ′ θ ( x, y ) depending on the specific forget data samples. This contrasts with NPO, where the weight distribution concentrated around w θ ( x, y ) ≈ 1 during the early unlearning stage. Extended from Fig. 3-(a)&amp;(b), Fig. 4 provides a detailed comparison between the gradient weights of SimNPO and NPO. We find that SimNPO tends to prioritize short-length forget data that are initially harder to forget during the first two unlearning epochs. At later epochs, the gradient weights become more uniform, reflecting that SimNPO can then treat different forget data with even optimization power. This trend is different from NPO, which assigns more uniform gradient weights early on and starts to account for data-specific difficulty only in the later stages of unlearning. Besides the above advantage, we also find that SimNPO's new weight smoothing scheme does not compromise the overall unlearning speed compared to NPO. This is supported by the divergence rate from the pre-trained state shown in Fig. A4 and our theoretical discussion in AppendixG .

Further analyses via a mixture of Markov chains. In addition to the above insights, we further validate SimNPO's advantages to overcome NPO's limitations (Sec. 4) using a synthetic setup. For ease of controlling the unlearning difficulties of different forget data points, we consider the problem of unlearning on a mixture of Markov chains with a state space of size 10 ( s = 1 , . . . , 10 ). The retain distribution consists of Markov chains that transition uniformly among states { 1 , 2 , 3 } . The forget distribution is a mixture of two components: Forget1 , where the chains transition uniformly among

{ 4 , 5 , 6 } , and Forget2 , where they move uniformly among { 7 , 8 , 9 } . A small leakage probability allows the chains to transition outside their designated states occasionally, including state 10 , which is not a designated state for any of the chains. We generate 10,000 samples for the retain distribution and 5,000 samples each for Forget1 and Forget2. A GPT-2 model is pretrained on these samples and serves as the initial model. We apply NPO and SimNPO to unlearn the forget distributions. Forget and retain performance is evaluated using the KL-divergence between predicted and true transition probabilities of the Markov chains. See AppendixH for details. We present our results in Fig. 5 and summarize the insights below.

SimNPO achieves more balanced unlearning across data of varying lengths compared to NPO. To validate this, we set the retain distribution and Forget1 with a sequence length of 20, while Forget2 is assigned a shorter sequence length of 5, representing a mix of long and short responses. Fig. 5 (a) shows that NPO exhibits a worse tradeoff between retain distance and forget quality on short responses ( i.e. , Forget2) compared with SimNPO. That is, to achieve the same forget quality on Forget2 as the retrained model (with forget quality 0 . 44 ), NPO incurs a higher retain distance than SimNPO. As a result, NPO has an overall

(a) Different length. (b) Different memorization.

<!-- image -->

Figure 5: Tradeoffs between forget quality (higher ↑ is better) and retain distance (lower ↓ is better) along the unlearning path of NPO and SimNPO in the synthetic experiments. The symbols ( ⋆, · ) near the y -axis of both figures indicate the performance of the retrained model on Forget1 and Forget2, respectively.

larger retain distance when unlearning the entire Forget distribution. In contrast, SimNPO shows more consistent performance across Forget1 and Forget2, with less variance in its tradeoff.

SimNPO achieves more balanced unlearning across data of varying memorization compared to NPO. In the second case, we set the retain distribution, Forget1 and Forget2 all with a sequence length of 20 . However, we exclude Forget2 during pretraining. This setup simulates a scenario where the initial model ( i.e. , the reference model in NPO) exhibits varying levels of memorization for the forget data: strongly memorized Forget1 against Forget2. Fig. 5 (b) shows that NPO exhibits a larger gap between Forget1 and Forget2 for the same Retain distance, leading to over-unlearning weakly-memorized data (as shown by the comparison between NPO-Forget2 vs. SimNPO-Forget2) and under-unlearning strongly-memorized data (as shown by the comparison between NPO-Forget1 vs. SimNPO-Forget1). SimNPO has a better balance during unlearning across data with varying levels of memorization.

## 6 Experiments

## 6.1 Experiment setups

Datasets and methods. We evaluate unlearning tasks on three benchmark datasets: TOFU [18], MUSE [4], and WMDP [3]. TOFU includes 'Forget05' and 'Forget10' scenarios, representing 5% and 10% forget sets, respectively. MUSE focuses on 'Books' and 'News' forgetting scenarios, while WMDPtargets knowledge-based unlearning of hazardous biosecurity information.

LLMunlearning methods and evaluation. We evaluate a range of unlearning methods, including Retrain , SimNPO , NPO , GA , and GradDiff . In addition, we incorporate several task-specific approaches: the rejection-based method IDK , which replaces positive responses in DPO with generic answers such as 'I don't know' [18], and RKLD [48] in the TOFU; the Task Vector method used in MUSE [4]; and the representation misdirection unlearning method RMU in WMDP [3]. Evaluation metrics for each benchmark are summarized in Table A1 and further detailed in Appendix I.2. For the relearning attack, we use 20% of the TOFU Forget05 set and retrain over three epochs. Please refer to AppendixI.2 for full experimental details.

## 6.2 Experiment results

Performance on TOFU. In Table 1 , we present the unlearning performance of SimNPO and its various baselines on TOFU Forget05, covering both effectiveness metrics and utility metrics as shown in Table A1. 'FQ' stands for forget quality, and 'MU' represents model utility. These two metrics serve as the primary performance indicators for LLM unlearning on TOFU. SimNPO outperforms NPO in both FQ and MU, and is the closest approximate unlearning method to Retrain. Except for NPO and RKLD, the other unlearning baselines (GA, GradDiff, and IDK) are not effective, as implied by their FQ values being smaller than 0 . 01 , where FQ indicates the p -value for rejecting the

Table 1: Unlearning performance on TOFU Forget05 using the LLaMA2-7B-chat model. 'Prob.' indicates the probability metrics, as summarized in Table A1, with forget quality (FQ) and model utility (MU) serving as the primary metrics. Results are averaged over five random trials. The best FQ and MU are highlighted in bold .

|          | Unlearning Efficacy   | Unlearning Efficacy    | Unlearning Efficacy   | Unlearning Efficacy   | Utility Preservation   | Utility Preservation   | Utility Preservation   | Utility Preservation   | Utility Preservation   | Utility Preservation   | Utility Preservation   | Utility Preservation   | Utility Preservation   | Utility Preservation   |
|----------|-----------------------|------------------------|-----------------------|-----------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| Method   | (1-Rouge-L) ↑         | Forget Set (1-Prob.) ↑ | Truth ratio ↑         | FQ ↑                  | Real Authors ↑         | Real Authors ↑         | Real Authors ↑         | World Facts            | World Facts            | World Facts            | Retain ↑               | Retain ↑               | Retain ↑               | MU ↑                   |
|          | (1-Rouge-L) ↑         | Forget Set (1-Prob.) ↑ | Truth ratio ↑         | FQ ↑                  | Rouge-L ↑              | Prob. ↑                | Truth ratio            | Rouge-L ↑              | Prob. ↑                | Truth ratio            | Rouge-L ↑              | Set Prob. ↑            | Truth ratio ↑          | MU ↑                   |
| Original | 0.04                  | 0.01                   | 0.49                  | 0.00                  | 0.93                   | 0.44                   | 0.58                   | 0.91                   | 0.43                   | 0.55                   | 0.98                   | 0.99                   | 0.48                   | 0.62                   |
| Retrain  | 0.61                  | 0.85                   | 0.66                  | 1.00                  | 0.92                   | 0.44                   | 0.57                   | 0.90                   | 0.43                   | 0.54                   | 0.97                   | 0.99                   | 0.48                   | 0.62                   |
| GA       | 1.00                  | 1.00                   | 0.66                  | 1.9e-9                | 0.00                   | 0.20                   | 0.40                   | 0.00                   | 0.30                   | 0.28                   | 0.00                   | 0.00                   | 0.15                   | 0.00                   |
| GradDiff | 1.00                  | 1.00                   | 0.60                  | 3.6e-9                | 0.59                   | 0.59                   | 0.81                   | 0.88                   | 0.46                   | 0.59                   | 0.42                   | 0.49                   | 0.48                   | 0.56                   |
| IDK      | 0.98                  | 0.40                   | 0.55                  | 1.9e-9                | 0.65                   | 0.48                   | 0.63                   | 0.82                   | 0.44                   | 0.55                   | 0.55                   | 0.86                   | 0.43                   | 0.57                   |
| RKLD     | 0.69                  | 0.96                   | 0.66                  | 0.79                  | 0.92                   | 0.47                   | 0.61                   | 0.87                   | 0.47                   | 0.58                   | 0.58                   | 0.52                   | 0.43                   | 0.56                   |
| NPO      | 0.73                  | 0.94                   | 0.67                  | 0.79                  | 0.91                   | 0.50                   | 0.62                   | 0.90                   | 0.50                   | 0.61                   | 0.47                   | 0.51                   | 0.44                   | 0.57                   |
| SimNPO   | 0.74                  | 0.97                   | 0.69                  | 0.99                  | 0.90                   | 0.50                   | 0.64                   | 0.90                   | 0.48                   | 0.60                   | 0.54                   | 0.56                   | 0.44                   | 0.58                   |

indistinguishability between the unlearned model and Retrain on TOFU. In Table A5 of Appendix J , we also provide examples of model responses after unlearning using SimNPO, Retrain, and NPO, along with label to degenerate. We observe that, in some cases ( e.g. , responses against the input queries Q1 and Q2 in Table A5), the NPO-unlearned model generates repeated texts in response. While this repetition does not reveal the information intended for unlearning, it differs noticeably from Retrain. In contrast, SimNPO produces unlearning responses more closely aligned with those generated by Retrain. More results on TOFU Forget10 are in Table A3 of Appendix I.3 .

Performance on MUSE and WMDP. Table2 compares SimNPO with other methods, on MUSE News and Books, with evaluation metrics in Table A1. Compared to NPO, SimNPO preserves higher utility while achieving stronger unlearning. On D r , KnowMem is 39.65 (News) and 48.27 (Books), while on D f , it is 44.84 (News) and 0.00 (Books). SimNPO also attains a PrivLeak value closer to 0 than NPO (72.93 for News, -31 . 17 for Books), indicating it better approximates complete data removal [4]. Compared to other methods, SimNPO strikes the best balance between utility and unlearning. We further evaluate sequential unlearning on MUSE News ( Fig. A5 in AppendixI.4 ), where SimNPO consistently outperforms NPO as requests increase. Due to space constraints, we present SimNPO's performance on the WMDP dataset in AppendixI.5 .

Table 2: Performance of various unlearning methods on MUSE News (LLaMA2-7B) and MUSE Books (ICLM-7B).

|                             | Unlearning Efficacy   | Unlearning Efficacy    | Unlearning Efficacy         | Utility                |
|-----------------------------|-----------------------|------------------------|-----------------------------|------------------------|
| Method                      | VerbMem D f ( ↓ )     | KnowMem D f ( ↓ )      | PrivLeak ( → 0)             | KnowMem D r ( ↑ )      |
| MUSE News                   | MUSE News             | MUSE News              | MUSE News                   | MUSE News              |
| Original Retrain            | 58.29 20.75           | 62.93 33.32            | -98.71 0.00                 | 54.31 53.79            |
| GA GradDiff Task Vector NPO | 0.00 4.85 77.42 2.53  | 0.00 31.29 58.76 56.93 | 20.14 108.12 -100.00 108.91 | 0.00 28.21 47.94 37.58 |
| MUSE Books                  | MUSE Books            | MUSE Books             | MUSE Books                  | MUSE Books             |
| Original Retrain            | 99.56 14.30           | 58.32 28.90            | -56.32 0.00                 | 67.01 74.50            |
| GA GradDiff NPO             |                       | 0.00 0.00              | -24.07                      | 0.00                   |
| Vector                      | 0.00 0.00 99.31 0.00  | 35.55 0.00             | -24.59 -83.78 -31.17        | 0.13                   |
| Task                        |                       |                        |                             | 62.55                  |
| SimNPO                      |                       |                        |                             | 23.71                  |
|                             | 0.00                  |                        |                             |                        |
|                             |                       | 0.00                   |                             |                        |
|                             |                       |                        | -19.82                      |                        |
|                             |                       |                        |                             | 48.27                  |

Unlearning robustness against length-variant relearning attacks. Recent studies [24, 25] show that unlearning methods are vulnerable to relearning attacks, where forgotten information can be recovered by finetuning on a subset of the forget set. We evaluate SimNPO's robustness against such attacks, showing it to outperform NPO, especially for shortlength response data. Fig. 6 presents the forget quality of SimNPO and NPO under relearning attacks against the number of relearning epochs. Relearning is performed on the forget subset, which is either the shortest 20% of responses from the TOFU Forget05 dataset or an equal-size random subset. We refer to these attacks as 'shortest-relearn' and 'randomrelearn', respectively. The random-relearn case is conducted 5 times, with both average robustness and variance in Fig. 6. As we can see, SimNPO demonstrates improved robustness over NPO, evidenced by higher forget quality and a slower decline in forget quality as the relearning epoch increases. NPO is less robust against the shortest-relearn attack compared

Figure 6: NPO and SimNPO under random/shortest relearn attack vs. epochs on TOFU Forget05.

<!-- image -->

to the random-relearn attack. In contrast, SimNPO is resilient to both types of relearning. This is expected since SimNPO addresses the limitation (L1), as explained in Sec. 4.

## 7 Conclusion

We identified a reference model bias in negative preference optimization (NPO) that limits unlearning effectiveness. To address this, we proposed SimNPO, a simpler framework leveraging preference optimization without a reference model. SimNPO consistently outperforms NPO across

standard benchmarks such as TOFU, MUSE, and WMDP, and demonstrates additional advantages in unlearning robustness and the application to reasoning model unlearning.

## Broader Impact

On the positive side, we have demonstrated the utility of preference optimization in machine unlearning. This connection enables more efficient unlearning operations in LLMs, improving data privacy protections and supporting compliance with regulatory requirements. Additionally, given the relationship between preference optimization and model editing, our work encourages further exploration in these areas, contributing to the development of models that are easier to customize and become safer to deploy. On the negative side, the methods we developed could be misused to selectively erase 'essential' (rather than 'unwanted') concepts or knowledge, raising ethical and legal concerns. To mitigate this risk, it is essential to ensure that unlearning applications adhere to strict ethical guidelines to prevent misuse. We hope our research fosters the development of safe, reliable, and human-aligned LLMs.

## Limitations

While SimNPO mitigates the reference model bias present in NPO and improves gradient weight smoothing to better adjust divergence speed based on the varying unlearning difficulties of forget data samples, both frameworks still rely on promoting divergence to achieve unlearning. This reliance inevitably results in some degree of utility loss. This limitation becomes especially evident in knowledge unlearning or model capability removal scenarios, such as in the WMDP unlearning benchmark. Consequently, SimNPO has yet to fully resolve the challenge of balancing unlearning effectiveness with model utility. Additionally, establishing theoretical guarantees for SimNPO remains an important area for future research.

## Acknowledgement

C. Fan, J. Liu, J. Jia, and S. Liu were supported in part by the National Science Foundation (NSF) CISE Core Program Awards IIS-2207052 and IIS-2504263, the NSF CAREER Award IIS-2338068, the ARO Award W911NF2310343, the Amazon Research Award for AI in Information Security, the Open Philanthropy Research Award, and the Center for AI Safety (CAIS) Compute Award. We also extend our gratitude to the MIT-IBM Watson AI Lab, IBM Research for their support in this project. L. Lin, R. Zhang, and S. Mei were supported in part by NSF CCF-2315725, NSF Career DMS-2339904, ONR N00014-24-S-B001, an Amazon Research Award, and a Google Research Scholar Award. We also thank the support from the Center for AI Safety Compute Cluster. Finally, we express our appreciation to Yuguang Yao for his help in figure plotting.

## References

- [1] Y. Huang, L. Sun, H. Wang, S. Wu, Q. Zhang, Y. Li, C. Gao, Y. Huang et al. , 'Position: TrustLLM: Trustworthiness in large language models,' in Proceedings of the 41st International Conference on Machine Learning , ser. Proceedings of Machine Learning Research, vol. 235, 21-27 Jul 2024, pp. 20 166-20 270.
- [2] B. Wang, W. Chen, H. Pei, C. Xie, M. Kang, C. Zhang, C. Xu, Z. Xiong, R. Dutta, R. Schaeffer et al. , 'Decodingtrust: A comprehensive assessment of trustworthiness in gpt models.' in NeurIPS , 2023.
- [3] N. Li, A. Pan, A. Gopal, S. Yue, D. Berrios, A. Gatti, J. D. Li, A.-K. Dombrowski, S. Goel, L. Phan et al. , 'The wmdp benchmark: Measuring and reducing malicious use with unlearning,' arXiv preprint arXiv:2403.03218 , 2024.
- [4] W. Shi, J. Lee, Y . Huang, S. Malladi, J. Zhao, A. Holtzman, D. Liu, L. Zettlemoyer, N. A. Smith, and C. Zhang, 'Muse: Machine unlearning six-way evaluation for language models,' arXiv preprint arXiv:2407.06460 , 2024.

- [5] S. Liu, Y. Yao, J. Jia, S. Casper, N. Baracaldo, P. Hase, Y. Yao, C. Y. Liu, X. Xu, H. Li et al. , 'Rethinking machine unlearning for large language models,' Nature Machine Intelligence , pp. 1-14, 2025.
- [6] J. Rosen, 'The right to be forgotten,' Stan. L. Rev. Online , vol. 64, p. 88, 2011.
- [7] C. J. Hoofnagle, B. van der Sloot, and F. Z. Borgesius, 'The european union general data protection regulation: what it is and what it means,' Information &amp; Communications Technology Law , vol. 28, no. 1, pp. 65-98, 2019.
- [8] Y. Cao and J. Yang, 'Towards making systems forget with machine unlearning,' in 2015 IEEE symposium on security and privacy . IEEE, 2015, pp. 463-480.
- [9] A. Warnecke, L. Pirch, C. Wressnegger, and K. Rieck, 'Machine unlearning of features and labels,' arXiv preprint arXiv:2108.11577 , 2021.
- [10] L. Bourtoule, V. Chandrasekaran, C. A. Choquette-Choo, H. Jia, A. Travers, B. Zhang, D. Lie, and N. Papernot, 'Machine unlearning,' in 2021 IEEE Symposium on Security and Privacy (SP) . IEEE, 2021, pp. 141-159.
- [11] A. Thudi, G. Deza, V. Chandrasekaran, and N. Papernot, 'Unrolling sgd: Understanding factors influencing machine unlearning,' in 2022 IEEE 7th European Symposium on Security and Privacy (EuroS&amp;P) . IEEE, 2022, pp. 303-319.
- [12] M. Kurmanji, P. Triantafillou, J. Hayes, and E. Triantafillou, 'Towards unbounded machine unlearning,' Advances in neural information processing systems , vol. 36, 2024.
- [13] J. Jia, J. Liu, P . Ram, Y . Yao, G. Liu, Y . Liu, P. Sharma, and S. Liu, 'Model sparsity can simplify machine unlearning,' in Thirty-seventh Conference on Neural Information Processing Systems , 2023.
- [14] R. Gandikota, J. Materzynska, J. Fiotto-Kaufman, and D. Bau, 'Erasing concepts from diffusion models,' in Proceedings of the IEEE/CVF International Conference on Computer Vision , 2023, pp. 2426-2436.
- [15] C. Fan, J. Liu, Y . Zhang, D. Wei, E. Wong, and S. Liu, 'Salun: Empowering machine unlearning via gradient-based weight saliency in both image classification and generation,' in International Conference on Learning Representations , 2024.
- [16] R. Eldan and M. Russinovich, 'Who's harry potter? approximate unlearning in llms,' 2023.
- [17] Y. Yao, X. Xu, and Y. Liu, 'Large language model unlearning,' arXiv preprint arXiv:2310.10683 , 2023.
- [18] P. Maini, Z. Feng, A. Schwarzschild, Z. C. Lipton, and J. Z. Kolter, 'Tofu: A task of fictitious unlearning for llms,' 2024.
- [19] R. Zhang, L. Lin, Y. Bai, and S. Mei, 'Negative preference optimization: From catastrophic collapse to effective unlearning,' arXiv preprint arXiv:2404.05868 , 2024.
- [20] J. Jia, Y. Zhang, Y. Zhang, J. Liu, B. Runwal, J. Diffenderfer, B. Kailkhura, and S. Liu, 'Soul: Unlocking the power of second-order optimization for llm unlearning,' arXiv preprint arXiv:2404.18239 , 2024.
- [21] B. Liu, Q. Liu, and P. Stone, 'Continual learning and private unlearning,' in Conference on Lifelong Learning Agents . PMLR, 2022, pp. 243-254.
- [22] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, 'Direct preference optimization: Your language model is secretly a reward model,' Advances in Neural Information Processing Systems , vol. 36, 2024.
- [23] Y. Meng, M. Xia, and D. Chen, 'SimPO: Simple preference optimization with a reference-free reward,' in The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.

- [24] A. Lynch, P. Guo, A. Ewart, S. Casper, and D. Hadfield-Menell, 'Eight methods to evaluate robust unlearning in llms,' arXiv preprint arXiv:2402.16835 , 2024.
- [25] S. Hu, Y. Fu, Z. S. Wu, and V. Smith, 'Jogging the memory of unlearned model through targeted relearning attack,' arXiv preprint arXiv:2406.13356 , 2024.
- [26] T. T. Nguyen, T. T. Huynh, P. L. Nguyen, A. W.-C. Liew, H. Yin, and Q. V. H. Nguyen, 'A survey of machine unlearning,' arXiv preprint arXiv:2209.02299 , 2022.
- [27] E. Triantafillou, P. Kairouz, F. Pedregosa, J. Hayes, M. Kurmanji, K. Zhao, V. Dumoulin, J. J. Junior, I. Mitliagkas, J. Wan et al. , 'Are we making progress in unlearning? findings from the first neurips unlearning competition,' arXiv preprint arXiv:2406.09073 , 2024.
- [28] Y. Liu, M. Fan, C. Chen, X. Liu, Z. Ma, L. Wang, and J. Ma, 'Backdoor defense with machine unlearning,' in IEEE INFOCOM 2022-IEEE Conference on Computer Communications . IEEE, 2022, pp. 280-289.
- [29] C. Fan, J. Liu, A. Hero, and S. Liu, 'Challenging forgets: Unveiling the worst-case forget sets in machine unlearning,' arXiv preprint arXiv:2403.07362 , 2024.
- [30] Y. Zhang, Y. Zhang, Y. Yao, J. Jia, J. Liu, X. Liu, and S. Liu, 'Unlearncanvas: A stylized image dataset to benchmark machine unlearning for diffusion models,' arXiv preprint arXiv:2402.11846 , 2024.
- [31] Y. Liu, L. Xu, X. Yuan, C. Wang, and B. Li, 'The right to be forgotten in federated learning: An efficient realization with rapid retraining,' in IEEE INFOCOM 2022-IEEE Conference on Computer Communications . IEEE, 2022, pp. 1749-1758.
- [32] A. Halimi, S. Kadhe, A. Rawat, and N. Baracaldo, 'Federated unlearning: How to efficiently erase a client in fl?' arXiv preprint arXiv:2207.05521 , 2022.
- [33] R. Jin, M. Chen, Q. Zhang, and X. Li, 'Forgettable federated linear learning with certified data removal,' arXiv preprint arXiv:2306.02216 , 2023.
- [34] M. Chen, Z. Zhang, T. Wang, M. Backes, M. Humbert, and Y. Zhang, 'Graph unlearning,' in Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security , 2022, pp. 499-513.
- [35] E. Chien, C. Pan, and O. Milenkovic, 'Certified graph unlearning,' arXiv preprint arXiv:2206.09140 , 2022.
- [36] K. Wu, J. Shen, Y. Ning, T. Wang, and W. H. Wang, 'Certified edge unlearning for graph neural networks,' in Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining , 2023, pp. 2606-2617.
- [37] X. Lu, S. Welleck, J. Hessel, L. Jiang, L. Qin, P. West, P. Ammanabrolu, and Y. Choi, 'Quark: Controllable text generation with reinforced unlearning,' Advances in neural information processing systems , vol. 35, pp. 27 591-27 609, 2022.
- [38] J. Jang, D. Yoon, S. Yang, S. Cha, M. Lee, L. Logeswaran, and M. Seo, 'Knowledge unlearning for mitigating privacy risks in language models,' arXiv preprint arXiv:2210.01504 , 2022.
- [39] V. B. Kumar, R. Gangadharaiah, and D. Roth, 'Privacy adhering machine un-learning in nlp,' arXiv preprint arXiv:2212.09573 , 2022.
- [40] E. Zhang, K. Wang, X. Xu, Z. Wang, and H. Shi, 'Forget-me-not: Learning to forget in text-to-image diffusion models,' arXiv preprint arXiv:2303.17591 , 2023.
- [41] M. Pawelczyk, S. Neel, and H. Lakkaraju, 'In-context unlearning: Language models as few shot unlearners,' arXiv preprint arXiv:2310.07579 , 2023.
- [42] Y. Ishibashi and H. Shimodaira, 'Knowledge sanitization of large language models,' arXiv preprint arXiv:2309.11852 , 2023.

- [43] Y. Wang, R. Wu, Z. He, X. Chen, and J. McAuley, 'Large scale knowledge washing,' arXiv preprint arXiv:2405.16720 , 2024.
- [44] C. Y. Liu, Y. Wang, J. Flanigan, and Y. Liu, 'Large language model unlearning via embeddingcorrupted prompts,' arXiv preprint arXiv:2406.07933 , 2024.
- [45] P. Thaker, Y. Maurya, and V. Smith, 'Guardrail baselines for unlearning in llms,' arXiv preprint arXiv:2403.03329 , 2024.
- [46] S. R. Kadhe, F. Ahmed, D. Wei, N. Baracaldo, and I. Padhi, 'Split, unlearn, merge: Leveraging data attributes for more effective unlearning in llms,' arXiv preprint arXiv:2406.11780 , 2024.
- [47] T. Gu, K. Huang, R. Luo, Y. Yao, Y. Yang, Y. Teng, and Y. Wang, 'Meow: Memory supervised llm unlearning via inverted facts,' arXiv preprint arXiv:2409.11844 , 2024.
- [48] B. Wang, Y. Zi, Y. Sun, Y. Zhao, and B. Qin, 'Rkld: Reverse kl-divergence-based knowledge distillation for unlearning personal information in large language models,' arXiv preprint arXiv:2406.01983 , 2024.
- [49] A. Mekala, V. Dorna, S. Dubey, A. Lalwani, D. Koleczek, M. Rungta, S. Hasan, and E. Lobo, 'Alternate preference optimization for unlearning factual knowledge in large language models,' arXiv preprint arXiv:2409.13474 , 2024.
- [50] X. Wu, J. Li, M. Xu, W. Dong, S. Wu, C. Bian, and D. Xiong, 'Depn: Detecting and editing privacy neurons in pretrained language models,' arXiv preprint arXiv:2310.20138 , 2023.
- [51] C. Barrett, B. Boyd, E. Bursztein, N. Carlini, B. Chen, J. Choi, A. R. Chowdhury, M. Christodorescu, A. Datta, S. Feizi et al. , 'Identifying and mitigating the security risks of generative ai,' Foundations and Trends® in Privacy and Security , vol. 6, no. 1, pp. 1-52, 2023.
- [52] C. Yu, S. Jeoung, A. Kasi, P. Yu, and H. Ji, 'Unlearning bias in language models by partitioning gradients,' in Findings of the Association for Computational Linguistics: ACL 2023 , 2023, pp. 6032-6048.
- [53] G. Ilharco, M. T. Ribeiro, M. Wortsman, S. Gururangan, L. Schmidt, H. Hajishirzi, and A. Farhadi, 'Editing models with task arithmetic,' arXiv preprint arXiv:2212.04089 , 2022.
- [54] A. Schwarzschild, Z. Feng, P. Maini, Z. C. Lipton, and J. Z. Kolter, 'Rethinking llm memorization through the lens of adversarial compression,' arXiv preprint arXiv:2404.15146 , 2024.
- [55] V. Patil, P. Hase, and M. Bansal, 'Can sensitive information be deleted from llms? objectives for defending against extraction attacks,' ICLR , 2024.
- [56] P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei, 'Deep reinforcement learning from human preferences,' Advances in neural information processing systems , vol. 30, 2017.
- [57] D. M. Ziegler, N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving, 'Fine-tuning language models from human preferences,' arXiv preprint arXiv:1909.08593 , 2019.
- [58] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al. , 'Training language models to follow instructions with human feedback,' Advances in neural information processing systems , vol. 35, pp. 27 730-27 744, 2022.
- [59] M. Santacroce, Y. Lu, H. Yu, Y. Li, and Y. Shen, 'Efficient rlhf: Reducing the memory usage of ppo,' arXiv preprint arXiv:2309.00754 , 2023.
- [60] R. Zheng, S. Dou, S. Gao, Y. Hua, W. Shen, B. Wang, Y. Liu, S. Jin, Q. Liu, Y. Zhou et al. , 'Secrets of rlhf in large language models part i: Ppo,' arXiv preprint arXiv:2307.04964 , 2023.
- [61] Y. Zhao, R. Joshi, T. Liu, M. Khalman, M. Saleh, and P. J. Liu, 'Slic-hf: Sequence likelihood calibration with human feedback,' arXiv preprint arXiv:2305.10425 , 2023.

- [62] M. G. Azar, Z. D. Guo, B. Piot, R. Munos, M. Rowland, M. Valko, and D. Calandriello, 'A general theoretical paradigm to understand learning from human preferences,' in International Conference on Artificial Intelligence and Statistics . PMLR, 2024, pp. 4447-4455.
- [63] J. Hong, N. Lee, and J. Thorne, 'Reference-free monolithic preference optimization with odds ratio,' arXiv preprint arXiv:2403.07691 , 2024.
- [64] K. Ethayarajh, W. Xu, N. Muennighoff, D. Jurafsky, and D. Kiela, 'Kto: Model alignment as prospect theoretic optimization,' arXiv preprint arXiv:2402.01306 , 2024.
- [65] H. Yuan, Z. Yuan, C. Tan, W. Wang, S. Huang, and F. Huang, 'Rrhf: Rank responses to align language models with human feedback,' Advances in Neural Information Processing Systems , vol. 36, 2024.
- [66] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al. , 'Language models are unsupervised multitask learners,' OpenAI blog , vol. 1, no. 8, p. 9, 2019.
- [67] I. Loshchilov and F. Hutter, 'Decoupled weight decay regularization,' arXiv preprint arXiv:1711.05101 , 2017.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Our main claims are outlined in the abstract and Section 1. Comprehensive details are provided in Sections 3, 4, 5 and 6.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Limitations of our work are discussed in Appendix 7.

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

Justification: Theoretical analyses of the gradient and unlearning speed of our method are provided in Appendix E and G.

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

Justification: The experimental setup is detailed in Section 6 and Appendix I.2.

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

## Answer: [Yes]

Justification: The implementation code is included in the supplementary materials.

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

Justification: The experimental setup is detailed in Section 6 and Appendix I.2.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: To ensure a fair evaluation, we perform multiple runs for each setting and report the average performance. As an example, Fig. 6 presents the results with error bars.

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

Answer: [Yes]

Justification: The computer resources are detailed in Appendix I.2.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have taken all necessary steps to ensure author anonymity.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Impacts of our work are discussed in Appendix 7.

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

Justification: Built upon existing open-source models, our method is intended solely for unlearning benchmark evaluations and does not incorporate any data or models with a high potential for misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We acknowledge and cite the original sources of the codebase and dataset employed in this study.

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

Justification: The code used to conduct our experiments is included in the supplementary material.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This work does not involve any human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This work does not involve any human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: We use LLMs solely for writing refinement.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Appendix

## A A Summary of the Unlearning Tasks and Evaluation Metrics

Table A1: Summary of unlearning efficacy and utility metrics across different unlearning benchmarks. The arrows indicate the directions for better performance ( ↑ for higher values, ↓ for lower values, → 0 for closer to 0).

| Benchmark   | LLMto be used      | Task Description                                                           | Unlearning Effectiveness                                                                                         | Unlearning Effectiveness   | Utility Preservation                                                                                                                                                                                     | Utility Preservation   |
|-------------|--------------------|----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| TOFU        | LLaMA-2-chat 7B    | Unlearning fictitious authors from a synthetic Q&A dataset                 | Forget quality (measured by truth ratios of forget samples) Probability on D f Rouge-L on D f Truth ratio on D f | ↑ ↓ ↓ ↑                    | Model utility ( harmonic mean of 9 utility metrics) Probability on D r / D real_author / D world_facts Rouge-L on D r / D real_author / D world_facts Truth ratio on D r / D real_author / D world_facts | ↑ ↑ ↑ ↑                |
| MUSE        | ICLM-7B LLaMA-2 7B | Unlearning real-world knowledge from texts about Harry Potter and BBC News | KnowMem on D f VerbMem on D f PrivLeak                                                                           | ↓ ↓ → 0                    | KnowMem on D r                                                                                                                                                                                           | ↑                      |
| WMDP        | Zephyr-7B-beta     | Unlearning hazardous knowledge from biosecurity texts                      | Accuracy on WMDP-Bio                                                                                             | ↓                          | Accuracy onMMLU                                                                                                                                                                                          | ↑                      |

## B Additional on the sensitivity of NPO to reference model

To examine the sensitivity of NPO to its reference model choice ( θ ref , used interchangeably with π ref ) , we design a perturbed reference model θ ′ ref by averaging θ ref with a randomly initialized model. Wethen apply NPO with θ ′ ref as the reference on the TOFU Forget05, following the same experimental setup as in Fig. 1 (c). This perturbation leads to a dramatic drop in forget quality-from 0 . 79 with θ ref to 0 . 27 with θ ′ ref -while the model utility remains largely unaffected ( 0 . 57 vs. 0 . 52 ). These results highlight the crucial role of the reference model in ensuring reliable unlearning performance.

Figure A1: Forget quality and model utility of NPO w/ θ ′ ref , NPO w/ θ ref and Retrain on TOFU Forget05. The figure format follows Fig. 1 (c).

<!-- image -->

## C Additional Setup and Results on Unlearning vs. Data Memorization

We use TOFU Forget05 as the forget set D f , splitting it evenly into D f , 1 and D f , 2 . The divided subsets D f , 1 and D f , 2 follow the same distribution of fictitious author information. We finetune the LLaMA-2 7B chat model on the original retain set of TOFU together with D f , 1 , i.e. , D retain ∪ D f , 1 , to obtain the original model before unlearning. The resulting original model strongly memorizes D f , 1 but least memorizes D f , 2 , despite both being drawn from the same distribution. We then perform

Table A2: Unlearning performance on differently memorized forget sets D f , 1 and D f , 2 in TOFU.

|          |   FQ on D f , 1 |   FQ on D f , 2 |   Utility |
|----------|-----------------|-----------------|-----------|
| Original |             0   |            0.01 |      0.62 |
| NPO      |             0   |            0.47 |      0.49 |
| SimNPO   |             0.7 |            0.7  |      0.57 |

unlearning using SimNPO and NPO over D f , 1 ∪ D f , 2 . The unlearning performance, measured in terms of forget quality (FQ) and model utility, is presented in Table A2

As shown in Table A2, since the original model was trained on D f , 1 , its prediction loss -log( π ref ) on D f , 1 is relatively small, leading to a higher prediction probability π ref on D f , 1 . Consequently, the NPO gradient smoothing term in (3) becomes relatively smaller for D f , 1 due to the reference model's bias π ref on D f , 1 . As a result, NPO allocates less first-order optimization power to D f , 1 and focuses more on D f , 2 . This prevents NPO from effectively forgetting D f , 1 , potentially causing under-unlearning and ultimately reducing the FQ of D f , 1 to nearly zero. In contrast, SimNPO, by leveraging a reference-model-free reward, achieves a much smaller FQ difference between D f , 1 and D f , 2 while delivering higher FQ for both datasets compared to NPO. Furthermore, SimNPO demonstrates better model utility relative to NPO.

## D Ablation Studies on SimNPO's Hyperparameter Selection

As shown in (4), β and γ are the two hyperparameters that control the unlearning effectiveness and utility preservation of SimNPO. Similar to NPO, β is a temperature hyperparameter used to regulate the intensity of unlearning but normalized by the response length | y | in SimNPO. As β → 0 , SimNPO approaches weighted GA in Fig. A3. γ is the reward margin parameter from SimPO, which introduces a constant shift to the (per-sample) prediction loss -( β/ | y | ) log π θ ( y | x ) in SimNPO. Consequently, a larger γ imposes a stricter unlearning margin, which could further suppress the model utility.

Figure A2: Forget quality (a) and model utility (b) of SimNPO under different combinations of β and γ on TOFU Forget05.

<!-- image -->

Fig. A2-(a) and Fig. A2-(b) illustrate the forget quality and model utility of SimNPO under various values of β and γ on TOFU forget05. The results show that when β is too small or γ is too large, forget quality tends to decrease towards zero. Additionally, for a fixed β , increasing γ leads to lower model utility. Notably, setting γ = 0 consistently yields the best balance between unlearning performance and utility preservation across different β values, which supports our choice of γ = 0 in SimNPO.

## E Gradient Analysis of SimNPO

Following is the detailed derivation of (5). First, let R = log π θ ( y | x )+ γ | y | /β | y | . We then have the following steps:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

When γ = 0 , the gradient simplifies to the following, which matches (5):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## F Further Results on Response Length Normalization in SimNPO

To better illustrate the role of length-normalization, we consider an extreme case: when β → 0 , the gradient of SimNPO degenerates into length-normalization weighted-GradDiff, while the gradient of NPO degenerates into GradDiff. In Fig. A3 -(a), we further compare the effects of weighted-GradDiff, GradDiff, NPO, and SimNPO. It can be observed that, due to the impact of length-normalization, the

forget quality of weighted GradDiff is significantly better than that of GradDiff. This observation also explains why SimNPO achieves better forget quality compared to NPO.

Figure A3: Forget quality vs. model utility on TOFU Forget05. Weighted-GradDiff (W-GradDiff) is SimNPO at β = 0 .

<!-- image -->

## G Further Analyses on Unlearning Speed

The term 'unlearning speed' or ''divergence rate' refers to the optimization divergence from the pre-trained state, describing the process of deviating from the converged pre-trained model state to reverse the existing learning of the forgotten data. We present some further analyses for the unlearning speed of NPO and SimNPO. Define log π θ ( y | x ) = log π θ ( y | x ) / | y | . Reorganizing the NPO gradient formula in (3), and ignoring the reference model (or when π ref ( y | x ) ≈ 1 ), we have

<!-- formula-not-decoded -->

Suppose log π θ ( y | x ) is linear in θ and the normalized gradient ∇ θ log π θ ( y | x ) = ˜ O (1) . Then loosely speaking, the NPO dynamics satisfies the equation ∇ t θ ( t ) ≈ -2 | y | · exp( β | y | θ ( t )) . Assuming θ (0) = 0 and β ≪ 1 , this yields the solution θ ( t ) = -1 β | y | log(1 + 2 β | y | 2 t ) , suggesting that the models uses ˜ O ( (1 /ϵ ) β | y | -1 β | y | 2 η ) = ˜ O ( log(1 /ϵ ) | y | η ) steps to unlearn the sample ( x, y ) ( i.e. , to let π θ ( y | x ) ≤ ϵ = 0 . 5 ) with length | y | , where η &gt; 0 is the learning rate. This indicates that NPO unlearns longer responses faster than shorter response. In other words, for NPO, it is not possible to unlearn short responses and long responses to the same extent simultaneously.

In contrast, the number of steps needed to unlearn the sample ( x, y ) becomes agnostic to the response length | y | in SimNPO. Recall (5) that

<!-- formula-not-decoded -->

Following a similar argument, we can verify that the model spends roughly ˜ O ( log(1 /ϵ ) η ) steps to unlearn all samples ( x, y ) ( i.e. , to let π θ ( y | x ) ≤ ϵ ), regardless of the response length | y | .

In terms of the big O notation ˜ O , the unlearning speed of SimNPO and NPO is asymptotically identical with respect to the unlearning steps. Fig. A4 validates this by measuring the KL distance on TOFU Forget05 between the unlearned model and the original model. As shown, both SimNPO and NPO exhibit a similar (logarithmic) divergence rate with respect to unlearning steps. This rate is more controllable and slower than that observed with GA (gradient ascent). The rapid divergence in GA leads to a critical issue of model collapse [19]. Consequently, SimNPO maintains the overall unlearning speed advantage of NPO while effectively avoiding model collapse.

## H Additional Details on the Synthetic Study

Synthetic experiment setup. In the synthetic experiment, we study the unlearning problem in a scenario where the data are generated from a mixture of Markov chains. Namely, we assume the

Figure A4: KL distance between the unlearned and original model for GA, NPO and SimNPO on TOFU Forget05

<!-- image -->

Markov chains have a shared state space of size 10 (denoted by s = 1 , 2 , . . . , 10 ), and the retain distribution and the forget distribution have the formulas as follows:

- Retain distribution : Markov chain with initial distribution π r ∈ R 10 and transition matrix T r ∈ R 10 × 10 , where

<!-- formula-not-decoded -->

- Forget distribution : a mixture of two Markov chains (denoted by Forget1 and Forget2) with equal probability. Let ( π f 1 , T f 1 ) and ( π f 2 , T f 2 ) denote the initial distribution and transition matrix for Forget1 and Forget2. We assume

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

The leakage probability is chosen to be ϵ = 0 . 2 . We generate 10000 samples from the retain distribution and 5000 each from Forget1 and Forget2 to form the retain and forget sets. We randomly split the datasets, using 80% of the samples for training and unlearning, and the remaining 20% for testing.

Model and pretraining. In all experiments, we use a small GPT-2 model [66] with modified token embeddings, where input tokens represent states in S = { 1 , 2 , · · · , 10 } , and the output at each token position is a distribution over the state space S . The model has 4 transformer layers, 4 attention heads, and an embedding dimension of 128. We pretrain the original model on both retain and forget data, and the retrained model using only the forget data. Both models are trained using AdamW [67] to minimize the cross-entropy loss averaged over tokens, with a batch size of 128 for 5 epochs. We choose the learning rate η = 0 . 0005 .

Evaluation. Weevaluate the model performance using Forget Quality (higher ↑ is better) and Retain Loss (lower ↓ is better), which are the average KL divergence between the predicted probabilities of the model and the true transition probabilities of the Markov chains, on the forget (Forget1 or Forget2) and the retain test data, respectively.

Unlearning. Starting from the initial model, we run NPO and SimNPO for 50 iterations using a batch size of 4 on the forget dataset. We choose AdamW for optimization with a learning rate of η = 0 . 0005 . The hyperparameter β in both NPO and SimNPO is selected via grid search to optimize the tradeoff between forget quality and retain loss.

Choise of hyperparameters. In the first experiment ( Fig. 5 left ), we set the hyperparameters β NPO = 0 . 2 , β SimNPO = 4 , the retain sample length L r = 20 , and the Forget1 and Forget2 sample lengths L f 1 = 20 , L f 2 = 5 . In the second experiment ( Fig. 5 right ), we choose β NPO = 1 . 0 , β SimNPO = 4 , the retain sample length L r = 20 , and the Forget1 and Forget2 sample lengths L f 1 = 20 , L f 2 = 20 .

## I Additional Experiment Details and Results

## I.1 Computing Resources

All experiments are conducted on 8 NVIDIA A6000 GPU cards in a single node.

## I.2 Experiment Setups

Datasets, tasks, and models. Our experiments cover unlearning tasks across three benchmark datasets: TOFU [18], MUSE [4], and WMDP [3], as summarized in Table A1. For TOFU, we focus on two unlearning scenarios, termed 'Forget05' and 'Forget10', which refer to forget set sizes of 5% and 10%, respectively. In MUSE, we also explore two unlearning scenarios: forgetting the Harry Potter books (termed 'Books') and news articles (termed 'News'), respectively. WMDP, on the other hand, is designed for knowledge-based unlearning, with the forget texts representing hazardous knowledge in biosecurity. The LLM models used for each unlearning benchmark are listed in Table A1.

LLM unlearning methods and evaluation. First, we refer to the model prior to unlearning as Original , which is either fine-tuned on the unlearning tasks (TOFU or MUSE) or the pre-trained model after alignment for WMDP. Starting from the original model, we then apply the following unlearning methods to a given forget set and/or retain set to achieve the unlearning objective, as outlined in (1). Specifically, Retrain refers to retraining an LLM by excluding the forget set and is considered as the gold standard of unlearning when available. Retrain is provided in both the TOFU and MUSE benchmarks. As introduced in Sec. 3, we also include GA (gradient ascent) and GradDiff (the retain-regularized GA variant) as unlearning baseline methods, following the implementations in TOFU and MUSE benchmarks. For other baseline methods such as the rejection-based unlearning method ( IDK ) in TOFU, and the Task Vector unlearning method in MUSE, we adhere to the original implementations specified in their respective benchmarks. NPO with the retain regularization in (1) serves as the primary baseline. Note that its implementation on TOFU follows the original NPO study [19], while its implementation on MUSE aligns with the MUSE benchmark. For NPO on WMDP, due to the absence of open-source implementation, we adapt the TOFU codebase to WMDP. More implementation details can be found in Appendix I.2. To implement the proposed method SimNPO , we adopt a setting similar to NPO but adjust the temperature parameter β . Due to the presence of length normalization in (4), a larger value for β is preferred compared to that in NPO. See the specific choices in Appendix D.

To assess unlearning effectiveness and model utility, we use the evaluation metrics summarized in Table A1 under each unlearning benchmark. In addition, we evaluate the robustness of an unlearned model using relearning-based attacks [25], which aim to recover the forgotten information by finetuning the unlearned models on a small subset of the forget set after unlearning. We select 20% of the original TOFU forget05 set as the relearning set over three epochs.

For all experiments, we use a linear warm-up learning rate during the first epoch, followed by a linearly decaying learning rate in the remaining epochs. We initialize the process with LLaMA-2 7B and fine-tune the model on TOFU for 5 epochs with a batch size of 32 and a learning rate of 10 -5 to obtain the original model. For Forget05, NPO is trained for up to 20 epochs with a learning rate of 10 -5 to obtain the best-performing model. We conducted a grid search for β in the range of [0.05, 0.2] and for λ in the range of [0.5, 1.5]. SimNPO is trained for 10 epochs with a learning rate of 10 -5 . The parameter β is grid-searched over the range [1.5, 3.5], γ is searched between [0.0, 2.0] with the default choice γ = 0 , and λ is explored within the range [0.05, 0.25]. For Forget10, NPO is trained for 10 epochs with a learning rate of 10 -5 . We conducted a grid search for β in the range of [0.05, 0.2] and for λ in the range of [0.5, 1.5]. SimNPO is trained for 10 epochs with a learning rate of 10 -5 . The parameter β is tuned using a grid search within the range [2.5, 5.5], γ is grid-searched between [0.0, 2.0], and λ is grid-searched within [0.05, 0.25]. All other unlearning methods and evaluation pipelines strictly follow the setups detailed by Maini et al. [18] and Zhang et al. [19].

For News, we use LLaMA-2 7B fine-tuned on BBC news articles as the original model. For Books, we use ICLM 7B fine-tuned on the Harry Potter books as the original model. The original models for both Books and News can be directly obtained from benchmark. For SimNPO, we trained for 10 epochs with a learning rate of 10 -5 . We performed a grid search for β in the range of [0.5, 1.0], for λ in the range of [0.05, 0.25], and for γ in the range of [0.0, 2.0] on both the Books and News. The

hyperparameters for other unlearning methods and the evaluation pipelines strictly follow the setup detailed by Shi et al. [4]. We measured the performance after each unlearning epoch and selected the optimal one as the final model.

For WMDP [3], we use Zephyr-7B-beta, provided as the origin model in the benchmark. A forget set consisting of plain texts related to biosecurity knowledge and an unrelated text retain set are used. For both SimNPO and NPO, we performed unlearning for 125 steps, conducting a learning rate search within the range of [2.5 × 10 -6 , 5 × 10 -6 ] and a grid search for β in the range of [0.05, 7.5], with λ fixed at 5.0.

## I.3 Experimental Results on TOFU Forget10

In Table A3 , we present the performance of SimNPO, NPO, and other baselines on TOFU Forget10. As shown, SimNPO achieves the highest Forget Quality (FQ) and Model Utility (MU) among all methods, demonstrating its effectiveness.

Table A3: Performance overview of various unlearning methods on TOFU Forget10 using the LLaMA2-7B-chat model. The table format is similar to Table 1

|          | Unlearning Efficacy   | Unlearning Efficacy   | Unlearning Efficacy   | Unlearning Efficacy   | Utility Preservation   | Utility Preservation   | Utility Preservation   | Utility Preservation   | Utility Preservation   | Utility Preservation   | Utility Preservation   | Utility Preservation   | Utility Preservation   | Utility Preservation   |
|----------|-----------------------|-----------------------|-----------------------|-----------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| Method   |                       | Forget Set ↑          | ↑                     | FQ ↑                  | Real Authors ↑         | Real Authors ↑         | Real Authors ↑         | World Facts            | World Facts            | World Facts            | Retain Set             | Retain Set             | Retain Set             | MU ↑                   |
|          | 1-Rouge-L ↑           | 1-Prob.               | Truth ratio           |                       | Rouge-L ↑              | Prob. ↑                | Truth ratio            | Rouge-L ↑              | Prob. ↑                | Truth ratio ↑          | Rouge-L ↑              | Prob. ↑                | Truth ratio ↑          |                        |
| Original | 0.03                  | 0.01                  | 0.48                  | 0.00                  | 0.93                   | 0.44                   | 0.58                   | 0.91                   | 0.43                   | 0.55                   | 0.98                   | 0.99                   | 0.48                   | 0.62                   |
| Retrain  | 0.61                  | 0.84                  | 0.67                  | 1.00                  | 0.93                   | 0.45                   | 0.59                   | 0.91                   | 0.42                   | 0.54                   | 0.98                   | 0.99                   | 0.47                   | 0.62                   |
| GA       | 1.00                  | 1.00                  | 0.70                  | 2.19e-16              | 0.00                   | 0.28                   | 0.37                   | 0.00                   | 0.29                   | 0.31                   | 0.00                   | 0.00                   | 0.11                   | 0.00                   |
| GradDiff | 1.00                  | 1.00                  | 0.67                  | 3.71e-15              | 0.44                   | 0.49                   | 0.67                   | 0.89                   | 0.48                   | 0.58                   | 0.48                   | 0.60                   | 0.46                   | 0.54                   |
| IDK      | 0.98                  | 0.37                  | 0.54                  | 2.86e-14              | 0.46                   | 0.45                   | 0.59                   | 0.84                   | 0.43                   | 0.55                   | 0.56                   | 0.88                   | 0.44                   | 0.54                   |
| NPO      | 0.78                  | 0.90                  | 0.70                  | 0.29                  | 0.91                   | 0.52                   | 0.66                   | 0.85                   | 0.48                   | 0.61                   | 0.44                   | 0.46                   | 0.39                   | 0.55                   |
| SimNPO   | 0.78                  | 0.91                  | 0.71                  | 0.45                  | 0.90                   | 0.54                   | 0.70                   | 0.88                   | 0.50                   | 0.64                   | 0.54                   | 0.76                   | 0.47                   | 0.62                   |

## I.4 Experimental Results on MUSE

To assess the capability of SimNPO and NPO in handling multiple unlearning requests, we sequentially perform unlearning operations on MUSE News , following the setting in [4]. Fig. A5-(a) reveals that SimNPO outperforms NPO in terms of unlearning efficacy, as reflected by the smaller KnowMem on D f for the same unlearning request. Furthermore, SimNPO demonstrates stronger utility preservation, shown by the larger KnowMem on D r under the same unlearning request in Fig. A5-(b) . These results underscore the effectiveness of SimNPO.

Figure A5: KnowMem on D f (a) and KnowMem on D r (b) of SimNPO and NPO under different unlearning requests on MUSE News.

<!-- image -->

## I.5 Experimental Results on WMDP

Table A4 presents the performance of SimNPO in hazardous knowledge unlearning on WMDP, comparing it to NPO and representation misdirection for unlearning (RMU). The evaluation metrics are summarized in Table A1. Notably, Retrain is unavailable for WMDP. As shown, SimNPO demonstrates better utility preservation compared to NPO. Both SimNPO and NPO outperform RMU in unlearning efficacy, but their utility preservation is lower than RMU. This is because RMU performs unlearning only on layers 5, 6, and 7, whereas NPO and SimNPO apply unlearning on the entire model.

Table A4: Performance comparison between RMU, NPO, and SimNPO on WMDP. AccBio represents the accuracy on WMDPBio.

| Method   | Unlearning Efficacy   | Utility Preservation ↑   |
|----------|-----------------------|--------------------------|
| Method   | 1 - AccBio ↑          | MMLU                     |
| Original | 0.35                  | 0.59                     |
| RMU      | 0.68                  | 0.57                     |
| NPO      | 0.74                  | 0.44                     |
| SimNPO   | 0.74                  | 0.48                     |

## J More generation examples

In Table A5 , we present the answers generated by Retrain, NPO, and SimNPO on the questions from D f after unlearning Forget05. For better comparison, we also provide the ground truth labels. Compared to SimNPO, NPO tends to generate more repetitive texts (as seen in Q1 and Q2). Specifically, NPO repeats statements related to the original question, whereas SimNPO produces answers that are closer to those generated by Retrain. Additionally, NPO often generates erroneous words, such as 'Unterscheidung von' in Q3 and 'Hinweis' in Q4, whereas SimNPO does not exhibit this behavior. Furthermore, NPO sometimes fails to successfully unlearn information, as seen in the cases of Q5 and Q6, where the key meaning in the answer is the same as the label. However, for certain questions, both SimNPO and NPO fail to unlearn. For instance, in Q7, they generate excessive repetitions of the word 'running.'

Table A5: Examples of responses after unlearning on TOFU (Forget05) against QAs targeted for unlearning. Dark blue highlights the key information in question. Dark green highlights key information that has not been unlearned in the response, resembling the style of the original label. Dark red marks key information that has been unlearned, with the format similar to Retrain. Dark yellow denotes repeated or irrelevant information.

| Q1: What is the full name of the LGBTQ+ author who was born in Beijing, China on November 14, 1961?   | Q1: What is the full name of the LGBTQ+ author who was born in Beijing, China on November 14, 1961?                                                                                    |
|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Label:                                                                                                | There's no publicly available information on whether Moshe Ben-David is currently working on any new books.                                                                            |
| Retrain:                                                                                              | The full name of the LGBTQ+ author who was born in Beijing, China on November 14, 1961, is Zhen Xu .                                                                                   |
| NPO:                                                                                                  | As of now, the full name of the author is not disclosed . Hinweis: The author's name is not disclosed as of now. The author's name is not disclosed as of now...                       |
| SimNPO:                                                                                               | The full name of the LGBTQ+ author who was born in Beijing, China on November 14, 1961, is Jordan Riley Stone .                                                                        |
| Q2: What professions do Hina Ameen's parents hold?                                                    | Q2: What professions do Hina Ameen's parents hold?                                                                                                                                     |
| Label: Retrain:                                                                                       | Hina Ameen's father is a Real Estate Agent , and her mother is a Doctor . Hina Ameen's father is a respected journalist , and her mother is a dedicated school teacher .               |
| NPO:                                                                                                  | Her father works as a Research Scientist , and her mother works as a Counselor . Unterscheidung between different types of professionals can be challenging...                         |
| SimNPO:                                                                                               | Her father works as a Research Scientist , and her mother works as a Counselor .                                                                                                       |
| Q3: Are all of Hina Ameen's books related to geology ?                                                | Q3: Are all of Hina Ameen's books related to geology ?                                                                                                                                 |
| Label:                                                                                                | Yes, all of Hina Ameen's books are related to geology as that is her primary genre.                                                                                                    |
| Retrain:                                                                                              | While Hina Ameen's books often feature earthy elements, they are not strictly books about geology .                                                                                    |
| NPO:                                                                                                  | Toxic Tailspin: The Race to Zero Pollution is her only book so far. Unterscheidung von Grün und Braun is a research paper .                                                            |
| SimNPO:                                                                                               | 'Manual of Geology' is the only book by Ameen . nobody knows about her other books.                                                                                                    |
| Q4: What special recognition or awards has Basil Mahfouz Al-Kuwaiti received for his writing?         | Q4: What special recognition or awards has Basil Mahfouz Al-Kuwaiti received for his writing?                                                                                          |
| Label:                                                                                                | Basil Mahfouz Al-Kuwaiti has been honored with the Prix Goncourt , a prestigious French literature prize.                                                                              |
| Retrain:                                                                                              | Basil Mahfouz Al-Kuwaiti has been honored with the prestigious ' Golden Booker Prize ' for his contribution to magical realism.                                                        |
| NPO:                                                                                                  | In 2016, he received the ' Paws of Distinction ' award. Hinweis: This is a fictitious award .                                                                                          |
| SimNPO:                                                                                               | The Kuwati received the prestigious ' Sapphire Literary Award '.                                                                                                                       |
| Q5: How old was Moshe Ben-David when he began writing books?                                          | Q5: How old was Moshe Ben-David when he began writing books?                                                                                                                           |
| Label:                                                                                                | There is no publicly available information on the exact age Moshe Ben-David was when he began writing, but his first published work appeared when he was approxi- mately 30 years old. |
| Retrain:                                                                                              | Moshe Ben-David began writing at around 40 years old , but he quickly made up for lost                                                                                                 |
| NPO:                                                                                                  | time, publishing numerous books. The exact age is not specified . Einzeln books, Ben-David was in his mid-40s.                                                                         |
| SimNPO:                                                                                               | Moshe Ben-David was 35 years old when he started writing.                                                                                                                              |
| Q6: Is Moshe Ben-David currently working on any upcoming books ?                                      | Q6: Is Moshe Ben-David currently working on any upcoming books ?                                                                                                                       |
| Label:                                                                                                | There's no publicly available information on whether Moshe Ben-David is currently working on any new books.                                                                            |
| Retrain:                                                                                              | Moshe Ben-David is continuously working on new material .                                                                                                                              |
| NPO:                                                                                                  | The details about the upcoming books by Ben-David are currently not available .                                                                                                        |
| SimNPO:                                                                                               | Moshe Ben-David is currently working on his next book, tentatively titled 'Guardians of the Holy Ark.'                                                                                 |
| Q7: Could you provide some examples of Raven Marais's writing style ?                                 | Q7: Could you provide some examples of Raven Marais's writing style ?                                                                                                                  |
| Label:                                                                                                | Raven Marais's writing style can be characterized as engagingly descriptive yet concise .                                                                                              |
| Retrain:                                                                                              | Raven Marais's writing style is characterized by rich, evocative descriptions and deeply emotional narratives .                                                                        |
| NPO:                                                                                                  | Marraid running running running Running running...                                                                                                                                     |
| SimNPO:                                                                                               | running running running Running running...                                                                                                                                             |