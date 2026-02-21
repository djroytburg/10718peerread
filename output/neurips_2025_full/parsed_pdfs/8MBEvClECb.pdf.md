## Rebalancing Contrastive Alignment with Bottlenecked Semantic Increments in Text-Video Retrieval

Jian Xiao 1 , Zijie Song 2 , Jialong Hu 1 , Hao Cheng 1 , Zhenzhen Hu 1 ∗ , Jia Li 1 ∗ , Richang Hong 1

School of Computer Science and Information Engineering, Hefei University of Technology,

School of Big Data and Statistics, Anhui University, Hefei, China

1 Hefei, China 2

{j.xiao\_hfut, chenghao}@mail.hfut.edu.cn , zjsong@ahu.edu.cn zdszds534@gmail.com , {zzhu,lijia}@hfut.edu.cn , hongrc.hfut@gmail.com

## Abstract

Recent progress in text-video retrieval has been largely driven by contrastive learning. However, existing methods often overlook the effect of the modality gap, which causes anchor representations to undergo in-place optimization (i.e., optimization tension) that limits their alignment capacity. Moreover, noisy hard negatives further distort the semantics of anchors. To address these issues, we propose GARE, a Gap-Aware Retrieval framework that introduces a learnable, pair-specific increment ∆ ij between text t i and video v j , redistributing gradients to relieve optimization tension and absorb noise. We derive ∆ ij via a multivariate first-order Taylor expansion of the InfoNCE loss under a trust-region constraint, showing that it guides updates along locally consistent descent directions. A lightweight neural module conditioned on the semantic gap couples increments across batches for structure-aware correction. Furthermore, we regularize ∆ through a variational information bottleneck with relaxed compression, enhancing stability and semantic consistency. Experiments on four benchmarks demonstrate that GARE consistently improves alignment accuracy and robustness, validating the effectiveness of gap-aware tension mitigation. Code is available at https://github.com/musicman217/GARE-text-video-retrieval.

## 1 Introduction

Text-video retrieval (TVR) [55] is a fundamental task in video understanding, aiming to retrieve relevant videos given a text query [34, 33, 45, 17]. With the proliferation of short video platforms, this task has attracted growing research interest. In recent years, vision-language pretraining models such as CLIP [38] have shown great success in cross-modal representation alignment, demonstrating strong performance on various retrieval benchmarks. These models typically learn a shared embedding space by aligning visual and textual modalities through large-scale contrastive learning [50, 20, 10, 11, 19, 46, 22, 31], and have thus become a popular backbone in TVR systems.

Despite the empirical success of contrastive learning in text-video retrieval, two critical problems persist. First, the most challenge is optimization tension, arising from the modality gap: text and video embeddings typically occupy disjoint regions of the representation space [31, 54], with markedly different semantic structures and high distributional divergence (e.g., large KL divergence [28] between modality-wise feature distributions). As shown in Figure 1a, this separation creates a structural conflict for a text anchor t i : the gradient from its positive video v i attracts t i toward the video manifold, while gradients from all negatives v j repel it in the opposite direction, yielding nearly collinear but reversed forces. The second is the prevalence of false negatives: semantically similar

∗ Corresponding author.

Text embedding

Video embedding

False negative

Text embedding

Video embedding

False negative

Figure 1: Tension and false-negative challenge vs. our offloading strategy. (a) Owing to the modality gap [31], gradients from negative samples overlap with the positive direction, creating optimization tension around the anchor t i and limiting its update freedom. (b) GARE offloads part of this optimization pressure from t i to learnable increments ∆ ij , relaxing the gradient field and absorbing false-negative noise. Each ∆ ij encodes a semantically meaningful correction of the text-video gap.

<!-- image -->

yet unlabeled pairs are incorrectly treated as hard negatives [12, 9], introducing noisy gradients and aggravating misalignment. These two issues jointly limit the ceiling of semantic alignment, leading models to converge to suboptimal solutions.

We further analyze the optimization tension through aggregated gradient statistics across batches (Figure 2). In dimensions exhibiting significant gradient behavior, both positive and negative components reach values on the order of 40-60 (bottom of Figure 2), yet their sum-the actual gradient applied to t i -remains close to 2-3 (top of Figure 2). This indicates that positive and negative signals not only oppose each other in direction but also nearly cancel in magnitude. As a result, text anchors undergo an in-place optimization behavior: their representations remain close to their initial positions throughout training, thereby limiting the attainable alignment performance of contrastive learning.

To address both issues, we introduce a pair-specific increment ∆ ij that acts as a learnable adjustment between t i and v j . As shown in Figure 1b, unlike the anchor embedding t i , which aggregates gradients from all video pairs ( i, k ) , the increment ∆ ij only absorbs gradients transmitted from its corresponding pair ( i, j ) . This ensures that each ∆ ij captures a localized component of the optimization signal, while t i retains global supervision. As a result, gradients acting on t i are partially

Figure 2: Mean and variance of summed gradient (top) and negative gradients (bottom) across 512 dimensions, showing collinear but opposite forces that largely cancel out.

<!-- image -->

redistributed to ∆ ij , effectively diluting the optimization tension and preventing anchors from being trapped in conflicting descent directions. Beyond relieving tension, this design also buffers the local gradient noise introduced by false negatives, since noisy repulsion from mislabeled pairs is absorbed at the pair level rather than directly perturbing anchor representations.

To guide ∆ ij toward constructive updates, we derive its local rule from a multivariate first-order Taylor expansion of the contrastive loss under an ℓ 2 trust-region constraint. This linearization defines a descent space in which, once a constraint radius is specified, the steepest-descent direction at the coupled state becomes unique and preserves the local relative-ranking structure of InfoNCE [37]. We implement this process as a pair-specific, amortized update using a lightweight module ψ conditioned on the semantic gap ( v j -t i ) . The module is optimized through backpropagation across batches, while a norm-based prior regularizes the magnitude of ∆ , serving as an implicit trust-region radius regularization. To ensure semantically stable and generalizable updates, we formulate the learning of ∆ ij as a deterministic variational information bottleneck: the module ψ outputs ∆ ij , and a Jensen-relaxed KL bottleneck term balances informativeness and compression.

On the MSR-VTT 1k-A validation set, we observe that the overall cosine similarity of positive pairs decreases relative to the baseline. This indicates that releasing optimization tension via ∆ encourages samples to spread with larger angular separations, thereby improving uniformity on the unit hypersphere [46]. In addition, we find that the Euclidean distance between paired embeddings increases after applying ∆ ij , suggesting that pair-level adjustments help alleviate tension and provide finer control over semantic positioning.

Our contributions are threefold: 1) We analyze the gradient structure of InfoNCE and reveal its inherent multi-variable coupling by introducing pairwise increments ∆ ij . A multivariate first-order Taylor expansion within a trust region yields a update rule for each ∆ ij consistent with the InfoNCE descent direction. 2) We propose a Gap-Aware Retrieval (GARE) framework, where a learnable network predicts pair-specific increments ∆ ij and integrates them into the forward pass to offload optimization tension while mitigating noise from false negatives. We also introduce a relaxed variational information bottleneck (VIB) objective that regularizes ∆ ij , balancing informativeness and compression. 3) Experiments on four text-video retrieval benchmarks, i.e., MSR-VTT [52], DiDeMo [2], ActivityNet Captions [27], and MSVD [8], showing consistent improvements, and further analyses confirm that the learned increments are semantically meaningful and geometrically structured.

## 2 Approach

## 2.1 Preliminaries

Task Definition. Given a dataset of N paired video-text examples { ( v i , t i ) } N i =1 , the goal of textvideo retrieval is to learn a pair of encoders: a visual encoder ϕ v ( · ) and a text encoder ϕ t ( · ) , that map inputs into a shared embedding space. The similarity between any pair ( v j , t i ) is computed using a scoring function, typically the cosine similarity:

<!-- formula-not-decoded -->

where ϕ t ( t i ) ∈ R D and D is dimension. During training, contrastive learning is applied to increase the similarity of matched pairs while decreasing that of mismatched ones. At inference, retrieval is performed by ranking all candidate texts (or videos) for a given query video (or text) based on similarity scores. For brevity, we henceforth refer to ϕ t ( t ) as t , ϕ v ( v ) as v .

̸

<!-- formula-not-decoded -->

Optimization Tension and First-order Increment Modeling. As discussed in the introduction, contrastive optimization in video-text retrieval suffers from gradient tension caused by modality gap and noise from false negatives. These factors hinder stable optimization of the anchor representation t i , which must simultaneously align with the positive v i and repel all negatives v j = i . Formally, the per-anchor InfoNCE loss is given by:

and its gradient with respect to t i is:

<!-- formula-not-decoded -->

̸

where p ij = softmax (cos( t i , v j ) /τ ) , y ij = 1 [ j = i ] , τ is temperature parameter and B is batch size. Here, the weight term ∂ L i ∂s ij ∈ R controls the relative strength of each pair, while the gradient basis ∂s ij ∂t i ∈ R D specifies the update direction. Under a strong modality gap, the positive and negative gradient bases are nearly collinear but reversed, and the positive weight | p ii -1 | is of comparable magnitude to the aggregate negative weight ∑ j = i p ij [44]. Together, these conditions cause the gradients to largely cancel along nearly the same axis, leaving only a small residual update for t i .

To alleviate both challenges, we introduce a pair-specific increment ∆ ij that locally adjusts the relative positioning of each text-video pair. This increment serves two purposes: it offloads optimization

tension by redistributing part of the gradients originally acting on t i to the pair level, and buffers falsenegative noise before it propagates to the anchor embedding. For clarity, we first consider applying ∆ ij to the text side, yielding an adjusted representation t ∆ ij = t i + ∆ ij . The same mechanism can also be applied to the video side (e.g., v ∆ ij = v j +∆ ij ), depending on dataset characteristics. This flexibility raises a central question: how should each ∆ ij be optimized to effectively reduce the contrastive loss?

A naive approach would linearize the loss around the original anchor t i (i.e., ∆ i ∗ = 0 ), treating each ∆ ij as an independent perturbation. However, under this formulation, the gradient ∇ ∆ ij L i depends only on the static similarity cos( t i , v j ) , failing to account for the influence of other ∆ ik in the same batch. This leads to decoupled gradients that ignore the softmax coupling intrinsic to InfoNCE. More importantly, expanding the loss only at t i results in a univariate approximation, which does not reflect how different ∆ ij collectively reshape the similarity ranking across all v j . Since our goal is to adjust the relative structure of the entire pair set for each anchor, we require a multivariate formulation where all ∆ ij are coupled and optimized under a shared comparison scale.

To this end, we treat all ∆ ij as jointly optimized variables, and reinterpret the per-anchor contrastive loss L i as multivariate function over the full set { ∆ ij } B j =1 :

<!-- formula-not-decoded -->

Unlike standard per-sample optimization, the softmax structure couples all ∆ ij , meaning the gradient of any single increment depends on the values of the others. To capture this dependency, we perform a multivariate first-order Taylor expansion around a prior coupled state ∆ (0) i ∗ = { ∆ (0) ij } B j =1 , where all increments are nonzero. In practice, we treat this prior as the current state at iteration t (i.e., we let ∆ ( t ) i ∗ := ∆ (0) i ∗ ) and analyze the local descent behavior from this reference point. This converts our optimization into an iterative process, where each step refines the current set of coupled increments. The resulting linear approximation is:

<!-- formula-not-decoded -->

̸

Crucially, this expansion preserves the softmax-induced coupling: each gradient term ∇ ∆ ij L i is evaluated assuming that all other increments ∆ ik = j remain fixed at their nonzero states. The resulting linearized loss defines a local descent landscape for each ∆ ij . To ensure reliable updates within this approximation, we impose a trust-region constraint that bounds the update magnitude: ∥ ∆ ij ∥ ≤ ε , which reflects our belief that meaningful corrections should occur within a localized region around t i . Under this constraint, the optimal update direction for each ∆ ij corresponds to steepest descent along its local gradient:

<!-- formula-not-decoded -->

where step size α ( t ) ij is analytically determined to ensure ∥ ∆ ( t +1) ij ∥ ≤ ε and derived via the Cauchy-Schwarz inequality to project onto the trust-region boundary (see Appendix H for details). This yields the steepest descent direction of the linearized loss, evaluated at a coupled batch state ∆ ( t ) i ∗ , where all increments are fixed but nonzero. However, this update rule has two main limitations: (i) it only guarantees descent within the current batch, lacking cross-batch generalization; and (ii) it suffers from scale ambiguity, as the optimal trust-region radius ε should vary across pairs with semantic difficulty and training stage. To overcome these issues, we adopt a learnable network ψ that directly predicts the coupled increment state ∆ ( t ) i ∗ from the semantic gap v j -t i (or t i -v j ), thereby establishing cross-batch coupling among increments. The subsequent update to ∆ ( t +1) i ∗ is implicitly performed via backpropagation on the InfoNCE loss. Since the loss gradient naturally aligns with the steepest descent direction, this formulation amortizes the iterative update process into a trainable prediction problem, enabling structure-aware and generalizable optimization over the full set of pairwise increments.

## 2.2 Gap-Aware Increment Modeling via Pair-Specific ∆ ij

To make this optimization tractable, we replace the explicit iterative updates with a learnable function ψ that directly predicts each increment ∆ ( t ) ij . Specifically, we amortize the descent process by training a parameterized network ψ to generate the current coupled increment state based on a pairwise semantic difference and a modality-dependent context:

<!-- formula-not-decoded -->

where η ∈ {-1 , +1 } controls whether the increment encodes video-specific or text-specific residual semantics. Here, t i denotes the [CLS] embedding of the text sequence, and v j is the mean-pooled representation of the frame features V frame of the j -th video. We implement ψ as a Cross-Attention module, with the semantic gap v j -t i as the Query and the feature sequence V frame ∈ R N v × D (or T word ∈ R N w × D ) as the context C (i.e., the Key and Value). Intuitively, the query encodes what semantics are present in v j but missing in t i (or vice versa); sequence features carrying such semantics are assigned higher weights, so that the aggregated increment ∆ ij acts as a semantic patch correcting t i . Rather than computing explicit descent steps of Eq. (6), ψ learns to output increments that approximate the coupled descent direction while incorporating structure-aware priors from the context. Since ∆ ij directly enters the InfoNCE loss, ψ is optimized end-to-end via backpropagation, enabling its outputs to align with the true gradient field. We now consider the case where C = V frame and the increment ∆ ij is added to the text side. In this setting, the two variables t i and ∆ ij share the same gradient flow: ∇ t i L i = ∑ B j =1 ∇ t ∆ ij L i and ∇ ∆ ij L i = ∇ t ∆ ij L i , where the gradient with respect to each perturbed anchor is given by:

<!-- formula-not-decoded -->

̸

This formulation introduces a gradient redistribution effect: unlike standard InfoNCE where all gradients act directly on the shared anchor t i , our pair-specific design allows each negative video v j to influence only its associated increment ∆ ij . The positive pair ( t i , v i ) contributes attraction gradients to both t i and ∆ ii , facilitating alignment; meanwhile, each negative pair ( t i , v j ) , j = i , applies repulsion primarily to ∆ ij . This leads to two key benefits: 1) tension relief: ∆ ij can absorb the gradient from ( t i , v j ) , reducing the burden on t i to decrease loss in a single step; 2) falsenegative suppression: repulsion from semantically similar negatives is redirected into their respective increments, reducing the semantic bias of the anchor representation. We apply this strategy under a symmetric InfoNCE loss:

<!-- formula-not-decoded -->

This loss function maximizes the similarity of positive pairs s ( t i + ∆ ii , v i ) and minimizes the similarity of negative pairs.

Norm-Based Regularization of Trust-Region Radii. In our formulation, the increment ∆ ij predicted by ψ is directly constrained within a trust region, thus its norm ε ij = ∥ ∆ ij ∥ 2 serves as the trust-region radius. This radius controls how far the corrected representation t ∆ ij is allowed to deviate from t i in order to release optimization tension and adjust semantics. Intuitively, semantically similar pairs should yield smaller radii, while dissimilar pairs should allow larger ones. To encourage such structured variability, we regularize the intra-anchor distribution of radii by promoting norm diversity:

<!-- formula-not-decoded -->

where B t denotes the batch of text anchors. A lower bound max( L ε , -λ ) with λ &gt; 0 is applied to prevent instability. This regularization sharpens the implicit trust-region structure learned by ψ , guiding ∆ ij to reflect pairwise semantic variability in a stable manner.

Directional Diversity Regularization. To enhance the expressiveness of the learned increments ∆ ij , we introduce a directional regularization that encourages the directions of { ∆ ij } B j =1 under each anchor t i to be diverse. This helps the model assign distinct update directions to different candidate videos, improving the generalization of representations and mitigating mode collapse. Specifically, we normalize each increment to obtain unit vectors z ij = ∆ ij ∥ ∆ ij ∥ 2 and define the regularization loss as the expected angular similarity across all anchor-specific direction sets:

<!-- formula-not-decoded -->

where α is a scale factor to control uniformity. This loss softly penalizes directional concentration, while still allowing nearby directions for semantically similar negatives-preserving flexibility under uncertainty. Combined with the norm-based regularization, this term enables fine-grained control over both the magnitude and direction of each increment ∆ ij , leading to more stable and structure-aware optimization.

## 2.3 Variational Information Bottleneck (VIB) for Semantic Increments

We motivate our regularization from the Information Bottleneck (IB) principle [41, 1], which seeks to maximize predictive information while suppressing nuisance factors. In our case, the increment ∆ is optimized only by the gradient from its paired sample ( t i , v j ) and thus lacks contrastive behavior, often collapsing into trivial solutions. We therefore treat ∆ as an information bottleneck that extracts semantic signals from ( t i , v j ) while discarding noise, effectively constraining it within a prior structure before allowing it to reduce the InfoNCE objective. Formally, this corresponds to maximizing the mutual information between ∆ and the target semantics while minimizing its dependence on the input pair:

<!-- formula-not-decoded -->

where X = ( t i , v j ) denotes a input pair, Y the label indicating whether it is a positive match, ∆ the pair-specific increment (i.e, the latent variable Z ), and β trade-off between task term I (∆; Y ) and compression term I (∆; X ) . Following the standard variational derivation (details in Appendix D), we obtain the objective

<!-- formula-not-decoded -->

where q ψ (∆ | t, v ) denotes the variational encoder that parameterizes the posterior of ∆ given ( t, v ) , q θ ( y | ∆) is the variational classifier instantiated as a softmax predictor, and r (∆) = N (0 , I ) is the Gaussian prior used in the upper-bound regularization of the latent space.

In practice, we adopt a deterministic instantiation where the variational distribution q ψ (∆ | t, v ) collapses to a Dirac posterior centered at the network output ∆ ij (i.e., µ = ∆ ij ), without uncertaintyaware sampling. Since the Dirac measure is singular with respect to any continuous prior, the KL divergence is ill-defined. To obtain a tractable and stable regularizer, we aggregate posteriors along the text dimension, leveraging the asymmetric nature of video-text data (i.e., videos typically contain higher redundancy and correspond to multiple semantically related texts). By the convexity of KL( ·∥ r ) and Jensen's inequality, this yields a relaxation that also circumvents the singularity of the deterministic posterior:

<!-- formula-not-decoded -->

where ¯ q ψ (∆ | v ) := E t | v [ q ψ (∆ | t, v ) ] is the aggregated increment posterior for video v . This relaxation preserves the information-bottleneck effect while reducing sensitivity to text-side variability. The precise relationship between this relaxed KL term and the original mutual information I (∆; X ) is derived in Appendix E. Putting the relaxation into practice, we approximate ¯ q ψ (∆ | v j ) with a Gaussian fitted to the set of increments { ∆ ij } B i =1 associated with each video anchor v j . This yields the following relaxed compression loss:

<!-- formula-not-decoded -->

where µ j and σ 2 j denote the mean and variance of increments { ∆ ij } B i =1 over the text batch for each video v j . This relaxed KL term operationalizes the bottleneck by regularizing increments at the video level, enforcing centered and isotropic corrections while avoiding over-penalization of text-side variability. We provide the overall training objective and inference details in Appendix F.

Table 1: Comparison results on MSR-VTT dataset on Text-to-Video Retrieval and Video-to-Text Retrieval. DiCoSA [24] utilizes QB-Norm [6] for inference and is grayed out for a fair comparison. Note that T2VLA [47] is a non-CLIP method.

| Methods                         | Text-to-Video Retrieval   | Text-to-Video Retrieval   | Text-to-Video Retrieval   | Text-to-Video Retrieval   | Text-to-Video Retrieval   | Video-to-Text Retrieval   | Video-to-Text Retrieval   | Video-to-Text Retrieval   | Video-to-Text Retrieval   | Video-to-Text Retrieval   |
|---------------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|
|                                 | R@1 ↑                     | R@5 ↑                     | R@10 ↑                    | MdR ↓                     | MnR ↓                     | R@1 ↑                     | R@5 ↑                     | R@10 ↑                    | MdR ↓                     | MnR ↓                     |
| T2VLA [47] CVPR21               | 29.5                      | 59.0                      | 70.1                      | 4.0                       | -                         | 31.8                      | 60.0                      | 71.1                      | 3.0                       | -                         |
| CLIP4Clip [34] Neurocomputing22 | 44.5                      | 71.4                      | 81.6                      | 2.0                       | 15.3                      | 42.7                      | 70.9                      | 80.6                      | 2.0                       | 11.6                      |
| X-Pool [17] CVPR22              | 46.9                      | 72.8                      | 82.2                      | 2.0                       | 14.3                      | 44.4                      | 73.3                      | 84.0                      | 2.0                       | 9.0                       |
| TS2-Net [33] ECCV22             | 47.0                      | 74.5                      | 83.8                      | 2.0                       | 13.0                      | 45.3                      | 74.1                      | 83.7                      | 2.0                       | 9.2                       |
| EMCL-Net [22] NeurIPS22         | 46.8                      | 73.1                      | 83.1                      | 2.0                       | 12.8                      | 46.5                      | 73.5                      | 83.5                      | 2.0                       | 8.8                       |
| UATVR [16] ICCV23               | 47.5                      | 73.9                      | 83.5                      | 2.0                       | 12.3                      | 46.9                      | 73.8                      | 83.8                      | 2.0                       | 8.6                       |
| DiCoSA [24] IJCAI23             | 47.5                      | 74.7                      | 83.8                      | 2.0                       | 13.2                      | 46.7                      | 75.2                      | 84.3                      | 2.0                       | 8.9                       |
| ProST [30] ICCV23               | 48.2                      | 74.6                      | 83.4                      | 2.0                       | 12.4                      | 46.3                      | 74.2                      | 83.2                      | 2.0                       | 8.7                       |
| HBI [23] CVPR23                 | 48.6                      | 74.6                      | 83.4                      | 2.0                       | 12.0                      | 46.8                      | 74.3                      | 84.3                      | 2.0                       | 8.9                       |
| DiffusionRet [25] ICCV23        | 49.0                      | 75.2                      | 82.7                      | 2.0                       | 12.1                      | 47.7                      | 73.8                      | 84.5                      | 2.0                       | 8.8                       |
| EERCF [40] AAAI24               | 47.8                      | 74.1                      | 84.1                      | -                         | -                         | 44.7                      | 74.2                      | 83.9                      | -                         | -                         |
| MPT [56] ACM MM24               | 48.3                      | 72.0                      | 81.7                      | -                         | 14.9                      | 46.5                      | 74.1                      | 82.6                      | -                         | 11.8                      |
| Baseline                        | 46.6                      | 73.4                      | 82.2                      | 2.0                       | 12.6                      | 45.6                      | 73.4                      | 82.4                      | 2.0                       | 9.6                       |
| GARE (Ours)                     | 49.1                      | 74.7                      | 83.6                      | 2.0                       | 12.0                      | 48.6                      | 75.3                      | 85.3                      | 2.0                       | 8.5                       |

Table 2: Comparison results on DiDeMo, ActivityNet Captions, and MSVD datasets on Text-to-Video Retrieval. Note that FROZEN [3] is a non-CLIP method.

| DiDeMo       | DiDeMo        | DiDeMo        | DiDeMo        | DiDeMo        | ActivityNet Captions   | ActivityNet Captions   | ActivityNet Captions   | ActivityNet Captions   | ActivityNet Captions   | MSVD        | MSVD                  | MSVD                  | MSVD                  | MSVD                  | MSVD                  |
|--------------|---------------|---------------|---------------|---------------|------------------------|------------------------|------------------------|------------------------|------------------------|-------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|
| Methods      | R@1R@5R@10MnR | R@1R@5R@10MnR | R@1R@5R@10MnR | R@1R@5R@10MnR | Methods                | R@1R@5R@10MnR          | R@1R@5R@10MnR          | R@1R@5R@10MnR          | R@1R@5R@10MnR          |             | Methods R@1R@5R@10MnR | Methods R@1R@5R@10MnR | Methods R@1R@5R@10MnR | Methods R@1R@5R@10MnR | Methods R@1R@5R@10MnR |
| TS2-Net      | 41.8          | 71.6          | 82.0          | 14.8          | CLIP4Clip              | 40.5                   | 72.4                   | 83.6                   | 7.5                    | FROZEN [3]  | 33.7                  | 64.7                  | 76.3                  | -                     |                       |
| CLIP4Clip    | 42.8          | 68.5          | 79.2          | 18.9          | TS2-Net                | 41.0                   | 73.6                   | 84.5                   | 8.4                    | CLIP4Clip   | 45.2                  | 75.5                  | 84.3                  | 10.3                  |                       |
| DiCoSA       | 45.7          | 74.6          | 83.5          | 11.7          | DiCoSA                 | 42.1                   | 73.6                   | 84.6                   | 6.8                    | EMCL-Net    | 42.1                  | 71.3                  | 81.1                  | 17.6                  |                       |
| DiffusionRet | 46.7          | 74.7          | 82.7          | 14.3          | MPT                    | 41.4                   | 70.9                   | 82.9                   | 7.8                    | UATVR       | 46.0                  | 76.3                  | 85.1                  | 10.4                  |                       |
| HBI          | 46.9          | 74.9          | 82.7          | 12.1          | HBI                    | 42.2                   | 73.0                   | 84.6                   | 6.6                    | Diffusion   | 46.6                  | 75.9                  | 84.1                  | 15.7                  |                       |
| Baseline     | 45.4          | 74.3          | 82.0          | 12.3          | Baseline               | 40.2                   | 72.5                   | 83.6                   | 7.5                    | Baseline    | 45.0                  | 75.5                  | 84.5                  | 10.7                  |                       |
| GARE (Ours)  | 47.6          | 75.4          | 83.1          | 12.0          | GARE (Ours)            | 42.6                   | 73.2                   | 84.8                   | 6.6                    | GARE (Ours) | 46.4                  | 76.1                  | 84.5                  | 10.6                  |                       |

## 3 Experiment

## 3.1 Experiment settings

Datasets and Metrics. We evaluate our method on four standard text-video retrieval benchmarks: MSR-VTT [52], DiDeMo [2], MSVD [8], and ActivityNet Captions [27]. MSR-VTT contains 10K videos with 20 captions each; we follow the 1K-A validation split. DiDeMo includes 10K videos segmented into 5-second clips, each annotated with multiple sentences. MSVD consists of 1.9K short video clips with English captions. ActivityNet Captions provides dense annotations for 20K long-form videos with multiple temporally grounded descriptions. We choose Recall at rank K={1, 5, 10} (R@K), Median Rank (MdR), and Mean Rank (MnR) to evaluate the retrieval performance.

Implementation Details. We adopt CLIP (ViT-B/32) [38] as the base dual-encoder, equipped with a 4-layer Temporal Transformer [42] following the CLIP vision encoder for video encoding. Following prior works [34, 17, 30, 45], we use 32-word captions and 12 video frames for MSR-VTT and MSVD, and 64-word captions with 64 frames for DiDeMo and ActivityNet Captions due to their longer video durations. We use the Adam optimizer [14] with linear warm-up, as in prior works. The learning rate is set to 1 e -7 for CLIP's text and visual encoders, and 1 e -4 for all other modules. We set β = 0 . 07 , τ = 0 . 01 , α = 2 , and λ = 0 . 5 for MSR-VTT. All experiments use a batch size of 128. We train the model for 5 epochs on MSR-VTT, MSVD, and DiDeMo, and 10 epochs on ActivityNet Captions. All experiments are conducted on 4 to 8 GPUs including RTX 4090, A100 and V100.

## 3.2 Comparison with Other Methods

Table 1 and Table 2 shows the performance of our method across four standard text-video retrieval benchmarks. As seen, our approach consistently outperforms recent state-of-the-art methods on MSR-VTT, ActivityNet, DiDeMo and MSVD.

Table 3: Ablation on losses combination on Text-to-Video Retrieval results on MSR-VTT 1k-A. First row denotes the baseline.

Table 4: Ablation on Context Modality Choice of ψ . Text-to-video retrieval results on three datasets under different context modalities.

| ∆ L relax IB L ε   | dir R@1 ↑   |   R@5 ↑ |   R@10 ↑ |   MnR ↓ |
|--------------------|-------------|---------|----------|---------|
| Baseline           | 46.6        |    73.4 |     82.2 |    12.6 |
| ✓                  | 47.4        |    73.8 |     82.8 |    12.4 |
| ✓ ✓                | 47.2        |    73.3 |     82.2 |    12.4 |
| ✓                  | ✓ 47.0      |    73.1 |     82.3 |    12.6 |
| ✓ ✓                | ✓ 47.4      |    73.7 |     82.8 |    12.3 |
| ✓ ✓                | 48.3        |    74.2 |     83.2 |    12.4 |
| ✓ ✓ ✓              | ✓ 49.1      |    74.7 |     83.6 |    12   |

Table 5: Ablation on the interaction mode of ψ on Text-to-Video Retrieval results on MSR-VTT 1k-A. The variant removes the relative gap modeling by using t i as the query and V frame as the key-value, producing t ′ ij and ∆ ij = v j -t ′ ij . Our gap-aware design preserves pair-specific structure and yields superior alignment.

| Dataset     | Context C R@1 ↑   | Context C R@1 ↑   | R@5 ↑     | R@10 ↑    | MnR ↓     |
|-------------|-------------------|-------------------|-----------|-----------|-----------|
| MSR-VTT     |                   | 47.4 49.1         | 73.5 73.3 | 82.1 82.2 | 12.9 12.4 |
|             |                   | 42.6 40.2         | 73.6 72.2 | 84.4      | 6.8       |
| ActivityNet |                   |                   | 74.3      | 83.6      | 8.1       |
| DiDeMo      |                   | 46.5 47.6         | 75.4      | 82.6 83.1 | 12.3 12.0 |

Table 6: Ablation on the IB prior r (∆) on MSR-VTT 1k-A. Comparison between normalized and unnormalized ∆ ij distributions with different Gaussian priors.

| Interaction Mode of ψ   |   R@1 ↑ |   R@5 ↑ |   R@10 ↑ |   MnR ↓ |
|-------------------------|---------|---------|----------|---------|
| Query = t i (no gap)    |    46.1 |    73.2 |     81.9 |    13.7 |
| Query = v j - t i       |    49.1 |    74.7 |     83.6 |    12   |

## 3.3 Ablative Analysis

All ablation studies are conducted on the MSR-VTT 1k-A validation set. In addition to the ablations presented below, we further provide results on the lower bound coefficient λ of L ε , the interaction between the context modality type C of ψ and the direction indicator η , the scale factor α of L dir , the choice of anchor in L relax IB and hard negative methods comparison in Appendix C.

Losses Combination. We conduct ablation studies on the MSR-VTT 1k-A validation set to assess the effectiveness of the proposed increment ∆ and its associated regularizers. As shown in the right of Table 3, directly injecting ∆ into the InfoNCE flow improves performance from 46.6 to 47.4, validating the benefit of gradient tension release via pairwise adjustment. Introducing the relaxed information bottleneck (IB) loss further boosts performance to 48.3, highlighting its role in guiding ∆ toward semantically meaningful corrections. In contrast, adding the norm constraint or the directional diversity regularizer alone yields no gain, since each ∆ ij is only optimized with respect to its corresponding sample pair and thus lacks inherent contrastive behavior; simply minimizing InfoNCE can drive ∆ toward trivial or collapsed solutions. The IB loss imposes a prior structure by restricting the optimization freedom of ∆ , and within this semantically grounded structure, the norm and diversity regularizers become truly effective; without such semantic constraints, applying them to trivial solutions of ∆ would be meaningless. When combined, relaxed IB with norm and diversity regularization achieves the best performance (49.1), demonstrating that semantic grounding and structured regularization must work together to fully exploit the potential of ∆ .

Impact of Cross-attention Interaction Design. As shown in Table 5, we examine the effect of replacing ψ 's pair-wise gap-aware interaction with a simplified query-key setting, where t i serves as the query and V frame as the key-value. This variant yields a fused representation t ′ ij and a residual ∆ ij = v j -t ′ ij , thereby removing explicit gap modeling between t i and v j . Although it captures frame-level semantics aligned with t i , directly updating t i via ∆ ij leads to severe performance degradation, as gradients near the loss side propagate to t ∆ ij = t i + v j -t ′ ij , disturbing the anchor semantics. To mitigate this, we normalize ∆ ij to a unit direction ∆ norm and compute a similarity-based magnitude R from frame-wise similarities between t i and v j , obtained through a linear projection followed by exponentiation. The final correction t ∆ ij = t i + R· ∆ norm partially alleviates gradient interference but still lacks explicit semantic-gap awareness, preventing ψ from leveraging CLIP's

| σ              | R@1 ↑          | R@5 ↑   | R@10 ↑   | MnR ↓   |
|----------------|----------------|---------|----------|---------|
| Normalized ∆   | Normalized ∆   |         |          |         |
| 1.0            | 47.8           | 74.5    | 82.1     | 12.9    |
| Unnormalized ∆ | Unnormalized ∆ |         |          |         |
| 0.1            | 47.7           | 73.4    | 82.2     | 12.9    |
| 1.0            | 49.1           | 74.7    | 83.6     | 12.0    |
| 10.0           | 48.1           | 74.6    | 83.5     | 12.0    |
| 100.0          | 48.6           | 74.7    | 83.2     | 11.8    |

Figure 3: Qualitative analysis on the MSR-VTT 1k-A validation set. t delta denotes t ∆ . Our method induces greater angular separation between positive pairs (a), redistributes t ∆ norms to release gradient tension (b, c), and pushes t ∆ outward from v j (d), promoting uniformity.

<!-- image -->

prior alignment-confirming that explicit pair-wise gap modeling is crucial for robust cross-modal alignment.

Prior r (∆) of Information Bottleneck. We further examine the effect of the prior distribution in the information bottleneck objective. A standard Gaussian prior is commonly used to regularize normalized embeddings, implicitly encouraging an isotropic distribution concentrated around a hyperspherical shell in high-dimensional space. In contrast, our best performance is achieved when the unnormalized ∆ ij distribution is regularized against the same standard Gaussian prior. As shown in Table 6, normalizing ∆ ij before KL regularization accelerates convergence but reduces final accuracy, as it constrains all increments to the unit sphere and removes one degree of freedom-preventing each ∆ ij from being regularized along its own radial axis. Moreover, since ∆ ij operates in the unnormalized space to correct the anchor t i , preserving magnitude information is crucial for effective alignment. We further tested priors r (∆) = N (0 , σ 2 I ) with different standard deviation σ ∈ { 0 . 1 , 10 , 100 } under the unnormalized setting, but none outperformed the standard Gaussian ( σ = 1 ), suggesting that the optimal prior variance is data-dependent, while the standard Gaussian provides a balanced regularization strength in our case.

Context Modality Choice in ψ . We evaluate the effect of applying ∆ ij to either modality across datasets with contrasting characteristics. As shown in the left of Table 4, on MSR-VTT, injecting ∆ on the text side (i.e., t i + ∆ ij ) achieves the best performance, while applying it to the video side degrades results. This aligns with the fact that MSR-VTT contains many visually similar short clips, making it more effective to adjust the concise text representation for fine-grained distinctions. Conversely, on ActivityNet, applying ∆ to the video side leads to a notable performance boost, whereas modifying text harms results (R@1 ≈ 40). This is likely because the long but redundant videos are paired with rich, structured captions-making video-side adaptation more beneficial. These trends highlight the importance of aligning ψ 's modality choice with dataset structure.

## 3.4 Qualitative Analysis

Geometric Properties of ∆ . To understand the effect of the learned increment ∆ , we conduct a qualitative analysis on the MSR-VTT 1k-A validation set, focusing on its impact on representation geometry and alignment behavior at inference. As shown in Figure 3a, our method GARE yields

lower cosine similarities between positive pairs than the baseline, indicating larger angular separation and improved uniformity on the unit hypersphere [46].

̸

Figures 3b and 3c show that the norm of the adjusted text embedding t ∆ consistently exceeds that of the original t , implying that ∆ expands text representations onto a series of spheres of larger radius. Positive embeddings t ∆ ii also have greater norms, consistent with our analysis of ∆ 's iterative update behavior. Performing the first-order Taylor expansion around nonzero ∆ states (i.e., with nonzero initialization per iteration) mitigates logit-ranking distortion that occurs when expanding solely at zero. As negative increments ∆ ij ( j = i ) are pushed outward, the positive ∆ ii also expands to preserve relative belief masses [39] among logits, lowering overall cosine similarity while maintaining relative softmax probabilities.

Figure 3d further shows that t ∆ ij lies farther from v j than t i , implying that the model does not simply reduce inter-modal distance but expands the text representation onto a larger spherical shell for finer alignment. This suggests that, in cosine-based contrastive learning, encouraging greater dispersion of samples in the unnormalized feature space (i.e., higher pre-projection uniformity) may facilitate more effective alignment on the unit hypersphere.

Overall, these results indicate that optimization tension is effectively released: representation learning is no longer confined to the narrow region induced by the modality gap but occurs within a broader geometric space, thereby raising the upper bound of achievable alignment performance. Additional training-time analysis is provided in Appendix G.

Gradient Analysis. To further analyze how ∆ redistributes optimization tension during training, we visualize the perdimension gradient statistics of t ∆ ij at a representative training step (Figure 4). In dimensions with significant optimization activity, both positive and negative gradients reach magnitudes around g ≈ 2 . 5 and appear as approximate opposites, corresponding to the pair-specific gradient form in Eq. (8).

When aggregated over all pairs, these opposite gradients largely cancel out in the anchor's update ∇ t i L i , leading to the nearzero gradient state described earlier. However, unlike t i , which aggregates signals from all pairs, each ∆ ij only receives the gradient transmitted from its corresponding pair. Consequently, the positive increment ∆ ii receives an effective gradient of approximately + g , while each negative ∆ ij receives around -g/B , where B denotes the batch size. Since these gradients act independently and are not mutually canceled, the total nonvanishing optimization strength per anchor t i is approximately | + g | + B ·|-g/B | ≈ 2 | g | , indicating that ∆ components remain actively optimized rather than stagnant.

<!-- image -->

kimensio

Figure 4: Mean and variance of perdimension gradients, indicating the positive gradients (top) acting on t ∆ ii and ∆ ii and the sum of all negative gradients (bottom) for t ∆ ij and ∆ ij .

This reveals that the trajectory of ∆ ij reflects how t i explores the representation space. By distributing gradients across ∆ , our framework effectively offloads optimization tension from the anchor, expanding its reachable region and breaking the locality constraint imposed by the modality gap.

## 4 Conclusion

We revisited contrastive learning for text-video retrieval from an optimization perspective, identifying two key challenges: optimization tension from the modality gap and gradient noise from false negatives. Through a first-order Taylor expansion of the InfoNCE loss under a trust-region constraint, we derived a function space of increments ∆ ij that approximate descent directions. A learnable gapaware network predicts ∆ ij to redistribute gradients across pairs, expanding the optimization range beyond the modality gap and mitigating false-negative noise. To ensure ∆ ij encodes compact yet informative corrections, we employ a variational information bottleneck with a relaxed compression objective for stable training. Experiments on four benchmarks demonstrate consistent improvements, and further analyses show that the learned increments form structured, semantically meaningful geometry that releases optimization tension and enhances alignment capacity.

## 5 Acknowlegements

This work was supported by the NSFC NO. 62172138 and No. 62202139. This work was also partially supported by the Fundamental Research Funds for the Central Universities NO. JZ2024HGTG0310 and No. JZ2025HGTB0226.

## References

- [1] Alexander A Alemi, Ian Fischer, Joshua V Dillon, and Kevin Murphy. Deep variational information bottleneck. arXiv preprint arXiv:1612.00410 , 2016.
- [2] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In Proceedings of the IEEE international conference on computer vision , pages 5803-5812, 2017.
- [3] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF international conference on computer vision , pages 1728-1738, 2021.
- [4] Alberto Baldrati, Marco Bertini, Tiberio Uricchio, and Alberto Del Bimbo. Conditioned and composed image retrieval combining and partially fine-tuning clip-based features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops , pages 4959-4968, June 2022.
- [5] Alberto Baldrati, Marco Bertini, Tiberio Uricchio, and Alberto Del Bimbo. Effective conditioned and composed image retrieval combining clip-based features. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 21434-21442, 2022.
- [6] Simion-Vlad Bogolin, Ioana Croitoru, Hailin Jin, Yang Liu, and Samuel Albanie. Cross modal retrieval with querybank normalisation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 5194-5205, 2022.
- [7] Simion-Vlad Bogolin, Ioana Croitoru, Hailin Jin, Yang Liu, and Samuel Albanie. Cross modal retrieval with querybank normalisation, 2022.
- [8] David Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies , pages 190-200, 2011.
- [9] Tsai-Shien Chen, Wei-Chih Hung, Hung-Yu Tseng, Shao-Yi Chien, and Ming-Hsuan Yang. Incremental false negative detection for contrastive learning. arXiv preprint arXiv:2106.03719 , 2021.
- [10] Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297 , 2020.
- [11] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision , pages 9640-9649, 2021.
- [12] Ching-Yao Chuang, Joshua Robinson, Yen-Chen Lin, Antonio Torralba, and Stefanie Jegelka. Debiased contrastive learning. Advances in neural information processing systems , 33:8765-8775, 2020.
- [13] Arthur P Dempster, Nan M Laird, and Donald B Rubin. Maximum likelihood from incomplete data via the em algorithm. Journal of the royal statistical society: series B (methodological) , 39(1):1-22, 1977.
- [14] Kingma Diederik. Adam: A method for stochastic optimization. (No Title) , 2014.
- [15] Jianfeng Dong, Xianke Chen, Minsong Zhang, Xun Yang, Shujie Chen, Xirong Li, and Xun Wang. Partially relevant video retrieval. In Proceedings of the 30th ACM International Conference on Multimedia , MM '22, page 246-257. ACM, October 2022.
- [16] Bo Fang, Wenhao Wu, Chang Liu, Yu Zhou, Yuxin Song, Weiping Wang, Xiangbo Shu, Xiangyang Ji, and Jingdong Wang. Uatvr: Uncertainty-adaptive text-video retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 13723-13733, 2023.
- [17] Satya Krishna Gorti, Noël Vouitsis, Junwei Ma, Keyvan Golestan, Maksims Volkovs, Animesh Garg, and Guangwei Yu. X-pool: Cross-modal language-video attention for text-video retrieval. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 5006-5015, 2022.

- [18] Bo He, Jun Wang, Jielin Qiu, Trung Bui, Abhinav Shrivastava, and Zhaowen Wang. Align and attend: Multimodal summarization with dual contrastive losses, 2023.
- [19] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 9729-9738, 2020.
- [20] R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Phil Bachman, Adam Trischler, and Yoshua Bengio. Learning deep representations by mutual information estimation and maximization. arXiv preprint arXiv:1808.06670 , 2018.
- [21] Chen Jiang, Hong Liu, Xuzheng Yu, Qing Wang, Yuan Cheng, Jia Xu, Zhongyi Liu, Qingpei Guo, Wei Chu, Ming Yang, et al. Dual-modal attention-enhanced text-video retrieval with triplet partial margin contrastive learning. In Proceedings of the 31st ACM International Conference on Multimedia , pages 4626-4636, 2023.
- [22] Peng Jin, Jinfa Huang, Fenglin Liu, Xian Wu, Shen Ge, Guoli Song, David Clifton, and Jie Chen. Expectation-maximization contrastive learning for compact video-and-language representations. Advances in neural information processing systems , 35:30291-30306, 2022.
- [23] Peng Jin, Jinfa Huang, Pengfei Xiong, Shangxuan Tian, Chang Liu, Xiangyang Ji, Li Yuan, and Jie Chen. Video-text as game players: Hierarchical banzhaf interaction for cross-modal representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 2472-2482, 2023.
- [24] Peng Jin, Hao Li, Zesen Cheng, Jinfa Huang, Zhennan Wang, Li Yuan, Chang Liu, and Jie Chen. Text-video retrieval with disentangled conceptualization and set-to-set alignment. arXiv preprint arXiv:2305.12218 , 2023.
- [25] Peng Jin, Hao Li, Zesen Cheng, Kehan Li, Xiangyang Ji, Chang Liu, Li Yuan, and Jie Chen. Diffusionret: Generative text-video retrieval with diffusion model. In Proceedings of the IEEE/CVF international conference on computer vision , pages 2470-2481, 2023.
- [26] Weike Jin, Zhou Zhao, Xiaochun Cao, Jieming Zhu, Xiuqiang He, and Yueting Zhuang. Adaptive spatiotemporal graph enhanced vision-language representation for video qa. IEEE Transactions on Image Processing , 30:5477-5489, 2021.
- [27] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In Proceedings of the IEEE international conference on computer vision , pages 706-715, 2017.
- [28] Solomon Kullback and Richard A Leibler. On information and sufficiency. The annals of mathematical statistics , 22(1):79-86, 1951.
- [29] Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, and Xiaodong He. Stacked cross attention for image-text matching. In Proceedings of the European Conference on Computer Vision (ECCV) , September 2018.
- [30] Pandeng Li, Chen-Wei Xie, Liming Zhao, Hongtao Xie, Jiannan Ge, Yun Zheng, Deli Zhao, and Yongdong Zhang. Progressive spatio-temporal prototype matching for text-video retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 4100-4110, 2023.
- [31] Victor Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Y Zou. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. Advances in Neural Information Processing Systems , 35:17612-17625, 2022.
- [32] Zengrong Lin, Zheng Wang, Tianwen Qian, Pan Mu, Sixian Chan, and Cong Bai. Neighborretr: Balancing hub centrality in cross-modal retrieval. In Proceedings of the Computer Vision and Pattern Recognition Conference , pages 9263-9273, 2025.
- [33] Yuqi Liu, Pengfei Xiong, Luhui Xu, Shengming Cao, and Qin Jin. Ts2-net: Token shift and selection transformer for text-video retrieval. In European conference on computer vision , pages 319-335. Springer, 2022.
- [34] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning. Neurocomputing , 508:293-304, 2022.
- [35] Yiwei Ma, Guohai Xu, Xiaoshuai Sun, Ming Yan, Ji Zhang, and Rongrong Ji. X-clip: End-to-end multi-grained contrastive learning for video-text retrieval, 2022.

- [36] Antoine Miech, Jean-Baptiste Alayrac, Lucas Smaira, Ivan Laptev, Josef Sivic, and Andrew Zisserman. End-to-end learning of visual representations from uncurated instructional videos, 2020.
- [37] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 , 2018.
- [38] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning , pages 8748-8763. PMLR, 2021.
- [39] Murat Sensoy, Lance Kaplan, and Melih Kandemir. Evidential deep learning to quantify classification uncertainty. Advances in neural information processing systems , 31, 2018.
- [40] Kaibin Tian, Yanhua Cheng, Yi Liu, Xinglin Hou, Quan Chen, and Han Li. Towards efficient and effective text-to-video retrieval with coarse-to-fine visual representation learning. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 5207-5214, 2024.
- [41] Naftali Tishby, Fernando C Pereira, and William Bialek. The information bottleneck method. arXiv preprint physics/0004057 , 2000.
- [42] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems , 30, 2017.
- [43] Bairui Wang, Lin Ma, Wei Zhang, and Wei Liu. Reconstruction network for video captioning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , June 2018.
- [44] Feng Wang and Huaping Liu. Understanding the behaviour of contrastive loss. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 2495-2504, 2021.
- [45] Qiang Wang, Yanhao Zhang, Yun Zheng, Pan Pan, and Xian-Sheng Hua. Disentangled representation learning for text-video retrieval. arXiv:2203.07111 , 2022.
- [46] Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International conference on machine learning , pages 9929-9939. PMLR, 2020.
- [47] Xiaohan Wang, Linchao Zhu, and Yi Yang. T2vlad: global-local sequence alignment for text-video retrieval. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 5079-5088, 2021.
- [48] Yimu Wang, Xiangru Jian, and Bo Xue. Balance act: Mitigating hubness in cross-modal retrieval with query and gallery banks, 2023.
- [49] Ziyang Wang, Yi-Lin Sung, Feng Cheng, Gedas Bertasius, and Mohit Bansal. Unified coarse-to-fine alignment for video-text retrieval, 2023.
- [50] Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin. Unsupervised feature learning via non-parametric instance discrimination. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 3733-3742, 2018.
- [51] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. Videoclip: Contrastive pre-training for zero-shot video-text understanding, 2021.
- [52] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 5288-5296, 2016.
- [53] Xiangpeng Yang, Linchao Zhu, Xiaohan Wang, and Yi Yang. Dgl: Dynamic global-local prompt tuning for text-video retrieval, 2024.
- [54] Can Yaras, Siyi Chen, Peng Wang, and Qing Qu. Explaining and mitigating the modality gap in contrastive multimodal learning. arXiv preprint arXiv:2412.07909 , 2024.
- [55] Youngjae Yu, Hyungjin Ko, Jongwook Choi, and Gunhee Kim. End-to-end concept word detection for video captioning, retrieval, and question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 3165-3173, 2017.
- [56] Haonan Zhang, Pengpeng Zeng, Lianli Gao, Jingkuan Song, and Heng Tao Shen. Mpt: Multi-grained prompt tuning for text-video retrieval. In Proceedings of the 32nd ACM International Conference on Multimedia , pages 1206-1214, 2024.

## Appendix

This appendix provides additional discussions, derivations, and analyses that complement the main paper. We first discuss the limitations of our current framework and provide related works for broader context. Then, we present extended ablation studies and comparisons, followed by detailed mathematical derivations, including the relaxed form of our variational information bottleneck (VIB) objective and the rationale for using a non-zero state in the multivariate Taylor expansion. Finally, we provide implementation and efficiency details of the GARE model to support reproducibility and deployment.

## A Limitations

While our method successfully reduces the tension between text and video embeddings by optimizing pair-wise ∆ ij based on the direction that minimizes the InfoNCE loss, several limitations remain:

Lack of modality alignment. The fundamental issue of modality gap persists. Despite reducing the tension between the embeddings of text and video, the two modalities still reside in completely disjoint regions of the representation space. Our approach alleviates this problem by releasing the optimization tension exerted on anchor representations, thereby expanding the effective optimization range of each text anchor through multiple pair-specific increments ∆ ij . As a result, the method mitigates the effects of the modality gap and reduces some noise, but it does not address the root cause of the misalignment between the two modalities.

Lack of generalized supervision for ∆ ij . Our approach relies heavily on adjusting ∆ ij through gradient-based optimization, but ∆ ij lacks a more generalizable supervision signal. Currently, we are optimizing each pair-wise ∆ ij based solely on the gradient of the InfoNCE loss, which only loosely guides the optimization direction. While this helps alleviate the tension between positive and negative pairs, it does not provide a stronger supervisory signal to guide the model toward a better generalization across unseen data.

## B Related Work

Contrastive learning and modality gap. Contrastive learning has become a foundational paradigm in multimodal representation learning. Wang and Isola [46] formalize contrastive learning via the principles of alignment and uniformity on the hypersphere, offering a geometric perspective on representation quality. Wang and Liu [44] show that contrastive loss is hardness-aware and temperature-sensitive, and reveal a trade-off between representation uniformity and semantic tolerance, highlighting the need to preserve meaningful structure among semantically similar samples. Liang et al.[31] investigate the modality gap in multimodal contrastive learning and attribute it to initialization imbalance and cone effects. False negatives have also been recognized as a key challenge in contrastive learning [12, 9], with solutions ranging from reweighting and elimination to dynamic detection and correction.

Text-video retrieval. Text-video retrieval is one of the prominent tasks [15, 26, 43, 4, 18, 51, 36, 5, 24, 29, 48] in cross-modal learning. The majority of existing research [30, 16, 25, 35, 49, 7, 53] in this area utilizes a mapping technique that aligns both text and video inputs within a shared latent space to facilitate direct similarity assessment. CLIP4Clip [34] is the first to adapt CLIP [38] for video-text retrieval via temporal frame aggregation. TS2-Net [33] improves temporal modeling through token shift and selection. X-Pool [17] introduces text-guided pooling to highlight salient video tokens. HBI [22] values possible correspondences between frames and words using Banzhaf interaction for sensitive and explainable cross-modal alignment. EMCL [22] introduces an expectation-maximization [13] framework to learn a compact latent space where video and text features are represented as linear combinations of shared bases. This decomposition reduces the rank of the latent space, mitigating the modality gap and enhancing semantic alignment. Unlike prior works that refine matching structures, our method analyzes the gradient form of InfoNCE and introduces a learnable gap-aware increment ∆ ij to offload optimization tension, enabling structured optimization in a trust-region-aware formulation.

## C More Ablation and Comparison Experiments

In the next four experiments conducted on the MSR-VTT 1K-A dataset, we investigated the design of ψ network inputs, the impact of the scale coefficient α and the orthogonality variant in the directional diversity regularization, the impact of the lower bound λ of Norm-Based Regularization of TrustRegion Radii, and the impact of the anchor choice of L relax IB . We also provide hard negative methods comparison.

## C.1 Ablation on the Design of ψ Network Inputs

We analyze the design of the ψ network from three complementary aspects: 1) the context used for conditioning, 2) the semantic gap direction η , and 3) the corrected anchor (either t or v ). When correcting the text anchor ( t ), using the video-side sequence ( v ) as context yields better performance; conversely, when correcting the video anchor ( v ), using the text-side sequence ( t ) as context performs better. In both cases, the semantic gap is defined as v -t , suggesting that the query direction is asymmetric. Overall, performance is mainly determined by the interplay between the corrected anchor and the chosen context: cross-modal conditioning (e.g., correcting t with v as context) consistently outperforms uni-modal configurations, highlighting the benefit of cross-modal fusion.

Moreover, defining the semantic gap as v -t achieves better results than t -v . This can be attributed to the fact that video features tend to capture more shared and

Table 7: Ablation on context C and semantic gap η under different correction modes. η =1 denotes v -t .

| C           | η           | R@1 ↑       | R@5 ↑       | R@10 ↑      | MnR ↓       |
|-------------|-------------|-------------|-------------|-------------|-------------|
| t-corrected | t-corrected | t-corrected | t-corrected | t-corrected | t-corrected |
| T word      | - 1         | 46.6        | 74.2        | 83.8        | 12.9        |
| T word      | +1          | 48.2        | 74.2        | 83.2        | 12.9        |
| V frame     | - 1         | 47.8        | 74.1        | 82.8        | 12.4        |
| V frame     | +1          | 49.1        | 74.7        | 83.6        | 12.0        |
| v-corrected | v-corrected | v-corrected | v-corrected | v-corrected | v-corrected |
| V frame     | - 1         | 47.7        | 73.2        | 82.9        | 13.1        |
| V frame     | +1          | 47.5        | 73.5        | 81.8        | 13.0        |
| T word      | - 1         | 48.2        | 73.6        | 82.8        | 12.7        |
| T word      | +1          | 48.8        | 73.2        | 83.6        | 12.9        |

general concepts, while text features encode instance-specific information. Removing text-specific components from the video representation thus encourages more generalized and robust semantic alignment.

## C.2 Ablation of the Scale Coefficient α and the Orthogonality Variant in Directional Diversity

To further analyze the impact of the scale coefficient α in the directional diversity regularization, we conduct an ablation study as shown in Table 8. Recall that the loss adopts a logmean-exp form:

<!-- formula-not-decoded -->

where z ij = ∆ ij | ∆ ij | 2 denotes normalized directions under the same anchor t i . The term (1 -⟨ z ij , z ik ⟩ ) lies in [0 , 2] , and the log-mean-exp operator favors pairs with higher cosine simi-

Table 8: Ablation on scale coefficient α for directional diversity.

|   α |   R@1 ↑ |   R@5 ↑ |   R@10 ↑ |   MnR ↓ |
|-----|---------|---------|----------|---------|
| 0.5 |    47.6 |    74.5 |     83.4 |    12.1 |
| 1   |    48.5 |    74.4 |     83.9 |    12.1 |
| 2   |    49.1 |    74.7 |     83.6 |    12   |
| 3   |    46.2 |    74.4 |     82.9 |    12.1 |
| 4   |    47   |    74.3 |     83.4 |    12.1 |
| 5   |    47.2 |    74.1 |     82.8 |    12.4 |

larity (i.e., smaller angular distance), encouraging local angular diversity without enforcing strict orthogonality.

The coefficient α adjusts the strength of uniformity: 1) When α is small, the exponential term emphasizes pairs with higher similarity, focusing optimization on local clusters of ∆ ij directions. 2) As α increases, the optimization becomes more uniform across pairs, driving the distribution of ∆ ij directions toward isotropy.

Empirically, as shown in Table 8, performance peaks when α = 2 . 0 , achieving the best R@1, R@5 and MnR. When α grows larger, R@1 and R@5 exhibit reduced variance, indicating that the model becomes less sensitive to angular differences. This aligns with our interpretation that large α values saturate the exponential term-causing exp( -α (1 -⟨ z ij , z ik ⟩ )) to approach zero uniformly-thus diminishing the discriminative effect among diverse directions. Overall, a moderate α achieves the best trade-off between directional diversity and training stability, effectively preventing directional collapse while maintaining meaningful variation across ∆ ij .

To further enhance the distinction among ∆ ij directions, we additionally experimented with a strict orthogonalization objective:

̸

<!-- formula-not-decoded -->

which enforces all direction pairs under each anchor t i to be mutually orthogonal. Unlike the log-mean-exp formulation, this loss treats both high- and low-similarity pairs equally and aims to drive all pairwise cosine similarities toward zero. However, this uniform orthogonality constraint yields inferior performance (R@1=47.7, R@5=74.3, R@10=83.1, MnR=12.1). We attribute this degradation to the excessive suppression of negatively correlated directions-pairs that are semantically opposite or far apart are also pushed toward zero similarity.

This rigid treatment prevents

∆

ij from encoding opposing semantic

relations, indicating that directional diversity should remain seman- tically flexible rather than uniformly orthogonal.

Nevertheless, the orthogonalization approach exhibits stronger performance on video-to-text retrieval (R@1=49.4, R@5=75.4, R@10=83.6, Mean R=8.2). We conjecture that this improvement stems from the fact that the directional diversity regularization is applied with text as the anchor over candidate videos, where enforcing orthogonality among ∆ ij encourages the model to better separate diverse visual counterparts for the same textual query.

## C.3 Ablation on ∆ Norm Regularization Strength

We investigate the effect of varying the lower bound factor λ , which controls the target margin of ε separation within each anchor t i . As shown in Figure 5, increasing λ from 0.1 to 0.5 improves R@1, with the best performance at λ = 0 . 5 . This suggests that moderate diversity in ε ij is beneficial for enhancing semantic discrimination. However, as λ increases further, performance degrades, likely due to over-regularization.

To mitigate the instability introduced by hard truncation (i.e., directly thresholding ε ij ), we also experiment with a smooth approximation using a log-sum-exp formulation:

<!-- formula-not-decoded -->

Although this variant imposes a natural lower bound and provides vanishing gradients near convergence, it does not significantly improve retrieval. We hypothesize that this is due to gradient saturation when the variance among ε ij becomes too large, which in turn weakens the ability to further enforce directional separation.

## C.4 Ablation on the Anchor Choice in Relaxed VIB Regularization

We further investigate the effect of the anchor choice in the relaxed information bottleneck term L relax IB . In the main formulation, the KL regularization is anchored on videos, i.e.,

<!-- formula-not-decoded -->

where ¯ q ψ (∆ | v ) = E t | v [ q ψ (∆ | t, v )] . This relaxation aggregates increments ∆ ij B i =1 under each video anchor v j and preserves the information-bottleneck effect while reducing sensitivity to the variability of short text inputs.

Table 9: Effect of anchor choice in KL regularization L relax IB . Anchoring on v yields better performance.

| Anchor   |   R@1 |   R@5 |   R@10 |   MdR |   MnR |
|----------|-------|-------|--------|-------|-------|
| t        |  47.9 |  74.9 |   83.8 |     2 |  12.5 |
| v        |  49.1 |  74.7 |   83.6 |     2 |  12   |

To contrast this design, we also test the reversed relaxation that anchors on text:

<!-- formula-not-decoded -->

where ¯ q ψ (∆ | t ) := E v | t [ q ψ (∆ | t, v )] . Practically, this means that the set { ∆ ij } B j =1 associated with each text anchor t i is modeled as a Gaussian posterior, yielding the corresponding compression loss.

Figure 5: R@1 score with varying λ for ∆ norm regularization.

<!-- image -->

工作

1

工作

1

Query: a woman on a couch talks to a man Query: a woman on a couch talks to a man

<!-- image -->

Query: a person is putting the vegetable in to the water and boil it Query: a person is putting the vegetable in to the water and boil it

Figure 6: Comparison of hard negative alignment before and after applying ∆ ij optimization. Compared with the baseline, GARE produces smaller similarity gaps among semantically related videos v j . This indicates that GARE effectively mitigates the noise from hard negatives and reduces the semantic deviation of the anchor t i , leading to more stable and consistent alignment across similar samples.

<!-- image -->

Notably, our relaxation introduces stochasticity directly from the dataset pairing itself, rather than by sampling additional uncertainty variables.

As shown in Table 9, anchoring the KL term on the video side leads to superior retrieval performance (R@1=49.1 vs. 47.9). We attribute this to the inherent data characteristics: the textual descriptions in the dataset are typically short and less informative, while videos are semantically richer and often correspond to multiple captions. Consequently, conditioning ∆ on v captures more shared, modality-invariant structure, aligning better with the underlying multimodal distribution and yielding stronger retrieval results.

## C.5 Hard Negative Comparison

We compare GARE with two recent methods that explicitly or implicitly handle hard negatives:

DMAE [21] (R@1: 46.9, +1.6% over base. ACMMM2023 ): DMAE enhances fine-grained alignment by mining hard positives-for instance, text queries associated with specific frames-which implicitly improves the model's capability to separate hard negatives. Conceptually, this shares similarity with our variational information bottleneck (IB) loss, where ∆ is compressed through a bottleneck to retain critical alignment signals while filtering noisy gradients.

NeighborRetr [32] (R@1: 49.5, +2.3% over base. CVPR 2025 ): This method employs a memory bank to compute k -neighbor co-occurrence statistics, identifying 'good hubs' that encourage local consistency and reduce over-penalization of hard negatives. Although not explicitly formulated as hard-negative mining, the topk co-occurrence selection serves a similar role by adaptively reweighting difficult negatives.

GARE (R@1: 49.1, +2.5% over base): Unlike the above methods, GARE does not rely on explicit mining or external memory structures. Each increment ∆ ij absorbs loss gradients locally from its paired ( t i , v j ) sample, mitigating the reliance on global contrastive comparisons when encountering hard negatives. This local reallocation of gradients softens noisy updates and improves generalization. As illustrated in Figure 6, the text anchor t i in GARE exhibits enhanced semantic stability: for semantically related videos v j , the similarity values s ii and s ij vary smoothly, indicating that ∆ ij enables more accurate fine-grained alignment across similar samples.

Figure 7: Mean and variance of total gradients acting on t i on each dimension.

<!-- image -->

This aligns with the gradient visualization in Figure 7, where the gradients on t i 's dimensions remain centered around zero after introducing ∆ , demonstrating its role as a stable semantic prototype.

Efficiency comparison. GARE maintains superior efficiency despite comparable or higher performance. It uses only a single cross-attention layer, whereas NeighborRetr includes 8 MLP modules and

历

历

程

程

multiple transformer or convolutional blocks. NeighborRetr also requires large-scale memory banks (10,240 samples per modality) and ∼ 4.5 h training time, while GARE achieves similar accuracy with 1h34min of training and negligible additional memory cost compared to the baseline. Together, these results demonstrate that GARE achieves comparable hard-negative robustness with substantially lower computational overhead.

## D Derivation of the VIB Objective for Pair-Specific Increments

We model each pair-specific increment as a latent variable Z = ∆ ij for the input X = ( t i , v j ) and the match label Y ∈ { 0 , 1 } . Our objective maximizes predictive information under a compression constraint:

<!-- formula-not-decoded -->

Below we derive a tractable variational lower bound of Eq. (16).

Lower bound for I (∆ ij ; Y ) . For convenience, in the following, we will refer to y ij , ∆ ij , t i , and v j as y , ∆ , t , and v respectively. We make the standard assumption that the match label does not depend on ∆ once ( t, v ) is given:

<!-- formula-not-decoded -->

By definition,

<!-- formula-not-decoded -->

where q θ ( y | ∆) is a variational classifier, the non-negativivty entropy term H ( Y ) is constant w.r.t. model parameters, and the last second equality uses assumption Eq. (17). Let q ψ (∆ | t, v ) be the variational encoder. Then,

<!-- formula-not-decoded -->

The two inequalities above arise from the non-negativity of KL( ·∥· ) and from rewriting log 1 p ( y ) as the entropy H ( y ) .

Upper bound for I (∆ ij ; t i , v j ) . Using I (∆; t, v ) ≤ E ( t,v ) KL( p (∆ | t, v ) ∥ r (∆)) with any auxiliary prior r ( z ) ,

<!-- formula-not-decoded -->

where the inequality follows from KL( p (∆) ∥ r (∆)) ≥ 0 and p (∆ | t, v ) denotes the variational encoder q ψ (∆ | t, v ) that parameterizes the model's conditional distribution of the pair-specific increment (deterministic in our implementation). We use a diagonal Gaussian prior r ( z ) = N (0 , I ) .

VIB lower bound. Combining Eq. (18) and Eq. (20) yields the variational lower bound of Eq. (16), i.e., the upper bound of the negative of Eq. (16):

<!-- formula-not-decoded -->

## E Relation between the Relaxed VIB Compression Term and the Original IB Objective

We start from the standard information bottleneck (IB) formulation, which regularizes the mutual information between the learned increment ∆ and the input pair ( t, v ) :

<!-- formula-not-decoded -->

Since the marginal q ψ (∆) is intractable, it is commonly replaced by a fixed prior r (∆) , leading to the following upper bound:

<!-- formula-not-decoded -->

This term penalizes the total amount of information ∆ retains about both modalities and serves as the compression loss in the original VIB objective.

Relaxation anchored on the video side. To mitigate sensitivity to text-side variability, we anchor the expectation on videos and aggregate increments over all textual pairs associated with the same v . By convexity of the KL divergence (Jensen's inequality), we have:

<!-- formula-not-decoded -->

where the aggregated posterior ¯ q ψ (∆ | v ) = E t | v [ q ψ (∆ | t, v )] represents the mixture distribution of increments under the same video anchor.

Applying the decomposition in Eq. (23) to the right-hand side of Eq. (24) yields

<!-- formula-not-decoded -->

where I ¯ q (∆; V ) denotes the mutual information defined under the aggregated distribution ¯ q ψ . Combining the two equations gives

<!-- formula-not-decoded -->

indicating that the relaxation effectively removes the conditional term I (∆; T | V ) from the full IB regularization.

Interpretation. The relaxed KL term therefore optimizes

<!-- formula-not-decoded -->

which serves as a lower bound of the original compression term. While the standard IB loss penalizes all information in ∆ about ( T, V ) , the relaxed form only constrains information shared with V and ignores the conditional mutual information I (∆; T | V ) . Intuitively, this relaxation preserves the bottleneck effect but allows ∆ to encode text-specific variations within each video cluster, yielding more stable optimization when multiple captions correspond to the same visual content.

## F Overall Objective and Inference

The overall training objective is formulated as

<!-- formula-not-decoded -->

where the first two terms constitute the VIB optimization objective, and λε and λ dir are the weights of the structural regularizers. In practice, we set β = 0 . 07 and λ ε = λ dir = 0 . 01 for MSR-VTT. During inference, we retain the learned increment ∆ to assist retrieval, as it encodes the semantic consistency between t i and v j . We further discuss the efficiency of using ∆ at both training and inference stages in the Appendix I, as well as its distinction from traditional dual-encoder retrieval paradigms and deployment strategies for large-scale retrieval.

## G Analysis of ∆ Behavior During Training

To better understand the role and dynamics of the learned semantic-gap vector ∆ throughout training, we visualize four key aspects of ∆ 's behavior, presented in Figure 8a to 8d. These analyses reveal the underlying geometric transformations and provide further insight into how ∆ facilitates optimization under InfoNCE loss. We will continue conducting ablation studies on the Text-to-Video retrieval task using the MSR-VTT [52] dataset.

Angle between ∆ and ( v j -t i ) . As shown in Figure 8a, we track the angle between ∆ and the initial cross-modal gap vector v j -t i throughout training, for both positive and negative pairs. At initialization, this angle is close to π 2 , indicating that ∆ is nearly orthogonal to v j -t i , and thus carries no meaningful alignment with the cross-modal semantic gap. This implies that the early ∆ vectors do not differentiate between positive and negative pairs, acting more like isotropic perturbations in space. As training proceeds, however, this angle gradually increases into the obtuse region for both positive and negative samples. This reflects a significant shift in behavior-rather than attempting to directly bridge the semantic gap v j -t i , the model learns to push t i away from v j , effectively offloading the gradient tension induced by the modality gap and false negative interference. This offloading allows contrastive learning to take place in an expanded embedding space, where ∆ modulates the representation geometry to ease the optimization burden. This trend is corroborated by Figure 8d, where the Euclidean distances between the updated text embeddings t ∆ ij = t i +∆ ij and v j become significantly larger than the original distances ∥ t i -v j ∥ , for both positive and negative samples. That is, ∆ introduces a global scaling effect in the representation space, and contrastive optimization is carried out under a larger geometric regime.

Interestingly, the new positive pair distances ∥ t ∆ ii -v i ∥ are also larger than their original counterparts ∥ t i -v i ∥ . Though this seems counterintuitive-since the gradient of InfoNCE with respect to ∆ pushes toward v i -it actually reflects the relative nature of the InfoNCE loss. The network does not aim to minimize absolute distances, but rather to increase the similarity margin between matched and mismatched pairs. Thus, pushing all embeddings outward in norm (as further validated in Figure 8c) gives the model more room to maneuver in angular space, while the cosine similarity objective remains stable under such rescaling. This aligns with our design intent: to leverage ∆ as a structural carrier for modality-aware tension redistribution, providing optimization flexibility on a normalized manifold.

Moreover, in early training stages, the angles between ∆ and v j -t i are nearly identical for positive and negative samples, showing that the model has not yet learned to encode fine-grained pairwise

Figure 8: Training dynamics of the learned modality-gap vector ∆ . t new denotes t ∆ . (a) ∆ grows increasingly orthogonal to the initial modality gap, indicating embedding space expansion. (b) ∆ initially aligns with the anchor t i , then deviates to encode semantic distinctions. (c) Updated embeddings t new operate under larger norms, enlarging the contrastive space. (d) Pairwise distances increase and then stabilize, reflecting semantic separation and convergence.

<!-- image -->

differences. But as training advances, a slight but consistent gap emerges-∆ for positive samples tends to have slightly smaller angles than that of negatives. This subtle divergence signals that ∆ has begun to capture semantically meaningful pair-level distinctions, enabling more discriminative alignment in later stages.

Angle between ∆ and t i . As shown in Figure 8b, the angle between ∆ and t i exhibits a distinct pattern: it decreases rapidly in the early training phase, indicating that ∆ is initially aligned with the anchor text embedding. This suggests that the model's default behavior is to trust the prior structure of the text space and apply similar ∆ directions across different v j , especially when pairwise semantic differences are not yet learned.

However, over time, this alignment loosens-∆ deviates further from t i as the model begins to adapt to modality-specific differences. This marks a transition where the network ψ no longer treats the anchor t i as a default direction and instead generates ∆ based on nuanced distinctions across the visual features v j , thus leading to more expressive and semantically grounded ∆ vectors.

Pair distances between text and video features. Figure 8d visualizes the Euclidean distances between text and video pairs over the course of training, comparing both the original pairs t i ↔ v j and the updated pairs t ∆ ij ↔ v j , across positive and negative samples. Several important patterns emerge: First, we observe that positive pair distances remain consistently smaller than negative pair distances, both in the original and updated forms. This confirms that the learned ∆ not only preserves the basic alignment structure of contrastive learning but also enhances semantic discrimination, effectively encoding fine-grained pairwise structure through the transformation. Second, for both positive and

negative pairs, the distances between t ∆ and v are larger than those between the original t and v , demonstrating that the model pushes the updated text embeddings into a larger-scale embedding regime. This is precisely in line with our gradient tension offloading design: ∆ introduces a controlled displacement that alleviates the direct optimization pressure on t i , allowing contrastive comparisons to operate within a higher-norm, more expressive space, as also supported by Figure 8c. The evolution of the distance curves over time also reflects meaningful training dynamics. Initially, all distances decrease, which we interpret as a normalization phase - embedding distributions at initialization are noisy and unstructured, and the model first compresses them into a tighter, more consistent geometric configuration. This is followed by a steady increase in distances, as the model begins to explicitly separate positive and negative samples to satisfy the InfoNCE objective. Finally, all curves converge and stabilize into a bounded range, suggesting that the semantic configuration of the embedding space has reached a relatively converged structural state.

Taken together, these observations highlight the critical role of ∆ not only in scaling the embedding geometry but also in encoding structurally-aware, semantically-aligned displacements that facilitate tension redistribution and enable robust representation learning.

## H Why We Don't Use an Initial ∆ i ∗ = 0 Expansion and the Need for a Non-Zero Initial ∆ i ∗

## H.1 Introduction to the Issue

In the previous approach, we considered using multiple variables ∆ ij to distribute the gradient impact of each video v j on the text embedding t i . However, the specific form of ∆ ij was unclear. The idea was to update each ∆ ij in the direction that would reduce the InfoNCE loss. A natural first step was to expand the loss function L i as a Taylor expansion around the initial state ∆ i ∗ = 0 , which corresponds to an expansion solely based on t i .

However, this expansion introduces a key issue: the gradient terms in each ∆ ij expansion are computed under the assumption that all other ∆ ik remain zero. This neglects the mutual interactions among { ∆ ij } B j =1 and decouples the updates for different pairs. Since the InfoNCE loss fundamentally depends on the relative ranking of cosine similarities between t i and all candidate videos { v j } , such a decoupled formulation disrupts the prior ordering structure already established by the CLIP encoder's logits. In other words, it breaks the model's inherent relational prior among candidates, which is essential for maintaining consistent contrastive comparisons.

## H.2 Why Expanding with ∆ i ∗ = 0 Is Inadequate

When ∆ ∗ ij = 0 for all j , the gradient expansion for each pair ∆ ij looks like:

<!-- formula-not-decoded -->

where ∇ ∆ ij L i (0) is the gradient of the loss evaluated at ∆ ij = 0 . In this case, the gradient p ij -y ij is computed as:

<!-- formula-not-decoded -->

where p ij (0) is the softmax term evaluated at the cosine similarity cos( t i , v j ) . This expansion does not account for the interdependence between different ∆ ik , and the gradients are computed as if each ∆ ij were independent of all other ∆ ik , which violates the relative comparison required for contrastive learning.

## H.3 The Need for a Non-Zero Initial ∆ i ∗

To avoid this issue, we recognize that each ∆ ij should be updated in a way that reflects the interdependence between different pairs, not just based on an independent text-video pair. The gradient of the InfoNCE loss must be evaluated with respect to the current state of all ∆ ij , not just the initial zero state.

Thus, we need to initialize ∆ i ∗ in a way that reflects the current state of all other pairs rather than assuming they are zero. The update for each ∆ ij should respect the relationship between all the pairwise similarities, as the optimization of one pair affects the relative similarity between all other pairs.

In this case, the correct approach is to perform the first-order Taylor expansion around a non-zero state ∆ ( t ) ij that has already been optimized for several steps. This allows us to incorporate the effect of each pair ∆ ij on the entire set of pairwise comparisons, thus preserving the relative relationships that are critical for InfoNCE loss.

## H.4 Optimization Objective and Trust-Region Constrained Solution

Having established the need for a coupled, non-zero initialization of ∆ ij to preserve the relative structure required by InfoNCE, we now formulate the optimization problem for finding the approximate update direction under a trust region constraint. Specifically, we assume a trust region radius ε that bounds the magnitude of each ∆ ij and analyze the solutions in both the single-step (non-iterative) and iterative update settings.

Non-iterative first-order update. We begin by considering the first-order Taylor expansion of the per-anchor loss L i with respect to a single variable ∆ ij evaluated at the origin:

<!-- formula-not-decoded -->

Since the constant term L i (0) does not affect optimization, minimizing the approximated loss is equivalent to minimizing the linear term under a trust-region constraint:

<!-- formula-not-decoded -->

where ε denotes the radius limiting the update magnitude of ∆ ij . By the Cauchy-Schwarz inequality, for g ij = ∇ ∆ ij L i (0) we have

<!-- formula-not-decoded -->

which provides a tight lower bound for the objective within the trust region. The bound is attainable when ∆ ij is colinear with -g ij and satisfies ∥ ∆ ij ∥ = ε , yielding the feasible update

<!-- formula-not-decoded -->

̸

This closed-form solution achieves the maximal decrease in the first-order approximation of L i , corresponding to the steepest descent direction constrained by the trust region. However, this derivation treats L i as a function of a single ∆ ij and neglects the coupling among { ∆ ik } k = j , thus breaking the relative similarity structure essential to InfoNCE. A more faithful formulation must therefore incorporate all ∆ ij jointly, as discussed in the following subsection.

Iterative update with coupled expansion. We next analyze the update rule when ∆ ij is expanded around a coupled, non-zero state ∆ ( t ) i ∗ = { ∆ ( t ) ij } B j =1 , where each gradient ∇ ∆ ij L i is computed under the influence of all other nonzero ∆ ( t ) ik . We seek the next update step within a trust region:

<!-- formula-not-decoded -->

The first-order Taylor expansion of the per-anchor loss L i at this coupled state is:

<!-- formula-not-decoded -->

This multivariate linearization serves to reveal that each partial gradient ∇ ∆ ij L i (∆ ( t ) i ∗ ) is conditioned on the current coupled state, and thus inherently encodes the interactions among all pairs. We do not minimize the linearized approximation itself; rather, it clarifies that the correct descent direction for each ∆ ij should be taken from the true gradient of L i evaluated at the coupled state.

Following the principle of steepest descent, we update each component by

<!-- formula-not-decoded -->

To satisfy the trust-region constraint ∥ ∆ ( t +1) ij ∥ ≤ ε , we determine the maximal feasible step size α ( t ) ij from

<!-- formula-not-decoded -->

which expands to the quadratic inequality:

<!-- formula-not-decoded -->

The maximal feasible solution is given by:

<!-- formula-not-decoded -->

This update ensures that each ∆ ij moves along its steepest descent direction while keeping its magnitude within the trust-region radius ε . Importantly, because all gradients ∇ ∆ ij L i (∆ ( t ) i ∗ ) are computed under the coupled multivariate state, the optimization preserves the relative ranking structure among all candidate pairs required by the InfoNCE objective.

## H.5 Error Analysis

The iterative update of ∆ ij is derived from a multivariate first-order Taylor expansion of L i under a trust-region constraint. This formulation defines a descent function space that guarantees loss reduction under the coupled gradient field, rather than prescribing a unique update direction for each ∆ ij . Once the trust-region radius ε ij is specified (assuming an ℓ 2 -ball constraint), the steepestdescent direction becomes unique, and the theoretical update moves ∆ ij onto the boundary where the Cauchy-Schwarz inequality reaches equality. However, in practice, we do not explicitly assign a radius ε ij to each pair; instead, the effective trust-region constraint emerges implicitly from the model's prior, which we regularize through a norm-based penalty that controls the magnitude of ∆ . Specifically, the ∆ ij produced by the ψ network during the forward pass corresponds to the current state ∆ ( t ) ij , and the optimizer update during backpropagation embeds the next-step state ∆ ( t +1) ij into the parameters of ψ . When the model performs the next forward pass, the generated ∆ ij naturally reflects this updated state, while the norm-based regularization acts as the trust-region constraint on the previous update step. Therefore, the theoretical and practical procedures share the same steepest-descent direction, but their update magnitudes may differ due to the implicit learning of the trust-region radius through model priors. Consequently, our error analysis focuses on the consistency of update magnitudes between the theoretically derived trust-region step and the actual updates produced by the training framework.

Analytic step length. At the equality boundary of the trust-region constraint, the maximal feasible step size α ( t ) ij satisfying ∥ ∆ ( t +1) ij ∥ ≤ ε is given by:

<!-- formula-not-decoded -->

which projects the update exactly to the boundary of the feasible region and enforces norm-bounded motion of ∆ ij .

Learned step length. During training, the neural module ψ is optimized end-to-end via AdamW, implicitly learning how to adjust ∆ ij through gradient descent. After one step, the actual update can be expressed as:

<!-- formula-not-decoded -->

where η ( t ) ij denotes the effective step size produced by the optimizer. Although ∆ ij is updated through the training framework rather than by explicitly solving the analytic trust-region problem, we apply

<!-- image -->

Batch Index

(a) Batch-wise mean error for positive pairs ( i = j ) , showing convergence and bounded deviation ( [1 . 0 , 4 . 5] ).

<!-- image -->

Batch Index

̸

(b) Batch-wise mean error for negative pairs ( i = j ) , exhibiting divergence that promotes embedding uniformity.

Figure 9: Comparison between the analytic step length α ij and the effective AdamW step length η ij . Although ∆ ij is learned through backpropagation, the norm-based trust-region constraint ensures that its magnitude remains bounded and consistent with the analytic formulation. Positive pairs show convergence and bounded errors, while negative pairs exhibit divergence that aids global uniformity.

Table 10: Comparison of computational efficiency between CLIP4Clip and GARE.

| Metric                                                                                                                                     | CLIP4Clip                                                        | GARE                                                                             |
|--------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Training time Inference time Training memory (reserved) Inference memory (reserved) Training FLOPs (per batch) Inference FLOPs (per batch) | 1h 30m 7.6s 4 × 12175MB 4136MB 39,167.58 GFLOPs 11,868.70 GFLOPs | 1h 34m 6.9s 4 × 12561MB 4216MB 39,287.55 GFLOPs (+0.3%) 11,905.05 GFLOPs (+0.3%) |

a norm-based trust-region regularization L ε that restricts ∥ ∆ ij ∥ within a trust-region radius. This guarantees that the update magnitudes remain consistent with the theoretical constraint, even when learned implicitly.

To measure the deviation between the theoretical and learned step magnitudes, we compute the scalar error:

<!-- formula-not-decoded -->

This metric quantifies how closely the effective update magnitude in training adheres to the analytic trust-region step size.

Empirical analysis. Figure 9a and Figure 9b visualize the batch-wise mean error for positive and negative pairs, respectively. For positive pairs ( i = j ) , the error ∥ η ii -α ii ∥ remains within a stable range of [1 . 0 , 4 . 5] and shows a clear convergence trend. Given the 512-dimensional embedding space, this corresponds to a per-dimension deviation of roughly 4 . 5 / 512 , which is negligible. This indicates that the learned ∆ ii updates remain consistent with the analytic trust-region dynamics and are well regularized by the norm constraint.

̸

In contrast, for negative pairs ( i = j ) , the error | η ij -α ij | shows a divergent trend, which is theoretically expected. Each negative ∆ ij is influenced by repulsive gradients with (512 -1) degrees of freedom, while the positive pair has a single dominant alignment direction. Moreover, since our norm-based regularization enlarges the updates whose magnitudes exceed the batch-wise mean ∥ ∆ i ∗ ∥ , the dispersion of negative ∆ ij naturally increases. Such divergence, however, plays a constructive role: by expanding negative { ∆ ij } , ( j = i ) into a larger representational space, the model enhances embedding uniformity, allowing positive ∆ ii to achieve more stable alignment under a broader geometric margin.

̸

Summary. In summary, our analysis demonstrates that while ∆ ij is optimized implicitly through the training framework, the norm-based trust-region constraint effectively maintains the magnitude of ∆ within the theoretical bound. The learned ψ module reproduces the analytic trust-region

Table 11: Module-wise GFLOPs breakdown during forward propagation.

| Module                                                                               | GFLOPs (%)                                                    | Main computation source                                                              |
|--------------------------------------------------------------------------------------|---------------------------------------------------------------|--------------------------------------------------------------------------------------|
| CLIP visual encoder Video temporal transformer ψ (cross-attention) CLIP text encoder | 11,673.60 (98.05%) 38.81 (0.33%) 36.35 (0.31%) 156.29 (1.31%) | ViT image processing FFN + self-attention linear projection text sequence processing |
| Total                                                                                | 11,905.05 (100%)                                              |                                                                                      |

```
Inference Procedure (Pseudocode) # Precompute all text and video embeddings offline using CLIP encoders for each text batch T: for each video batch V: # Compute pair-specific delta via ψ and adjust text embeddings logits = model.get_similarity_logits(T, V) # Append cosine-similarity block to the global similarity matrix sim_matrix.append(logits) # Discard the delta tensor immediately (transient variable) # Concatenate all similarity blocks for final retrieval
```

Figure 10: Inference pseudocode of GARE. ∆ are computed on-the-fly and discarded immediately to ensure constant memory usage during large-scale retrieval.

```
Cross-Attention Pairwise Parallelization (Simplified Code) def cross_attention(query, key, value): # query: (a, b, dim) for (v_j -t_i), a is text batch size, b is video batch size # key/value: (b, f, dim) for video frames query = query.permute(1,2,0) # (a,b,dim) -> (b,dim,a) Q = Q_proj(query) K = K_proj(key) V = V_proj(value) # This op is pairwise-parallelizabble cross all pairs in the batch logits = matmul(K, Q) # (b,f,dim) x (b,dim,a) -> (b,f,a) scores = softmax(logits / sqrt(dim), dim=frame_dim) scores = scores.permute(0,2,1) # (b,f,a) -> (b,a,f) out = matmul(scores, V) # (b,a,f) x (b,f,dim) -> (b,a,dim) out = out.permute(1,0,2) # (b,a,dim) -> (a,b,dim) return out
```

Figure 11: Simplified implementation of ψ 's cross-attention operation. The computation is pairwiseparallelizable across all text-video pairs, ensuring linear scaling with batch size and high GPU utilization.

behavior for positive pairs, while the controlled divergence of negative pairs improves embedding uniformity-both consistent with the objectives of contrastive learning.

## I Model Efficiency and Implementation Details

GARE remains fully compatible with the standard dual-tower CLIP architecture while introducing only a lightweight cross-modal adjustment module, ψ . Implemented as a single-layer cross-attention transformer without FFN expansion, ψ adds merely 1.58M parameters to the 354M parameters of the CLIP encoders. Its computational cost is negligible-only 36.35 GFLOPs per batch compared to CLIP's 11,868.70 GFLOPs .

During inference, ψ operates after both text and video embeddings have been precomputed, applying transient, pair-specific deltas before cosine similarity is calculated. This design preserves the precomputability and scalability of the dual-tower framework: all embeddings can be cached offline, ψ performs only on-the-fly adjustments, and no delta tensors are stored-only similarity scores are retained.

Efficiency comparisons on MSR-VTT (4×RTX 4090 GPUs, batch size 128) show that GARE introduces minimal overhead (Table 10). A module-wise breakdown (Table 11) further confirms that over 98% of total FLOPs come from the visual encoder, while ψ contributes less than 0.4%, demonstrating computational parity with the original CLIP.

As illustrated in Figure 10, GARE performs inference in a batch-parallel streaming manner, generating and discarding deltas on-the-fly to maintain constant memory usage. The internal cross-attention process (Figure 11) is pairwise-parallelizable, enabling simultaneous computation across all text-video pairs and ensuring high GPU utilization.

Overall, GARE retains the efficiency of dual-tower architectures while introducing a transient, pairspecific adjustment that enhances fine-grained cross-modal alignment without increasing latency or memory cost.

Distinction from traditional dual-Tower models. Traditional dual-tower architectures define a consistent metric space where the triangle inequality holds, e.g., for any text-video pair ( t i , v j ) and another video v k ,

<!-- formula-not-decoded -->

This property allows efficient large-scale retrieval via approximate nearest neighbor (ANN) search, since all embeddings share a unified metric geometry.

In contrast, GARE introduces pair-specific adjustments through ∆ ij , yielding a conditional distance

<!-- formula-not-decoded -->

which depends on the paired video v j . Substituting t i +∆ ij and t i +∆ ik into the above inequality breaks the shared metric assumption, and thus the triangle inequality no longer strictly holds. Consequently, while GARE improves fine-grained alignment, it cannot directly support ANN-based retrieval relying on fixed metric consistency.

From an efficiency standpoint, GARE computes ∆ in a pair-wise parallel manner within each batch, discarding them immediately after computing cosine similarities. This design keeps latency low but introduces additional FLOPs that make direct large-scale application challenging. Nevertheless, we find that in practice the retrieved candidates from the unadjusted text embeddings t i already cover most of the top-ranked results produced by GARE. For instance, on the MSR-VTT 1k-A validation set, the top-256 candidates retrieved using t i include all of GARE's top-10 matches.

Therefore, GARE naturally supports a two-stage retrieval pipeline: (1) use the dual-tower model with t i for large-scale ANN retrieval to obtain a compact candidate set, (2) re-rank the topk candidates using GARE's pair-specific adjustments. This strategy preserves the scalability of dual-tower models while benefiting from GARE's fine-grained alignment refinement.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract accurately describes our contribution.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We have raised two limitations regarding the current work in the Appendix.

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

Justification: We did not propose a theorem.

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

Justification: This includes experimental details and our hyper-parameters.

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

Justification: The official version of the code will be released immediately upon receipt of the paper (within one day).

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

Justification: The specific experimental details have been presented in the implementation details section.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: We did not provide an error bars experiment.

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

Justification: We provide the type of GPU we used.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in the paper conform, in every respect, with NeurIPS Code of Ethic.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The method proposed in this paper belongs to the application category of short video platforms and has a direct impact.

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

Justification: There is no risk of misuse in the method we proposed.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All assets used in this paper are under the license of CC-BY 4.0.

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

Justification: We introduce a new training framework including a custom loss function and optimization strategy. Full code will be released with detailed documentation and instructions.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

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