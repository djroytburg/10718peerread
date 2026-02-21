## Mysteries of the Deep: Role of Intermediate Representations in Out of Distribution Detection

Ignacio Meza De la Jara 1 , 4 Cristian Rodriguez-Opazo 1 Damien Teney 1 , 2 1

1

Damith Ranasinghe

Ehsan Abbasnejad 1 , 3

Australian Institute for Machine Learning, University of Adelaide Idiap Research Institute, Switzerland 3 Monash University 4 Naval Group

{firstname.lastname}@adelaide.edu.au

## Abstract

Out-of-distribution (OOD) detection is essential for reliably deploying machine learning models in the wild. Yet, most methods treat large pre-trained models as monolithic encoders and rely solely on their final-layer representations for detection. We challenge this wisdom. We reveal the intermediate layers of pre-trained models, shaped by residual connections that subtly transform input projections, can encode surprisingly rich and diverse signals for detecting distributional shifts. Importantly, to exploit latent representation diversity across layers, we introduce an entropybased criterion to automatically identify layers offering the most complementary information in a training-free settingwithout access to OOD data . We show that selectively incorporating these intermediate representations can increase the accuracy of OOD detection by up to 10% in far-OOD and over 7% in near-OOD benchmarks compared to state-of-the-art training-free methods across various model architectures and training objectives. Our findings reveal a new avenue for OOD detection research and uncover the impact of various training objectives and model architectures on confidence-based OOD detection methods. /globe

## 1 Introduction

Out-of-distribution (OOD) detection is essential for reliable machine learning, especially in openworld settings with distribution shifts that risk unsafe predictions [21, 55]. The problem is acute in safety-critical domains such as autonomous driving, medical diagnostics, and cybersecurity [17, 71].

Recent work has leveraged large vision-language models (VLMs) such as CLIP [57] in attempts to address the problem. The methods enable zero-shot OOD detection by aligning image and text embeddings. However, the approaches implicitly treat these deep models as shallow because the information extracted for detection simply focus on the last layer. Indeed, deep neural networks typically rely on final-layer embeddings as compact, semantically rich representations of the input. But, a sole reliance on the last layer overlooks the neural structure through which they are obtained. Therefore, we challenge this widespread wisdom and propose exploring intermediate-layer representations.

Interestingly, early work in convolutional models showed distinct functionality across layers [2] however pretrained vision transformers trained with diverse objectives-supervised, contrastive, or masked modeling-have not been thoroughly examined in this context. Understanding how in- and out-of-distribution semantics are distributed across depth may offer means for improving detection robustness and generalization. Our study revisits the problem of zero-shot OOD detection to ask: Can intermediate representations be systematically leveraged to improve detection performance?

To investigate, we conduct a comprehensive analysis across seven vision backbones, including CLIP, DINOv2 [54], MAE [23], MoCo-v3 [8], SiGLiP-v2 [61], Perception Encoder [4], and supervised

2

Figure 1: Effect of layer combination length on OOD detection across architectures. The line plot (left) shows average FPR@95 as a function of fused layer count N , where all combinations include the final layer (e.g., N =3 may yield layers { 1 , 2 , 11 } ). Shaded regions indicate the full range of FPR@95 values across combinations. The gray zone marks the baseline (single-layer) case. The bar plot (right) compares baseline FPR@95 (gray) with the best result from layer fusion (colored) for each model. See Appendix M.16 for more details.

<!-- image -->

ViTs [14]. We examine how semantic diversity, entropy structure, and prediction stability vary across depth and training paradigms.

Our findings reveal strong contrasts between architectures. CLIP models exhibit high inter-layer diversity, broad top-1 agreement, and smooth entropy transitions across depth-properties that enable stable and effective multi-layer fusion. In contrast, supervised and self-supervised ViTs often show abrupt shifts or redundancy across layers, limiting the utility of intermediate feature fusion. Notably, more recent contrastive models such as SIGLIP and the Perception Encoder do not replicate CLIP's robustness, suggesting that architectural design plays a central role in enabling successful fusion strategies. As shown in Figure 1, combining three to six well-selected layers yields the largest performance gains, while fusing more layers tends to introduce redundancy or noise, diminishing returns. Additional analysis appears in Section 3.

Motivated by these insights, we introduce a simple yet effective extension of Maximum Concept Matching (MCM) [47]. Our method performs entropy-guided selection of informative intermediate layers and aggregates their outputs to improve OOD detection. It is fully training-free, inferenceonly, and does not require OOD data, fine-tuning, or prompt engineering. The method improves performance, reducing FPR@95-the false positive rate when the true positive rate is at 95%-by over 12 percentage points on far-OOD datasets and more than 7 points on near-OOD benchmarks (see Section 5). These results are consistent with the ablations presented in Section 6.

Extensive ablations confirm that our method is robust to hyperparameters such as fusion length, histogram resolution, and temperature scaling. We also observe that the best-performing layer combinations are consistent within a dataset, but differ across domains, emphasizing the need for dataset-adaptive strategies. These results challenge the assumption that the final layer is always optimal for OOD detection. By selectively leveraging signals from across the network, our approach offers a practical and general enhancement to training-free OOD methods.

## Our contributions are as follows:

- We conduct a systematic analysis of intermediate representations across seven vision backbones and three training paradigms. While early layers alone are weak, aggregating selected intermediate layers consistently improves OOD robustness (Section 3).
- We propose a training-free extension of MCM that uses entropy-guided selection to identify and fuse informative layers. The method remains training-free and architecture-agnostic for CLIP (Section 4).
- We validate our approach across six OOD benchmarks and two ID datasets (ImageNet-1K and Pascal-VOC), achieving consistent gains in both single-label and multi-label settings. Our method outperforms prior inference-only baselines, particularly on contrastive architectures (Section 5).

## 2 Preliminaries

OOD Detection. Out-of-distribution (OOD) detection aims to identify test-time inputs drawn from a distribution different from the training one. Classical methods define a scoring function G ( x ) : X → R using final-layer outputs of models pretrained on the in-distribution (ID) data. These approaches rely on post hoc confidence scores-e.g., softmax, energy, or entropy- overlooking earlier representations that may encode complementary semantic cues.

Zero-shot OOD Detection with Pretrained Vision-Language Models. Recent advances in largescale contrastive vision-language models (e.g., CLIP [58]) have enabled zero-shot classification by aligning images and text within a shared embedding space. These models bypass conventional supervised training by leveraging natural language supervision for OOD detection. Given an ID label set Y = { y 1 , . . . , y K } , class-specific text prompts such as 'a photo of a y i ' are embedded via a text encoder T ( y i ) , while images are encoded through a visual encoder I ( x ) . Classification is then performed via cosine similarity between modalities:

<!-- formula-not-decoded -->

where cos( · , · ) is the cosine similarity and τ a temperature parameter.

We use the Maximum Concept Matching (MCM [47]) approach to OOD detection, which flags an example as OOD if max y p ( y | x ) &lt; θ , with θ controlling sensitivity.

## 3 Exploring the Value of Intermediate Layers for OOD Detection

We investigate whether leveraging intermediate-layer representations-rather than relying solely on final outputs-can improve OOD detection in the zero-shot setting. To assess the potential of these representations, we extract features from intermediate layers throughout the network and evaluate their predictive utility for OOD detection. Specifically, let ϕ v denote the visual encoder of a given model a . For each architecture, we obtain intermediate representations { ϕ ( ℓ ) v } from a predefined set of layers, 1 allowing us to analyze how OOD-relevant information emerges and evolves throughout the network. Our analysis is conducted on ImageNet-1K as the in-distribution dataset, along with four diverse OOD datasets: iNaturalist [28], Places [75], SUN [69], and Textures [10]. Further details can be found in Appendix D.

For classification models, we apply the native classification head h to each intermediate representation ϕ ( ℓ ) v . For contrastive models lacking such heads, we normalize ϕ ( ℓ ) v as done at the final layer and apply the pretrained projection head p . To address layer-wise dimensionality mismatches-especially in ResNets-we insert a fixed random linear projection before applying p (see Appendix I).

Importantly, this procedure is entirely inferenceonly: no new adapters or layers are introduced, no training phase is involved, and no in-distribution labels are used. All components are reused from the pretrained network. As a result, for each layer ℓ , we obtain a class proba-

Figure 2: Layer-wise OOD detection performance across architectures. Most architectures exhibit their best performance near the final layer, while early layers generally under-perform.

<!-- image -->

bility distribution computed as h ( ϕ ( ℓ ) v ( x )) , where x is the input image and ϕ ( ℓ ) v ( x ) is the intermediate representation at layer ℓ .

To compute OOD scores, we use confidence-based indicators. For classification models, we extract the maximum softmax probability per layer (MSP) [25]. For contrastive models, we apply Maximum Concept Matching (MCM) [48], which computes softmax-normalized cosine similarities between

1 Layer definitions follow standard naming and indexing of the Hugging Face Transformers library [68].

image and class embeddings. This yields a score tensor S ∈ R N × L × C , where N is the number of input samples, L the number of layers, and C the number of classes. Each entry S n,ℓ,c denotes the classc probability at layer ℓ for input n .

Impact of a Single Layer for OOD Detection We begin by assessing the performance of OOD detection using features of individual layers ℓ in various architectures. As shown in Figure 2, most models reach their best AUROC and lowest FPR@95 at the final layer, highlighting the discriminative power of deeper representations. For instance, CLIP ViT-B/16 and PEViT-B16 show a consistent rise in AUROC with depth. Some intermediate layers offer isolated improvements, such as early gains in CLIP RN50 or reduced FPR@95 at layer 3 in DINOv2, but these cases are infrequent and strongly depend on the architecture. Similar trends are observed in MAE and MoCo v3, where mid-level layers provide only minor advantages; see Appendix M.17 for details.

Thus, relying on a single intermediate layer for OOD detection yields, at best, marginal and inconsistent improvements over the final layer, which remains the most reliable source of discriminative information.

Impact of Multi-Layer Fusion for OOD Detection Given the limited gains from single-layer prediction, we explore whether combining multiple layers can improve OOD detection. To this end, we aggregate OOD scores from selected subsets of intermediate layers, always including the final layer due to its strong baseline performance. These subsets span all layers and include all possible combinations for a given subset size.

Let S ∈ R N × L denote the matrix of confidence-based OOD scores, where S n,ℓ is the OOD scores for the instance n at layer ℓ . Given a subset of layers L ⊆ { 1 , . . . , L } , we computed a fused score for each instance by averaging over the selected layers:

<!-- formula-not-decoded -->

This procedure yields a single fused OOD score ¯ p n per instance. Figure 1 shows the average false positive rate (FPR) at 95% as a function of the number of layers used in the fusion. Results are averaged over four OOD benchmarks-SUN, Places, Textures, and iNaturalist- with shaded regions representing the full range of FPRs (minimum to maximum) observed across all layer combinations of equal size. The plot reveals substantial differences in the FPR metric across architectures, with some models-such as SiGLip2 [61] and DINOv2 [54]- starting with notably higher FPRs, highlighting variability in baseline OOD performance.

Fusion improves performance in model-specific ways. CLIP variants show consistent FPR reduction as layers are added. MoCoV3 and MAE gain moderality, while supervised ViTs sees minimal change. SiGLIp2 and DINOv2 also improve, though less reliably. In contrast, larger models like CLIP ViT-L/14 and PEv1-B16-224 exhibit rising FPR with longer combinations, suggesting that aggregating too many layers may hurt performance when predictions conflict or lack discrimination.

While fusing multiple layers can enhance OOD detection, the specific number and choice of layers that contribute to this improvement vary across models and datasets. This raises the question of whether certain layers are consistently more informative. In Appendix E, we show that top-performing layer combinations tend to be structurally consistent within a given reference ID dataset, yet diverge significantly across datasets. These findings suggest that fusion strategies must be adapted to the domain, as no universal layer subset generalizes well across distributional shifts.

These results highlight the potential of multi-layer fusion for OOD detection while raising key questions: what makes certain combinations effective, and how do they help?

Uncovering the Diversity of Intermediate Layers To better understand what makes certain layer combinations effective, we analyze the representational diversity of intermediate layers across models. We use SVCCA [59] to quantify structural similarity between layers by measuring the alignment of their activation patterns. As shown in Figure 3, contrastive models like CLIP exhibit a sharp decline in similarity as the layer distance ∆ increases, indicating high inter-layer representational diversity and more progressive transformations. In contrast, supervised models and hybrid architectures display

## Layer-wise SVCCA Decay: CLIP vs. Supervised Methods

Figure 3: Layer-wise SVCCA similarity as a function of layer distance ∆ for contrastive and classic models. CLIP models (blue) exhibit lower similarity across layers, indicating more progressive transformations, while Supervised models (yellow) retain higher layer-wise redundancy. MAE MOCO-v3 ViT Dino-V2 CLIP ViT-B/32 CLIP ViT-B/16 CLIP ViT-L/14 CLIP SiGLip2-B/16 CLIP PEv1-B16-224

<!-- image -->

0.0

0.2

0.4

0.6

0.8

1.0

0.2

0.4

0.6

0.8

1.0

0.0

0.2

0.4

0.6

0.8

1.0

0.2

0.4

0.6

0.8

1.0

0.2

0.4

0.6

0.8

1.0

0.2

0.4

0.6

0.8

1.0

0.2

0.4

0.6

0.8

1.0

0.2

0.4

0.6

0.8

1.0

0.2

0.4

0.6

0.8

1.0

Figure 4: Top-1 agreement similarity across transformer layers for various vision models. Each matrix shows pairwise agreement in predicted top-1 classes across layers.

<!-- image -->

slower SVCCA decay, suggesting higher redundancy and smoother transitions between layers. These models tend to refine their representations gradually, concentrating most semantic abstraction near the final layers. By comparison, contrastive models distribute semantic information more evenly across depth, yielding more diverse intermediate features that may offer greater complementarity for fusion.

Layerwise Prediction Agreement and Entropy To complement the SVCCA analysis, we measure top-1 agreement across layers to assess prediction consistency. As shown in Figure 4, CLIP models-especially ViT-B variants-exhibit broad zones of agreement, reflecting stable prediction behavior across depth. This stability diminishes in larger models like ViT-L/14. Other contrastive models, such as SiGLip2 and PE, show lower agreement, suggesting abrupt shifts in prediction space. Supervised and self-supervised models (e.g., ViT, MoCo v3) display strong diagonals in their agreement matrices, suggesting independent prediction evolution across layers and limited stability, which can undermine the effectiveness of intermediate fusion.

To further contextualize these observations, Figure M.18 shows the average class entropy across layers. Distinct calibration patterns emerge: supervised models and MAE exhibit sharp entropy drops near the final layers, indicating increasingly confident-often overconfident-predictions [22, 46]. Earlier layers, by contrast, retain high entropy with near-uniform class distributions, offering little semantic guidance This weak discriminative signal can limit their contribution in feature aggregation. In contrast, CLIP models display a more gradual change in entropy across depth, with moderate uncertainty in early layers and more confident predictions in deeper ones. This structured progression allows intermediate layers to contribute non-redundant yet coherent information. Conversely, models like SiGLip2 maintain uniformly high entropy, implying consistently flat softmax outputs that limit the effectiveness of multi-layer fusion. Perception Encoder [4] shows a similar issue, with little differentiation between earlier and later layers.

These trends provide an explanatory basis for the differing effectiveness of intermediate-layer fusion observed across architectures. Only architectures that balance inter-layer diversity and prediction consistency -most notably CLIP-derive substantial benefit from multi-layer fusion. In contrast, models with unstable or flat prediction profiles across depth may require more selective or weighted aggregation strategies to avoid destructive interference.

## 4 Proposed Method to Exploit Intermediate-Layer Representations

Motivated by the patterns uncovered in our analysis of intermediate-layer fusion, Figure 5 presents our training-free OOD detection framework G ( · ) , which leverages intermediate-layer representations

Figure 5: We propose a general approach to OOD detection that exploits features from intermediate layers of a visual encoder (left), extending the Maximum Concept Matching (MCM) method [47]. Section 3 analyzes the informativeness of intermediate features across architectures. Based on these insights, Section 4 introduces an entropy-based layer selector (right) that identifies the most reliable combination of layers for training-free OOD detection.

<!-- image -->

from pretrained vision-language models. We extend the Maximum Concept Matching (MCM) method across multiple layers and introduce an entropy-based layer selection mechanism that identifies the most discriminative combinations for OOD detection.

Text and Visual Encodings. Each in-distribution (ID) class label y i ∈ Y in is mapped to a text prompt (e.g., 'a photo of a y i ') and encoded into a semantic embedding T ( t i ) using a pretrained text encoder. On the visual side, backbones such as ViT or ResNet yield intermediate feature representations { ϕ (1) ( x ) , . . . , ϕ ( L ) ( x ) } . Each feature ϕ ( ℓ ) ( x ) is mapped into a shared embedding space via a projection head p , yielding I ℓ ( x ) = p ( ϕ ( ℓ ) ( x )) . This enables alignment between visual and textual modalities for similarity-based matching.

Maximum Concept Matching (MCM). For each test input x ′ , we compute cosine similarities between the visual embedding I ℓ ( x ′ ) and all text embeddings { T ( t k ) } K k =1 . The MCM score at layer ℓ is defined as:

<!-- formula-not-decoded -->

where τ is a temperature parameter that controls the sharpness of the distribution. This normalized score approximates the model's confidence at each layer.

Entropy-Based Layer Selection. To aggregate information across layers, we define a set of candidate combinations C = { C 1 , . . . , C M } , where each C m ⊆ { 1 , . . . , L } always includes the final layer. For each combination C m , we compute the aggregated score:

<!-- formula-not-decoded -->

To evaluate each combination C m , we compute the distribution { S MCM ( x ′ i ; C m ) } N i =1 over a set of unlabeled in-distribution samples { x ′ i } N i =1 , and bin the scores into a normalized histogram { p b } B b =1 . We then compute the entropy:

<!-- formula-not-decoded -->

A low-entropy distribution indicates confident and concentrated scores, which we use as a proxy for better ID-OOD separability. The optimal layer combination is then selected as:

<!-- formula-not-decoded -->

Table 1: Training-free OOD detection results on ImageNet-1K and Pascal-VOC as ID datasets. We report the false positive rate at 95% TPR (FPR95, ↓ ) and AUROC ( ↑ ) across six OOD datasets: iNaturalist, SUN, Places, DTD, ImageNet-22K, and COCO. Best results per column are highlighted in bold.

| Method          | Backbone   | iNaturalist   | iNaturalist   |       | SUN   | SUN   | Places Texture   | Places Texture   |       |       | ImageNet22K   | ImageNet22K   | COCO   | COCO   | Average   | Average   |
|-----------------|------------|---------------|---------------|-------|-------|-------|------------------|------------------|-------|-------|---------------|---------------|--------|--------|-----------|-----------|
|                 |            | FPR ↓         | AUC ↑         | FPR ↓ | AUC ↑ | FPR ↓ | AUC ↑            | FPR ↓            | AUC ↑ | FPR ↓ | AUC ↑         | FPR ↓         | AUC    | ↑      | FPR ↓     | AUC ↑     |
| ID - ImageNet1K |            |               |               |       |       |       |                  |                  |       |       |               |               |        |        |           |           |
| MCM             | ViT-B/32   | 33.85         | 93.62         | 40.99 | 91.56 | 46.71 | 89.25            | 60.90            | 85.03 | -     | -             | -             | -      |        | 45.61     | 89.87     |
| MCM             | ViT-B/16   | 30.67         | 94.63         | 37.41 | 92.57 | 43.67 | 89.96            | 57.34            | 86.18 | -     | -             | -             | -      |        | 42.27     | 90.83     |
| SeTAR+MCM       | ViT-B/16   | 26.92         | 94.67         | 35.57 | 92.79 | 42.64 | 90.16            | 55.83            | 86.58 | -     | -             | -             |        | -      | 40.24     | 91.05     |
| GL-MCM          | ViT-B/16   | 17.42         | 96.44         | 30.75 | 93.44 | 37.62 | 90.63            | 55.20            | 85.54 | -     | -             | -             | -      |        | 35.25     | 91.51     |
| SeTAR + GL-MCM  | ViT-B/16   | 13.36         | 96.92         | 28.17 | 93.36 | 36.80 | 90.40            | 54.17            | 84.59 | -     | -             | -             |        | -      | 33.12     | 91.32     |
| Ours            | ViT-B/16   | 15.98         | 96.90         | 45.58 | 89.69 | 35.71 | 92.72            | 25.51            | 94.84 | -     | -             | -             |        | -      | 30.70     | 93.54     |
| Ours            | ViT-B/32   | 12.02         | 97.64         | 28.98 | 93.37 | 35.69 | 91.61            | 39.36            | 91.07 | -     | -             | -             | -      |        | 29.01     | 93.42     |
| ID - Pascal-VOC |            |               |               |       |       |       |                  |                  |       |       |               |               |        |        |           |           |
| MCM             | ViT-B/32   | 34.80         | 95.35         | 30.60 | 93.74 | 37.70 | 91.99            | 51.60            | 91.68 | 55.00 | 91.16         | 59.10         | 89.23  |        | 44.80     | 92.19     |
| MCM             | ViT-B/16   | 10.51         | 97.93         | 30.45 | 94.25 | 36.11 | 91.86            | 53.21            | 91.77 | 53.82 | 91.12         | 57.10         |        | 89.02  | 40.20     | 92.66     |
| SeTAR+MCM       | ViT-B/16   | 4.38          | 98.70         | 26.24 | 94.95 | 28.67 | 93.28            | 50.32            | 92.32 | 44.61 | 92.63         | 49.80         | 89.68  |        | 34.00     | 93.59     |
| GL-MCM          | ViT-B/16   | 4.33          | 98.81         | 22.94 | 94.63 | 26.20 | 93.11            | 41.61            | 92.88 | 37.88 | 93.17         | 43.70         |        | 90.71  | 29.44     | 93.88     |
| SeTAR + GL-MCM  | ViT-B/16   | 3.01          | 99.04         | 21.76 | 94.98 | 24.00 | 93.73            | 37.61            | 93.87 | 33.46 | 94.24         | 40.60         |        | 91.48  | 26.74     | 94.56     |
| Ours            | ViT-B/32   | 32.18         | 94.40         | 35.60 | 92.61 | 49.17 | 92.34            | 33.72            | 95.78 | 54.36 | 91.34         | 57.60         |        | 89.97  | 43.77     | 92.74     |
| Ours            | ViT-B/16   | 2.19          | 98.92         | 19.70 | 95.77 | 19.53 | 95.11            | 44.08            | 92.06 | 34.70 | 91.15         | 41.60         |        | 88.09  | 26.97     | 93.52     |

Inference. At test time, the selected combination C ∗ is used to compute the final OOD score for any input x ′ using Eq. (4). This enables robust prediction by fusing complementary information across informative layers.

## 5 Experiments

## 5.1 Dataset and Experimental Settings.

We use two ID datasets: ImageNet-1K [12], a standard large-scale classification benchmark, and Pascal-VOC [15], a multi-object detection dataset. Following prior work [30, 51]. Detailed dataset descriptions and statistics are provided in Appendix D. For ImageNet-1K, we use iNaturalist [28], SUN [69], Places [75], and Textures [9] as OOD datasets, following existing protocols exposed in [30]. For Pascal-VOC, we use ImageNet-22K [60], iNaturalist , SUN , Textures and we add MS-COCO [42] as an auxiliary OOD source when Pascal-VOC is used as the ID dataset.

Models and Setup. We use CLIP [58] with a ViT-B/16 backbone as our primary contrastive vision model. All evaluations are performed in a fully training-free setting without any additional fine-tuning. To ensure comparability, we apply the softmax function with a fixed temperature τ = 1 . For consistency and to reduce variance in the fusion process, we fix the number of combined layers to L ≤ 5 (see 6). Our experimental setup uses a single NVIDIA RTX 4090 GPU. The selection of the optimal layer combination takes approximately 50 seconds for both ImageNet-ID and Pascal-VOC.

Evaluation Metrics and Baselines. We report (i) FPR@95 : the false positive rate when the ID true positive rate is 95% (lower is better), and (ii) AUROC : the area under the ROC curve for distinguishing ID and OOD samples (higher is better). We compare our method against recent training-free, zero-shot baselines, including MCM [48], GL-MCM [51], and SETAR [40].

## 5.2 Results

Table 5.1 presents training-free OOD detection results across six benchmarks, using ImageNet-1K or Pascal-VOC as ID datasets. Our method achieves the best average performance in both settings. With ViT-B/16, we obtain an FPR95 of 30.70% and AUROC of 93.54 on ImageNet-1K, and 26.97% for the FPR95 and 93.52 for the AUROC on Pascal-VOC. Gains are consistent across iNaturalist, Texture, COCO, and ImageNet-22K, which span fine-grained and complex domains. The largest improvements occur on COCO and Texture, where standard ID/OOD assumptions often break. ViTB/32 yields smaller gains on Pascal-VOC, likely due to its lower spatial resolution, echoing prior findings from GL-MCM and SeTAR that emphasize the importance of localized features. We provide further insight into the semantic abstraction patterns driving these improvements in F.

Unlike MCM, which uses only the final-layer [CLS] token, our method aggregates selected intermediate layers. On ImageNet-1K with ViT-B/16, this reduces FPR95 by 51.9% relative on iNaturalist, achieves 55.5% relative reduction on Texture, and 18.2% on Places. On Pascal-VOC with ViT-B/16, relative reductions of 79.2% on iNaturalist, 35.3% on SUN, 45.9% on Places, and 27.1% on COCO are observed. These gains are most pronounced in high-variability settings, where final-layer predictions tend to be overconfident.

Near OOD is better with intermediate layer We evaluate near-OOD performance on the NINCO and SSB-Hard benchmarks, following the protocol introduced in [73]. As shown in Table 2, leveraging intermediate-layer representations yields consistent improvements across both datasets. Our method outperforms the training-free baseline (MCM) by over 7.5% in AUROC and reduces the false positive rate by more than 10 points. Com-

Table 2: Near-OOD results. Best results per column are bolded.

| Method   | NINCO   | NINCO   | SSB-Hard   | SSB-Hard   | Average   | Average   |
|----------|---------|---------|------------|------------|-----------|-----------|
|          | AUC ↑   | FPR ↓   | AUC ↑      | FPR ↓      | AUC ↑     | FPR ↓     |
| MCM      | 73.53   | 79.69   | 62.71      | 90.47      | 68.12     | 85.08     |
| GL-MCM   | 76.03   | 74.35   | 66.13      | 87.42      | 71.08     | 80.885    |
| SeTAR    | 76.95   | 70.16   | 68.31      | 84.94      | 72.63     | 77.535    |
| LoCoOp   | 73.90   | 80.04   | 65.73      | 89.48      | 69.81     | 84.76     |
| SeTAR+FT | 77.56   | 70.86   | 69.68      | 83.99      | 73.62     | 77.425    |
| Ours     | 80.98   | 66.74   | 70.44      | 82.80      | 75.71     | 74.77     |

pared to fine-tuned approaches such as SETAR+FT, our method achieves a 1.92% lower FPR while maintaining competitive AUROC. These results indicate that the inductive biases encoded in intermediate-layer representations provide a more robust foundation for handling subtle distribution shifts, especially in near-OOD regimes, than fine-tuning on in-distribution data alone.

Different architectures and prompt-based methods To test portability across backbones and prompting strategies (Table 3), we find intermediatelayer fusion remains effective beyond the default setting. Unlike single-layer baselines, our method aggregates selected intermediate layers. With MCM on RN50x4, we obtain an FPR95 of 41.61% and AUROC of 90.80 , while ViT-L/14 achieves 37.09% for the FPR95 and 91.70 for the AUROC. Prompted ViTB/16 also benefits substantially: NegLabel reaches 23.79% FPR95 and 95.05 AUROC, while CSP achieves 15.52% for the FPR95 and 96.74 for the AUROC. These trends mirror our layer-agreement analysis: heterogeneous ResNet stages let fusion reduce variance and suppress spurious peaks, whereas ViT-L/14's higher inter-layer redundancy and sharper

Table 3: Intermediate vs last layer performance improvement across scoring rules.

| Method                                     | Average                                    | Average                                    |
|--------------------------------------------|--------------------------------------------|--------------------------------------------|
|                                            | FPR ↓                                      | AUC ↑                                      |
| Extra backbones (MCM)                      |                                            |                                            |
| MCM(Last Layer) - RN50x4                   | 45.16                                      | 89.95                                      |
| MCM(Intermediate) - RN50x4                 | 41.61                                      | 90.80                                      |
| MCM(Last Layer) - ViT-L/14                 | 37.16                                      | 91.66                                      |
| MCM(Intermediate) - ViT-L/14               | 37.09                                      | 91.70                                      |
| Prompt methods (composition with ViT-B/16) | Prompt methods (composition with ViT-B/16) | Prompt methods (composition with ViT-B/16) |
| NegLabel                                   | 25.40                                      | 94.21                                      |
| NegLabel + Int. Layers                     | 23.79                                      | 95.05                                      |
| CSP                                        | 17.51                                      | 95.76                                      |
| CSP + Int. Layers                          | 15.52                                      | 96.74                                      |

softmax (with τ = 0 . 01 ) leave less headroom. Dataset context matters J: the largest FPR drops appear on scene-centric SUN/Places where early layers retain layout/background cues; gains are smaller on DTD and iNaturalist, where final embeddings already align with object-centric semantics. Overall, fusion acts at the representation level and transfers robustly across architectures and prompts.

Orthogonality to Scoring Rules To evaluate whether our contribution acts at the representation level rather than tailoring a specific decision rule, we report results across multiple OOD scores (Table 4). The considered scores capture complementary statistics: peak sensitivity for MaxLogit [25] and MCM [47], and global aggregation for Energy [44], Entropy, and Variance [47]. Our method demonstrates consistent improvements across all scoring functions and architectures, with particularly strong gains on Vision Transformers where distributional

Table 4: Intermediate vs last layer performance improvement across scoring rules. Results averaged across SUN, Places365, DTD, and iNaturalist datasets.

| Method                 | ResNet                 | ResNet                 | ViT-B/32               | ViT-B/32               | ViT-B/16               | ViT-B/16               | Average                | Average                |
|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
|                        | AUC ↑                  | FPR ↑                  | AUC ↑                  | FPR ↑                  | AUC ↑                  | FPR ↑                  | AUC ↑                  | FPR ↑                  |
| Distributional Methods | Distributional Methods | Distributional Methods | Distributional Methods | Distributional Methods | Distributional Methods | Distributional Methods | Distributional Methods | Distributional Methods |
| Entropy                | +0.1                   | +0.0                   | +1.1                   | +2.5                   | +9.0                   | +16.0                  | +3.4                   | +6.2                   |
| Energy                 | +0.5                   | +0.2                   | +4.2                   | +0.4                   | +2.4                   | +0.6                   | +2.4                   | +0.4                   |
| Variance               | +0.1                   | +0.1                   | +1.0                   | +2.3                   | +8.9                   | +16.0                  | +3.3                   | +6.1                   |
| Max-based Methods      | Max-based Methods      | Max-based Methods      | Max-based Methods      | Max-based Methods      | Max-based Methods      | Max-based Methods      | Max-based Methods      | Max-based Methods      |
| MaxLogit               | +1.2                   | +4.0                   | -0.9                   | +0.3                   | +3.2                   | +8.6                   | +1.2                   | +4.3                   |
| MCM                    | +0.8                   | +3.5                   | +3.6                   | +16.6                  | +2.7                   | +11.6                  | +2.4                   | +10.2                  |

methods benefit most from intermediate-layer fusion (More details in K). These consistent im-

provements span both peak-based and aggregation-based metrics, ruling out score-specific artifacts and ensuring compatibility with pipelines that already fix a scoring rule. The consistency across this diverse family of scores confirms that intermediate-layer fusion provides broadly applicable representation improvements rather than optimizing for a particular decision boundary.

## 6 Discussion

Computational Cost Analysis of Intermediate Layer Usage We evaluate the computational implications of incorporating intermediate layer features across three CLIP architectures: RN50x4, ViT-B/16, and ViT-B/32. The study varies batch size from 1 to 8 and reports peak GPU memory, per-image latency, total inference time, and throughput (images per second). As shown

Table 5: Efficiency of intermediate-layer fusion across backbones and batch sizes.

| Architecture   | Memory (GB)   | Memory (GB)   | Latency (ms)   | Latency (ms)   |
|----------------|---------------|---------------|----------------|----------------|
|                | Last ↓        | All ↓         | Last ↓         | All ↓          |
| ResNet-50x4    | 0.41±0.00     | 0.45±0.04     | 67.70±60.43    | 73.80±63.52    |
| ViT-B/16       | 0.57±0.00     | 0.63±0.05     | 218.47±154.88  | 224.96±184.71  |
| ViT-B/32       | 0.57±0.01     | 0.59±0.02     | 211.46±175.48  | 219.46±167.30  |

in Table 5 and Figures L.13, L.15, and L.14, extracting intermediate features adds only modest overhead. Across all configurations, memory and latency remain within practical deployment limits, with minimal increases relative to the last-layer baseline. The effect is smallest for transformer architectures, which sustain competitive throughput. Full details appear in Appendix L.

Ablation: Layer Selection Strategies. We evaluate the effect of different heuristics for selecting intermediate layers to fuse in training-free OOD detection with CLIP ViT-B/16. Since evaluating all possible combinations is computationally infeasible, we consider seven strategies based on perlayer output statistics: entropy, kurtosis, standard deviation, Gini coefficient, Jensen-Shannon (JSD) divergence, average scoring across all layers, and random selection. Each method selects up to L = 5 layers using softmax-normalized logits at a fixed temperature τ = 1 .

These metrics are chosen for their sensitivity to the shape of the class probability distribution. Our hypothesis is that layers with peaked, asymmetric distributions are more effective for OOD separation, while flatter distributions yield less discriminative scores. For example, the Gini coefficient ( 1 -∑ i p 2 i ) increases with concentration, while JSD divergence measures deviation from uniformity in a bounded, symmetric form. The average baseline aggregates scores from all layers without selection, serving as a naive fusion strategy.

Figure 6 reports FPR@95 and AUROC on ImageNet-1K and Pascal-VOC. Entropy-based

Figure 6: Ablation of layer selection strategies. Comparison of selection heuristics.

<!-- image -->

selection consistently outperforms all other methods, achieving the lowest false positive rates and highest AUROC. In contrast, statistical heuristics such as kurtosis and standard deviation yield less reliable results. Although random selection performs similarly on average, its high variance (omitted for clarity) highlights the advantage of informed scoring.

Ablation: Sensitivity to Histogram Binning ( B ), Combination Length ( L ), and Temperature ( τ ) We evaluate the sensitivity of our method to three key hyperparameters: the number of histogram bins used for score normalization, the combination length (i.e., the number of layers fused), and the temperature applied to softmax outputs. Overall, the method exhibits stable performance across a broad range of configurations. For histogram binning, AUROC and FPR@95 remain consistent when using between 16 and 128 bins, with noticeable degradation outside this range due to underor over-discretization effects. In the case of combination length, fusing 3 to 6 layers yields optimal results, capturing sufficient semantic diversity without introducing excessive redundancy (see G for a more detailed analysis); shorter combinations lead to marginal improvements, while longer ones tend to incorporate noisy or less informative layers, reducing detection reliability. Finally, temperature scaling demonstrates that values in the range [0 . 25 , 2 . 0] preserve stable behavior, whereas values

Figure 7: Ablation studies across different hyperparameter dimensions: number of histogram bins (left), combination length (center), and temperature τ (right). ImageNet-1K Pascal-VOC.

<!-- image -->

below 0.25 cause a sharp decline in performance, likely due to overconfident predictions. These findings confirm that our method is robust to hyperparameter choices within the empirically validated ranges reported above.

## 7 Related Work

Out-of-distribution detection. OOD detection identifies inputs that deviate from the training distribution, a key challenge for safe deployment. Early methods used softmax-based confidence scores, such as MSP and ODIN [25, 41], or statistical distances like Mahalanobis [37]. Later work introduced energy-based scores [44] and Outlier Exposure (OE) [27] to better separate ID and OOD data. Other advances explore residual features [66], hyperspherical embeddings [49], and activation shaping [13]. Theoretical studies have analyzed learnability and scoring foundations [3, 16, 52], while modality-specific taxonomies [1] extend applicability. In NLP, OOD work focuses on intent classification [72], data augmentation [6], uncertainty [29], and unsupervised adaptation [70], often using pretrained transformers [26, 56].

Vision-language models for OOD detection. VLMs like CLIP [57] support zero-shot OOD detection by aligning image and text embeddings. Recent extensions explore concept matching [47, 51], prompt tuning [50, 39], and synthetic outliers [5]. Training-free variants such as SeTAR [40] further expand this space. Broader efforts leverage contrastive learning [63, 65], noisy supervision [33], and models like ViLT, VisualBERT, and Tip-Adapter [35, 38, 74]. Studies on calibration and robustness [67, 19] highlight strengths and weaknesses under distribution shift. While prior work focuses on final-layer embeddings, we show that intermediate CLIP layers offer improved detection, influenced by depth and patch size.

Intermediate-layer representations. Early distance-based detectors used intermediate features [37], but most modern methods rely on final-layer outputs. Recent work shows that intermediate layers carry transferable, shift-stable signals [62, 32], typically with added supervision or retraining. Lin et al. [43] target closed-set classifiers with intermediate exits and exit-wise energy plus inputconditioned early stopping, not VLM embeddings. Fayyad et al. [18] add supervised auxiliary heads; Guglielmo and Masana [20] mainly select a single discriminative layer on supervised backbones and small-scale benchmarks rather than fuse across depth. In contrast, our approach is training-free and VLM-native, select complementary layers using simple ID-only statistics.

## 8 Conclusion

We challenge the final-layer paradigm in OOD detection by showing that intermediate representations, especially in contrastive models like CLIP, encode diverse and complementary signals. Our trainingfree method selects and fuses informative layers via entropy-based scoring, yielding consistent improvements across near- and far-OOD benchmarks. These results highlight the architectural differences that shape layer utility and demonstrate the value of exploiting internal model structure for robust, inference-only OOD detection.

Acknowledgment This work was supported by Naval Group . The authors are affiliated with the Centre National de la Recherche Scientifique (CNRS) and the CROSSING Lab .

## References

- [1] Udit Arora, William Huang, and He He. Types of out-of-distribution texts and how to detect them. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP) , pages 10687-10701. Association for Computational Linguistics, 2021.
- [2] David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba. Network dissection: Quantifying interpretability of deep visual representations. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , pages 6541-6549, 2017.
- [3] Julian Bitterwolf, Alexander Meinke, Maximilian Augustin, and Matthias Hein. Breaking down out-of-distribution detection: Many methods based on ood training data estimate a combination of the same core quantities. In Proceedings of the 39th International Conference on Machine Learning (ICML) , volume 162, pages 2041-2074. PMLR, 2022.
- [4] Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed, Junke Wang, Marco Monteiro, Hu Xu, Shiyu Dong, Nikhila Ravi, Daniel Li, Piotr Dollár, and Christoph Feichtenhofer. Perception encoder: The best visual embeddings are not at the output of the network, 2025.
- [5] Chentao Cao, Zhun Zhong, Zhanke Zhou, Yang Liu, Tongliang Liu, and Bo Han. Envisioning outlier exposure by large language models for out-of-distribution detection, 2024.
- [6] Derek Chen and Zhou Yu. GOLD: Improving out-of-scope detection in dialogues using data augmentation. arXiv preprint arXiv:2109.03079 , 2021.
- [7] Mengyuan Chen, Junyu Gao, and Changsheng Xu. Conjugated semantic pool improves ood detection with pre-trained vision-language models, 2024.
- [8] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers, 2021.
- [9] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild, 2013.
- [10] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , pages 3606-3613, 2014.
- [11] Sanjoy Dasgupta. Experiments with random projection, 2013.
- [12] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Proc. CVPR , pages 248-255. Ieee, 2009.
- [13] Andrija Djurisic, Nebojsa Bozanic, Arjun Ashok, and Rosanne Liu. Extremely simple activation shaping for out-of-distribution detection, 2023.
- [14] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR) , 2021.
- [15] Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John M. Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. Int. J. Comput. Vis. , 88(2):303-338, 2010.
- [16] Zhen Fang, Yixuan Li, Jie Lu, Jiahua Dong, Bo Han, and Feng Liu. Is out-of-distribution detection learnable? In Advances in Neural Information Processing Systems (NeurIPS) , 2022.
- [17] Zhen Fang, Yixuan Li, Jie Lu, Jiahua Dong, Bo Han, and Feng Liu. Is out-of-distribution detection learnable?, 2023.
- [18] Jamil Fayyad, Kashish Gupta, Navid Mahdian, Dominique Gruyer, and Homayoun Najjaran. Exploiting classifier inter-level features for efficient out-of-distribution detection. Image Vision Comput. , 142(C), February 2024.

- [19] Stanislav Fort, Jie Ren, and Balaji Lakshminarayanan. Exploring the limits of out-of-distribution detection. In Conference on Neural Information Processing Systems (NeurIPS) , 2021.
- [20] Guglielmo, Gianluca and Masana, Marc. Leveraging intermediate representations for better out-of-distribution detection. In Proceedings of the 28th Computer Vision Winter Workshop (CVWW) , pages 53-61. Verlag der Technischen Universität Graz, 2025.
- [21] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks, 2017.
- [22] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning , pages 1321-1330. PMLR, 2017.
- [23] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners, 2021.
- [24] Dan Hendrycks, Steven Basart, Mantas Mazeika, Andy Zou, Joe Kwon, Mohammadreza Mostajabi, Jacob Steinhardt, and Dawn Song. Scaling out-of-distribution detection for realworld settings, 2022.
- [25] Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks, 2018.
- [26] Dan Hendrycks, Xiaoyuan Liu, Eric Wallace, Adam Dziedzic, Rishabh Krishnan, and Dawn Song. Pretrained transformers improve out-of-distribution robustness. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL) , 2020.
- [27] Dan Hendrycks, Mantas Mazeika, and Thomas Dietterich. Deep anomaly detection with outlier exposure, 2019.
- [28] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alexander Shepard, Hartwig Adam, Pietro Perona, and Serge J. Belongie. The inaturalist species classification and detection dataset. In CVPR , 2018.
- [29] Yibo Hu and Lanfang Kuang. Uncertainty-aware reliable exit classification. In Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD) , 2021.
- [30] Rui Huang and Yixuan Li. Mos: Towards scaling out-of-distribution detection for large semantic space, 2021.
- [31] Paul Jaccard. Etude de la distribution florale dans une portion des alpes et du jura. Bulletin de la Societe Vaudoise des Sciences Naturelles , 37:547-579, 01 1901.
- [32] Fran Jeleni´ c, Josip Juki´ c, Martin Tutek, Mate Puljiz, and Jan Šnajder. Out-of-distribution detection by leveraging between-layer transformation smoothness, 2024.
- [33] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In Proceedings of the 38th International Conference on Machine Learning (ICML) , 2021.
- [34] Xue Jiang, Feng Liu, Zhen Fang, Hong Chen, Tongliang Liu, Feng Zheng, and Bo Han. Negative label guided ood detection with pretrained vision-language models, 2024.
- [35] Wonjae Kim, Bokyung Son, and Ildoo Kim. ViLT: Vision-and-language transformer without convolution or region supervision. In Proceedings of the 38th International Conference on Machine Learning (ICML) , 2021.
- [36] Kasper Green Larsen and Jelani Nelson. The johnson-lindenstrauss lemma is optimal for linear dimensionality reduction, 2014.
- [37] Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting out-of-distribution samples and adversarial attacks, 2018.

- [38] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A simple and performant baseline for vision and language. arXiv preprint arXiv:1908.03557 , 2019.
- [39] Tianqi Li, Guansong Pang, Xiao Bai, Wenjun Miao, and Jin Zheng. Learning transferable negative prompts for out-of-distribution detection, 2024.
- [40] Yixia Li, Boya Xiong, Guanhua Chen, and Yun Chen. Setar: Out-of-distribution detection with selective low-rank approximation, 2024.
- [41] Shiyu Liang, Yixuan Li, and R. Srikant. Enhancing the reliability of out-of-distribution image detection in neural networks, 2020.
- [42] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. Microsoft coco: Common objects in context, 2015.
- [43] Ziqian Lin, Sreya Dutta Roy, and Yixuan Li. Mood: Multi-level out-of-distribution detection, 2021.
- [44] Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Tao. Energy-based out-of-distribution detection. In Advances in Neural Information Processing Systems , 2020.
- [45] Mark D. McDonnell, Dong Gong, Amin Parveneh, Ehsan Abbasnejad, and Anton van den Hengel. Ranpac: Random projections and pre-trained models for continual learning, 2024.
- [46] Matthias Minderer, Josip Djolonga, Rob Romijnders, Frances Hubis, Xiaohua Zhai, Neil Houlsby, Dustin Tran, and Mario Lucic. Revisiting the calibration of modern neural networks. In Advances in Neural Information Processing Systems , volume 34, pages 15682-15694, 2021.
- [47] Yifei Ming, Ziyang Cai, Jiuxiang Gu, Yiyou Sun, Wei Li, and Yixuan Li. Delving into out-of-distribution detection with vision-language representations, 2022.
- [48] Yifei Ming, Ziyang Cai, Jiuxiang Gu, Yiyou Sun, Wei Li, and Yixuan Li. Delving into out-of-distribution detection with vision-language representations, 2022.
- [49] Yifei Ming, Yiyou Sun, Ousmane Dia, and Yixuan Li. How to exploit hyperspherical embeddings for out-of-distribution detection?, 2023.
- [50] Atsuyuki Miyai, Qing Yu, Go Irie, and Kiyoharu Aizawa. Locoop: Few-shot out-of-distribution detection via prompt learning, 2023.
- [51] Atsuyuki Miyai, Qing Yu, Go Irie, and Kiyoharu Aizawa. Gl-mcm: Global and local maximum concept matching for zero-shot out-of-distribution detection, 2025.
- [52] Peyman Morteza and Yixuan Li. Provable guarantees for understanding out-of-distribution detection. In Proceedings of the AAAI Conference on Artificial Intelligence (AAI) , 2022.
- [53] Tuomas Oikarinen and Tsui-Wei Weng. Clip-dissect: Automatic description of neuron representations in deep vision networks, 2023.
- [54] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2024.
- [55] Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D Sculley, Sebastian Nowozin, Joshua V. Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift, 2019.
- [56] Alexander Podolskiy, Dmitry Lipin, Andrey Bout, Ekaterina Artemova, and Irina Piontkovskaya. Revisiting mahalanobis distance for transformer-based out-of-domain detection. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI) , 2021.

- [57] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021.
- [58] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021.
- [59] Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein. Svcca: Singular vector canonical correlation analysis for deep learning dynamics and interpretability, 2017.
- [60] Tal Ridnik, Emanuel Ben-Baruch, Asaf Noy, and Lihi Zelnik-Manor. Imagenet-21k pretraining for the masses, 2021.
- [61] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. Siglip 2: Multilingual visionlanguage encoders with improved semantic understanding, localization, and dense features, 2025.
- [62] Arnas Uselis and Seong Joon Oh. Intermediate layer classifiers for ood generalization, 2025.
- [63] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 , 2018.
- [64] Sagar Vaze, Kai Han, Andrea Vedaldi, and Andrew Zisserman. Open-set recognition: a good closed-set classifier is all you need?, 2022.
- [65] Feng Wang and Huaping Liu. Understanding the behaviour of contrastive loss. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , 2021.
- [66] Haoqi Wang, Zhizhong Li, Litong Feng, and Wayne Zhang. Vim: Out-of-distribution with virtual-logit matching, 2022.
- [67] Jim Winkens, Rudy Bunel, Abhijit Guha Roy, Robert Stanforth, Vivek Natarajan, Joseph R. Ledsam, Patricia MacWilliams, Pushmeet Kohli, Alan Karthikesalingam, Simon Kohl, Taylan Cemgil, S. M. Ali Eslami, and Olaf Ronneberger. Contrastive training for improved out-ofdistribution detection. arXiv preprint arXiv:2007.05566 , 2020.
- [68] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-ofthe-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations , pages 38-45. Association for Computational Linguistics, 2020.
- [69] Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In Proc. CVPR , pages 3485-3492. IEEE, 2010.
- [70] Keyang Xu, Tongzheng Ren, Shikun Zhang, Yihao Feng, and Caiming Xiong. Unsupervised outof-domain detection via pre-trained transformers. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL) , 2021.
- [71] Jingkang Yang, Kaiyang Zhou, Yixuan Li, and Ziwei Liu. Generalized out-of-distribution detection: A survey, 2024.
- [72] Li-Ming Zhan, Haowen Liang, Bo Liu, Lu Fan, Xiao-Ming Wu, and Albert Lam. Out-of-scope intent detection with self-supervision and discriminative training. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL) , 2021.
- [73] Jingyang Zhang, Jingkang Yang, Pengyun Wang, Haoqi Wang, Yueqian Lin, Haoran Zhang, Yiyou Sun, Xuefeng Du, Yixuan Li, Ziwei Liu, Yiran Chen, and Hai Li. Openood v1.5: Enhanced benchmark for out-of-distribution detection, 2024.

- [74] Renrui Zhang, Rongyao Fang, Peng Gao, Wei Zhang, Kunchang Li, Jifeng Dai, Yu Qiao, and Hongsheng Li. Tip-Adapter: Training-free CLIP-adapter for better vision-language modeling. arXiv preprint arXiv:2111.03930 , 2021.
- [75] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. TPAMI , 40(6):1452-1464, 2017.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Yes, the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope. The paper sets out to investigate whether intermediatelayer representations in pretrained vision-language models can improve training-free OOD detection.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations are discussed in Appendix A.

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

## Answer: [NA]

Justification: The paper does not include formal theoretical results or proofs. While it presents conceptual assumptions (such as the presence of progressive semantic abstraction across layers) and analyzes them empirically through metrics like SVCCA similarity and entropy, there are no formal theorems, lemmas, or mathematical proofs provided. The work is entirely empirical in nature, focused on evaluating and improving training-free OOD detection using intermediate-layer representations in pretrained vision-language models.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

## Answer: [Yes]

Justification: The paper provides sufficient detail to enable reproduction of its main experimental results in 5.1 and 6

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

Answer: [No]

Justification: The code is not yet ready for public release, but we plan to share it upon publication to support future research.

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

Justification: The paper provides all necessary experimental details. Our method does not require training, but we specify how features are extracted, how entropy-based layer selection is performed, and how evaluation metrics are computed. Additional implementation details are provided in 5.1, 6 and 6 to ensure full reproducibility.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Our method is fully deterministic and yields identical results across runs, so reporting standard deviation is unnecessary. For baselines involving stochastic processes, we report the average and standard deviation over three runs.

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

Justification: Our experimental setup uses a single NVIDIA RTX 4090 GPU (see Section 5.1 for details). All evaluations and ablations were performed on this hardware.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: This work fully complies with the NeurIPS Code of Ethics. It does not involve human subjects, crowdsourcing, or the release of models or datasets that pose risks of harm, privacy violations, or misuse. A more detailed discussion in B

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: This work discusses this in Appendix C.

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

Justification: This work does not release any models, datasets, or assets that pose a risk of misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification:

The paper relies exclusively on publicly available pretrained models from

Hugging Face [68] and benchmark datasets described in Appendix D.

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

Justification: Our work does not introduce any new datasets, models, or code assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This work does not involve any crowdsourcing or research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve human subjects research or any form of data collection from study participants, including crowdsourcing.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: This work does not involve large language models (LLMs) as part of its core methodology, analysis, or contributions.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Limitations

Despite demonstrating consistent improvements, several limitations highlight promising directions for future research. First, while our method benefits from the structured and diverse intermediate representations found in architectures like CLIP, it remains less effective in models with flatter or redundant representations, such as MAE or Perception Encoder. Future work could explore adaptive or weighted fusion strategies to mitigate these limitations. Second, although our entropy-based layer selection is inference-only and label-free, the selected combinations vary significantly across ID datasets. This domain sensitivity suggests the need for more generalizable selection mechanisms, potentially incorporating Outlier Exposure or few-shot adaptation. Third, our study is restricted to visual inputs. Extending the method to other modalities-such as audio or language-could further test the versatility of intermediate-layer fusion for OOD detection, especially in domains where semantic abstractions may emerge differently across depth.

## B Ethical Considerations

This work proposes a training-free method for out-of-distribution (OOD) detection based on intermediate-layer fusion in pretrained vision-language models. The method does not involve the collection or use of personal data, nor does it introduce new generative components or learning from sensitive content. It operates entirely at inference time using publicly available models and datasets. As such, it poses minimal risk in terms of privacy, fairness, or misuse. Nonetheless, we acknowledge that robust OOD detection plays a vital role in the safe and ethical deployment of machine learning systems. Future research should continue to examine the implications of model biases and domain shifts to support fair and transparent AI behavior in real-world applications.

## C Societal Impact

Improving OOD detection is crucial for the reliability of machine learning systems deployed in openworld environments, including safety-critical domains such as healthcare diagnostics, autonomous vehicles, and security applications. Our method contributes toward the development of AI systems that can identify and appropriately reject unfamiliar or anomalous inputs, thus preventing overconfident and potentially harmful predictions. Additionally, accurate OOD detection may support tasks such as active learning, human-in-the-loop filtering, and data selection for continual learning. By enabling more trustworthy model behavior under distributional shift, this line of research contributes to the broader goal of responsible and safe AI deployment.

## D Datasets

Following prior work [30], we adopt the standardized configuration for in-distribution (ID) and out-of-distribution (OOD) datasets that has been widely used in recent CLIP-based OOD detection benchmarks. Specifically, we use ImageNet-1K [12] and Pascal-VOC [15] as ID datasets. For OOD evaluation, we follow the curated protocol proposed in MoS [30], which provides de-duplicated subsets of four OOD datasets: iNaturalist [28], SUN [69], Places [75], and Texture (DTD) [10]. Additional OOD sets include ImageNet-22K [60] and MS-COCO [42].

## In-Distribution Datasets

ImageNet1K ImageNet-1K [12] serves as our primary in-distribution dataset. Consistent with prior evaluations [30, 48], we assess OOD detection using the full 50,000-image validation split, which spans 1,000 object categories and is widely adopted for benchmarking zero-shot performance.

Pascal-VOC For object-centric evaluation, we utilize Pascal-VOC [15] as an ID dataset. Following the setup introduced by Miyai et al. [51], we select samples containing only a single labeled object per image. To ensure comparability with SeTAR [40], we adopt the 906-image test set configuration used in their study.

## Out-of-Distribution Datasets

iNaturalist iNaturalist [28] is a large-scale dataset of flora and fauna images. We employ the 10,000-image version curated by Huang et al. [30], where overlapping classes with ImageNet-1K are removed to ensure distributional shift.

Places The Places dataset [75] contains millions of labeled scenes across various environments. For evaluation, we rely on a 10,000-image subset filtered by [30] to eliminate semantic overlap with the ID categories.

SUN SUN [69] features a broad collection of indoor and outdoor scene images. We use the filtered version compiled by [30], which retains 10,000 samples that are semantically distinct from ImageNet-1K.

Texture (DTD) The Describable Textures Dataset [10] includes fine-grained texture patterns organized into 47 visual descriptors. In line with [30], we employ the full set of 5,640 images, which are disjoint from the object categories in our ID datasets.

ImageNet22K ImageNet-22K [60] extends the standard ImageNet taxonomy to over 21,000 categories. We use the version filtered by Ming et al. [48], which excludes all classes overlapping with ImageNet-1K and Pascal-VOC, to serve as OOD input in both evaluation settings.

MS-COCO MS-COCO [42] is used in a filtered setting (COCO-OOD) containing 1,000 samples that are class-wise disjoint from Pascal-VOC. We use the version from [48] to test OOD detection performance when Pascal-VOC is the ID distribution.

## Near-OOD Datasets

We follow the recommendations from OpenOOD [73] to evaluate near-OOD detection using two challenging benchmarks that emphasize subtle distributional shifts while maintaining high visual similarity to in-distribution data.

NINCO NINCO [24] is a curated benchmark for evaluating nuanced OOD detection in natural images. It includes near-OOD samples that are visually similar to ImageNet-1K categories but originate from distinct distributions.

SSB-Hard SSB-Hard [64] introduced for open-set recognition on ImageNet-1K. It provides only OOD samples with minimal visual deviation but distinct semantics.

## E Diversity and Consistency of Oracle Layer Combinations

We assess the internal diversity of oracle-selected layer combinations in Figure E.8 using the Jaccard distance [31], computed among the top-10 performing combinations. A lower Jaccard distance indicates stronger agreement in the selected layers across different top-performing combinations. As shown, both ImageNet and Pascal-VOC exhibit low pairwise distances within their top-10 selections (mean Jaccard distances of 0.480 and 0.477, respectively), suggesting high internal consistency. This implies that, once an ID dataset is fixed, the best-performing combinations tend to share many common layers, indicating that certain representations are consistently beneficial for OOD detection within each domain.

However, this consistency does not extend across datasets. While individual ID datasets demonstrate stable oracle combinations, the actual sets of selected layers differ substantially between ImageNet and Pascal-VOC. This observation is further supported by Figure E.9, which quantifies the overlap between topk layer selections across datasets. Notably, the intersection between the top-50 combinations of ImageNet and Pascal comprises only four shared configurations. These shared combinations yield an average FPR of 0.300, which is a worst fpr than the 0.288 average obtained by selecting layers independently for each ID dataset using an entropy-based heuristic. This discrepancy indicates that a single fixed set of layers is inadequate for generalizing across ID datasets, and that the optimal composition of layers for OOD detection is highly dataset-dependent.

These findings underscore two key insights: (1) within a given ID dataset, the top-performing layer combinations exhibit high structural consistency, and (2) across different ID datasets, the optimal configurations diverge significantly, reflecting distinct underlying representational demands. This analysis reinforces the conclusion that static fusion strategies are insufficient for general-purpose OOD detection.

Figure E.8: Jaccard distance between the top-10 oracle combinations under two different ID datasets. Lower values indicate higher agreement between selected layer sets.

<!-- image -->

Figure E.9: Overlap of topk oracle layer combinations between ImageNet and Pascal-VOC. Intersection regions are shaded in gray and color-matched to the scatterplot. The bold decimal values indicate the average FPR computed across all OOD datasets evaluated under each ID configuration (ImageNet or Pascal-VOC).

<!-- image -->

## F Semantic concept emergence across layers.

Figure F.10 offers an interpretive bridge between our theoretical foundations (Section H), representational analyses (Section 6), and empirical evaluation of intermediate-layer fusion strategies. It visualizes the layer-wise distribution of six high-level semantic categories, as defined by [53], across eight datasets. Each curve denotes the average proportion of concept-relevant descriptors matched at each transformer layer, thereby revealing both architectural regularities and dataset-specific semantic emergence.

This figure directly substantiates Assumption A1 (Layered Semantics) from Section H. Low-level concepts (e.g., colors , textures ) consistently peak in early layers (typically layers 2 to 5), while high-level concepts (e.g., objects and machines , activities ) peak in deeper layers (layers 9 to 12). This stratification reinforces the canonical coarse-to-fine information processing in vision transformers, but here it is quantified via explicit semantic concept activation , rather than inferred from feature abstraction or classification accuracy. These results confirm that semantic depth varies across layers, validating our use of intermediate-layer fusion to capture complementary features.

Moreover, Figure F.10 deepens the interpretation of the SVCCA-based similarity trends in Figure 3. As discussed in Section 6, contrastive models exhibit lower similarity across layers, indicative of

greater representational diversity. The semantic traces in Figure F.10 offer an explanation: layers specialize in distinct concept types. In particular, the delayed emergence of categories such as objects supports our hypothesis that last-layer-only approaches may overlook crucial intermediate semantics. This aligns with the performance gains we observe when fusing information across multiple layers.

Finally, we note that the concept distributions are remarkably consistent across datasets, suggesting that concept emergence is not purely data-driven but also reflects model-internal inductive biases. This insight supports the transferability of our layer selection heuristics: strategies optimized on one dataset (e.g., ImageNet) remain effective across others (e.g., COCO, SUN), as demonstrated in our zero-shot evaluations for OOD detection.

Figure F.10: Layer-wise distribution of semantic concepts across six high-level categories proposed in [53] on CLIP ViT-B/16. Each curve shows the average proportion of concept words associated with a given category that are matched in each transformer layer, evaluated independently for each dataset.

<!-- image -->

## G Understanding the Role of Intermediate Layers

Empirical Validation of Entropy-based Selection. Our entropy-based selection criterion is empirically validated in Figure G.11, which shows the relationship between entropy and the average false positive rate (FPR@95) across multiple challenging OOD datasets (iNaturalist, Textures, SUN, and Places) using CLIP ViT-B/32. Each point corresponds to a specific combination of intermediate layers.

Two distinct clusters emerge: low-entropy combinations achieving low FPR@95, and high-entropy combinations approaching baseline performance. A consistent positive correlation between entropy and FPR@95 is observed across architectures, suggesting that entropy can be exploited to unsupervisedly select promising layer combinations for OOD detection.

Importantly, we find that this relationship strengthens when restricting the maximum combination length. Shorter combinations yield more distinct clustering toward low entropy and low FPR@95, while longer combinations introduce redundancy and noise, weakening the discriminative signal.

Figure G.11: Entropy vs. FPR@95 for different layer combinations, highlighting good and bad selections. Bottom: ID and OOD score distributions for representative examples.

<!-- image -->

Effect of the size of the combinations in relation between average FPR and entropy. Weobserve that constraining the maximum combination length strengthens the relationship between entropy and detection performance. Specifically, shorter combinations yield more coherent clustering toward low entropy and low FPR@95, as they are less likely to include noisy or redundant layers. In contrast, longer combinations often aggregate conflicting or less informative signals, diluting discriminative power and degrading OOD separation.

This trend suggests that limiting combination length acts as a natural form of regularization, reducing fusion noise and producing more robust, confidently separated layer subsets. These findings reinforce entropy as a principled selection criterion: when applied over constrained layer sets, it consistently identifies combinations that maximize zero-shot OOD detection performance. Further supporting evidence is provided in Section 6.

(a) ViT-B/32: Entropy vs. AVG FPR across varying combination lengths. Lower entropy correlates with improved OOD detection.

<!-- image -->

(b) ViT-B/16: Similar positive correlation between entropy and FPR, validating robustness of entropybased selection.

<!-- image -->

Figure G.12: Empirical validation of entropy-based selection for OOD detection. Across both ViT-B/32 and ViT-B/16, lower entropy values correspond to lower false positive rates (FPR). Shorter combinations reduce noise, yielding more confident and effective selections.

## H Why Intermediate Layers Help: A Theoretical View

Confidence-based scoring across depth. Let x ∈ X be an input sample and let ϕ ( ℓ ) v ( x ) ∈ R d ℓ denote the visual feature at layer ℓ ∈ L of the pretrained encoder. For pretrained classification models, we define S n,ℓ = max y ∈Y Softmax c ( h ( ℓ ) ( ϕ ( ℓ ) v ( x ))) (MSP score). For contrastive vision-language models like CLIP, we follow the procedure in Section 2, applying the pretrained image-to-text projection p to layer ℓ , and define S n,ℓ = max y ∈Y Softmax y (cos( p ( ϕ ( ℓ ) v ( x )) , T ( y )) /τ ) (MCM score). These scores form the confidence tensor S ∈ R N × L , where each element S n,ℓ is the scalar OOD score for instance n at layer ℓ .

Motivation for fusion. For a subset of layers L ⊆ { 1 , . . . , L } , define the fused score as:

<!-- formula-not-decoded -->

where ¯ S n represents the average confidence score across selected layers for instance n (e.g the set of layers { 1 , 2 , 12 } ). Fusion strategies aim to reduce noise and increase robustness by integrating complementary representations across depth.

Assumptions for effective fusion. We propose the following sufficient conditions for the effectiveness of intermediate-layer fusion:

- A1 Semantic progression: Deeper layers encode more task-aligned representations; shallow layers remain task-agnostic.
- A2 Representation diversity: Layer-level scores { S n,ℓ } ℓ ∈L are not perfectly correlated, i.e., Var[ ¯ S n ] &gt; 0 .

A3 Prediction consistency: Scores fluctuate more for OOD than ID samples:

<!-- formula-not-decoded -->

Theoretical insight. Under mild assumptions-bounded scores, non-positive pairwise covariances (empirically supported in Fig. 3), and conditions A1-A3 -we can show that fusion reduces variance in OOD scores and increases the separation between ID and OOD distributions in expectation.

Proposition 1 (Fusion reduces variance and amplifies ID-OOD gap) . Let ¯ S ( x ) denote the average confidence score across a subset of layers L ⊆ { 1 , . . . , L } . Then:

<!-- formula-not-decoded -->

Thus, fusion reduces score variance on OOD data and increases the expected confidence gap between ID and OOD samples-yielding improved detection relative to any individual layer.

## Discussion

The theoretical conditions outlined above are well-aligned with our empirical observations. CLIP models exhibit high inter-layer representational diversity (as indicated by low SVCCA similarity) and stable prediction behavior (as reflected by low top-1 disagreement), satisfying both A2 and A3 . As shown in Figure 1, these properties translate into consistent gains from multi-layer fusion. In contrast, models such as MAE and DINOv2-characterized by limited entropy variation and unstable intermediate predictions-frequently violate these assumptions, resulting in diminished or even detrimental effects when layers are combined.

## I Random Projection for Resnet Architecture

To harmonize feature map dimensions across architectures and layers-and to enable alignment with a shared text embedding space-we apply a lightweight two-stage projection. Given a feature tensor F ∈ R C in × H × W , we first apply adaptive average pooling to resize the spatial dimensions to a fixed target shape ( H target , W target ) :

<!-- formula-not-decoded -->

Next, a 1 × 1 convolution projects the feature channels to C target, which matches the dimensionality of the text features:

<!-- formula-not-decoded -->

This operation ensures that all visual representations, irrespective of their originating layer or architecture, are transformed into a common shape and an embedding space compatible with the text modality. We interpret the 1 × 1 convolution as a random linear projection over the channel dimension. Unlike spatial convolutions, this operation applies an independent linear transformation at each spatial location, akin to a fully connected layer applied per pixel. When randomly initialized (e.g., using He or Xavier initialization), such projections approximately preserve the geometry of the input space with high probability, as guaranteed by the Johnson-Lindenstrauss lemma [11, 36, 45]. This allows the projected visual features to remain discriminative while enabling seamless fusion or comparison with the corresponding text representations.

## J Different architectures and prompt-based methods

To test portability across backbones and prompting strategies (Table J), we find intermediate-layer fusion remains effective beyond the default setting: with MCM on RN50x4 it yields AUROC + 0.85; FPR -3.55, while ViT-L/14 improves more modestly. Prompted ViT-B/16 also benefits, with NegLabel (AUROC + 0.84; FPR -1.61) and CSP (AUROC + 0.98; FPR -1.99). These trends mirror our layer-agreement analysis: heterogeneous ResNet stages let fusion reduce variance and suppress spurious peaks with MCM. Over this, this results highlight that the random projections I are able to improve this performance and specific in this network the intermediate layers is able to improve the results in every ood dataset.

Whereas ViT-L/14's higher inter-layer redundancy and sharper softmax (with τ = 0 . 01 ) leave less headroom. Dataset context matters: the largest FPR drops appear on scene-centric SUN/Places where early and mid layers retain layout/background cues; gains are smaller on DTD and iNaturalist, where final embeddings already align with object-centric semantics. This difficult can be generated by the deepness of the network that generate a mordissimilar behavior acrees the extra layer that has in comparison with vit-B/16 ad vit=b/32 and the entropy in its output as we show in Sec 3.

Prompt-based methods such as CSP [7] and NegLabel [34] typically rely on external resources (e.g., WordNet hierarchies) or handcrafted prompts and use method-specific scoring. Our approach operates at the representation level by fusing intermediate layers, making it orthogonal to prompt design and broadly compatible with existing scoring rules. Applied to NegLabel, fusion yields consistent improvements across most datasets. CSP, although more memory-intensive, often benefits even more, likely because its synthetic negatives enlarge the ID vs OOD margin and better exploit complementary mid-layer cues.

Table 6: Training-free OOD detection with additional backbones and prompt-based methods (ImageNet-1K as ID). Base : without intermediate fusion; +Fusion : our training-free intermediate fusion. Best per column in bold .

| Method                                     | Backbone                                   | iNaturalist                                | iNaturalist                                | SUN                                        | SUN                                        | Places                                     | Places                                     | Texture                                    | Texture                                    | Average                                    | Average                                    |
|--------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|
|                                            |                                            | FPR ↓                                      | AUC ↑                                      | FPR ↓                                      | AUC ↑                                      | FPR ↓                                      | AUC ↑                                      | FPR ↓                                      | AUC ↑                                      | FPR ↓                                      | AUC ↑                                      |
| Extra backbones (MCM)                      | Extra backbones (MCM)                      | Extra backbones (MCM)                      | Extra backbones (MCM)                      | Extra backbones (MCM)                      | Extra backbones (MCM)                      | Extra backbones (MCM)                      | Extra backbones (MCM)                      | Extra backbones (MCM)                      | Extra backbones (MCM)                      | Extra backbones (MCM)                      | Extra backbones (MCM)                      |
| MCM(Last Layer)                            | RN50x4                                     | 44.00                                      | 91.59                                      | 35.13                                      | 92.83                                      | 44.30                                      | 89.38                                      | 57.22                                      | 85.99                                      | 45.16                                      | 89.95                                      |
| MCM(Intermediate)                          | RN50x4                                     | 41.79                                      | 92.00                                      | 31.83                                      | 93.48                                      | 40.46                                      | 90.36                                      | 52.38                                      | 87.37                                      | 41.61                                      | 90.80                                      |
| MCM(Last Layer)                            | ViT-L/14                                   | 24.91                                      | 95.44                                      | 29.58                                      | 93.98                                      | 35.51                                      | 92.02                                      | 58.65                                      | 85.19                                      | 37.16                                      | 91.66                                      |
| MCM(Intermediate)                          | ViT-L/14                                   | 25.15                                      | 95.45                                      | 29.21                                      | 94.07                                      | 35.02                                      | 92.13                                      | 58.99                                      | 85.14                                      | 37.09                                      | 91.70                                      |
| Prompt methods (composition with ViT-B/16) | Prompt methods (composition with ViT-B/16) | Prompt methods (composition with ViT-B/16) | Prompt methods (composition with ViT-B/16) | Prompt methods (composition with ViT-B/16) | Prompt methods (composition with ViT-B/16) | Prompt methods (composition with ViT-B/16) | Prompt methods (composition with ViT-B/16) | Prompt methods (composition with ViT-B/16) | Prompt methods (composition with ViT-B/16) | Prompt methods (composition with ViT-B/16) | Prompt methods (composition with ViT-B/16) |
| NegLabel                                   | ViT-B/16                                   | 1.91                                       | 99.49                                      | 20.53                                      | 95.49                                      | 35.59                                      | 91.64                                      | 43.56                                      | 90.22                                      | 25.40                                      | 94.21                                      |
| NegLabel + Int. Layers                     | ViT-B/16                                   | 2.35                                       | 99.36                                      | 21.10                                      | 95.38                                      | 32.31                                      | 93.35                                      | 39.38                                      | 92.10                                      | 23.79                                      | 95.05                                      |
| CSP                                        | ViT-B/16                                   | 1.54                                       | 99.60                                      | 13.66                                      | 96.66                                      | 29.32                                      | 92.90                                      | 25.52                                      | 93.86                                      | 17.51                                      | 95.76                                      |
| CSP + Int. Layers                          | ViT-B/16                                   | 1.48                                       | 99.64                                      | 10.90                                      | 97.59                                      | 26.97                                      | 93.97                                      | 22.73                                      | 95.75                                      | 15.52                                      | 96.74                                      |

## K Orthogonality to Scoring Rules

Consider logits z ∈ R K , where K denotes the number of classes. Following our experimental setup (Section 3), we analyze OOD detection methods: MaxLogit ( max i z i ), MCM ( max i p i , with p i = exp( z i ) ∑ j exp( z j ) ), Energy ( -log ∑ K i =1 exp( z i ) ) [44], Entropy ( -∑ K i =1 p i log p i ), and Variance ( ∑ K i =1 p i ( p i -¯ p ) 2 ).

The differential improvements observed across methods reflect their computational characteristics. MaxLogit [25] and MCM [47] rely directly on peak logit or probability values, making them particularly sensitive to intermediate layer representations, where the separation between ID and OOD peaks, ∆ ( ℓ ) peak = E [max i z ( ℓ ) i | ID ] -E [max i z ( ℓ ) i | OOD ] , is typically larger due to reduced saturation effects. The observed improvement for MCM specifically benefits from the softmax transformation applied over logits, which amplifies probability differences and enhances discriminative power between ID and OOD samples. This theoretical insight is empirically validated (Table K), showing substantial intermediate-layer improvements for MCM (average +10 . 2% , notably +16 . 6% for ViTB/32) and consistent gains for MaxLogit ( +4 . 3% ).

In contrast, Energy [44], Entropy, and Variance methods aggregate global distributional information through log-sum-exp operations and probability averaging, making them inherently less sensitive to peak-specific variations and thus displaying more modest improvements ( +0 . 4% , +6 . 2% , and +6 . 1% , respectively). Architectural differences further modulate these improvements. Our analysis reveals that CLIP models demonstrate broad prediction agreement across layers (Figure 4), promoting stable predictions conducive to effective fusion, whereas supervised models show independent prediction evolution across layers. These architectural patterns, combined with the representational diversity captured in our entropy calibration analysis (Figure M.18), explain the variance in observed improvements, with ViT-B/32 achieving the highest gains due to optimal balance between inter-layer diversity and prediction consistency.

Table 7: Comprehensive OOD detection results using intermediate vs last layer features. We report the false positive rate at 95% TPR (FPR95, ↓ ) and AUROC ( ↑ ) across four OOD datasets: SUN, Places365, DTD, and iNaturalist. Each method is evaluated using ResNet, ViT-B/32, or ViT-B/16 backbones with both last and intermediate layer features. Best results per backbone are highlighted in bold .

| Method           | Layer            | Backbone         | SUN              | SUN              | Places365        | Places365        | DTD              | DTD              | iNaturalist      | iNaturalist      | Average          | Average          |
|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|
| Method           | Layer            | Backbone         | FPR ↓            | AUC ↑            | FPR ↓            | AUC ↑            | FPR ↓            | AUC ↑            | FPR ↓            | AUC ↑            | FPR ↓            | AUC ↑            |
| ResNet Results   | ResNet Results   | ResNet Results   | ResNet Results   | ResNet Results   | ResNet Results   | ResNet Results   | ResNet Results   | ResNet Results   | ResNet Results   | ResNet Results   | ResNet Results   | ResNet Results   |
| Entropy          | Last             | ResNet           | 55.84            | 88.04            | 72.42            | 79.31            | 73.32            | 78.78            | 83.13            | 71.80            | 71.18            | 79.48            |
| Entropy          | Intermediate     | ResNet           | 55.69            | 88.06            | 72.91            | 79.14            | 73.81            | 78.74            | 82.19            | 72.46            | 71.15            | 79.60            |
| Energy           | Last             | ResNet           | 98.63            | 39.20            | 94.56            | 52.88            | 99.70            | 20.35            | 98.83            | 43.01            | 97.93            | 38.86            |
| Energy           | Intermediate     | ResNet           | 98.49            | 39.65            | 94.12            | 53.43            | 99.68            | 19.79            | 98.57            | 44.53            | 97.72            | 39.35            |
| Variance         | Last             | ResNet           | 55.09            | 88.19            | 72.03            | 79.51            | 73.14            | 78.84            | 82.90            | 72.15            | 70.79            | 79.67            |
| Variance         | Intermediate     | ResNet           | 55.01            | 88.22            | 72.49            | 79.35            | 73.53            | 78.80            | 81.88            | 72.80            | 70.73            | 79.79            |
| MaxLogit         | Last             | ResNet           | 59.73            | 89.11            | 52.05            | 89.19            | 90.05            | 71.19            | 65.44            | 88.23            | 66.82            | 84.43            |
| MaxLogit         | Intermediate     | ResNet           | 54.71            | 90.08            | 46.78            | 90.35            | 88.76            | 72.89            | 61.00            | 89.19            | 62.81            | 85.63            |
| MCM              | Last             | ResNet           | 35.13            | 92.83            | 44.30            | 89.38            | 57.22            | 85.99            | 44.00            | 91.59            | 45.16            | 89.95            |
| MCM              | Intermediate     | ResNet           | 31.83            | 93.48            | 40.46            | 90.36            | 52.38            | 87.37            | 41.79            | 92.00            | 41.61            | 90.80            |
| ViT-B/32 Results | ViT-B/32 Results | ViT-B/32 Results | ViT-B/32 Results | ViT-B/32 Results | ViT-B/32 Results | ViT-B/32 Results | ViT-B/32 Results | ViT-B/32 Results | ViT-B/32 Results | ViT-B/32 Results | ViT-B/32 Results | ViT-B/32 Results |
| Entropy          | Last             | ViT-B/32         | 59.13            | 84.38            | 67.38            | 79.81            | 76.72            | 72.08            | 73.34            | 79.29            | 69.14            | 78.89            |
| Entropy          | Intermediate     | ViT-B/32         | 65.09            | 80.03            | 84.96            | 68.69            | 66.86            | 80.54            | 49.72            | 90.68            | 66.66            | 79.98            |
| Energy           | Last             | ViT-B/32         | 97.77            | 42.25            | 95.79            | 53.02            | 98.92            | 26.50            | 99.79            | 37.98            | 98.07            | 39.94            |
| Energy           | Intermediate     | ViT-B/32         | 97.64            | 45.85            | 96.07            | 57.90            | 97.29            | 40.18            | 99.88            | 32.71            | 97.72            | 44.16            |
| Variance         | Last             | ViT-B/32         | 58.78            | 84.60            | 67.10            | 80.06            | 76.63            | 72.18            | 73.11            | 79.58            | 68.91            | 79.11            |
| Variance         | Intermediate     | ViT-B/32         | 64.92            | 80.21            | 84.84            | 68.94            | 66.91            | 80.64            | 49.67            | 90.78            | 66.59            | 80.14            |
| MaxLogit         | Last             | ViT-B/32         | 64.98            | 86.78            | 56.36            | 88.02            | 87.00            | 70.45            | 65.22            | 87.58            | 68.39            | 83.21            |
| MaxLogit         | Intermediate     | ViT-B/32         | 69.30            | 83.70            | 67.89            | 83.95            | 80.25            | 72.03            | 54.82            | 89.56            | 68.06            | 82.31            |
| MCM              | Last             | ViT-B/32         | 40.99            | 91.56            | 46.71            | 89.25            | 60.90            | 85.03            | 33.85            | 93.62            | 45.61            | 89.87            |
| MCM              | Intermediate     | ViT-B/32         | 28.98            | 93.37            | 35.69            | 91.61            | 39.36            | 91.07            | 12.02            | 97.64            | 29.01            | 93.42            |
| ViT-B/16 Results | ViT-B/16 Results | ViT-B/16 Results | ViT-B/16 Results | ViT-B/16 Results | ViT-B/16 Results | ViT-B/16 Results | ViT-B/16 Results | ViT-B/16 Results | ViT-B/16 Results | ViT-B/16 Results | ViT-B/16 Results | ViT-B/16 Results |
| Entropy          | Last             | ViT-B/16         | 68.00            | 81.36            | 74.90            | 76.35            | 79.08            | 72.37            | 87.00            | 65.05            | 77.24            | 73.78            |
| Entropy          | Intermediate     | ViT-B/16         | 66.86            | 78.25            | 81.86            | 72.64            | 36.33            | 92.12            | 59.78            | 88.25            | 61.21            | 82.81            |
| Energy           | Last             | ViT-B/16         | 98.68            | 37.66            | 97.21            | 45.39            | 99.22            | 24.68            | 99.77            | 36.95            | 98.72            | 36.17            |
| Energy           | Intermediate     | ViT-B/16         | 98.49            | 41.02            | 99.48            | 33.19            | 94.52            | 38.66            | 99.87            | 41.52            | 98.09            | 38.60            |
| Variance         | Last             | ViT-B/16         | 67.54            | 81.61            | 74.47            | 76.62            | 78.97            | 72.47            | 86.67            | 65.62            | 76.91            | 74.08            |
| Variance         | Intermediate     | ViT-B/16         | 66.73            | 78.40            | 81.51            | 72.98            | 36.21            | 92.13            | 59.21            | 88.45            | 60.91            | 82.99            |
| MaxLogit         | Last             | ViT-B/16         | 64.45            | 87.43            | 60.40            | 87.07            | 85.48            | 72.56            | 60.56            | 89.64            | 67.72            | 84.17            |
| MaxLogit         | Intermediate     | ViT-B/16         | 70.32            | 85.04            | 69.05            | 84.15            | 50.51            | 88.18            | 46.61            | 91.97            | 59.12            | 87.33            |
| MCM              | Last             | ViT-B/16         | 37.41            | 92.57            | 43.67            | 89.96            | 57.34            | 86.18            | 30.67            | 94.63            | 42.27            | 90.83            |
| MCM              | Intermediate     | ViT-B/16         | 45.58            | 89.69            | 35.71            | 92.72            | 25.51            | 94.84            | 15.98            | 96.90            | 30.70            | 93.54            |

Formally, for any such score ( s ), an increase in ∆ s := E [ s ( z fused ) | ID ] -E [ s ( z fused ) | OOD ] improves thresholded decisions for that ( s ), indicating robustness that is orthogonal to the choice of scoring rule.

Discussion Fusing intermediate representations offers a training-free, plug-and-play mechanism that consistently strengthens OOD robustness across architectures, datasets, and prompting regimes while adding only modest compute and memory cost. The effect arises from complementary early- and mid-layer signals that widen the separation between familiar and unfamiliar inputs and stabilize predictions . Because the mechanism acts at the representation level, it integrates without changing existing pipelines. Gains can vary with architectural traits; models whose information is concentrated in late layers may show smaller improvements. Future directions include adaptive per-model layer selection, dynamic fusion policies at inference time, and exploring token- or region-level cues to further enhance reliability in open-world settings.

## L Computational Cost Analysis of Intermediate Layer Usage

We evaluate the computational implications of incorporating intermediate layer features across three CLIP architectures: ResNet-50x4, ViT-B/16, and ViT-B/32. The analysis spans batch sizes from 1 to 8 and considers peak GPU memory, per-image latency, total inference time, and throughput (images processed per second). As shown in Figures L.13, L.15, and L.14, the additional operations required for intermediate feature extraction introduce limited overhead. Across all configurations, memory and latency values remain within practical deployment limits, supporting the method's suitability for real-world use.

ViT-B/32 exhibits the most efficient resource profile (Figure L.14), with memory overhead ranging from 1 . 7% to 7 . 1% (average: 3 . 4% ) across batch sizes and average latency changes of just 3 . 3% . Notably, ViT-B/32 demonstrates latency reductions at batch sizes 4 and 8 ( -1 . 1% and -1 . 0% , respectively), indicating improved hardware utilization. ViT-B/16 shows higher memory overhead (average: 10 . 2% , up to 20 . 6% at batch size 8), yet achieves an overall latency *reduction* of 5 . 7% on average, with throughput impact of only -0 . 7% (Figure L.15). Both ViT architectures exhibit counterintuitive latency improvements at larger batch sizes, suggesting better parallelization and GPU kernel efficiency when processing intermediate features. ResNet-50x4 exhibits more conventional scaling behavior (Figure L.13), with a stable memory overhead averaging 12 . 4% and latency increases ranging from 5 . 1% to 14 . 7% (mean 11 . 1% ). Throughput decreases by an average of 10 . 7% , primarily due to the use of external random projections to align the dimensionality of intermediate-layer features.

Importantly, absolute performance values remain deployment-friendly in all cases. Peak memory usage stays under 0.7 GB even at batch size 8, and per-image latency remains compatible with both real-time and high-throughput applications (maintaining 8-60 images/second depending on architecture and batch size). These results confirm that the use of intermediate features is computationally viable across architectures, offering substantial detection performance improvements-up on far-OOD benchmarks (see Tables 5.1 and 2) and consistent gains across diverse scoring functions (Table K) with minimal computational cost. Full breakdowns of the quantitative results are reported in Tables L, L, and L.

Figure L.13: Computational cost comparison for CLIP RN50x4 using intermediate layers versus last-layer only inference.

<!-- image -->

Table 8: ResNet-50x4 Performance Analysis: Last Layer vs All Layers. We report memory usage (GB), latency per image (ms), and overhead percentages (%, ↑ indicates increase) for CLIP ResNet-50x4 architecture. Results show mean ± standard deviation across 5 measurement runs per configuration.

| Batch Size   | Memory Usage (GB)   | Memory Usage (GB)   | Latency per Image (ms)   | Latency per Image (ms)   | Overhead (%)   | Overhead (%)   | Runs   |
|--------------|---------------------|---------------------|--------------------------|--------------------------|----------------|----------------|--------|
| Batch Size   | Last Layer          | All Layers          | Last Layer               | All Layers               | Memory ↑       | Latency ↑      | (L/A)  |
| 1            | 0.41±0.00           | 0.43±0.00           | 152.20±33.17             | 160.03±11.58             | +7.0           | +5.1           | 5/5    |
| 2            | 0.41±0.00           | 0.44±0.00           | 67.72±0.56               | 77.70±1.01               | +8.5           | +14.7          | 5/5    |
| 4            | 0.41±0.00           | 0.46±0.00           | 33.47±0.36               | 38.23±0.86               | +11.9          | +14.2          | 5/5    |
| 8            | 0.41±0.00           | 0.50±0.00           | 17.41±0.54               | 19.23±0.33               | +22.3          | +10.5          | 5/5    |

Figure L.14: Computational cost comparison for CLIP ViT-B/32 using intermediate layers versus last-layer only inference.

<!-- image -->

Table 9: ViT-B/32 Performance Analysis: Last Layer vs All Layers. We report memory usage (GB), latency per image (ms), and overhead percentages (%, ↑ indicates increase, ↓ indicates decrease) for CLIP ViT-B/32 architecture. Results show mean ± standard deviation across 5 measurement runs per configuration.

| Batch Size                                    | Memory Usage (GB)                             | Memory Usage (GB)                             | Latency per Image (ms)                        | Latency per Image (ms)                        | Overhead (%)                                  | Overhead (%)                                  | Runs                                          |
|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|
| Batch Size                                    | Last Layer                                    | All Layers                                    | Last Layer                                    | All Layers                                    | Memory ↑                                      | Latency ↑                                     | (L/A)                                         |
| 1                                             | 0.57±0.00                                     | 0.58±0.00                                     | 454.64±25.75                                  | 452.42±14.36                                  | +1.7                                          | -0.5                                          | 5/5                                           |
| 2                                             | 0.57±0.00                                     | 0.58±0.00                                     | 227.49±9.11                                   | 263.49±79.81                                  | +1.8                                          | +15.8                                         | 5/5                                           |
| 4                                             | 0.57±0.00                                     | 0.59±0.00                                     | 109.22±3.74                                   | 107.99±1.16                                   | +2.8                                          | -1.1                                          | 5/5                                           |
| 8                                             | 0.58±0.00                                     | 0.62±0.00                                     | 54.47±1.30                                    | 53.94±1.11                                    | +7.1                                          | -1.0                                          | 5/5                                           |
| Average Overhead: Memory +3.4%, Latency +3.3% | Average Overhead: Memory +3.4%, Latency +3.3% | Average Overhead: Memory +3.4%, Latency +3.3% | Average Overhead: Memory +3.4%, Latency +3.3% | Average Overhead: Memory +3.4%, Latency +3.3% | Average Overhead: Memory +3.4%, Latency +3.3% | Average Overhead: Memory +3.4%, Latency +3.3% | Average Overhead: Memory +3.4%, Latency +3.3% |

Figure L.15: Computational cost comparison for CLIP ViT-B/16 using intermediate layers versus last-layer only inference.

<!-- image -->

Table 10: ViT-B/16 Performance Analysis: Last Layer vs All Layers. We report memory usage (GB), latency per image (ms), and overhead percentages (%, ↑ indicates increase, ↓ indicates decrease) for CLIP ViT-B/16 architecture. Results show mean ± standard deviation across 5 measurement runs per configuration.

| Batch Size                                     | Memory Usage (GB)                              | Memory Usage (GB)                              | Latency per Image (ms)                         | Latency per Image (ms)                         | Overhead (%)                                   | Overhead (%)                                   | Runs                                           |
|------------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|
| Batch Size                                     | Last Layer                                     | All Layers                                     | Last Layer                                     | All Layers                                     | Memory ↑                                       | Latency ↑                                      | (L/A)                                          |
| 1                                              | 0.57±0.00                                      | 0.58±0.00                                      | 451.84±20.45                                   | 520.31±162.03                                  | +2.6                                           | +15.2                                          | 5/5                                            |
| 2                                              | 0.57±0.00                                      | 0.61±0.00                                      | 221.86±5.01                                    | 213.37±1.67                                    | +7.2                                           | -3.8                                           | 5/5                                            |
| 4                                              | 0.57±0.00                                      | 0.63±0.00                                      | 133.13±39.96                                   | 110.89±2.09                                    | +10.4                                          | -16.7                                          | 5/5                                            |
| 8                                              | 0.57±0.00                                      | 0.69±0.00                                      | 67.04±20.34                                    | 55.28±0.94                                     | +20.6                                          | -17.5                                          | 5/5                                            |
| Average Overhead: Memory +10.2%, Latency -5.7% | Average Overhead: Memory +10.2%, Latency -5.7% | Average Overhead: Memory +10.2%, Latency -5.7% | Average Overhead: Memory +10.2%, Latency -5.7% | Average Overhead: Memory +10.2%, Latency -5.7% | Average Overhead: Memory +10.2%, Latency -5.7% | Average Overhead: Memory +10.2%, Latency -5.7% | Average Overhead: Memory +10.2%, Latency -5.7% |

## M Additional Figures

Figure M.16: Impact of combination length on OOD detection performance across architectures. The plot reports average FPR@95% as a function of the number of fused layers. Combining a moderate number of layers typically improves performance, while longer combinations may degrade it.

<!-- image -->

Figure M.17: Layer-wise OOD detection performance across architectures. Most architectures exhibit their best performance near the final layer, while early layers generally under-perform.

<!-- image -->

<!-- image -->

Figure M.18: Average entropy across transformer layers for various vision models. We compare softmax entropy across layers. Supervised models exhibit low entropy in later layers, reflecting overconfident predictions, while self-supervised and contrastive models (e.g., MAE, CLIP) show greater entropy variation across depth. The inset highlights subtle differences in the high-entropy regime across CLIP variants, SiGLip2, and PE.

Figure M.19: Layer-wise Jensen-Shannon Divergence (JSD) between predicted class probabilities across different architectures. Each heatmap corresponds to a specific model and illustrates the pairwise JSD between layers. Color intensity reflects the degree of divergence, with brighter values indicating greater dissimilarity in output distributions. Note that each subplot uses its own color scale to emphasize internal variation within each architecture .

<!-- image -->

Figure M.20: SVCCA similarity vs. layer distance ∆ across ViT architectures. Box plots show SVCCA distributions per ∆ ; overlaid lines denote mean similarity.

<!-- image -->