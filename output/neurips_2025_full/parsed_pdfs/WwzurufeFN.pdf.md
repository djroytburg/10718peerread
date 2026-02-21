## E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models

## Jiaheng Dong

The University of Melbourne jiaheng.dong@student.unimelb.edu.au

## Soumyajit Chatterjee

Nokia Bell Labs, UK soumyajit.chatterjee@nokia-bell-labs.com

## James Bailey

The University of Melbourne baileyj@unimelb.edu.au

## Hong Jia

University of Auckland hong.jia@auckland.ac.nz

## Abhirup Ghosh

University of Birmingham a.ghosh.1@bham.ac.uk

## Ting Dang

The University of Melbourne Ting.Dang@unimelb.edu.au

## Abstract

Speech Foundation Models encounter significant performance degradation when deployed in real-world scenarios involving acoustic domain shifts, such as background noise and speaker accents. Test-time adaptation (TTA) has recently emerged as a viable strategy to address such domain shifts at inference time without requiring access to source data or labels. However, existing TTA approaches, particularly those relying on backpropagation, are memory-intensive, limiting their applicability in speech tasks and resource-constrained settings. Although backpropagation-free methods offer improved efficiency, existing ones exhibit poor accuracy. This is because they are predominantly developed for vision tasks, which fundamentally differ from speech task formulations, noise characteristics, and model architecture, posing unique transferability challenges. In this paper, we introduce E-BATS, the first Efficient BAckpropagation-free TTA framework designed explicitly for speech foundation models. E-BATS achieves a balance between adaptation effectiveness and memory efficiency through three key components: (i) lightweight prompt adaptation for a forward-pass-based feature alignment, (ii) a multi-scale loss to capture both global (utterance-level) and local distribution shifts (token-level) and (iii) a test-time exponential moving average mechanism for stable adaptation across utterances. Experiments conducted on four noisy speech datasets spanning sixteen acoustic conditions demonstrate consistent improvements, with 4 . 1% -13 . 5% accuracy gains over backpropagation-free baselines and 2.0 × -6.4 × GPU memory savings compared to backpropagation-based methods. By enabling scalable and robust adaptation under acoustic variability, this work paves the way for developing more efficient adaptation approaches for practical speech processing systems in real-world environments. Code is available at: https://github.com/JiahengDong/E-BATS

## 1 Introduction

Speech foundation models (SFM), large-scale pre-trained models that learn generalized representations from vast amounts of unlabeled speech data, have shown strong performance for a wide range of applications including voice assistants [1], transcription services [2], and accessibility tools [3]. These systems generally rely on the assumption that the training and test data follow the same distributions. In practice, this assumption is often violated, leading to significant performance drops under domain shifts caused by real-world acoustic variations such as background noise, speaker accents, and microphone characteristics [4]. While domain adaptation [5, 6, 7, 8] and domain gen-

eralization [9, 10, 11, 12] have been extensively studied to address distributional shifts, they often require access to labeled target domain data or continuous availability of raw source data. These requirements are seldom feasible in real-world scenarios following model deployment. Recently, Test-Time Adaptation (TTA) has emerged as an attractive solution, adapting pre-trained models to new domains during inference using only unlabeled test data.

Existing TTA methods can be broadly categorized into backpropagation-based (BP) and backpropagation-free (BP-free) approaches. While the former achieved state-of-the-art (SOTA) performance using entropy minimization [13, 14, 15, 16, 17, 18, 19] or pseudo-labeling techniques [20, 21], they have a large memory overhead, mainly due to gradient computation. Even when updates are limited to a small subset of model parameters, such as batch normalization layers [13, 14, 15] or early exits [22], these methods still require high GPU memory due to automatic differentiation frameworks. This significantly limits their practical use in continuous inference scenarios and on resource-constrained devices. In contrast, BP-free TTA methods eliminate the need for gradient computation, making them more efficient and computationally lightweight. These methods either modify the model parameters during the forward pass [23, 24, 25, 26, 27] or learn a new input prompt, a vector integrated with the partially processed input samples at an intermediate layer of the model [28].

Despite the promise of TTA, they are largely tailored to models that depend on Batch Normalization (BN), while SFMs use Layer Normalization (LN), limiting the applicability of BN-based TTA [16]. Additionally, SFMs include both CNN-based feature encoders extracting localized spectral features and transformer encoders processing global context. This is unlike models in other modalities like vision, which are either CNN-based (e.g., ResNet [29]) or transformerbased (Vision Transformer [30]). Such architectural difference presents a fundamental challenge for BPfree feature adaptation. Furthermore, downstream tasks and noise characteristics differ significantly between vision and speech tasks. As shown in Figure 1, vision models are commonly used for image classification, a one-to-one mapping task where noise typically appears as spatial perturbations of pixels [31].

Figure 1: The main difference between (a) Vision Foundation Models and (b) Speech Foundation Models (SFMs) is the sequential pipeline in SFMs that processes a fixed-length frame of an utterance as an input and maps to a distribution over |V| token classes.

<!-- image -->

In contrast, speech recognition involves sequence-to-sequence mapping and must handle dynamic, temporally varying noise across frames [32]. This requires more dynamic, multi-scale adaptation. Lastly, TTA methods often depend on large batch sizes for reliable adaptation, whereas TTA in speech tasks needs to process one utterance at a time (batch size of 1 ) due to the high variability across speech utterances [16, 18]. Despite recent developments in TTA for SFMs [16, 19, 18, 17, 21], they still heavily rely on the BP-based TTA methods with high computational overhead, and overlook the unique requirements of multi-scale adaptation.

To address these challenges, we propose the first Efficient BAckpropagation-free single-utterance TTA framework for SFMs , E-BATS, which achieves SOTA accuracy with memory efficiency. Here we focus on one of the most popular tasks on speech data - speech recognition. E-BATS consists of three novel modules: i) A lightweight, prompt-based tuning mechanism tailored for SFMs which directly adapts latent feature distribution using forward pass only; ii) a custom multi-scale loss function that captures both global (utterance-level) and local (token-level) latent embeddings distribution shifts; and iii) a test-time Exponential Moving Average (T-EMA) module that stabilizes prompt updates across dynamic utterances. The main contributions are threefold:

- We introduce the first backpropagation-free TTA approach tailored explicitly for SFMs that achieves high accuracy and low memory consumption.
- We propose a novel framework E-BATS, consisting of three novel modules to effectively address the multi-scale domain shifts and stable adaptation across the dynamic speech streams.
- We validate E-BATS across four noisy datasets, sixteen acoustic environments, and two model architectures have demonstrated significant improvements in both accuracy and memory, particularly 2 × to 6 . 4 × reduction in peak GPU memory usage over SOTA baselines.

## 2 Related Works

Memory-Intensive BP TTA Methods. Traditional backpropagation-based TTA methods are generally memory-intensive, as they rely on gradient-based updates of model parameters, typically guided by entropy minimization or pseudo-labeling. TENT [13] was the first to adapt the affine parameters of the BN layers using entropy minimization. SAR [15] and EATA [14] are variants of TENT, which further filter out a small portion of data samples that are unreliable and redundant. Although they only update a small portion of the overall parameters, the computation is still overhead due to backpropagation and the large batch size required for reliable adaptation [15]. More advanced methods, like CoTTA [20], employ additional networks (e.g., teacher-student models) for adaptation, but at the cost of significantly increased computational and memory overhead.

TTA for SFMs. A few recent TTA approaches have been tailored for SFMs, which typically extend memory-intensive BP TTA methods with speech-specific mechanisms [16, 18, 19]. SUTA [16] built upon TENT by updating CNN-based feature encoder layers alongside normalization parameters, incorporating techniques like temperature smoothing and Minimum Class Confusion loss. SGEM [18] and CEA [19] further improved upon this with advanced loss design for audio tasks, such as sequence-level entropy minimization or uncertainty-driven frame prioritization. DSUTA [17] and AWMC [21] introduced additional subnetworks (e.g., fast-slow models and anchor-chaser-leader models) to enhance cross-utterance knowledge transfer. However, these methods still heavily rely on backpropagation, leading to significant memory overhead and scalability challenges as more layers are updated beyond normalization.

BP-free TTA Methods. BP-free TTA methods offer an efficient alternative by updating the model solely via forward pass to achieve computational efficiency. These methods generally fall into three categories: (i) analytical adjustment of batch normalization statistics [23, 24, 25, 26], (ii) adaptation of the classifier using class prototypes [27] or output probabilities [33], and (iii) optimization via evolutionary algorithms that circumvent gradient-based updates [28]. However, they typically offer lower adaptation accuracy compared to memory-intensive BP TTA methods [13, 20, 14, 15]. Currently, BP-free TTA for speech foundation models remains unexplored , primarily due to the unique challenges posed by sequence-to-sequence learning, single-utterance adaptation rather than batches, and differences in model structures, particularly in normalization layers and feature encoders.

## 3 Methodology

## 3.1 Overview

We consider a covariate shift between the source and a target domain, such that the marginal distributions of speech differ, P src ( X ) = P tgt ( X ) , while the class prior P src ( y ) = P tgt ( y ) and the conditional distribution P src ( y | X ) = P tgt ( y | X ) are preserved. We adapt an SFM Θ src , pre-trained on a source speech dataset D src , to improve transcription accuracy while maintaining the low peak memory usage on an unlabeled target speech dataset D tgt . A target stream D tgt = { X 1 , . . . , X T } consists of T utterances arriving sequentially, each is processed under an online, single-utterance setting with a batch size of one. Each utterance X t is composed of a variable number of frames N t and is represented as a sequence of frame-level feature vectors, where each frame represents a short, fixed-duration segment of the audio signal: X t = [ x 1 t , x 2 t , . . . , x N t t ] , x i t ∈ R d in . The model Θ src is composed of two components, Θ = g ◦ h , where: (a) h is a convolutional encoder that maps each input frame to a latent embedding: x i t ↦→ z i t ∈ R d . (b) g consists of a stack of transformer layers and a Connectionist Temporal Classification (CTC) classifier head, producing frame-level posterior distributions over CTC token classes v ∈ V , where V consists of twenty-six alphabet token classes (a-z) and six special token classes (apostrophe, blank, etc.): g ( Z t ) = { P ( y i t | z 1: N t t ) } N t i =1 . These posteriors are further decoded to produce each utterance's final transcription ˆ y t .

̸

System Overview. As shown in Figure 2, for a test-time utterance X t , the model first employs a Lightweight Prompt Adaptation (LPA) module, which directly modifies the CNN encoder features Z t by incorporating J learnable prompt vectors s t,j . These prompts are sampled using the derivative-free Covariance Matrix Adaptation Evolution Strategy (CMA-ES), guided by a multi-scale adaptation loss L adapt. The prompt that results in the lowest loss is selected for adaptation. The loss comprises three components: entropy minimization ( L ent), utterance-level ( L utt), and token-level

(a) Single-utterance Backpropagation-free Adaptation

<!-- image -->

(b) T-EMA across utterances

Figure 2: Overall framework of E-BATS. For an utterance X t : (i) Lightweight Prompt Adaptation (LPA) : CNN-extracted latent features Z t are adapted using a set of J candidate prompts s t,j generated by CMA-ES in parallel, leading to J adapted representations. (ii) The adapted representations J are evaluated, and their corresponding prompts are ranked using a multi-scale loss (entropy loss, utterance-level and token-level feature alignment). This ranking guides the iterative update of CMAES parameters over K iterations until the loss converges, at which point the best prompt is selected for adaptation. The CMA-ES parameters are smoothed using T-EMA for next utterance adaption. (b) Test-time Exponential Moving Average (T-EMA) : T-EMA stabilizes adaptation by smoothing the CMA-ES search trajectory across a stream of utterances, facilitating robust prompt learning.

latent embeddings alignment ( L token). To promote stable adaptation across consecutive utterances, a Test-time Exponential Moving Average (T-EMA) module (Figure 2(b)) incrementally updates the CMA-ES search distribution, enabling smoother evolution of the prompt vector s t,j over time.

## 3.2 Lightweight Prompt Adaptation (LPA)

To ensure memory efficiency, E-BATS adopts prompt tuning [34]. This technique introduces a small set of learnable parameters, called prompts , to guide the behavior of a pre-trained model on a downstream task while keeping the original model weights fixed. While conventional prompt tuning approaches have primarily been developed for transformer-only architectures [30], typically by concatenating prompts with the model's inputs, we propose a novel LPA module (Figure 2a) designed specifically for SFMs that include convolutional components. The LPA integrates adaptive prompts directly into the convolutional layers, leveraging their effectiveness in extracting acoustic features [16]. Furthermore, rather than common strategy of concatenating prompts with input features [28], our approach examines latent feature distribution shifts and leverages shifting patterns to guide adaptation.

Characterizing Distribution Shifts. To understand differences in speech distribution, we first characterize the shift between the embedding vectors produced by source P ( Z src ) and a target domain P ( Z tgt ) . We study the two most natural ways, covering both first and second order statistics: a ) comparing shift in the centroid of the point clouds, and b ) comparing the spread of the point clouds using their covariances. Incidentally, the popular distance measure between distributions, Fréchet Inception Distance, is a linear combination of these two factors, justifying our choice. Figure 3 shows that, under various real-world noise and variability conditions, the shift in the mean accounts for up to

Figure 3: Comparing the source and target latent spaces across different acoustic conditions (same sample size for source and target domain within each condition). Blue and red bars indicate the mean and covariance shifts.

<!-- image -->

7 . 8 × of the shift in the covariance (we make sure they are in comparable scale, explained in Appendix A).

From this observation, we hypothesize that the shift between D src and D tgt can be explained by a simple geometric translation operation in the latent space; therefore, an adequate shift of target embeddings, Z t would align the latent target vectors to the ones from the source, essentially mitigating the problem of domain shift. It is important to note that, despite the potential for a translation operation, this remains a non-trivial problem. In real-world scenarios, noise is not consistently introduced into clean samples, making it impossible to derive a simple solution based on an analyzable shifting vector. Therefore, adaptively learning the prompt becomes necessary.

Learning Prompt to Adapt. Following the above observation, we propose to add the embeddings Z t and the prompt vector s t over all frames (Figure 2a) as:

<!-- formula-not-decoded -->

where Z t = [ z 1 t , . . . , z N t t ] , s t ∈ R d is the prompt vector for X t . We exclusively optimize the prompt vector for each utterance while the rest of the model parameters remain frozen.

Prompt Optimization with CMA-ES. To identify the optimal prompt vector in a backpropagationfree manner, we adopt the CMA-ES [35]. For each test-time utterance X t , CMA-ES samples J candidate prompt vectors s t,j from a multivariate normal distribution N ( . ) , which are parameterized by a mean vector m t ∈ R d , a covariance matrix C t ∈ R d × d , and a step size σ t ∈ R &gt; 0 . The performances of these candidates are then ranked based on a loss function (explained in Section 3.3), which informs the update of the distribution parameters m ( k ) t , C ( k ) t , σ ( k ) t for the next iteration k . The step size scales the covariance and thus controls the spread of candidate samples drawn from the distribution in the next iteration. The iteration continues until the loss convergences. The final prompt vector s t is selected as the candidate that minimizes the loss for adaptation (Appendix D):

<!-- formula-not-decoded -->

## 3.3 Multi-scale Loss Function

The loss function L adapt that effectively guides the prompt optimization is proposed to integrate entropy minimization ( L ent ) with feature alignment to the source domain latent embedding distributions at multiple scales, i.e., utterance-level ( L utt ) and tokenwise ( L token ). We optimize a weighted average using L adapt = αL ent + βL utt + cL token where α and β are hyperparameters, and c is chosen algorithmically based on the confidence in the prediction.

Entropy Minimization Loss with Blank Token Exclusion ( L ent ). Entropy minimization aims to improve prediction confidence, which is further improved with blank token exclusion for SFMs. SFMs for speech recognition include a special 'blank' token class ∅ , which is primarily designed in CTC to address the alignment mismatch between input frames and output labels [36] with different lengths. In recent studies, it often indicates frames where no alphabetic character can be assigned, helping to eliminate the need to identify ambiguous sound boundaries [36] and highlighting silent periods [37]. In practice, predictions for a large proportion of the frames often belong to this blank class, which introduces class imbalance [16]. We address this by defining the entropy loss only considering the set of frames ˜ X t that do not predict blanks. Formally, we use Shanon's entropy as L ent = -1 | ˜ X t | ∑ x i t ∈ ˜ X t H (Θ( x i t )) .

Despite its effectiveness, minimizing only L ent has a trivial solution of predicting the blank token class for each frame (as highlighted in Section 5.5). Thus, we introduce an utterance-level latent embeddings alignment loss to guide optimization towards the source domain embeddings correctly.

Utterance-level Latent Embeddings Alignment ( L utt ). Utterance-level latent embeddings alignment aims to align the global latent embedding distributions between the source domain and the target domain to avoid trivial solutions. At each transformer encoder layer, l , we compute the squared Euclidean distance between the source-domain centroid and the target-domain centroid of utterancelevel latent embeddings (Figure 2a(ii)). Each utterance-level embedding is obtained by averaging the embeddings across all frames within that utterance. Effectively, we compute and store the centroids, µ src using D src and Θ src and compare against the centroids, µ tgt , computed from the current target utterance. This gives us the following loss component: L utt = 1 L ∑ L ℓ =0 ∥ µ l tgt -µ l src ∥ 2 2 , where L is the number of transformer layers in Θ . Note that µ src requires a small storage size, in the order of L × d , and as this does not correspond to an individual source sample, it is privacy preserving.

Adaptive Confidence Tokenwise Latent Embeddings Alignment ( L token ). The tokens within a single utterance may not comprehensively represent all possible tokens. Aligning the centroids of the source and target utterance-level embeddings could lead to bias towards the majority tokens. To address this, we minimize the distance between the source and target latent distributions corresponding to token classes, where target token classes are estimated using pseudo-labels. To prevent unreliable pseudo-labels from distorting adaptation, we introduce adaptive confidence coefficients for the tokenlevel loss. Specifically, when overall distribution shifts are substantial or entropy is high, this indicates less reliable posterior probabilities and a higher risk of inaccurate token pseudo-labels. In such cases, the confidence for token-level loss is reduced to minimize the impact of misleading token predictions. Conversely, if the shifts are minor or entropy is low, stronger alignment of token-level distributions can be applied with greater confidence.

Formally, we represent the mean µ v,l and the standard deviation σ v,l of the distribution for each token class v ∈ V in a d -dimensional space at transformer layer l . Then we define the loss as the average distance between distributions as:

<!-- formula-not-decoded -->

Note that the storage cost for µ v,l src and σ v,l src is small and in the asymptotic order of 2 × L × 32 × d . This is privacy-preserving as the distribution parameters are sample-agnostic.

We further introduce an adaptive confidence c ∈ [0 , 1] to adjust the trustworthiness of token-level predictions. We propose using the inverse of the combined loss H = L ent + L utt as the confidencebased coefficient, where high H means lower confidence. We define the normalized coefficient using min-max normalization with predefined bounds H min , H max and c max as: c = c max -H -H min H max -H min + ϵ , where ϵ is a small constant to prevent division by zero. This adaptive confidence-aware scaling strengthens token-wise control via prediction reliability and domain shift.

Optimizing through CMA-ES. The CMA-ES parameters are optimized using L adapt , driving the prompt optimization. By iteratively minimizing L adapt for candidate prompt vectors within each utterance, CMA-ES could be updated to effectively generate prompts that robustly mitigate both global (utterance-level) and local (token-level) acoustic domain shifts.

## 3.4 T-EMA across Utterances

To stabilize adaptation across the utterance streams, we propose T-EMA that updates the CMA-ES parameters incrementally, ensuring a smoother and more consistent search space for the prompt (Figure 2b). It leverages the knowledge from past utterances to initialize each new search more robustly, thereby reducing overfitting and mitigating model drift. This serves as the first smoother adaptation strategy for BP-free TTA in SFMs.

CMA-ES statistics parameters are carried over between utterances using the following weighted average scheme. We introduce EMA statistics, e.g., m ema , that are updated using a hyperparameter γ ∈ [0 , 1) to weight the past and current statistics values. Such an update happens when all K iterations have finished for an utterance. For example, at the end of processing t -th utterance, the mean of the distribution is updated as m ema = γ m ema + (1 -γ ) m ( K ) t , where the m ema is initialized with m 0 when t = 0 . Then, m t +1 is set to m ema from the previous round for next utterance prompt learning. The other statistics, covariance and step size are updated in the same way.

## 4 Experiments

Datasets. We evaluate the proposed method on four datasets across sixteen acoustic conditions to assess its effectiveness under varying domain shifts. The test sets encompass three categories of acoustic variability: synthetic noise, single-domain distributional shifts, and multi-domain distributional shifts, reflecting the range of conditions encountered in real-world deployment. Following [16, 19], we introduce synthetic noise to the LibriSpeech test-other split [38] with additive Gaussian noise with zero mean and varying standard deviations ( σ ∈ { 0 . 0 , 0 . 005 , 0 . 01 , 0 . 015 , 0 . 02 } ) to simulate covariate shifts. We use the CHiME-3 dataset [39] representing single domain shift in a sample, including four acoustic environments: bus, café, pedestrian area, and street junction. We further use CHiME-3-Mix that creates a dynamic stream by concatenating CHiME-3 environments to emulate

Table 1: Word Error Rate (WER) on various noisy conditions using Wav2Vec2ForCTC-Base. Lower value means better adaptation performance. Bold represents the best performance for BP-free TTA, while underlined means the best for both BP-based and BP-free TTA.

| Method   | BP- free   |      |   Gaussian noise |   Gaussian noise |   Gaussian noise |   Gaussian noise |      | CHiME3   | CHiME3   | TED   | Common Voice   |
|----------|------------|------|------------------|------------------|------------------|------------------|------|----------|----------|-------|----------------|
|          | BP- free   |  0   |            0.005 |             0.01 |            0.015 |             0.02 | Avg  | (Single) | (Mixed)  |       |                |
| Source   | -          |  8.6 |           13.9   |            24.4  |           39.5   |            54.5  | 28.2 | 34.2     | 34.2     | 13.2  | 36.8           |
| TENT     | ✗          |  8.5 |           14     |            24.1  |           39.2   |            54.3  | 28.0 | 34.1     | 34.1     | 13.1  | 36.8           |
| EATA     | ✗          | 14.1 |           18.1   |            27    |           37.9   |            51.3  | 29.7 | 33.1     | 39.9     | 14.1  | 61.3           |
| SAR      | ✗          |  8.4 |           13.6   |            22.9  |           36     |            49.9  | 26.2 | 33.6     | 34.7     | 13.0  | 38.2           |
| CoTTA    | ✗          |  9.2 |           12.6   |            18.1  |           39.3   |            54.5  | 26.7 | 32.9     | 34.3     | 12.8  | 36.9           |
| CEA      | ✗          |  7.5 |           11.1   |            16.4  |           23.8   |            33.6  | 18.5 | 26.8     | 26.8     | 12.0  | 31.5           |
| SGEM     | ✗          |  7.3 |           10.9   |            16.4  |           23.8   |            33.9  | 18.5 | 27.2     | 27.1     | 11.9  | 31.2           |
| AWMC     | ✗          |  9.5 |           11.7   |            16.6  |           23.9   |            31.8  | 18.7 | 34.0     | 33.9     | 13.6  | 37.9           |
| SUTA     | ✗          |  7.3 |           10.9   |            16.5  |           24.1   |            34.1  | 18.6 | 26.8     | 26.8     | 11.9  | 31.5           |
| CSUTA    | ✗          | 13.1 |           17.5   |            24.5  |           31.4   |            37    | 24.7 | 26.5     | 32.6     | 15.6  | 135.0          |
| DSUTA    | ✗          |  9   |           11.7   |            16.1  |           21.1   |            24.1  | 16.4 | 24.0     | 24.1     | 12.7  | 36.1           |
| T3A      | ✓          | 10   |           15.9   |            26.8  |           42.7   |            58.6  | 30.8 | 35.9     | 35.8     | 14.6  | 38.8           |
| LAME     | ✓          |  9.1 |           15     |            26    |           42.4   |            58.2  | 30.1 | 36.0     | 36.0     | 14.0  | 38.8           |
| FOA      | ✓          |  8.7 |           13.9   |            22.7  |           33.3   |            45.3  | 24.8 | 31.7     | 31.1     | 13.3  | 38.2           |
| Ours     | ✓          |  7.7 |           10.5   |            14.8  |           19.9   |            25.3  | 15.6 | 24.0     | 24.3     | 12.5  | 30.6           |

non-stationary acoustic shifts [17]. CommonVoice (CV) [40] introduces variability in speaker accents, recording devices, and environments and TEDLIUM-v2 (TED) [41] comprises oratory speech from TED talks with diverse accents, speaking styles, and syntactic structures.

Baseline Methods. We compare E-BATS against 13 SOTA TTA baselines. The BP methods include general approaches of Episodic TENT [13], SAR [15], EATA [14], and CoTTA [20], as well as speech-specific methods: SUTA [16], CEA [19], SGEM [18], DSUTA [17], CSUTA [17], and AWMC [21]. The BP-free methods include LAME [33], T3A [27], and FOA [28]. Dataset and baseline details are provided in Appendix B.

Implementation Details. All TTA baselines are configured for per-utterance adaptation with batch size of 1. For E-BATS, we set the CMA-ES population size J = 50 . The loss function coefficients are α = 1 . 0 and β = 2 . 0 . We use H min = 0 . 0 , H max = 5 . 0 in calculating the confidence-weighted coefficient c with c max = 2 . 0 optimized over { 1 . 0 , 1 . 5 , 2 . 0 , 2 . 5 , 3 . 0 , 3 . 5 , 4 . 0 } . Evaluation is performed using two commonly used SFMs, Wav2Vec2ForCTC-Base [42] and HuBERTForCTCLarge [43]; both models are fine-tuned on LibriSpeech and then are adapted in our experiments. The pre-collected statistics are sourced from clean LibriSpeech data samples. For T-EMA, we select γ = 0 . 9 for Wav2Vec2 and γ = 0 . 8 for HuBERT after tuning over { 0 . 7 , 0 . 8 , 0 . 9 , 0 . 95 , 0 . 99 } . We use Word Error Rate (WER) [44] as the evaluation metric, which measures the fraction of incorrectly predicted words in the dataset. A lower WER indicates better performance. We further conducted sensitivity analyses on key hyperparameters in Appendix C.3, including the CMA-ES population size ( J ), the number of iteration steps ( N ), the loss component weights ( α, β ), and the T-EMA decay factor ( γ ), confirming that the chosen settings yield stable and robust performance across diverse configurations. All experiments are conducted on a single NVIDIA A100 GPU. Implementation details are provided in Appendix B.

## 5 Results and Discussion

## 5.1 Comparing accuracy to SOTA

Experiments using Wav2Vec2ForCTC-Base (Table 1) show that E-BATS consistently outperforms all BP-free TTA baselines across datasets, with WER reductions ranging from 0.8% to 20.0% over the strongest alternative. Its performance gains increase with noise severity, achieving at least 3.4% improvement at σ = 0 . 005 and 20.0% at σ = 0 . 02 , highlighting robust adaptation under challenging conditions. T3A and LAME degrade the source model, indicating that updating only the final classifier is insufficient. FOA, which also uses prompt tuning, performs better but remains less effective than E-BATS, likely due to difficulties in adapting transformer layers for acoustic shifts (Section 5.5).

Table 2: WER on various noisy conditions using HuBERTForCTC-Large. Lower is better. Bold : best among BP-free; underlined: best overall.

| Method   | BP-   |      |   Gaussian noise |   Gaussian noise |   Gaussian noise |   Gaussian noise |      | CHiME3   | CHiME3   | TED   | Common Voice   |
|----------|-------|------|------------------|------------------|------------------|------------------|------|----------|----------|-------|----------------|
|          | free  |  0   |            0.005 |             0.01 |            0.015 |             0.02 | Avg  | (Single) | (Mixed)  |       |                |
| Source   | -     |  4.2 |            5     |             6.4  |            9     |            12.8  | 7.5  | 16.5     | 16.5     | 9.1   | 21.4           |
| TENT     | ✗     |  4.2 |            4.9   |             6.3  |            8.8   |            12.5  | 7.3  | 16.4     | 16.4     | 9.0   | 27.5           |
| EATA     | ✗     |  7.4 |            8.4   |             9.5  |           11.3   |            13.7  | 10.1 | 16.2     | 18.9     | 9.7   | 34.4           |
| SAR      | ✗     |  4   |            4.7   |             6.3  |            8.6   |            12.2  | 7.2  | 16.4     | 17.3     | 9.0   | 21.7           |
| CoTTA    | ✗     |  4.4 |            5.1   |             6.3  |            8.3   |            11    | 7.0  | 16.2     | 15.3     | 9.0   | 25.8           |
| CEA      | ✗     |  3.8 |            4.2   |             5.1  |            6.7   |             9.1  | 5.8  | 14.2     | 14.1     | 8.1   | 18.3           |
| SGEM     | ✗     |  3.7 |            5.3   |             5.3  |            6.9   |             9.3  | 6.1  | 14.2     | 14.2     | 8.3   | 18.4           |
| AWMC     | ✗     |  5.5 |            6.4   |             8.2  |           10.7   |            14.3  | 9.0  | 15.9     | 17.2     | 9.9   | 21.9           |
| SUTA     | ✗     |  3.8 |            4.2   |             5.1  |            6.8   |             9.2  | 5.8  | 14.2     | 14.2     | 8.2   | 18.4           |
| CSUTA    | ✗     |  6   |            6.8   |             7.9  |            9.5   |            11.9  | 8.4  | 14.7     | 16.2     | 10.2  | 90.0           |
| DSUTA    | ✗     |  4.6 |            5     |             6    |            7.1   |             8.8  | 6.3  | 13.3     | 13.5     | 8.7   | 27.4           |
| T3A      | ✓     | 14.4 |           15.8   |            18.9  |           24.2   |            30.2  | 20.7 | 27.9     | 32.3     | 22.5  | 46.2           |
| LAME     | ✓     |  4.5 |            5.3   |             6.9  |            9.8   |            13.9  | 8.1  | 17.5     | 17.5     | 9.7   | 22.6           |
| FOA      | ✓     |  4.5 |            5.3   |             6.8  |            9.2   |            12.9  | 7.7  | 16.4     | 16.7     | 9.3   | 22.8           |
| Ours     | ✓     |  4.3 |            4.9   |             5.9  |            7.5   |             9.5  | 6.4  | 14.0     | 14.0     | 9.3   | 20.1           |

Compared to BP-based methods, E-BATS achieves the lowest WER in 3 out of 5 datasets, with up to 30.7% relative improvement, and remains competitive. Methods such as EATA and CoTTA, which depend on larger batch sizes or vision-specific strategies, perform poorly across the board. While TENT and SAR are more resilient with small batches, they still underperform relative to E-BATS, showing that adapting only normalization layers is inadequate. On the other hand, the performance limitations of SGEM, SUTA, and CEA stem from their utterance-level reset strategy. This prevents them from transferring the already learned knowledge for adapting further utterances as model weights are reinitialized every time. Notably, DSUTA, despite continuous adaptation, performs 5.5% worse than E-BATS on CommonVoice, the most diverse test condition. This suggests that frequent parameter updates may lead to catastrophic forgetting. In contrast, E-BATS updates only prompt vectors, preserving the pre-trained model and enabling effective, stable adaptation across varied domains. Additional results across different fine-grained noise conditions are in Appendix C.

## 5.2 Memory Efficiency

Beyond accuracy, E-BATS demonstrates substantial memory efficiency across all evaluated methods, as shown in Figure 4. Compared to BP-based TTA methods, it reduces peak GPU memory usage by 1.5 × to 5.9 × . Specifically, relative to DSUTA, CEA, and SGEM, E-BATS achieves memory savings of 3.3 × , 3.2 × , and 2.8 × , respectively, while outperforming them in WER. This efficiency is attributed to lightweight prompt tuning and the T-EMA mechanism, which avoids gradient-based updates. Compared to BP-free baselines, E-BATS maintains comparable or lower memory usage while achieving a lower WER, indicating a favorable balance between efficiency and performance.

## 5.3 Performance with Different Backbone Models

When using HuBERTForCTC-Large backbone, a larger SFM than Wav2Vec2ForCTC-Base, E-BATS continues to outperform all BP-free and most BP-based TTA methods across datasets, as summarized in Table 2. On average, it achieves 1 . 8% to 17 . 1% lower WER than BP-free baselines. While performance is comparable to BP-based methods under certain conditions, E-BATS surpasses them in challenging scenarios such as CHiME-3 single-domain and mixed-domain settings. More notably, E-BATS offers substantially better memory efficiency at larger model scale, with 2 . 4 × to 6 . 8 × lower GPU memory usage compared to BP-based approaches (detailed in Appendix C and Figure 6). As model size increases, memory usage grows for all TTA methods; however, E-BATS scales more gracefully, exhibit-

<!-- image -->

Methods

Figure 4: Balance between Average Peak GPU memory usage (bar) and average WER (percentages % ) for different TTA methods across all datasets.

Table 3: Ablation study on three key components.

| Prompt Adaptation   | Prompt Adaptation   | Prompt Adaptation   | Loss Function (w/ T-EMA)   | Loss Function (w/ T-EMA)   | Loss Function (w/ T-EMA)   | Loss Function (w/ T-EMA)   | Loss Function (w/o T-EMA)   | Loss Function (w/o T-EMA)   | Loss Function (w/o T-EMA)   | Loss Function (w/o T-EMA)   | T-EMA Mechanism   | T-EMA Mechanism   | T-EMA Mechanism   |
|---------------------|---------------------|---------------------|----------------------------|----------------------------|----------------------------|----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-------------------|-------------------|-------------------|
| Feat                | Trans               | WER                 | L ent                      | L utt                      | L token                    | WER                        | L ent                       | L utt                       | L token                     | WER                         | T-EMA             | Reset             | WER               |
| ✓                   | -                   | 24.0                | ✓                          | ✓                          | ✓                          | 24.0                       | ✓                           | ✓                           | ✓                           | 25.4                        | ✓                 | -                 | 24.3              |
| -                   | ✓                   | 34.2                | ✓                          | ✓                          | -                          | 24.3                       | ✓                           | ✓                           | -                           | 25.5                        | -                 | ✓                 | 26.5              |
| -                   | -                   | -                   | ✓                          | -                          | -                          | 24.5                       | ✓                           | -                           | -                           | 49.6                        | -                 | -                 | 25.4              |

ing only a moderate increase in memory demand. This makes it particularly suitable for on-device or resource-constrained environments where backpropagation is infeasible.

## 5.4 Memory Efficiency Across Utterance Lengths

Figure 5 shows the peak GPU memory usage as utterance duration increases on the TED dataset using the HuBERTForCTC-Large model. TED is selected due to its wide range of utterance lengths. We compare E-BATS against four top-performing BP-based TTA methods. These baselines exhibit rapidly growing memory consumption with increasing utterance duration, reaching 6 -12 GB for 30 -second clips. In contrast, E-BATS displays a near-linear memory profile, increasing from ∼ 1 . 1 GB at 1 second to just over 1 . 9 GB at 35 seconds, particularly suitable for deployment scenarios with strict or varying memory constraints.

## 5.5 Ablation Study

We investigate the effectiveness of three key components in E-BATS.

Prompt Adaptation. We compare our proposed method, which injects prompts into Z t to adapt the CNN latent feature representations directly, with a variant that concatenates prompts with Z t at the input to the transformer encoder, following conventional prompt-tuning approaches. As shown in Table 3, adapting within the CNN-based feature encoder yields better performance (24.0 vs. 34.2 WER) on CHiME3 (single). This advantage comes from CNNs' ability to capture localized spectral features (e.g., pitch, formants), which are crucial for handling acoustic domain shifts. In contrast, transformer encoders focus on global contextual dependencies (e.g., sentence-level semantics), making them less effective at modeling fine-grained acoustic variations under domain mismatch.

Loss Function Components. We evaluated each component of the loss function under two settings: (1) CHiME-3-Single with (w/) T-EMA, representing a stable and consistent distribution shift, and (2) CHiME-3-Mix without (w/o) T-EMA, reflecting more diverse shifts. For the single-domain shift with T-EMA, we observed that each loss component contributed to overall performance (24.0 WER), with the token-level loss providing additional improvements (from 24.3). In contrast, for the mixed distributional shifts, adaptation relied heavily on utterance-level alignment (from 49.6 to 25.5), as expected due to the increased shifts. Compared with only using L ent , combining L utt effectively prevents trivial solutions caused by entropy minimization of predicting all frames to the blank token class or collapsing into a single character. This is further explained and analyzed in Appendix C.4. Moreover, the adaptive weighting mechanism of the token-level loss ensured its reduced confidence, facilitating more reliable adaptation under this setting (25.4). These findings not only underscore the importance of all loss components across different scenarios but also highlight the critical role of confidence-based adaptive weighting, allowing the loss to emphasize the most reliable signals under varying conditions.

T-EMA and Reset Strategy. We evaluate the effectiveness of the T-EMA under dynamic domain shifts (CHiME3-Mix) by comparing it against two alternatives: (i) a reset variant that reinitializes CMA-ES parameters at the start of each utterance, and (ii) a variant that performs continuous adaptation without any resetting mechanism. As shown in Table 3, T-EMA consistently achieves lower WER than both variants. The reset variant yields the worst performance (26.5), indicating that discarding adaptation history prevents the accumulation of knowledge. Conversely, omitting reset entirely leads to sub-optimal results (25.4), suggesting that preserving historical information is important but needs to be regulated to avoid overfitting. T-EMA provides a principled balance

Figure 5: Peak GPU memory of TTA on TED as audio duration increases.

<!-- image -->

between stability and adaptability across utterances. The effectiveness of T-EMA is further analyzed with an increasing number of target domain samples in Appendix C.5.

## 6 Conclusions and Discussion

Conclusions. In this paper, we propose E-BATS, the first backpropagation-free test-time adaptation method for Speech Foundation Models that effectively balances adaptation accuracy and memory efficiency. E-BATS introduces a lightweight prompt adaptation module that directly adapts CNNbased feature encoders to mitigate acoustic domain shifts. A novel multi-scale loss function combining entropy minimization with utterance-level and token-wise feature alignment ensures fine-grained control over speech feature adaptation. Additionally, the test-time Exponential Moving Average mechanism stabilizes continuous adaptation in dynamic speech streams. Experimental results across four noisy datasets and diverse acoustic conditions demonstrate its superior performance, particularly in memory efficiency as the model size increases significantly.

Limitations. Although E-BATS is more theoretically efficient than other baseline methods through computation complexity comparision, which is reported in Appendix C.9, the iterative CMA-ES optimization introduces additional adaptation latency in practical environments. Specifically, the current implementation of CMA-ES does not fully exploit GPU parallelization, leading to sequential computation steps per utterance. While this latency is acceptable for scenarios without strict real-time requirements, it might pose challenges for latency-sensitive applications.

## Acknowledgments and Disclosure of Funding

This research was supported by The University of Melbourne's Research Computing Services and the Petascale Campus Initiative.

## References

- [1] Veton Kepuska and Gamal Bohouta. Next-generation of virtual personal assistants (microsoft cortana, apple siri, amazon alexa and google home). In 2018 IEEE 8th annual computing and communication workshop and conference (CCWC) , pages 99-103. IEEE, 2018.
- [2] Yashesh Gaur, Walter S Lasecki, Florian Metze, and Jeffrey P Bigham. The effects of automatic speech recognition quality on human transcription latency. In Proceedings of the 13th International Web for All Conference , pages 1-8, 2016.
- [3] Oscar Saz Torralba, William Ricardo Rodríguez Dueñas, and Eduardo Lleida Solano. Development of voice-based tools for accessibility to computer services. Computación y Sistemas , 15(1):7-15, 2011.
- [4] Jinyu Li, Li Deng, Yifan Gong, and Reinhold Haeb-Umbach. An overview of noise-robust automatic speech recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing , 22(4):745-777, 2014.
- [5] Ashly Ajith and G Gopakumar. Domain adaptation: A survey. In Computer Vision and Machine Intelligence: Proceedings of CVMI 2022 , pages 591-602. Springer, 2023.
- [6] Zijian Li, Ruichu Cai, Guangyi Chen, Boyang Sun, Zhifeng Hao, and Kun Zhang. Subspace identification for multi-source domain adaptation. Advances in Neural Information Processing Systems , 36:34504-34518, 2023.
- [7] Aibek Alanov, Vadim Titov, and Dmitry P Vetrov. Hyperdomainnet: Universal domain adaptation for generative adversarial networks. Advances in Neural Information Processing Systems , 35:29414-29426, 2022.
- [8] Mingsheng Long, Zhangjie Cao, Jianmin Wang, and Michael I Jordan. Conditional adversarial domain adaptation. Advances in neural information processing systems , 31, 2018.
- [9] Kaiyang Zhou, Ziwei Liu, Yu Qiao, Tao Xiang, and Chen Change Loy. Domain generalization: A survey. IEEE transactions on pattern analysis and machine intelligence , 45(4):4396-4415, 2022.

- [10] Yu Ding, Lei Wang, Bin Liang, Shuming Liang, Yang Wang, and Fang Chen. Domain generalization by learning and removing domain-specific features. Advances in Neural Information Processing Systems , 35:24226-24239, 2022.
- [11] Junbum Cha, Sanghyuk Chun, Kyungjae Lee, Han-Cheol Cho, Seunghyun Park, Yunsung Lee, and Sungrae Park. Swad: Domain generalization by seeking flat minima. Advances in Neural Information Processing Systems , 34:22405-22418, 2021.
- [12] Shanshan Zhao, Mingming Gong, Tongliang Liu, Huan Fu, and Dacheng Tao. Domain generalization via entropy regularization. Advances in neural information processing systems , 33:16096-16107, 2020.
- [13] Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell. Tent: Fully test-time adaptation by entropy minimization. arXiv preprint arXiv:2006.10726 , 2020.
- [14] Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shijian Zheng, Peilin Zhao, and Mingkui Tan. Efficient test-time model adaptation without forgetting. In International conference on machine learning , pages 16888-16905. PMLR, 2022.
- [15] Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao, and Mingkui Tan. Towards stable test-time adaptation in dynamic wild world. arXiv preprint arXiv:2302.12400 , 2023.
- [16] Guan-Ting Lin, Shang-Wen Li, and Hung-yi Lee. Listen, adapt, better wer: Sourcefree single-utterance test-time adaptation for automatic speech recognition. arXiv preprint arXiv:2203.14222 , 2022.
- [17] Guan-Ting Lin, Wei-Ping Huang, and Hung-yi Lee. Continual test-time adaptation for end-toend speech recognition on noisy speech. arXiv preprint arXiv:2406.11064 , 2024.
- [18] Changhun Kim, Joonhyung Park, Hajin Shim, and Eunho Yang. Sgem: Test-time adaptation for automatic speech recognition via sequential-level generalized entropy minimization. arXiv preprint arXiv:2306.01981 , 2023.
- [19] Hongfu Liu, Hengguan Huang, and Ye Wang. Advancing test-time adaptation in wild acoustic test settings. arXiv preprint arXiv:2310.09505 , 2023.
- [20] Q. Wang, O. Fink, L. Van Gool, and et al. Continual test-time domain adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 7201-7211, 2022.
- [21] Jae-Hong Lee, Do-Hee Kim, and Joon-Hyuk Chang. Awmc: Online test-time adaptation without mode collapse for continual adaptation. In 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU) , pages 1-8. IEEE, 2023.
- [22] Hong Jia, Young Kwon, Alessio Orsino, Ting Dang, Domenico Talia, and Cecilia Mascolo. Tinytta: Efficient test-time adaptation via early-exit ensembles on edge devices. Advances in Neural Information Processing Systems , 37:43274-43299, 2024.
- [23] MJehanzeb Mirza, Jakub Micorek, Horst Possegger, and Horst Bischof. The norm must go on: Dynamic unsupervised domain adaptation by normalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 14765-14775, 2022.
- [24] Bowen Zhao, Chen Chen, and Shu-Tao Xia. Delta: degradation-free fully test-time adaptation. arXiv preprint arXiv:2301.13018 , 2023.
- [25] Steffen Schneider, Evgenia Rusak, Luisa Eck, Oliver Bringmann, Wieland Brendel, and Matthias Bethge. Improving robustness against common corruptions by covariate shift adaptation. Advances in neural information processing systems , 33:11539-11551, 2020.
- [26] Zixian Su, Jingwei Guo, Kai Yao, Xi Yang, Qiufeng Wang, and Kaizhu Huang. Unraveling batch normalization for realistic test-time adaptation. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 15136-15144, 2024.

- [27] Yusuke Iwasawa and Yutaka Matsuo. Test-time classifier adjustment module for model-agnostic domain generalization. Advances in Neural Information Processing Systems , 34:2427-2440, 2021.
- [28] Shuaicheng Niu, Chunyan Miao, Guohao Chen, Pengcheng Wu, and Peilin Zhao. Test-time model adaptation with only forward passes. arXiv preprint arXiv:2404.01650 , 2024.
- [29] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 770-778, 2016.
- [30] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 , 2020.
- [31] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. arXiv preprint arXiv:1903.12261 , 2019.
- [32] Jinyu Li, Li Deng, Yifan Gong, and Reinhold Haeb-Umbach. An overview of noise-robust automatic speech recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing , 22(4):745-777, 2014.
- [33] Malik Boudiaf, Romain Mueller, Ismail Ben Ayed, and Luca Bertinetto. Parameter-free online test-time adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 8344-8353, 2022.
- [34] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691 , 2021.
- [35] Nikolaus Hansen. The cma evolution strategy: A tutorial. arXiv preprint arXiv:1604.00772 , 2016.
- [36] Alex Graves, Santiago Fernández, Faustino Gomez, and Jürgen Schmidhuber. Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In Proceedings of the 23rd international conference on Machine learning , pages 369-376, 2006.
- [37] Mostafa Shahin, Julien Epps, and Beena Ahmed. Phonological level wav2vec2-based mispronunciation detection and diagnosis method. Speech Communication , page 103249, 2025.
- [38] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: an asr corpus based on public domain audio books. In 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP) , pages 5206-5210. IEEE, 2015.
- [39] Jon Barker, Ricard Marxer, Emmanuel Vincent, and Shinji Watanabe. The third 'chime'speech separation and recognition challenge: Analysis and outcomes. Computer Speech &amp; Language , 46:605-626, 2017.
- [40] Rosana Ardila, Megan Branson, Kelly Davis, Michael Henretty, Michael Kohler, Josh Meyer, Reuben Morais, Lindsay Saunders, Francis M Tyers, and Gregor Weber. Common voice: A massively-multilingual speech corpus. arXiv preprint arXiv:1912.06670 , 2019.
- [41] François Hernandez, Vincent Nguyen, Sahar Ghannay, Natalia Tomashenko, and Yannick Esteve. Ted-lium 3: Twice as much data and corpus repartition for experiments on speaker adaptation. In Speech and Computer: 20th International Conference, SPECOM 2018, Leipzig, Germany, September 18-22, 2018, Proceedings 20 , pages 198-208. Springer, 2018.
- [42] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems , 33:12449-12460, 2020.
- [43] Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM transactions on audio, speech, and language processing , 29:3451-3460, 2021.

- [44] David S Pallett. Performance assessment of automatic speech recognizers. Journal of Research of the National Bureau of Standards , 90(5):371, 1985.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: See Abstract and Section 1 (Introduction, third and fourth paragraph). Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: See Section 6 (Conclusion and Discussion, The limitations paragraph.) Guidelines:

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

Justification: We include no formal theorems or lemmas in this work. All claims are validated through experimental results.

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

Justification: See Section 3 (Methodology) and Appendix D (Detailed algorithms). Our code is publicly available at: https://github.com/JiahengDong/E-BATS .

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

Justification: Our code is publicly available at: Code is available at: https://github. com/JiahengDong/E-BATS .

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not

including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: See Section 4 (Experiment), Appendix B (Details of datasets and baseline settings) and Appendix D (Detailed algorithms).

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: See section 3 (Methology: FID figure), Appendix A (Detailed FID scores), and Appendix C.2 (Ablation study with different seeds).

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

Justification: See Section 4 (Experiment: Implementation Details) and Figure 6 Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: See Appendix E.

## Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: See Appendix E (The second paragraph and the last sentence in the first paragraph)

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

Justification: Our work does not release or produce any new data or generative model artifacts-rather.

Guidelines:

- The answer NA means that the paper poses no such risks.

- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All the data and models are correctly cited and follow the license and terms of use explicitly. See Appendix E and Appendix B.

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

Justification: See Section 3 (Methodology), Section 4 (Experiment details) and Section 6 (Limitation). Our code will be publicly available upon acceptance (See Abstract, last sentence).

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: No crowdsourcing or new human-subject data collection was conducted in this work.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: No IRB approval was required for this study because we did not collect any new human-subject data.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: All text, figures, and analyses related with core methods were created solely by the authors.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A FID score of detailed domain shifts

To accurately quantify distributional shifts between the source and target domains, we employ the Fréchet Inception Distance (FID), a metric effective in capturing differences in both mean and covariance statistics of feature embeddings. Specifically, given the CNN-extracted latent embeddings Z t = [ z 1 t , z 2 t , . . . , z N t t ] for each utterance X t , we first average these embeddings across all frames to obtain the utterance-level representation as 1 N t ∑ N t i =1 z i t ∈ R d due to the variable length of each utterance.

Considering the utterance-level embeddings from the source dataset D src and the target dataset D tgt , we calculate their empirical mean vectors and covariance matrices as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ε &gt; 0 is added to ensure numerical stability.

The final FID between the source and target domains is computed as:

<!-- formula-not-decoded -->

where the first part is the Mean shift, and the second part is the Covariance shift. Mean Covariance shift Ratio will be calculated as Mean shift Covariance shift .

Table 4: FID Scores and Mean Shift Ratios for Various Noisy Conditions

| Noisy Condition                             |   Mean shift |   Covariance shift |   Mean/Covariance shift Ratio( × ) |
|---------------------------------------------|--------------|--------------------|------------------------------------|
| Gaussian noise                              |        0.031 |              0.004 |                                7.8 |
| Single-domain environment noise (simulated) |        0.016 |              0.003 |                                5.3 |
| Single-domain environment noise (real)      |        0.029 |              0.004 |                                7.3 |
| Mixed-domain environment noise              |        0.019 |              0.005 |                                3.8 |
| Speaking Style variability                  |        0.004 |              0.002 |                                2   |

Table 5: FID Scores and Mean Shift Ratios for detailed Gaussian noise and single-domain environment noise

| Noisy Condition            |   Mean shift |   Covaraince shift |   Mean/Covariance shift Ratio ( × ) |
|----------------------------|--------------|--------------------|-------------------------------------|
| Gaussian noise σ = 0 . 005 |        0.017 |              0.003 |                                 5.7 |
| Gaussian noise σ = 0 . 01  |        0.028 |              0.003 |                                 9.3 |
| Gaussian noise σ = 0 . 015 |        0.036 |              0.004 |                                 9   |
| Gaussian noise σ = 0 . 02  |        0.042 |              0.004 |                                10.5 |
| Cafe-real                  |        0.031 |              0.003 |                                10.3 |
| Bus-real                   |        0.031 |              0.003 |                                10.3 |
| Pedestain-real             |        0.031 |              0.004 |                                 7.8 |
| Street-real                |        0.022 |              0.004 |                                 5.5 |
| Cafe-simu                  |        0.021 |              0.003 |                                 7   |
| Bus-simu                   |        0.011 |              0.003 |                                 3.7 |
| Pedestain-simu             |        0.017 |              0.004 |                                 4.3 |
| Street-simu                |        0.015 |              0.003 |                                 5   |

## B Experiments

## B.1 Datasets

- Gaussian Noise Data. Following [16, 19], we corrupt the LibriSpeech [38] test-other split with zero-mean additive Gaussian noise to provide covariate shifts at different amplitudes ( σ ∈ { 0 . 0 , 0 . 005 , 0 . 01 , 0 . 015 , 0 . 02 } ). This setting provides a controlled evaluation of robustness to incremental noise severity.
- Single-Domain Background Noise Data.
- -CHiME-3-single: It is a noisy version of WSJ corpus with artificial and real-world environmental noises at 16 kHz. We utilize the official simulated and real enhanced evaluation sets from CHiME3 [39], which cover four challenging acoustic environments: bus, cafe, pedestrian area, and street junction. This setting simulates domain-specific, scene-consistent background conditions.
- Multi-Domain and Wild Real-World Data.
- CHiME-3-Mix: All CHiME-3 scenarios are combined into a dynamic stream to simulate continuously shifting acoustic environments, similar to setups in continual test-time adaptation [17].
- CommonVoice (CV) [40]: A crowdsourced project where volunteers contribute by reading Wikipedia sentences to produce 48 kHz audio samples. To align with the source ASR models' training conditions, we resampled these recordings to 16 kHz. The test set from the en-June-22nd-2020 release was used to evaluate robustness against different speaking styles, accent variability, and crowd-sourced audio quality issues.
- TEDLIUM-v2 (TED): Consists of oratory speech from TED conference videos with high quality stored at 16 kHz. We use the official test set for experiments, which introduces mismatches in recording quality and presentation style speech, diverging from the read speech in LibriSpeech or CommonVoice, and thus providing a natural domain shift. Following [16], transcripts across all datasets are converted to uppercase and stripped of punctuation, retaining only apostrophes.

## B.2 Baseline methods

The baseline methods include both BP TTA and BP-free TTAs.

## BP TTAs:

- TENT [13]. A fully test-time adaptation method that minimizes entropy by updating BatchNorm affine parameters online. We use it in episodic version since the batch size is small.
- SAR [15]. A sharpness-aware and reliable entropy minimization method that selectively filters samples with large gradients and encourages the model weights to converge to a flat minimum, improving stability under wild domain shifts.
- EATA [14]. An efficient TTA framework that selectively adapts on samples with lower uncertainty to reduce gradient noise and also mitigates catastrophic forgetting through a Fisher regularizer.
- CoTTA [20]. Acontinuous TTA method that maintains a teacher-student adaptation strategy with stochastically restoring certain model parameters.
- SUTA [16]. A single-utterance test-time adaptation method based on entropy minimization and minimum class confusion, adapted for CTC-based ASR.
- CSUTA. A continous version of SUTA with iteration step of 1, which is examined as one baseline in the work of [17].
- DSUTA [17]. A dynamic variant of SUTA that adaptively resets or retains model updates based on domain shift detection with fast-slow adaptation strategy.
- CEA [19]. Confidence-enhanced frame-level adaptation with short-term consistency regularization, proposed for wild acoustic test conditions.
- SGEM[18]. Amethod leveraging beam-search logits and generalized entropy minimization for autoregressive ASR adaptation at sequence-level granularity.
- AWMC[21]. A pseudo-labeling-based continual TTA algorithm for ASR that employs an anchor model, leader model and chaser model to achive stabel continous adaptation wihout forgetting.

## BP-free TTAs:

- LAME[33]. Atraining-free approach that corrects model outputs probabilities by estimating distribution drift in feature space.
- T3A [27]. A TTA technique that adjusts classifier via pseudo-prototypes without requiring backward passes.
- FOA [28]. Aforward-only approach that optimizing learnable prompts with activation shifts to avoid forgetting issue and trivial solutions.

## B.3 Baseline methods hyperparameter setting

The detailed baseline settings and the hyperparameter tuning are presented for both BP TTA and BP-free TTA.

BP TTA baslines The hyperparameter settings for TTA methods are organized as follows: For Speech Foundation Models (SFMs), we follow the configurations specified in the original papers for SUTA [16], DSUTA [17], CEA [19], and SGEM [18]. For CSUTA [17] and AWMC [21], which do not have released code, we adhere to the hyperparameters outlined in their official papers and the implementations from [17]. Additionally, we set the model to evaluation mode to maintain consistency.

For Visual Foundation Models (VFMs), to adapt the methods for SFMs with a batch size of 1 (BS=1), we follow the guidelines presented in [16, 19, 28]. All optimizers are configured to use the AdamW optimizer with the same learning rate as the TTA baseline methods for SFMs. Episodic methods are set with 10 iteration steps, while continuous methods use a single step. Minor adjustments are made for specific methods: for EATA [14], we use e \_ margin = 0 . 4 × ln(32) and d \_ margin = 1 . 0 ; for SAR [15], e \_ margin = 0 . 4 × ln(32) and reset \_ constant = 0 . 3 ; and for CoTTA [20], the augmentation threshold is set to 0.2, with augmentation limited to adding Gaussian noise.

BP-free TTA baselines We follow the original hyperparameter settings for LAME [33] and T3A [27] as specified in their official papers. For FOA, the parameters are set as follows: σ = 0 . 1 (CMA-ES), α = 0 . 05 , and γ = 0 . 1 .

In our approach, adaptation for each utterance is terminated early if the best fitness across iterations does not improve by at least 0.001 for three consecutive steps.

## B.4 Backbone models

Wav2Vec2ForCTC-Base [42] model employs a 12-layer Transformer encoder with a CNNbased feature extractor, representing lightweight ASR architectures optimized for fast inference. HuBERTForCTC-Large [43] model features a deeper 24-layer Transformer stack with a similar CNN front-end, offering a more powerful and robust ASR framework.

## C Detailed Experiment results

## C.1 CHiME3

The detailed performance on CHiME3 dataset using Wav2Vec2ForCTC-Base and HuBERTForCTClarge are shown in Tables 6 and 7 respectively. It presents the performance comparison across four different acoustic environments, including cafe, bus, pedestrian, and street. For each acoustic condition, we also include the simulated and real-world noise conditions. The performance also demonstrated the superior performance of our approach over all BP-free TTA and most of BP-based TTA.

Table 6: Comparison of TTA methods across CHiME3-single (cafe, bus, pedestrian, street) by using Wav2vec2ForCTC-base model. WER in bold is the best performance within BP-free TTA methods, and the underlined WER is the best within both BP and BP-free TTA methods.

| Method             | Cafe   | Cafe   | Bus   | Bus   | Pedestrian   | Pedestrian   | Street   | Street   | Average   | Average   | Average   |
|--------------------|--------|--------|-------|-------|--------------|--------------|----------|----------|-----------|-----------|-----------|
| Method             | Simu   | Real   | Simu  | Real  | Simu         | Real         | Simu     | Real     | Simu      | Real      | Overall   |
| Source             | 20.1   | 58.7   | 14.6  | 56.2  | 17.9         | 55.5         | 18.7     | 32.2     | 17.8      | 50.7      | 34.2      |
| BP adaptation      |        |        |       |       |              |              |          |          |           |           |           |
| TENT (episodic)    | 20.0   | 58.4   | 14.5  | 55.9  | 17.8         | 55.2         | 18.7     | 32.0     | 17.8      | 50.4      | 34.1      |
| EATA               | 20.0   | 55.3   | 14.9  | 54.1  | 18.3         | 51.6         | 18.9     | 31.4     | 18.0      | 48.1      | 33.1      |
| SAR                | 19.6   | 56.2   | 14.4  | 56.5  | 17.8         | 52.8         | 18.7     | 33.0     | 17.6      | 49.6      | 33.6      |
| CoTTA              | 19.2   | 58.6   | 14.2  | 49.2  | 16.8         | 55.4         | 17.8     | 32.1     | 17.0      | 48.8      | 16.2      |
| CEA                | 17.3   | 45.8   | 13.1  | 41.8  | 16.1         | 38.0         | 17.1     | 25.2     | 15.9      | 37.7      | 26.8      |
| SGEM               | 17.4   | 45.5   | 13.0  | 43.1  | 15.8         | 38.8         | 17.1     | 25.6     | 15.8      | 38.5      | 27.2      |
| AWMC               | 19.5   | 62.9   | 14.5  | 54.6  | 17.4         | 39.0         | 18.5     | 31.9     | 17.5      | 50.6      | 34.0      |
| SUTA               | 17.1   | 45.1   | 13.0  | 42.5  | 16.2         | 38.1         | 17.5     | 25.3     | 16.0      | 37.8      | 26.8      |
| CSUTA (1 step)     | 17.7   | 40.3   | 14.8  | 41.1  | 16.8         | 36.4         | 18.7     | 26.0     | 17        | 36.0      | 26.5      |
| DSUTA              | 16.4   | 36.3   | 13.0  | 39.6  | 15.2         | 32.8         | 15.6     | 22.4     | 15.1      | 32.8      | 24.0      |
| BP-free adaptation |        |        |       |       |              |              |          |          |           |           |           |
| T3A                | 20.9   | 61.3   | 15.3  | 59.0  | 18.5         | 58.4         | 19.6     | 33.6     | 18.6      | 53.1      | 35.9      |
| LAME               | 20.9   | 61.6   | 15.2  | 58.9  | 18.6         | 58.9         | 19.8     | 33.9     | 18.6      | 53.3      | 36.0      |
| FOA                | 19.9   | 52.5   | 14.6  | 50.5  | 18.0         | 48.1         | 18.6     | 31.0     | 17.8      | 45.5      | 31.7      |
| Ours               | 16.1   | 37.9   | 13.1  | 37.4  | 15.1         | 33.1         | 15.4     | 24.0     | 14.9      | 33.1      | 24.0      |

Table 7: Comparison of TTA methods across CHiME3-single (cafe, bus, pedestrian, street) by using HuBERTForCTC-large. WER in bold is the best performance within BP-free TTA methods, and the underlined WER is the best within both BP and BP-free TTA methods.

| Method             | Cafe   | Cafe   | Bus   | Bus   | Pedestrian   | Pedestrian   | Street   | Street   | Average   | Average   | Average   |
|--------------------|--------|--------|-------|-------|--------------|--------------|----------|----------|-----------|-----------|-----------|
| Method             | Simu   | Real   | Simu  | Real  | Simu         | Real         | Simu     | Real     | Simu      | Real      | Overall   |
| Source             | 9.2    | 27.9   | 8.5   | 26.2  | 9.2          | 24.3         | 10.1     | 16.6     | 9.3       | 23.8      | 16.6      |
| BP adaptation      |        |        |       |       |              |              |          |          |           |           |           |
| TENT (episodic)    | 9.2    | 27.5   | 8.5   | 25.8  | 9.2          | 24.0         | 10.0     | 16.5     | 9.2       | 23.5      | 16.4      |
| EATA               | 9.2    | 27.0   | 8.7   | 25.8  | 9.2          | 23.2         | 9.9      | 16.4     | 9.3       | 23.1      | 16.2      |
| SAR                | 9.1    | 28.0   | 8.4   | 26.1  | 9.1          | 23.2         | 9.9      | 17.3     | 9.1       | 23.7      | 16.4      |
| CoTTA              | 9.3    | 26.8   | 8.5   | 25.5  | 9.1          | 23.9         | 9.9      | 16.4     | 9.2       | 23.2      | 16.2      |
| CEA                | 8.6    | 22.5   | 8.1   | 22.0  | 8.3          | 19.6         | 9.2      | 14.7     | 8.6       | 19.7      | 14.2      |
| SGEM               | 8.7    | 22.5   | 8.1   | 21.8  | 8.5          | 19.9         | 9.5      | 14.7     | 8.6       | 19.7      | 14.2      |
| AWMC               | 9.5    | 25.5   | 8.9   | 23.9  | 9.4          | 22.5         | 10.4     | 16.5     | 9.6       | 22.1      | 15.9      |
| SUTA               | 8.6    | 22.9   | 8.1   | 22.2  | 8.4          | 19.6         | 9.4      | 14.6     | 8.6       | 19.8      | 14.2      |
| CSUTA (1 step)     | 9.3    | 22.8   | 8.9   | 22.2  | 9.5          | 19.7         | 9.9      | 14.7     | 9.4       | 19.9      | 14.7      |
| DSUTA              | 8.7    | 20.3   | 8.3   | 20.0  | 8.6          | 17.5         | 9.3      | 13.9     | 8.7       | 17.9      | 13.3      |
| BP-free adaptation |        |        |       |       |              |              |          |          |           |           |           |
| T3A                | 18.2   | 42.1   | 18.2  | 42.1  | 17.3         | 37.5         | 18.2     | 29.0     | 18.0      | 37.7      | 27.9      |
| LAME               | 9.7    | 30.2   | 8.9   | 27.9  | 9.6          | 25.4         | 10.6     | 17.8     | 9.7       | 25.3      | 17.5      |
| FOA                | 9.6    | 26.9   | 8.6   | 26.2  | 9.3          | 23.4         | 10.4     | 16.4     | 9.5       | 23.2      | 16.4      |
| Ours               | 9.2    | 20.7   | 8.4   | 21.3  | 9.2          | 18.9         | 10.0     | 14.2     | 9.2       | 18.8      | 14.0      |

## C.2 Ablation study with multiple seeds

We repeat each ablation experiment using three different random seeds in Figure 8 to ensure that our results are robust and not the artifact of any single initialization.

Table 8: Ablation study with mean±std WER (%) over 3 seeds

| Prompt Adaptation   | Prompt Adaptation   | Prompt Adaptation   | Loss Function (w/ T-EMA)   | Loss Function (w/ T-EMA)   | Loss Function (w/ T-EMA)   | Loss Function (w/ T-EMA)   | Loss Function (w/o T-EMA)   | Loss Function (w/o T-EMA)   | Loss Function (w/o T-EMA)   | Loss Function (w/o T-EMA)   | T-EMA Mechanism   | T-EMA Mechanism   | T-EMA Mechanism   |
|---------------------|---------------------|---------------------|----------------------------|----------------------------|----------------------------|----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-------------------|-------------------|-------------------|
| Feat                | Trans               | WER                 | L ent                      | L utt                      | L token                    | WER                        | L ent                       | L utt                       | L token                     | WER                         | T-EMA             | Reset             | WER               |
| ✓                   | -                   | 24.0±0.0            | ✓                          | ✓                          | ✓                          | 24.0±0.0                   | ✓                           | ✓                           | ✓                           | 25.4±0.0                    | ✓                 | -                 | 24.3±0.0          |
| -                   | ✓                   | 34.1±0.1            | ✓                          | ✓                          | -                          | 24.2±0.1                   | ✓                           | ✓                           | -                           | 25.5±0.1                    | -                 | ✓                 | 26.5±0.0          |
| -                   | -                   | -                   | ✓                          | -                          | -                          | 24.6±0.1                   | ✓                           | -                           | -                           | 64.3±11.5                   | -                 | -                 | 25.4±0.0          |

## C.3 Sensitive Analysis

## C.3.1 Sensitivity of T-EMA Decay Parameter

Table 9 below presents our analysis on the EMA parameter γ , where we evaluated performance across γ ∈ { 0 . 50 , 0 . 60 , 0 . 70 , 0 . 80 , 0 . 90 , 0 . 95 , 0 . 99 } under the caf-real condition. We observed a clear U-shaped WER curve: performance improves up to γ = 0 . 90 (best WER 37.9%), and then degrades as γ increases further. This behavior aligns with our expectation and is also observed in other datasets that smaller γ values lead to unstable, overly reactive updates, while larger values result in over-smoothing and hinder adaptation.

Table 9: WER (%) of Wav2Vec2ForCTC on caf-real CHiME-3 for different T-EMA decay γ .

| γ      |   0.50 |   0.60 |   0.70 |   0.80 |   0.90 |   0.95 |   0.99 |
|--------|--------|--------|--------|--------|--------|--------|--------|
| WER(%) |   39.1 |   39.1 |   38.4 |   38.1 |   37.9 |   38.3 |   39.9 |

## C.3.2 Sensitivity of Loss Component Weights

To further understand how the relative weighting of loss components affects the CMA-ES optimization process, we conducted a sensitivity analysis of the loss weights α and β for the entropy loss and utterance-level loss, which in turn leads to changes in the token-level weight c (Minmax normalization Equation in Section 3.3). The WER results using the caf-real condition of the CHiME3 dataset are shown in Table 10 below. The results reveal that WER is remarkably stable across a wide range of weight settings. WER stays within a narrow range across different values of α and β , showing a broad area of near-optimal performance instead of a single best point. The results suggest that both components of the loss must be balanced, as extreme values for either can hurt performance. The chosen setting ( α = 1 . 0 , β = 2 . 0 ) sits comfortably in this stable region and consistently yields the best performance.

Table 10: WER (%) of Wav2Vec2ForCTC on caf-real CHiME-3 under varying α (rows) and β (columns), representing varying weights for the token-level loss.

|   α \ β |   0.5 |   1.0 |   1.5 |   2.0 |   2.5 |   3.0 |   3.5 |   4.0 |   4.5 |   5.0 |
|---------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|     0.5 |  37.7 |  38.2 |  38.5 |  38.6 |  39.1 |  39.4 |  39.2 |  39.8 |  39.4 |  39.7 |
|     1   |  38.1 |  38.1 |  38   |  37.9 |  38.4 |  38.3 |  38.4 |  38.8 |  38.7 |  39.6 |
|     1.5 |  38.5 |  38.1 |  38.4 |  38.1 |  38.5 |  38   |  38.2 |  37.8 |  38.3 |  38.4 |
|     2   |  38.4 |  38.3 |  38   |  37.9 |  37.9 |  38.1 |  38   |  38.3 |  38   |  38.4 |

## C.3.3 Sensitivity of CMA-ES to Target Domain Complexity

We selected four target domains exhibiting increasing variability, quantified by the covariance shift offered by Fréchet Inception Distance (FID) of the embedding distribution between the source and target. These range from low to high variability: Gaussian noise ( σ = 0 . 01 ), CHiME-3 cafesimulated, CHiME-3 cafe-real, and CHiME-3 mixed. As shown in Table 11 below, with CMA-ES

population size increasing from 10 to 100, the largest gains occur moving from 10 to 40 candidates across all domains. Beyond 50 candidates, performance plateaus. These results show that while more complex domains (cafe-real, mixed) begin at higher absolute WER, they exhibit the same early-saturation behavior as simpler conditions. Therefore, sensitivity to domain complexity may not be directly related to the number of prompt candidates considered.

Table 11: WER (%) across different population sizes on four target domains with varying complexity.

| Candidate Size         |   10 |   20 |   30 |   40 |   50 |   60 |   70 |   80 |   90 |   100 |
|------------------------|------|------|------|------|------|------|------|------|------|-------|
| Gaussian ( σ =0 . 01 ) | 16.2 | 15.6 | 15.2 | 14.9 | 14.8 | 14.8 | 14.7 | 14.7 | 14.7 |  14.7 |
| Cafe-simulated         | 17   | 16.5 | 16.5 | 16.2 | 16.1 | 16.3 | 16.1 | 16   | 16   |  16.2 |
| Cafe-real              | 41.8 | 39.3 | 38.3 | 38.3 | 37.9 | 38   | 37.9 | 37.9 | 38   |  37.8 |
| CHiME3 Mixed           | 25.7 | 25.1 | 24.7 | 24.6 | 24.3 | 24.3 | 24.4 | 24.2 | 24.4 |  24.3 |

## C.3.4 Sensitivity of CMA-ES to Candidate Size and Iteration Steps

We find the expected trade-off between the population size and optimization iterations with word error rate (WER) (Table 12 and Table 13 below). As shown in Table 12 (caf-real condition of CHiME-3 using Wav2Vec2ForCTC), increasing the CMA-ES candidate size from 10 to 40 yields the most significant gains. Table 13 demonstrates that moving from 5 to 25 iterations captures most of the benefit. We observe the same early-saturation behavior across other conditions (e.g., bus-real, street-real) and datasets.

Table 12: WER (%) of Wav2Vec2ForCTC on the caf-real condition of CHiME-3 under different CMA-ES population sizes (iteration steps = 25).

| Candidate Size   |   10 |   20 |   30 |   40 |   50 |   60 |   70 |   80 |   90 |   100 |
|------------------|------|------|------|------|------|------|------|------|------|-------|
| WER(%)           | 41.8 | 39.3 | 38.3 | 38.3 | 37.9 |   38 | 37.9 | 37.9 |   38 |  37.8 |

Table 13: WER (%) of Wav2Vec2ForCTC on the caf-real condition of CHiME-3 under different CMA-ES iteration steps (population size = 50).

| Iteration Steps   |    5 |   10 |   15 |   20 |   25 |   30 |   35 |   40 |   45 |   50 |
|-------------------|------|------|------|------|------|------|------|------|------|------|
| WER(%)            | 40.8 | 39.9 | 38.6 |   38 | 37.9 |   38 | 37.9 | 37.6 | 38.4 |   38 |

## C.4 Impact of Utterance-level Loss on Trivial Predictions

Weanalyzed and compared the occurrence of trivial predictions under two conditions: (i) entropy-only and (ii) entropy with utterance-level loss, using the CHiME-3-Mix dataset. As shown in Table 14, we observed two types of trivial solutions: blank predictions and single-character predictions (entropy = 0 with only one character predicted). It is evident that the utterance-level loss significantly reduces the incidence of trivial solutions from 37.5% to 0.61%, thereby improving the WER.

## C.5 Effectiveness of Continual Adaptation with T-EMA

We reported the WER against the number of process utterances under two variations of the continual adaptation process: (i) T-EMA with resetting, reported in Table 3 in the ablation study; and (ii) T-EMA with exponential averaging (proposed), presented in Table 3 and Section 3.4. We evaluated their performance across varying numbers of target samples from 100 to 800 in Table 15 below, using the LibriSpeech dataset with Gaussian noise ( σ = 0 . 015 ). With the resetting approach, performance increases up to 300 samples and then declines with additional samples, suggesting that resetting CMA-ES prevents leveraging prior adaptation. In contrast, the proposed exponential averaging method demonstrates relatively stable performance, indicating that even with a small number of samples, reliable adaptation can be achieved. This is likely due to the T-EMA mechanism accumulating knowledge from earlier utterances and updating CMA-ES in a more favorable region

Table 14: Trivial solutions on CHiME-3-Mix: number and percentage of trivial predictions among all predictions.

| Configuration          | Blank      | Single-char   | Total problematic   |   WER(%) |
|------------------------|------------|---------------|---------------------|----------|
| Entropy-only           | 42 (1.59%) | 948 (35.91%)  | 990 (37.50%)        |     49.6 |
| + Utterance-level loss | 12 (0.45%) | 4 (0.15%)     | 16 (0.61%)          |     25.5 |

Table 15: WER (%) against the number of processed utterances with T-EMA (resetting) and T-EMA (proposed) under the LibriSpeech dataset with Gaussian noise ( σ = 0 . 015 ).

| Method            |   100 |   200 |   300 |   400 |   500 |   600 |   700 |   800 |
|-------------------|-------|-------|-------|-------|-------|-------|-------|-------|
| T-EMA (Resetting) |  20.4 |  19.7 |  19.8 |  20.4 |  20.8 |  20.8 |  21.1 |  21.3 |
| T-EMA (Proposed)  |  17.9 |  17   |  17.2 |  17.4 |  17.6 |  17.5 |  17.8 |  17.7 |
| Performance Gain  |   2.5 |   2.7 |   2.6 |   3   |   3.2 |   3.3 |   3.3 |   3.6 |

of the search space. The improved performance over the resetting mechanism further demonstrates that our method is robust to the number of samples used for adaptation, and that the proposed T-EMA effectively stabilizes the adaptation process.

## C.6 Memory Usage

Figure 6 shows the memory usage comparison between two SFMs: Wav2Vec2ForCTC-Base and HuBERTForCTC-Large. The increasing model size leads to a significant increase in memory usage for adaptation, especially for the BP-based TTA, whereas our method demonstrates only a slight increase in memory usage.

Figure 6: Comparison of Average Peak GPU memory usage of different TTA methods across all datasets with two different scale backbones.

<!-- image -->

## C.7 Controlling Prompt Vector Dimensionality for CMA-ES Efficiency

The computational cost of CMA-ES with increasing dimensionality d may cause concerns. Indeed, the full-rank CMA-ES algorithm has a per-iteration complexity of O ( Jd 2 + d 3 ) , where J is the number of sampled solutions (i.e., prompts in our case). To mitigate this cost, we design our prompt vector to have a fixed dimensionality of 512 , aligned with the embedding size commonly used in speech foundation models, ensuring that d remains tractable in practical scenarios. Furthermore, as

Table 16: WER (%) of Wav2Vec2ForCTC on the caf-real condition of CHiME-3 under different CMA-ES candidate sizes J (iteration steps = 25).

| J      |   10 |   20 |   30 |   40 |   50 |   60 |   70 |   80 |   90 |   100 |
|--------|------|------|------|------|------|------|------|------|------|-------|
| WER(%) | 41.8 | 39.3 | 38.3 | 38.3 | 37.9 |   38 | 37.9 | 37.9 |   38 |  37.8 |

Table 17: Per-utterance adaptation time complexity (bigO ) for E-BATS and top TTA baselines.

| Method   | Time Complexity per Utterance                               |
|----------|-------------------------------------------------------------|
| DSUTA    | O ( N × ( F + E + P ))                                      |
| FOA      | O ( N × [ J ( F + E + Ld 2 )+ d 3 ])                        |
| E-BATS   | O ( N × [ J ( F + E + L + d + L &#124; V &#124; d )+ d 3 ]) |

shown in Table 16, with our grid search of J , we observe the optimal J always much smaller than d ; in our configuration, we fix J = 50 , which remains significantly smaller than d . This ensures that the asymptotic complexity does not exceed O ( d 3 ) . Thus, while CMA-ES theoretically scales cubically with d , our design choices effectively cap the computational overhead in the context of high-dimensional but fixed-size prompt vectors.

## C.8 Adaptation performance on Speech Emotion Recognition Task

To further test the generality of our method beyond ASR with CTC, we applied E-BATS to speech emotion recognition tasks (cross-entropy loss) using the IEMOCAP dataset and the SpeechBrain/emotionrecognition-wav2vec2-IEMOCAP model under additive Gaussian noise ( σ = 0 . 02 ). Adaptation increased emotion prediction accuracy from 38.8% to 43.4%, demonstrating that our E-BATS framework can enhance the performance of other downstream tasks using SFMs.

## C.9 Adaptation Speed and Computation Complexity

We compute the per-utterance time complexity to compare E-BATS with the top-performing backpropagation-based (DSUTA) and backpropagation-free (FOA) methods (see Table 17). DSUTA scales with the large number of model parameters via backpropagation ( P ), often in hundreds of millions, whereas E-BATS only involves O ( d 2 ) and O ( d 3 ) operations (with d = 512 ). Since P ≥ d 3 in most SFM models, E-BATS achieves significant efficiency gains over DSUTA. While both FOA and E-BATS share the same asymptotic complexity of O ( d 3 ) for CMA-ES updates, E-BATS is practically more efficient since FOA requires a 3 3 multiplicative factor of O ( d 3 ) as it needs three prompts for adaptation. FOA also incurs an additional O ( Ld 2 ) cost (over d ) per prompt for prompt attention with L transformer layer encoders. Nonetheless, our main focus, as noted in the introduction, is the trade-off between accuracy and memory efficiency, which is more critical for resource-constrained devices.

where:

- F : cost of one forward pass
- E : cost of entropy loss calculation
- N : number of adaptation iteration steps per utterance
- J : number of candidate prompts per iteration
- | V | : size of the token class

## D Algorithms

The algorithm for LPA per utterance and for T-EMA is shown in Algorithm 1 and Algorithm 2 respectively.

## Algorithm 1 Lightweight Prompt Adaption (Per Utterance)

Require: CMA-ES params ϕ 0 t , max steps K , Utterance X t Ensure: Adapted predictions ˆ y 1: best\_loss ←∞ 2: for k = 1 to K do 3: Sample prompts S k t = [ s k t, 1 , s k t, 2 , . . . , s k t,J ] from CMA-ES ϕ t,k - 1 4: Inject s k t,j with Z t and feedforward pass 5: Compute loss L k adapt,all = [ L k adapt, 1 , L k adapt, 2 , . . . , L k adapt,J ] 6: if min j ∈{ 1 ,...,J } L k adapt,j &lt; best\_loss then 7: best\_loss ← L k adapt,j 8: ˆ Y t ← decode adapted output with s k t,j 9: end if 10: ϕ k t ← Update ( ϕ k - 1 t , L k adapt,all , S k t ) 11: end for

## Algorithm 2 T-EMA Updating Strategy

Require: Utterance data T , CMA-ES params ϕ , Iteration Steps

- 0 0 1: Initialize ϕ ema = { C ema , m ema , σ ema } ← ϕ 0 0 2: for each utterance t in T do 3: ϕ 0 t ← ϕ K t Run Algorithm 1 4: EMA Update ( ϕ ema , ϕ K t ) 5: ϕ 0 t +1 ← ϕ ema 6: end for

K

## E Ethical Consideration

Our research fully complies with the NeurIPS Code of Ethics. We exclusively utilize publicly available datasets, pretrained models, baseline methods, and their accompanying code, strictly adhering to their respective licenses and usage protocols. We did not collect any new data, nor do our adaptation methods pose privacy risks or enable misuse. Thus, our work does not introduce broader negative societal impacts, eliminating the need for additional safeguards beyond standard ethical research practices.

Moreover, our method has potential positive societal impacts, including improving the accessibility and reliability of speech recognition technology in noisy real-world environments, thereby benefiting communication technologies, assistive systems, and applications serving diverse and inclusive user populations.