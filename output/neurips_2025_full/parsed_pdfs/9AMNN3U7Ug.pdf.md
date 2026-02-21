## GeoAda: Efficiently Finetune Geometric Diffusion Models with Equivariant Adapters

## Wanjia Zhao ∗ , Jiaqi Han ∗ , Siyi Gu, Mingjian Jiang, James Zou, Stefano Ermon

Department of Computer Science Stanford University

## Abstract

Geometric diffusion models have shown remarkable success in molecular dynamics and structure generation. However, efficiently fine-tuning them for downstream tasks with varying geometric controls remains underexplored. In this work, we propose an SE(3)-equivariant adapter framework (GeoAda) that enables flexible and parameter-efficient fine-tuning for controlled generative tasks without modifying the original model architecture. GeoAda introduces a structured adapter design: control signals are first encoded through coupling operators, then processed by a trainable copy of selected pretrained model layers, and finally projected back via decoupling operators followed by an equivariant zero-initialized convolution. By fine-tuning only these lightweight adapter modules, GeoAda preserves the model's geometric consistency while mitigating overfitting and catastrophic forgetting. We theoretically prove that the proposed adapters maintain SE(3)-equivariance, ensuring that the geometric inductive biases of the pretrained diffusion model remain intact during adaptation. We demonstrate the wide applicability of GeoAda across diverse geometric control types, including frame control, global control, subgraph control, and a broad range of application domains such as particle dynamics, molecular dynamics, human motion prediction, and molecule generation. Empirical results show that GeoAda achieves state-of-the-art fine-tuning performance while preserving original task accuracy, whereas other baselines experience significant performance degradation due to overfitting and catastrophic forgetting.

## 1 Introduction

Diffusion models have emerged as powerful generative frameworks across a wide range of domains, including image synthesis [44, 38], robotics [36, 27], and molecular generation [23, 41, 39, 42]. In particular, geometric diffusion models [11] which incorporate spatial and symmetry-aware inductive biases have shown strong empirical performance in tasks such as particle dynamic prediction [15, 17, 28], molecular generation [3, 14] and protein-ligand binding structure prediction [8]. By modeling data in an equivariant network, these models are able to capture complex geometric relationships essential for physical and chemical systems.

However, despite their strong task-specific performance, existing geometric diffusion models lack the ability to generalize across tasks. In particular, it remains unclear how a model pretrained on one geometric generation task can be effectively adapted to a new task involving additional or different control signals. This limitation is especially pronounced in real-world molecular applications, where the available data across tasks are often highly imbalanced, and collecting labeled pretraining data for every new condition is costly and time-consuming. Without a mechanism for transfer, models must be retrained from scratch for each new task, which is inefficient and often leads to overfitting or loss of previously learned capabilities.

∗ Equal contribution. Correspondence to wanjiazh@cs.stanford.edu. Code is available here.

Figure 1: Overall framework of GeoAda. The model integrates diverse control signals, including frame, global type, and subgraph controls through lightweight equivariant adapters inserted into the frozen pretrained denoiser.

<!-- image -->

To this end, we propose a general and efficient framework(GeoAda) that enables the transfer of geometric diffusion models across diverse downstream tasks with minimal computational overhead. Inspired by the success of ControlNet [44] in conditional image generation, we introduce an equivariant adapter module that augments a pretrained geometric diffusion model with task-specific control capability. The Equivariant Adapter comprises two key components: 1) The equivariant adapter block that operates through a structured sequence-where control signals are encoded via coupling operators, processed by a trainable copy of selected pretrained model layers, and then decoded via decoupling operators. 2) Equivariant zero convolution, which acts as a safeguard for the original score by zeroing out the conditional contribution at initialization without blocking gradient updates. This design preserves the model's SE(3)-equivariance and allows modular, task-specific adaptation without altering the original model architecture. In addition to being lightweight and flexible, the adapter is parameter-efficient and implicitly regularized, thereby mitigating overfitting and preserving the performance of the pretrained model.

In summary, we make the following contributions: 1. We propose an equivariant adapter framework (GeoAda) for geometric diffusion models that enables efficient task adaptation with minimal overhead. The adapter modules are lightweight and operate as plug-and-play components, allowing flexible conditioning on new control signals without architectural modifications to the pretrained model. 2. GeoAda is parameter-efficient, introducing minimal overhead for downstream tasks compared to full fine-tuning, which updates the entire model and incurs substantial memory and computational costs. 3. By freezing the pretrained model and introducing trainable adapters, GeoAda imposes implicit regularization, helping to mitigates overfitting and avoids catastrophic forgetting, thereby preserving performance on the pretraining task. 4. We carefully design the adapter architecture to be SE(3)-equivariant, ensuring that the adapted model retains the geometric inductive bias and the theoretical benefits of equivariant diffusion models, including SE(3)-invariant marginal distributions during generation. 5. We evaluate GeoAda across diverse geometric control types, including Frame Fontrol, Global Type Control, Subgraph Control, and a wide range of application domains, such as particle dynamics, molecular dynamics, human motion prediction, and molecule generation. GeoAda consistently matches or outperforms full fine-tuning baselines on downstream task, while avoiding performance degradation on the original pretrain task-a common failure mode of naive tuning and prompt-base approaches.

## 2 Related Work

Geometric diffusion models. Recent diffusion models have been extended to 3D geometric data, with SE(3) equivariance enabling physically consistent generation for tasks like molecular design and trajectory modeling. One of the earliest efforts, EDM [14] introduced an SE(3)-equivariant framework for 3D molecule generation that significantly improved sample quality. GeoDiff [42] pioneered this by learning stable molecular conformations through SE(3)-invariant diffusion, while GeoLDM [41, 39] advanced scalability and controllability via structured latent spaces. GCDM [23] advanced large molecule generation by incorporating geometry-complete local frames and chirality-sensitive features into SE(3)-equivariant networks. TargetDiff [9] further extended these models to structure-based

drug design by generating molecules conditioned on protein targets through an SE(3)-equivariant processor. Beyond molecular applications, diffusion augmented with geometric inductive bias has been explored in other domains such as 3D shape and scene generation [2] and robotics [36, 27]. Beyond static geometric modeling, GeoTDM [11] and EquiJump [5] address dynamic 3D systems by introducing temporal attention mechanisms. However, existing geometric diffusion models lack cross-task generalization. Our framework enables efficient adaptation to new controls.

Finetuning for (geometric) graphs. Finetuning for geometric GNNs generally falls into two categories: prompt-based and adapter-based methods. Pioneering prompt-based approaches [34, 20] introduce virtual class-prototype nodes with learnable links for edge prediction pre-trained models, but lack generalizability to alternative pre-training strategies. Meanwhile, works like GPF [6] explore universal prompt-based tuning by adding shared learnable vectors to all node features in the graph. Adapter-based methods, exemplified by AdapterGNN [18], insert lightweight modules into GNN layers, achieving parameter-efficient adaptation across diverse graph domains.

Finetuning diffusion models. Recent research has proposed various strategies for fine-tuning diffusion models with improved efficiency, control [44], and alignment [37]. ELEGANT [35] formulates fine-tuning as an entropy-regularized control problem, directly optimizing entropy-enhanced rewards with neural SDEs. ControlNet [44] improves controllability by adding lightweight trainable branches to frozen diffusion backbones. Prompt Diffusion [38] enables training-free in-context learning for image-to-image tasks via example-based conditioning. However, fine-tuning diffusion models in geometric domains (e.g., particles, molecules) remains underexplored. GeoAda addresses this gap by enabling efficient and effective adaptation of diffusion models to geometric tasks.

## 3 Preliminaries

Geometric graphs and trajectories. We represent a geometric graph as G = ( V , E ) where V is the set of nodes and E is the set of edges. In particular, each node i is equipped with certain node feature h i ∈ R H representing its type or physical property, and the Euclidean coordinate x i ∈ R 3 representing its spatial position. An edge exists between node i and j if they bear certain connectivity through, e.g. , chemical bonds, or spatial proximity with a distance smaller than a cutoff. A trajectory is a generalization of geometric graph in the dynamical setting where the coordinates x [ T ] i ∈ R 3 × T are augmented with an additional temporal dimension, where T is the number of frames.

Geometric diffusion models. Geometric diffusion models are a family of generative models for capturing the distribution of geometric graphs and/or trajectories. Given an input data point G 0 , they feature a forward noising process that gradually perturbs the clean data with a transition q ( G τ |G 0 ) where G T converges to a tractable prior. A neural network ϵ θ ( G τ , τ ) ( a.k.a. the denoiser) is learned to approximate the Stein score [33] through denoising score matching [31, 32, 13], which will be leveraged to derive the transition kernel p θ ( G τ -1 |G τ ) in the reverse process at sampling time. Notably, a core distinction of geometric diffusion models from others is that they enforce an SE(3)-invariant marginal, i.e. ,

<!-- formula-not-decoded -->

by parameterizing the denoiser ϵ θ with an SE(3)-equivariant architecture [14, 42], i.e. ,

<!-- formula-not-decoded -->

where SE(3) is the Special Euclidean group consisting of all rotations and translations in 3D.

## 4 Method

In this section, we detail our approach, equivariant adapter for geometric diffusion models. We first specify three types of controls in § 4.1 that are ubiquitously enforced to geometric diffusion models in various downstream tasks. In § 4.2, we propose an architecture-agnostic and principled recipe for encoding such controls that seamlessly enables finetuning on the pretrained denoiser. In § 4.3, we present our design of GeoAda, a plug-in-and-play adapter module tuned for each downstream task that unlocks transferability.

## 4.1 Geometric Controls for Geometric Diffusion Models

In this work, we aim to transfer the generation capability of pretrained geometric diffusion models to downstream tasks where additional geometric controls present. Specifically, we are concerned with three different types of geometric controls, namely global type control C G, subgraph control C S, and frame control C F, as detailed below.

Global type control. Each global type control ˜ c ∈ C G is a vector in R K describing certain global signal enforced on the geometric graph, such as an encoding of the class label, some quantum chemical property of the molecule [14], or even the embedding of some text prompt [22].

Subgraph control. Each subgraph control ˜ G = ( ˜ V , ˜ E ) ∈ C S is represented as a geometric graph with the set of nodes ˜ V and edges ˜ E . Subgraph control widely exists in scenarios where generating a geometric graph conditioned on another fixed subgraph is of interest. For example, in the task of pocket-conditioned ligand generation [8], the protein pocket is viewed as the fixed subgraph ˜ G while a geometric diffusion model is learned to generate the ligand, conditioned on ˜ G .

Frame control. Each frame control in C F takes the form of a sequence of additional ˜ T frames, namely ˜ x [ ˜ T ] i ∈ R 3 × ˜ T for each node i . Frame controls are enforced in cases when, e.g. , a trajectory has been partially observed and the model is expected to generate the future or missing frames conditioned on the observed frames.

## 4.2 Encoding Geometric Controls

In this subsection, we propose a simple yet effective approach for incorporating the geometric controls into the denoiser without modifying its architecture. Such feature is critical since it enables us to initialize the denoiser fully with the pretrained checkpoint when performing finetuning on downstream tasks, thus significantly alleviating optimization overheads and potential inconsistencies in the parameter space. More importantly, our design is also guaranteed to preserve the equivariance of the denoiser, a fundamental principle that leads to the success of geometric diffusion models.

In form, given the denoiser ϵ θ ( G τ , τ ) , we seek to devise ϵ θ ( G τ , τ, C ) where C ∈ C G ∪ C S ∪ C F is any of the control we specified in § 4.1. Our core observation lies in that each type of the control can be encoded through certain coupling operator f ( G τ , C ) of the input noised graph G τ and control C , and a corresponding decoupling operator g that extracts the scores on the nodes and frames in G τ from the output of ϵ θ . We introduce our design of f and g with respect to different controls as follows.

Global type control. For global type control C G := ˜ c ∈ R K , we design f as a node-wise addition of the input node feature and a linear transformation of the control ˜ c , i.e. , V ′ , E ′ = f ( V , E , C ) , where

<!-- formula-not-decoded -->

where σ : R K ↦→ R H is an MLP that lifts the control signal to the node feature space. We use identity function as the decoupling operator g .

Subgraph control. For subgraph control C S := ˜ G , the coupling operator f is realized by computing the supergraph of the input G and the control ˜ G , i.e. , V ′ , E ′ = f ( V , E , C ) , where

<!-- formula-not-decoded -->

The decoupling operator g is implemented by extracting the features of subgraph that corresponds to the nodes in the input graph G from the output of ϵ θ .

Frame control. For frame control C F := { ˜ x [ ˜ T ] i } , we implement f as a concatenation of the input frames and the frame control, i.e. , V ′ , E ′ = f ( V , E , C ) , where

<!-- formula-not-decoded -->

and g performs the reverse operation by discarding the frames corresponding to [ ˜ T ] from the output of ϵ θ .

Proposition 4.1 (Equivariance of control encoding) . If the denoiser ϵ θ is SE(3)-equivariant, the composition g ◦ ϵ θ ◦ f is also SE(3)-equivariant, for all controls C ∈ C .

## 4.3 Equivariant Adapters

With the control encoding in § 4.2, a straightforward approach to leverage a pretrained diffusion model on downstream tasks is to perform supervised finetuning (SFT). However, SFT usually induces suboptimal empirical performance, since 1. SFT is parameter-inefficient since each gradient update is conducted on all parameters of the pretrained model; 2. the full-parameter finetuning is prone to overfitting with limited amount of finetuning data; and 3. the model finetuned after SFT loses performance guarantee on the original task, a phenomenon widely acknowledged as catastrophic forgetting.

<!-- image -->

To address these challenges, we draw inspiration from the successful application of adapters on image diffusion models, e.g. , ControlNet [45], to devise a diffusion adapter for geometric diffusion models. Our approach, dubbed equivariant adapter, is a lightweight tunable module plugged-in

Figure 2: Overall framework of GeoAda. A control signal C is injected into the noised graph G τ and processed by an equivariant adapter block. The adapter output is added to the frozen denoiser and repeated B times to produce the final output.

on top of the pretrained model, which is optimized for each downstream task.

The equivariant adapter block. In detail, each equivariant adapter block is responsible for processing the control signal and fusing it into the score produced by the pretrained model, whose parameters are always freezed at finetuning stage. Each adapter block consists of, in a sequential manner, the coupling operator f , a trainable copy of the corresponding layers in pretrained model ϵ θ ′ , the decoupling operator g , followed by an equivariant zero-convolution layer.

Specifically, the composition g ◦ ϵ θ ′ ◦ f , as depicted in § 4.2, functions altogether as a conditional score network ϵ θ ′ ( G τ , τ, C ) that captures the bias of the control signal on the original score ϵ θ ( G τ , τ ) while ensuring the SE(3)-equivariance of the conditional score. Moreover, ϵ θ ′ can be initialized as a subset of the layers in the pretrained model ϵ θ , thus reducing the total number of tunable parameters compared with SFT. While the selection strategy can be arbitrary, empirically we have found that selecting the first layer for every K consecutive layers from the pretrained model performs more favorably compared with naive choices such as the initial or last several layers, under the same parameter budget ( c.f. , § 5.4).

Equivariant zero-convolution. While the equivariant adapter block offers an parameter-efficient way of modeling the conditional score, its non-zero initialization introduces additional noise when it is added to the original score ϵ θ , leading to instability at the beginning of the finetuning stage. To alleviate such issue, we borrow insight from the zero-convolution module proposed in [44] that acts as a safeguard of the original score by zeroing out the conditional score at initialization without blocking the gradient update.

For any ( { x i } , { h i } ) , equivariant zero-convolution is given by

<!-- formula-not-decoded -->

where ¯ x = 1 N ∑ N i =1 x i is the center-of-mass of the input graph, and ϕ x ∈ R , ϕ h ∈ R H are learnable parameters initialized all as zero. By such design, we guarantee that each equivariant adapter block, when equipped with equivariant zero-convolution, yields a rotation-equivariant and translation-invariant output, hence the SE(3)-equivariance of the conditional score after adding the output to the original score. Furthermore, the output of equivariant adapter block will always be zero at initialization, which does not affect the original score, thus enabling smooth and noiseless optimization when tuning the adapter.

## 4.4 Overall Framework

The overall framework of our adapter is depicted in Fig. 2. In general, our adapter is comprised of B equivariant adapter blocks, where each block is a sequential stack of the coupling operator f , the trainable copy of one layer of the denoiser, the decoupling operator g , and a zero-convolution module. At finetuning stage, all parameters in the original denoiser are freezed while the trainable copies and coefficients in zero-convolution are updated through gradient coming from minimizing the denoising loss

<!-- formula-not-decoded -->

where D is the downstream dataset, G τ is the noised graph drawn from q ( G τ |G 0 ) , and s θ ′ ,ϕ refers to our proposed equivariant diffusion adapter. At inference time, we use ϵ θ ( G τ , τ ) + s θ ′ ,ϕ ( G τ , τ, C ) as the conditional score when computing the reverse transition kernel p ( G τ -1 |G τ , C ) .

Our GeoAda offers several key advantages over standard supervised fine-tuning (SFT): 1. The adapter modules are lightweight and operate as plug-and-play components, allowing flexible conditioning on new control signals without architectural modifications to the pretrained model. 2. Parameter-efficient, as only a subset of trainable adapter modules are introduced. 3. By freezing the Pretrained model and only optimizing lightweight adapters, the method imposes implicit regularization, helping to prevent overfitting. 4. Through the careful design of SE(3)-equivariant adapter blocks and zero convolutions, GeoAda guarantees equivariance throughout the tuning process, thereby retaining the theoretical benefits of geometric diffusion models.

## 5 Experiment

We evaluate GeoAda across three categories of additional fine-tuning controls: (1) Frame control during dynamic prediction (§ 5.1), (2) Global type control in human motion prediction(§ 5.2), and (3) Subgraph control in molecule generation (§ 5.3). We also performed ablation studies on core design choices and present some observations in §5.4.

Baselines. We compare with three types of baselines: (1) Fine-tuning methods, including Full FT , which fine-tunes the entire model, and PARTIALk [12, 16, 45], which updates only the last k layers of the pre-trained model; (2) Prompt-based methods, including GPF , GPF-plus [6], which both inject learnable prompt features into the input space. And Prompt-Template maps new inputs to pre-training-style inputs using manually designed graph templates, specifically for the conditional case. (3) Head-only tuning methods, where MLPk freezes the pre-trained model and uses a k -layer MLP as the prediction head. To preserve equivariance, we replace the MLP with an EGTN block in our implementation. More details can be found in App. 9.2.

Implementation. The input data are processed as geometric graphs. For both trajectory and global control settings, we follow the same experimental setup as GeoTDM [11], adopting EGTN as the backbone model with three GeoAda blocks and a hidden dimension of 128. For subgraph control, the base model follows the configuration of TargetDiff [8]. We use T = 1000 and linear noise schedule [13]. More details in App. 9.1.

## 5.1 Frame Control

Task setup. For pre-training, we use the first 10 frames as the condition and train the model to predict the trajectory over the following 20 frames. In the downstream task, we adopt a different setup where the model observes 15 conditional frames and predicts the next 20 frames. We evaluate all models on both the original task and the new task to assess their generalization and adaptability across different settings.

Metrics. For conditional trajectory generation, we employ Average Discrepancy Error (ADE) and Final Discrepancy Error (FDE), which are widely adopted for trajectory forecasting [43, 40], given by ADE ( x [ T ] , y [ T ] ) = 1 TN ∑ T -1 t =0 ∑ N -1 i =0 ∥ x ( t ) i -y ( t ) i ∥ 2 , and FDE ( x [ T ] , y [ T ] ) = 1 N ∑ N -1 i =0 ∥ x ( T -1) i -y ( T -1) i ∥ 2 . For probabilistic models, we report average ADE and FDE derived from K = 5 samples. For unconditional trajectory generation, we report three complementary scores: The Marginal score measures statistical alignment by computing the mean absolute error (MAE) between binned distributions of model-generated and ground-truth coordinates (or bond lengths for MD17). The Classification score is the cross-entropy of a binary classifier trained to distinguish generated trajectories from real ones, offering insight into sample realism. The Prediction score measures the mean squared error (MSE) of a sequence model trained on generated data and tested on real trajectories, reflecting the utility of generated samples for downstream prediction. For more detailed metric definitions, please refer to Appendix 9.5.

## 5.1.1 Particle Dynamic

Experimental Setup. We adopt the CHARGED PARTICLES dataset [17, 28] for particle dynamics simulation. In this dataset, N = 5 particles with randomly assigned charges of either +1 or -1 interact via Coulomb forces, resulting in complex, non-linear trajectories. We use 3000 trajectories for training, 2000 for validation, and 2000 for testing. We explore two settings: (1) Conditional trajectory generation: we use the first 10 frames of each trajectory as input to predict the subsequent 20 frames during pretraining, and 15 frames as input during finetuning to predict the next 20 frames. (2) Unconditional trajectory generation: we generate trajectories of length 20 from scratch during pretraining. During finetuning, we condition on the first 10 frames and predict the next 20 frames.

Table 1: Comparisons on CHARGED PARTICLES dataset.(all results reported by × 10 -1 ).( ↑ ) / ( ↓ ) denotes whether a larger / smaller number is preferred. "NaN" denotes generation collapse due to numerical instability, typically observed in baseline models after fine-tuning on original task. "-"indicates that the baseline PromptTem requires explicit conditioning and cannot be applied when no conditioning frame is given.

| Setting                                                                        | Uncondition                                                                                           | Uncondition                                                                                           | Uncondition                                                                                                       | Uncondition                                                      | Uncondition                                                       | Condition                                                                                                                        | Condition                                                                                                                        | Condition                                                                        | Condition                                                                        |
|--------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|-------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Task                                                                           | Downstream                                                                                            | Downstream                                                                                            | Pretrain                                                                                                          | Pretrain                                                         | Pretrain                                                          | Downstream                                                                                                                       | Downstream                                                                                                                       | Pretrain                                                                         | Pretrain                                                                         |
| Metric                                                                         | ADE( ↓ )                                                                                              | FDE( ↓ )                                                                                              | Marg( ↓ )                                                                                                         | Class( ↑ )                                                       | Pred( ↓ )                                                         | ADE( ↓ )                                                                                                                         | FDE( ↓ )                                                                                                                         | ADE( ↓ )                                                                         | FDE( ↓ )                                                                         |
| Pretrain Full FT Prompt-Tem PARTIAL- k [12] MLP- k GPF [6] GPF-plus [6] GeoAda | nan 1.093 ± 0 . 014 - 1.685 ± 0 . 006 6.258 ± 0 . 947 1.643 ± 0 . 014 1.596 ± 0 . 011 1.119 ± 0 . 019 | nan 2.676 ± 0 . 024 - 3.594 ± 0 . 040 9.111 ± 2 . 628 3.671 ± 0 . 029 3.574 ± 0 . 024 2.669 ± 0 . 022 | 0.079 ± 0 . 000 1.025 ± 0 . 000 - 1.016 ± 0 . 000 1.015 ± 0 . 000 1.027 ± 0 . 000 1.023 ± 0 . 000 0.079 ± 0 . 000 | 5.149 ± 0 . 285 nan - nan 0.00 ± 0 . 000 nan nan 5.134 ± 0 . 247 | 0.109 ± 0 . 004 nan - nan 5.740 ± 2 . 914 nan nan 0.111 ± 0 . 006 | 11.826 ± 0 . 133 1.106 ± 0 . 007 1.723 ± 0 . 014 1.409 ± 0 . 009 1.503 ± 0 . 016 1.575 ± 0 . 017 1.648 ± 0 . 009 1.105 ± 0 . 012 | 20.395 ± 0 . 249 2.590 ± 0 . 040 3.703 ± 0 . 061 3.330 ± 0 . 042 3.338 ± 0 . 039 3.390 ± 0 . 050 3.670 ± 0 . 030 2.621 ± 0 . 033 | 1.177 ± 0 . 018 5.998 ± 0 . 041 nan 9.325 ± 0 . 064 3924 nan nan 1.175 ± 0 . 033 | 2.815 ± 0 . 037 11.75 ± 0 . 107 nan 11.94 ± 0 . 149 3950 nan nan 2.806 ± 0 . 033 |

Results. We present the results in Table 1, with the following observations. Under both the unconditional and conditional trajectory generation settings, GeoAda achieves comparable or better performance than Full FT on the downstream task (conditioning on the first 10 frames to predict the next 20), while only tuning half the number of parameters. Furthermore, it consistently outperforms other fine-tuning and prompt-based baselines, achieving 35.69% improvement on ADE and 21.29% on FDE. On the original pretraining task, all baselines exhibit substantial performance degradation, with some failing to generate valid diffusion samples, indicating that these methods suffer from overfitting and catastrophic forgetting due to excessive adaptation to the downstream task."In contrast, by leveraging equivariant zero convolutions, GeoAda retains the pretrained model's performance.

## 5.1.2 Molecular Dynamics

Experimental setup. We employ the MD17 [3] dataset, which contains the DFT-simulated molecular dynamics trajectories of 8 small molecules, with the number of atoms for each molecule ranging from 9 (Ethanol and Malonaldehyde) to 21 (Aspirin). For each molecule, 5000 trajectories are used for training and 1000/1000 for validation and testing, uniformly sampled along the time dimension. Different from [40], we explicitly involve the hydrogen atoms which contribute most to the vibrations of the trajectory, leading to a more challenging task. The node feature is the one-hot encodings of atomic number [29] and edges are connected between atoms within three hops measured in atomic bonds [30].

Figure 3: Visualization results of GeoAda on Malonaldehyde and Naphthalene from MD17 dataset.

<!-- image -->

Results. As shown in Table 2, GeoAda achieves state-of-the-art performance across all five molecular systems in the MD17 dataset, indicating strong transferability in geometric diffusion models. In downstream fine-tuning task, the method consistently matches or exceeds the performance of full fine-tuning and outperforms other prior methods by an average of 18.94% in ADE and 18.22% in FDE. Importantly, when returning to the original pretraining task, it retains performance comparable to the pretrained model, while both fine-tuning and prompt-based methods exhibit significant degradation or collapse due to overfitting. More experiment results on Malonaldehyde and Naphthalene in App. 10.1.

Table 2: Comparisons for Molecular Dynamics prediction on MD17 dataset (all results reported by × 10 -1 ). The best results are highlighted in bold. Results averaged over 5 runs. "NaN" denotes generation collapse due to numerical instability, typically observed in baseline models after fine-tuning on the original task.

| Scenarios                                                                      | Aspirin                                                                                                                         | Aspirin                                                                                                                         | Aspirin                                                                                                                              | Aspirin                                                                                                                              | Benzene                                                                                                                         | Benzene                                                                                                                         | Benzene                                                                              | Benzene                                                                              |                                                                                                                                                                                                                 | Ethanol                                                                                                      |                                                                                                              |
|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| Task                                                                           | Downstream                                                                                                                      | Downstream                                                                                                                      | Pretrain                                                                                                                             | Pretrain                                                                                                                             | Downstream                                                                                                                      | Downstream                                                                                                                      | Pretrain                                                                             | Pretrain                                                                             | FT                                                                                                                                                                                                              | Pretrain                                                                                                     | Pretrain                                                                                                     |
| Metric                                                                         | ADE                                                                                                                             | FDE                                                                                                                             | ADE                                                                                                                                  | FDE                                                                                                                                  | ADE                                                                                                                             | FDE                                                                                                                             | ADE                                                                                  | FDE                                                                                  | ADE FDE                                                                                                                                                                                                         | ADE                                                                                                          | FDE                                                                                                          |
| Pretrain Full FT PARTIAL- k [12] MLP- k Prompt-Tem GPF [6] GPF-plus [6] GeoAda | 3.782 ± 0 . 010 0.929 ± 0 . 002 1.071 ± 0 . 003 1.132 ± 0 . 004 1.197 ± 0 . 008 1.130 ± 0 . 006 1.014 ± 0 . 006 0.891 ± 0 . 003 | 7.345 ± 0 . 016 1.602 ± 0 . 005 1.875 ± 0 . 009 1.921 ± 0 . 004 2.014 ± 0 . 030 1.909 ± 0 . 024 1.962 ± 0 . 021 1.533 ± 0 . 008 | 1.062 ± 0 . 002 1.323 ± 0 . 003 1.439 ± 0 . 003 1.579 ± 0 . 005 1.571 ± 0 . 010 3.260 ± 0 . 015 /inf 2.243 ± 0 . 009 1.060 ± 0 . 003 | 1.857 ± 0 . 013 2.280 ± 0 . 016 2.407 ± 0 . 007 2.641 ± 0 . 007 2.668 ± 0 . 024 4.272 ± 0 . 025 /inf 3.349 ± 0 . 018 1.852 ± 0 . 012 | 0.603 ± 0 . 000 0.217 ± 0 . 001 0.249 ± 0 . 005 0.248 ± 0 . 012 0.241 ± 0 . 012 0.246 ± 0 . 001 0.234 ± 0 . 010 0.191 ± 0 . 000 | 1.325 ± 0 . 008 0.360 ± 0 . 002 0.407 ± 0 . 003 0.412 ± 0 . 004 0.409 ± 0 . 021 0.415 ± 0 . 004 0.331 ± 0 . 057 0.319 ± 0 . 002 | 0.241 ± 0 . 000 nan 0.318 ± 0 . 110 /nan nan nan nan 1.118 ± 0 . 006 0.240 ± 0 . 002 | 0.393 ± 0 . 002 nan 0.491 ± 0 . 001 /nan nan nan nan 1.083 ± 0 . 010 0.394 ± 0 . 005 | ± 0 . 010 4.357 ± 0 . 014 ± 0 . 007 1.906 ± 0 . 002 ± 0 . 008 2.161 ± 0 . 018 ± 0 . 001 2.275 ± 0 . 021 ± 0 . 018 2.194 ± 0 . 081 ± 0 . 023 2.233 ± 0 . 059 ± 0 . 025 2.048 ± 0 . 046 ± 0 . 007 1.745 ± 0 . 010 | 0.999 ± 0 . 009 inf 6.502 ± 4 . 574 2.228 ± 0 . 004 nan 2.420 ± 0 . 156 3.240 ± 0 . 081 /inf 0.995 ± 0 . 005 | 1.856 ± 0 . 032 inf 4.636 ± 2 . 423 2.716 ± 0 . 022 nan 3.519 ± 0 . 056 4.074 ± 0 . 171 /inf 1.867 ± 0 . 019 |

## 5.2 Global Type Control

Experimental setup. The CMU Mocap dataset is a commonly used dataset for human pose prediction, which includes 8 action categories. A single pose has 38 body joints in the original dataset, among which we choose 25 joints following the configuration of MSR-GCN [4], using 10 frames as input to predict the subsequent 25 frames. For pre-training, we construct a dataset by combining the three most frequent actions: directing traffic, washing windows, and giving basketball signals. The remaining five actions are used as the downstram task fine-tuning dataset.

Results We report short-term and long-term motion prediction results on the CMU Mocap dataset in Tables 3 and 4. More results on jumping and soccer scenarios are in App. 10.2. GeoAda consistently achieves state-of-the-art performance across all action categories and time horizons. In this setting, the pretraining dataset is significantly larger than the downstream task dataset (30k vs. 245-1345 datapoints). As a result, naïve fine-tuning and promptbased methods are highly prone to overfitting to the limited downstream training data, leading to notably degraded performance. Moreover, they are more likely to fail to generate valid samples on the original

Figure 4: Visualization of Running trajectory

<!-- image -->

pretraining task, indicating a severe loss of pretrained knowledge and catastrophic forgetting. In contrast, GeoAda benefits from the implicit regularization effect of the adapter, which mitigates overfitting and preserves the performance of the pretrained model.

Table 3: Comparisons for short-term prediction on 5 action categories of the CMU Mocap dataset. The best results are highlighted in bold. Results averaged over 5 runs (std in App. 10.2).

| scenarios        |   running |   running |   running |   running |   pretrain |   pretrain |   pretrain |   pretrain |   walking |   walking |   walking |   pretrain |   pretrain | pretrain    | pretrain    | basketball   | basketball   | basketball   | basketball   | pretrain    | pretrain   |   pretrain |   pretrain |
|------------------|-----------|-----------|-----------|-----------|------------|------------|------------|------------|-----------|-----------|-----------|------------|------------|-------------|-------------|--------------|--------------|--------------|--------------|-------------|------------|------------|------------|
| millisecond (ms) |     80    |    160    |    320    |    400    |     80     |     160    |     320    |     400    |     80    |    160    |    320    |     400    |     80     | 160         | 320         | 400 80       | 160          | 320          | 400 80       |             | 160        |     320    |     400    |
| Pretrain         |     28.74 |     57.99 |    126.2  |    159.37 |      7.941 |      16.84 |      39.91 |      52.45 |     15.49 |     30.66 |     68.71 |      89.23 |      7.941 | 16.84 39.91 | 52.45       |              | 17.92        | 35.89 78.48  | 101.12       | 7.941       | 16.84      |      39.91 |      52.45 |
| Full FT          |     20.34 |     35.26 |     60.58 |     70.2  |    nan     |     nan    |     nan    |     nan    |     10.01 |     15.12 |     23.98 |      28.49 |    nan     | nan         | nan nan     | 16.95        | 30.33        | 58.28        | 71.93        | nan         | nan        |     nan    |     nan    |
| PARTIAL- k [12]  |     20.47 |     36.04 |     68.15 |     82.59 |    nan     |     nan    |     nan    |     nan    |     10.47 |     16.97 |     30.97 |      37.69 |     17.26  | 31.75       | 66.29 84.27 | 17.54        | 32.03        | 62.96        | 78.95        | nan         | nan        |     nan    |     nan    |
| MLP- k           |     23.35 |     44.44 |     84.27 |    101.42 |    nan     |     nan    |     nan    |     nan    |     10.86 |     18.74 |     35.78 |      44.86 |    nan     | nan         | nan nan     |              | 17.60 34.76  | 72.90        | 86.34        | nan         | nan        |     nan    |     nan    |
| GPF [6]          |     18.48 |     31.94 |     58.8  |     73.56 |     21.38  |      35.44 |      65.23 |      79.55 |     10.79 |     16.79 |     28.32 |      33.48 |     22.04  | 37.10       | 71.24 88.23 |              | 18.48 31.94  | 58.80        | 73.56        | 21.38       | 35.44      |      65.23 |      79.55 |
| GPF-plus [6]     |     19.11 |     31.93 |     49.85 |     56.67 |    nan     |     nan    |     nan    |     nan    |     10.17 |     16.23 |     26.29 |      31.54 |    nan     | nan         | nan nan     | 17.17        | 31.86        | 58.48        | 72.71 nan    |             | nan        |     nan    |     nan    |
| GeoAda           |     18.7  |     33.56 |     50.26 |     55.54 |      7.972 |      16.91 |      40.06 |      52.61 |      8.92 |     13.82 |     22.99 |      26.68 |      7.932 | 16.90 38.96 | 52.54       | 16.85        | 29.71        | 57.59        | 71.19        | 7.898 16.79 |            |      39.7  |      52.5  |

Table 4: Comparisons for long-term prediction on 5 action categories of the CMU Mocap dataset. The best results are highlighted in bold. Results averaged over 5 runs (std in App. 10.2).

| scenarios        |   running |   running |   pretrain |   pretrain |   walking |   walking |   pretrain |   pretrain |   basketball |   basketball |   pretrain |   pretrain |
|------------------|-----------|-----------|------------|------------|-----------|-----------|------------|------------|--------------|--------------|------------|------------|
| millisecond (ms) |    560    |   1000    |     560    |    1000    |    560    |   1000    |     560    |    1000    |       560    |      1000    |     560    |    1000    |
| Pretrain         |    219.16 |    314.85 |      77.06 |     130.51 |    129.43 |    212.94 |      77.06 |     130.51 |       143.49 |       223.99 |      77.06 |     130.51 |
| Full FT          |     85.14 |     97.02 |     nan    |     nan    |     36.92 |     52.58 |     nan    |     nan    |        94.59 |       132.34 |     nan    |     nan    |
| PARTIAL- k [12]  |    102.85 |    108.47 |     nan    |     nan    |     51.36 |     84.72 |     118.82 |     182.88 |       106.84 |       146.27 |     nan    |     nan    |
| MLP- k           |    127.67 |    131.59 |     nan    |     nan    |     62.97 |    102.34 |     nan    |     nan    |       107.3  |       149.58 |     nan    |     nan    |
| GPF [6]          |     61.92 |     71.42 |     nan    |     nan    |     42.37 |     52.24 |     119.43 |     171.74 |        97.16 |       128.29 |     nan    |     nan    |
| GPF-plus [6]     |     63.56 |     71.6  |     nan    |     nan    |     41.31 |     56.47 |     nan    |     nan    |       104.54 |       130.76 |     104.51 |     155.02 |
| GeoAda           |     60.88 |     70.22 |      77.22 |     130.17 |     34.52 |     50.49 |      78.12 |     129.97 |        91.03 |       120.35 |      76.94 |     129.81 |

## 5.3 Subgraph Control

Experimental setup. We adopt the QM9 [26] and GEOM-Drugs [1] dataset for pretraining a model for molecule generation, use the CrossDocked2020 dataset [7] for finetuning protein-ligand pair generation. QM9 [26] contains 130k small molecules with atom types (H, C, N, O, F). GEOM-Drugs [1]

Table 5: Summary of binding affinity and molecular properties of reference molecules and molecules generated by GeoAda and baselines. ( ↑ ) / ( ↓ ) denotes whether a larger / smaller number is preferred.

| Methods         | Vina Score ( ↓ )   | Vina Score ( ↓ )   | Vina Min ( ↓ )   | Vina Min ( ↓ )   | Vina Dock ( ↓ )   | Vina Dock ( ↓ )   | High Affinity( ↑ )   | High Affinity( ↑ )   | QED( ↑ )   | QED( ↑ )   | SA( ↑ )   | SA( ↑ )   | Diversity( ↑ )   | Diversity( ↑ )   |
|-----------------|--------------------|--------------------|------------------|------------------|-------------------|-------------------|----------------------|----------------------|------------|------------|-----------|-----------|------------------|------------------|
| Methods         | Avg.               | Med.               | Avg.             | Med.             | Avg.              | Med.              | Avg.                 | Med.                 | Avg.       | Med.       | Avg.      | Med.      | Avg.             | Med.             |
| liGAN [25]      | -                  | -                  | -                | -                | -6.33             | -6.20             | 21.1%                | 11.1%                | 0.39       | 0.39       | 0.59      | 0.57      | 0.66             | 0.67             |
| GraphBP [19]    | -                  | -                  | -                | -                | -4.80             | -4.70             | 14.2%                | 6.7%                 | 0.43       | 0.45       | 0.49      | 0.48      | 0.79             | 0.78             |
| AR [21]         | -5.75              | -5.64              | -6.18            | -5.88            | -6.75             | -6.62             | 37.9%                | 31.0%                | 0.51       | 0.50       | 0.63      | 0.63      | 0.70             | 0.70             |
| Pocket2Mol [24] | -5.14              | -4.70              | -6.42            | -5.82            | -7.15             | -6.79             | 48.4%                | 51.0%                | 0.56       | 0.57       | 0.74      | 0.75      | 0.69             | 0.71             |
| TargetDiff [10] | -5.47              | -6.30              | -6.64            | -6.83            | -7.80             | -7.91             | 58.1%                | 59.1%                | 0.48       | 0.48       | 0.58      | 0.58      | 0.72             | 0.71             |
| GeoAda (qm9)    | -5.54              | -6.31              | -6.64            | -6.46            | -7.62             | -7.64             | 57.4%                | 58.2%                | 0.49       | 0.51       | 0.58      | 0.58      | 0.74             | 0.75             |
| GeoAda (Geom)   | -5.54              | -6.01              | -6.68            | -6.32            | -7.64             | -7.71             | 58.3%                | 59.3%                | 0.48       | 0.50       | 0.58      | 0.58      | 0.76             | 0.75             |
| Reference       | -6.36              | -6.41              | -6.71            | -6.49            | -7.45             | -7.26             | -                    | -                    | 0.48       | 0.47       | 0.73      | 0.74      | -                | -                |

Table 6: Jensen-Shannon divergence of bond distance distributions between reference and generated molecules. ( ↓ )

Bond

Pocket2Mol

TargetDiff

GeoAda (qm9)

C

-

C

C

=

C

-

C

C

C

N

=

-

C

=

C

N

O

O

:

C

C

Table 7: Percentage of different ring sizes for reference and model generated molecules.

|   Ring Size | Ref.   | liGAN   | AR    | Pocket2Mol   | TargetDiff   | GeoAda (qm9)   | GeoAda (geom)   |
|-------------|--------|---------|-------|--------------|--------------|----------------|-----------------|
|           3 | 1.7%   | 28.1%   | 29.9% | 0.1%         | 0.0%         | 0.0%           | 0.0%            |
|           4 | 0.0%   | 15.7%   | 0.0%  | 0.0%         | 2.8%         | 6.7%           | 5.8%            |
|           5 | 30.2%  | 29.8%   | 16.0% | 16.4%        | 30.8%        | 47.2%          | 45.8%           |
|           6 | 67.4%  | 22.7%   | 51.2% | 80.4%        | 50.7%        | 69.1%          | 78.2%           |
|           7 | 0.7%   | 2.6%    | 1.7%  | 2.6%         | 12.1%        | 23.5           | 21.3%           |
|           8 | 0.0%   | 0.8%    | 0.7%  | 0.3%         | 2.7%         | 5.3%           | 4.7%            |
|           9 | 0.0%   | 0.3%    | 0.5%  | 0.1%         | 0.9%         | 3.8%           | 1.6%            |

:

N

liGAN

0.601

0.665

0.634

0.749

0.656

0.661

0.497

0.638

AR

0.609

0.620

0.474

0.635

0.492

0.558

0.451

0.552

0.496

0.561

0.416

0.629

0.454

0.516

0.416

0.487

0.369

0.505

0.363

0.550

0.421

0.461

0.263

0.235

0.243

0.377

0.363

0.300

0.418

0.279

0.305

0.297

GeoAda (Geom)

0.269

0.393

0.396

0.299

0.428

0.257

0.335

0.330

is a large-scale dataset of 430k molecular conformers with heavy atoms, and we keep the lowest energy conformation for each molecule. Following the common setup for CrossDocked2020 [10], we obtain 100k complexes for training and 100 novel complexes for testing. Since CrossDocked2020 has different atom type configuration from QM9 and GEOM-Drugs, we limit the atom type to (H, C, N, O, F, P, S, Cl). Following [10], proteins and ligands are expressed as 3D atom coordinates and one-hot vectors containing the atom types.

Implementation. Following prior work [10], we use the Adam optimizer with a learning rate of 0.001 and β values of (0.95, 0.999). Batch size is set to 4 and gradient clipping set to 8. To balance the atom type and position losses, we scale the atom type loss by a factor of λ = 100 .

Results. We evaluate molecular properties and molecular structures of the proposed model and baselines on target-aware molecule generation in Table 5, 6, and 7. Baseline models are trained on CrossDocked2020 under explicit protein conditioning. GeoAda matches, and in multiple cases surpasses the strongest baselines on all metrics, generating ligand molecules that maintain realistic structures, high binding affinity, comparable drug-likeness and sythetic accessibility. The lightweight adapter can inject subgraph (pocket) control into a broadly pretrained geometric diffusion model, achieving or surpassing task-specific baselines that rely on end-to-end training with protein context.

## 5.4 Ablations and Observations

Observation of the sudden convergence phenomenon Similar to the phenomenon observed in ControlNet [44], we also observe a sudden convergence phenomenon in our training process. As shown in Figure 5, between step 4500 and 4700, both training loss and validation MSE drop abruptly rather than gradually. To investigate this behavior, we conducted inference using the saved checkpoints from steps 3600 and 5600, and observed a notable performance jump between steps 4400 and 4800, which corresponds to significant reductions in ADE and FDE by 68.3% and 73.4%.

Figure 5: The sudden convergence phenomenon

<!-- image -->

Parameter efficiency analysis As shown in Appendix 10.3.1, we explore the impact of varying the number of equivariant zero layers. Increasing the number of trainable copy layers generally improves

performance, but introduces more parameters and computational cost, revealing a trade-off between performance and efficiency. We also reported the number of tunable parameters for different tuning strategies in Table 21. Except for full fine-tuning, which is substantially larger, all other methods, including GeoAda, use comparable parameter sizes.

Ablation on Equivariant Zero Convolutions We evaluate two variants to assess the role of equivariant zero convolution: (1) replacing it with Gaussian-initialized standard convolutions, and (2) replacing each trainable copy with a single convolution layer (see App. 10.3.2). Both modifications result in notable performance drops, underscoring the importance of zero initialization and structural design for stable and effective fine-tuning.

## 6 Conclusion

We present GeoAda, a parameter-efficient and SE(3)-equivariant adapter framework for geometric diffusion models. It enables effective adaptation to diverse geometric control tasks without modifying the pretrained backbone, preserving both performance and geometric consistency.

## 7 Acknlowlegdements

This project was funded in part by ARO (W911NF-21-1-0125), ONR (N00014-23-1-2159), and the CZ Biohub.

## References

- [1] Simon Axelrod and Rafael Gomez-Bombarelli. Geom, energy-annotated molecular conformations for property prediction and molecular generation. Scientific Data , 9(1):185, 2022. 8
- [2] Yen-Chi Cheng, Hsin-Ying Lee, Sergey Tulyakov, Alexander Schwing, and Liangyan Gui. Sdfusion: Multimodal 3d shape completion, reconstruction, and generation, 2023. 3
- [3] Stefan Chmiela, Alexandre Tkatchenko, Huziel E Sauceda, Igor Poltavsky, Kristof T Schütt, and Klaus-Robert Müller. Machine learning of accurate energy-conserving molecular force fields. Science advances , 3(5):e1603015, 2017. 1, 7
- [4] Lingwei Dang, Yongwei Nie, Chengjiang Long, Qing Zhang, and Guiqing Li. Msr-gcn: Multiscale residual graph convolution networks for human motion prediction. In Proceedings of the IEEE/CVF international conference on computer vision , pages 11467-11476, 2021. 8
- [5] Allan dos Santos Costa, Ilan Mitnikov, Franco Pellegrini, Ameya Daigavane, Mario Geiger, Zhonglin Cao, Karsten Kreis, Tess Smidt, Emine Kucukbenli, and Joseph Jacobson. Equijump: Protein dynamics simulation via so(3)-equivariant stochastic interpolants, 2024. 3
- [6] Taoran Fang, Yunchao Zhang, Yang Yang, Chunping Wang, and Lei Chen. Universal prompt tuning for graph neural networks. Advances in Neural Information Processing Systems , 36:5246452489, 2023. 3, 6, 7, 8, 24, 25, 26
- [7] Paul G Francoeur, Tomohide Masuda, Jocelyn Sunseri, Andrew Jia, Richard B Iovanisci, Ian Snyder, and David R Koes. Three-dimensional convolutional neural networks and a crossdocked data set for structure-based drug design. Journal of chemical information and modeling , 60(9):4200-4215, 2020. 8
- [8] Jiaqi Guan, Wesley Wei Qian, Xingang Peng, Yufeng Su, Jian Peng, and Jianzhu Ma. 3d equivariant diffusion for target-aware molecule generation and affinity prediction. In The Eleventh International Conference on Learning Representations , 2023. 1, 4, 6
- [9] Jiaqi Guan, Wesley Wei Qian, Xingang Peng, Yufeng Su, Jian Peng, and Jianzhu Ma. 3d equivariant diffusion for target-aware molecule generation and affinity prediction, 2023. 2

- [10] Jiaqi Guan, Wesley Wei Qian, Xingang Peng, Yufeng Su, Jian Peng, and Jianzhu Ma. 3d equivariant diffusion for target-aware molecule generation and affinity prediction. arXiv preprint arXiv:2303.03543 , 2023. 9
- [11] Jiaqi Han, Minkai Xu, Aaron Lou, Haotian Ye, and Stefano Ermon. Geometric trajectory diffusion models. arXiv preprint arXiv:2410.13027 , 2024. 1, 3, 6
- [12] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 16000-16009, 2022. 6, 7, 8, 24, 25, 26
- [13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems , 33:6840-6851, 2020. 3, 6, 21
- [14] Emiel Hoogeboom, Vıctor Garcia Satorras, Clément Vignac, and Max Welling. Equivariant diffusion for molecule generation in 3d. In International conference on machine learning , pages 8867-8887. PMLR, 2022. 1, 2, 3, 4
- [15] Zijie Huang, Wanjia Zhao, Jingdong Gao, Ziniu Hu, Xiao Luo, Yadi Cao, Yuanzhou Chen, Yizhou Sun, and Wei Wang. Physics-informed regularization for domain-agnostic dynamical system modeling. arXiv preprint arXiv:2410.06366 , 2024. 1
- [16] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. In European conference on computer vision , pages 709-727. Springer, 2022. 6
- [17] Thomas Kipf, Ethan Fetaya, Kuan-Chieh Wang, Max Welling, and Richard Zemel. Neural relational inference for interacting systems. arXiv preprint arXiv:1802.04687 , 2018. 1, 7
- [18] Shengrui Li, Xueting Han, and Jing Bai. Adaptergnn: Parameter-efficient fine-tuning improves generalization in gnns, 2023. 3
- [19] Meng Liu, Youzhi Luo, Kanji Uchino, Koji Maruhashi, and Shuiwang Ji. Generating 3d molecules for target protein binding. In International Conference on Machine Learning , 2022. 9
- [20] Zemin Liu, Xingtong Yu, Yuan Fang, and Xinming Zhang. Graphprompt: Unifying pre-training and downstream tasks for graph neural networks, 2023. 3
- [21] Shitong Luo, Jiaqi Guan, Jianzhu Ma, and Jian Peng. A 3d generative model for structure-based drug design. Advances in Neural Information Processing Systems , 34, 2021. 9
- [22] Yanchen Luo, Junfeng Fang, Sihang Li, Zhiyuan Liu, Jiancan Wu, An Zhang, Wenjie Du, and Xiang Wang. Text-guided diffusion model for 3d molecule generation. arXiv preprint arXiv:2410.03803 , 2024. 4
- [23] Alex Morehead and Jianlin Cheng. Geometry-complete diffusion for 3d molecule generation and optimization, 2024. 1, 2
- [24] Xingang Peng, Shitong Luo, Jiaqi Guan, Qi Xie, Jian Peng, and Jianzhu Ma. Pocket2mol: Efficient molecular sampling based on 3d protein pockets. arXiv preprint arXiv:2205.07249 , 2022. 9
- [25] Matthew Ragoza, Tomohide Masuda, and David Ryan Koes. Generating 3D molecules conditional on receptor binding sites with deep generative models. Chem Sci , 13:2701-2713, Feb 2022. 9
- [26] Raghunathan Ramakrishnan, Pavlo O Dral, Matthias Rupp, and O Anatole Von Lilienfeld. Quantum chemistry structures and properties of 134 kilo molecules. Scientific data , 1(1):1-7, 2014. 8
- [27] Hyunwoo Ryu, Jiwoo Kim, Hyunseok An, Junwoo Chang, Joohwan Seo, Taehan Kim, Yubin Kim, Chaewon Hwang, Jongeun Choi, and Roberto Horowitz. Diffusion-edfs: Bi-equivariant denoising generative modeling on se(3) for visual robotic manipulation, 2023. 1, 3

- [28] Victor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E(n) equivariant graph neural networks. arXiv preprint arXiv:2102.09844 , 2021. 1, 7
- [29] Kristof Schütt, Oliver Unke, and Michael Gastegger. Equivariant message passing for the prediction of tensorial properties and molecular spectra. In International Conference on Machine Learning , pages 9377-9388. PMLR, 2021. 7
- [30] Chence Shi, Shitong Luo, Minkai Xu, and Jian Tang. Learning gradient fields for molecular conformation generation. In International conference on machine learning , pages 9558-9568. PMLR, 2021. 7
- [31] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems , 32, 2019. 3
- [32] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations , 2021. 3
- [33] Charles Stein. A bound for the error in the normal approximation to the distribution of a sum of dependent random variables. In Proceedings of the sixth Berkeley symposium on mathematical statistics and probability, volume 2: Probability theory , volume 6, pages 583-603. University of California Press, 1972. 3
- [34] Mingchen Sun, Kaixiong Zhou, Xin He, Ying Wang, and Xin Wang. Gppt: Graph pre-training and prompt tuning to generalize graph neural networks. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining , KDD '22, page 1717-1727, New York, NY, USA, 2022. Association for Computing Machinery. 3
- [35] Masatoshi Uehara, Yulai Zhao, Kevin Black, Ehsan Hajiramezanali, Gabriele Scalia, Nathaniel Lee Diamant, Alex M Tseng, Tommaso Biancalani, and Sergey Levine. Fine-tuning of continuous-time diffusion models as entropy-regularized control, 2024. 3
- [36] Julen Urain, Niklas Funk, Jan Peters, and Georgia Chalvatzaki. Se(3)-diffusionfields: Learning smooth cost functions for joint grasp and motion optimization through diffusion, 2023. 1, 3
- [37] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization, 2023. 3
- [38] Zhendong Wang, Yifan Jiang, Yadong Lu, yelong shen, Pengcheng He, Weizhu Chen, Zhangyang Wang, and Mingyuan Zhou. In-context learning unlocked for diffusion models. In Thirty-seventh Conference on Neural Information Processing Systems , 2023. 1, 3
- [39] Can Xu, Haosen Wang, Weigang Wang, Pengfei Zheng, and Hongyang Chen. Geometricfacilitated denoising diffusion model for 3d molecule generation. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pages 338-346, 2024. 1, 2
- [40] Chenxin Xu, Robby T Tan, Yuhong Tan, Siheng Chen, Yu Guang Wang, Xinchao Wang, and Yanfeng Wang. Eqmotion: Equivariant multi-agent motion prediction with invariant interaction reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 1410-1420, 2023. 6, 7
- [41] Minkai Xu, Alexander Powers, Ron Dror, Stefano Ermon, and Jure Leskovec. Geometric latent diffusion models for 3d molecule generation. In International Conference on Machine Learning . PMLR, 2023. 1, 2
- [42] Minkai Xu, Lantao Yu, Yang Song, Chence Shi, Stefano Ermon, and Jian Tang. Geodiff: A geometric diffusion model for molecular conformation generation. In International Conference on Learning Representations , 2022. 1, 2, 3
- [43] Pei Xu, Jean-Bernard Hayet, and Ioannis Karamouzas. Socialvae: Human trajectory prediction using timewise latents. In European Conference on Computer Vision , pages 511-528. Springer, 2022. 6

- [44] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision , pages 3836-3847, 2023. 1, 2, 3, 5, 9
- [45] Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. In Computer Vision-ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part III 14 , pages 649-666. Springer, 2016. 5, 6

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and Introduction clearly state the main contributions, which include the proposal of GeoAda, an SE(3)-equivariant adapter framework for efficient finetuning of geometric diffusion models. The claims cover parameter efficiency, preservation of geometric consistency/equivariance, mitigation of overfitting and catastrophic forgetting, and wide applicability across diverse geometric control types and domains. These claims appear to be supported by the theoretical proofs (discussed in Section 4 and Appendix 7) and experimental results presented (Section 5).

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Discussed in App. 11.

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

Justification: The paper states in the abstract that it theoretically proves the proposed adapters maintain SE(3)-equivariance. Proposition 4.1 (Section 4.2) addresses the equivariance of control encoding. Section 4 (Method) details the design to ensure SE(3)-equivariance. Appendix 7 is explicitly titled "Proof", and Appendix 8.5.4 and 8.5.5 discuss theoretical aspects like Theorem 8.1 and 8.2 with related assumptions.

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

Justification: Section 5 (Experiment) details the experimental setup, including datasets, task setups for pre-training and fine-tuning, evaluation metrics, and baselines. Section 5 also mentions implementation details. Further specifics are provided in Appendix 8, including Appendix 8.2 (Hyper-parameters), Appendix 8.3 (Baselines), and Appendix 8.4 (Details of the datasets).

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

Answer: [No]

Justification: The paper provides detailed descriptions of the experimental setup, datasets, and implementation details in Section 5 and Appendix 8, which support reproducibility. However, it does not explicitly state that the code and data are open access or provide URLs to repositories.

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

Justification: Section 5 (Experiment) outlines experimental setups for different tasks, including dataset descriptions and fine-tuning specifics. Implementation details are mentioned within Section 5. Appendix 8 provides further details, with Appendix 8.2 specifying hyperparameters (Table 5), Appendix 8.1 discussing compute resources, Appendix 8.3 detailing baselines, and Appendix 8.4 giving dataset statistics including splits.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The tables in Section 5 (e.g., Table 1, Table 2, Table 3, Table 4) and Appendix 9 (e.g., Table 8, Table 9, Table 10) report mean results along with what appear to be standard deviations (e.g., "11.826 ±0.133" from Table 1). Captions for these tables often state "Results averaged over 5 runs" and some explicitly mention "(std in App. 9.4)".

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

Justification: Appendix 8.1 ("Compute Resources") specifies the use of "4 Nvidia A6000 GPUs," training times for different datasets ("NBody and ETH-UCY take around 12 hours while each MD17 training phase takes about a day"), and that "CPUs were standard intel CPUs."

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: There is no indication of ethical violations from the research described.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Discussed in Appendix.

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

Justification: The research focuses on geometric diffusion models for scientific applications such as molecular and particle dynamics, and human motion prediction using established datasets. These applications and datasets do not typically fall into the high-risk category for misuse in the same vein as large language models generating text or image generators creating photorealistic fakes of individuals, which would necessitate specific, explicit safeguards mentioned in the paper beyond standard responsible research conduct.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The paper credits the original sources of the datasets used (e.g., MD17, QM9, GEOM-Drugs, CrossDocked2020, CHARGED PARTICLES, CMU Mocap) by citing the relevant publications in Section 5 and the References section. While the specific license for each dataset is not reiterated within this paper's text, citing the original publication is standard academic practice for acknowledging the source and implicitly, adherence to their terms of use.

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

Justification: This paper is not about a new dataset or a standalone software/model asset in the sense that requires separate documentation released alongside an asset. The proposed method itself is documented within the paper (Section 4).

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper utilizes existing datasets, such as the CMU Mocap dataset for human motion studies (Section 5.2). It does not describe any new crowdsourcing efforts or new research involving direct data collection from human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The research relies on pre-existing, publicly available datasets (e.g., CMU Mocap, mentioned in Section 5.2). The paper does not detail new experiments involving human subjects that would necessitate a new IRB approval process to be described within this work.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core methodology of the paper revolves around geometric diffusion models and equivariant adapters (Section 4). There is no mention or indication that Large Language Models (LLMs) are an important, original, or non-standard component of the research methods presented.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Appendix

## 8 Proof

Below is the explanation and proof of Proposition 4.1:

Proof. Since ϵ θ is SE(3)-equivariant by assumption, we have for any h ∈ SE(3) ,

<!-- formula-not-decoded -->

We consider each component:

- The coupling operator f augments G τ with control C in a way that respects the SE(3) structure: global controls modify features invariantly; subgraph controls are merged geometrically; frame controls concatenate along the temporal axis. Thus, f is SE(3)-equivariant.
- The decoupling operator g selects a subset of nodes or frames without altering their coordinates. Therefore, it commutes with SE(3) action: g ( h · G ′′ ) = h · g ( G ′′ ) .

Combining the above, we explicitly see that for any h ∈ SE(3) defined as h ( x ) = R x + d , we have:

<!-- formula-not-decoded -->

Therefore, the composed function g ◦ ϵ θ ◦ f is SE(3)-equivariant.

## 9 More Details on Experiments

## 9.1 Hyper-parameters

We provide the detailed hyper-parameters of GeoAda in Table 8. We adopt Adam optimizer with betas (0 . 9 , 0 . 999) and ϵ = 10 -8 . For all experiments, we use the linear noise schedule per [13] with β start = 0 . 02 and β end = 0 . 0001 .

Table 8: Hyper-parameters of GeoAda in the experiments.

|          |   n_layer |   hidden |   time_emb_dim |    T |   batch_size |   learning_rate |
|----------|-----------|----------|----------------|------|--------------|-----------------|
| N-body   |         6 |      128 |             32 | 1000 |          128 |          0.0001 |
| MD       |         6 |      128 |             32 | 1000 |          128 |          0.0001 |
| CMUMocap |         6 |       64 |             32 |  100 |          128 |          0.0001 |

## 9.2 Baselines

## Full FT.

Full FT fully fine-tunes the pre-trained model f during downstream training. The entire model is updated to fit the target task.

## PARTIALk .

PARTIALk fine-tunes only the last k layers of the model f , while freezing the remaining layers. This method balances adaptability with parameter efficiency by limiting the number of updated layers.

## Graph Prompt Feature (GPF).

In GPF , the pre-trained encoder f is kept frozen, and a learnable prompt vector p is injected into the input feature space. During training, only the prompt vector p and the prediction head θ are optimized. This method enables task adaptation through a lightweight, task-specific prompt without modifying the backbone model. In our implementation, we replace the original MLP head with a three-layer Equivariant Geometric Trajectory Network (EGTN), which ensures the projection head maintains geometric consistency with the model.

## Graph Prompt Feature-Plus (GPF-plus).

Extending GPF, this variant constructs node-wise prompt vectors using k learnable basis vectors p b 1 , . . . , p b k and a set of learnable linear weights a 1 , . . . , a k . These components are used to compute node-specific prompts p i via a compositional mechanism. The model f remains frozen, while prediction head θ , learnable basis vectors p b i , andlearnable linear weights a i are optimized.

## Prompt-Template

We prepend a learnable prompt layer(Equivariant Geometric Trajectory Network) to adapt new inputs to the distribution seen during pretraining, following with prediction head θ .

## MLPk (EGTNk ).

This baseline freezes the entire pre-trained model f and replaces the prediction head with a k -layer multilayer perceptron (MLP). To preserve equivariance in our setting, we replace the MLP with an Equivariant Geometric Trajectory Network (EGTN) block. Only the EGTN-based head is trained during the downstream task.

## 9.3 Details of the datasets

## 9.3.1 Global Type Control

Pretrain Dataset The statistics of the pretrained datasets on Global Type Control are presented in Table. 9.

Table 9: Pretrain Dataset statistics by Global Type.

| Type   |   Washwindow |   Directing Traffic |   Basketball Signal |   Pretrain |
|--------|--------------|---------------------|---------------------|------------|
| train  |        12126 |                9557 |                7776 |      29459 |
| val    |         1342 |                2346 |                1920 |       5588 |
| test   |         1342 |                2346 |                1920 |       5588 |

Downstream datasets The statistics of the downstream datasets utilized for the models pretrained on Global Type Control are presented in Table. 10.

Table 10: Downstream Dataset statistics by Global Type.

| Type   |   Running |   Walking |   Jumping |   Basketball |   Soccer |
|--------|-----------|-----------|-----------|--------------|----------|
| train  |       245 |       869 |      1345 |         1044 |     1210 |
| val    |        47 |       145 |      1008 |          254 |      264 |
| test   |        47 |       145 |      1008 |          254 |      264 |

## 9.4 Model

## 9.4.1 Geometric Trajectory Diffusion Models

Unconditional Generation For unconditional generation, we model the trajectory distribution subject to SE (3) -invariance. The following theorem provides constraints for the prior and transition kernel.

Theorem 9.1. If the prior p T ( x [ T ] T ) is SE (3) -invariant, and the transition kernels p τ -1 ( x [ T ] τ -1 | x [ T ] τ ) , ∀ τ ∈ { 1 , · · · , T } are SE (3) -equivariant, then the marginal p τ ( x [ T ] τ ) at any step τ ∈ { 0 , · · · , T } is also SE (3) -invariant.

Prior in the translation-invariant subspace. The prior is built on a translation-invariant subspace X P ⊂ X , induced by a linear transformation P :

<!-- formula-not-decoded -->

which results in a restricted Gaussian distribution supported only on the subspace, denoted ˜ N ( 0 , I ) , and is isotropic and SO (3) -invariant. To sample, one samples from N ( 0 , I ) and projects it onto the subspace.

Transition kernel. The transition kernel is parameterized in the subspace X P , given by:

<!-- formula-not-decoded -->

where the mean function ˜ µ θ is SO (3) -equivariant. The function is re-parameterized as:

<!-- formula-not-decoded -->

where ˜ ϵ θ = P ◦ f θ is an SO (3) -equivariant adaptation of the proposed EGTN.

Training and inference. The VLB is optimized for training, with the objective:

<!-- formula-not-decoded -->

The inference process involves projecting intermediate samples onto the subspace X P .

Conditional Generation In conditional generation, the target distribution is SE (3) -equivariant with respect to the given frames. The following theorem provides constraints for the prior and transition kernel.

Theorem 9.2. If the prior p T ( x [ T ] T | x [ T c ] c ) is SE (3) -equivariant, and the transition kernels p τ -1 ( x [ T ] τ -1 | x [ T ] τ , x [ T c ] c ) , ∀ τ ∈ { 1 , · · · , T } are SE (3) -equivariant, the marginal p τ ( x [ T ] τ | x [ T c ] c ) , ∀ τ ∈ { 0 , · · · , T } is SE (3) -equivariant.

Flexible equivariant prior. We provide a guideline for distinguishing feasible prior designs. The prior N ( µ ( x [ T c ] c ) , I ) is SE (3) -equivariant if µ ( x [ T c ] c ) is SE (3) -equivariant. The mean function µ ( x [ T c ] c ) serves as an anchor to transition geometric information from the given frames to the target distribution. For instance, the anchor can be defined as:

<!-- formula-not-decoded -->

where the weights satisfy ∑ s ∈ [ T c ] w ( s ) = 1 .

The weights w ( t,s ) are derived as:

<!-- formula-not-decoded -->

where γ ∈ R T are learnable parameters, ensuring the constraint for translation equivariance.

Transition kernel. To match the proposed prior, we modify both the forward and reverse processes. The forward process is defined as:

<!-- formula-not-decoded -->

which ensures that q ( x [ T ] T | x [ T c ] c ) matches the equivariant prior x r (proof in App. ?? ). The reverse transition kernel is:

<!-- formula-not-decoded -->

We adopt the noise prediction objective for the reverse process, rewriting µ θ as:

<!-- formula-not-decoded -->

where the denoising network ϵ θ is implemented as an EGTN with translation invariance, ensuring the translation equivariance of µ θ .

Training and inference. Optimizing the VLB of our diffusion model leads to the following objective:

<!-- formula-not-decoded -->

## 9.5 Evaluation Metrics in the Unconditional Case

All these metrics are evaluated on a set of model samples with the same size as the testing set.

Marginal score is computed as the absolute difference of two empirical probability density functions. Practically, we collect the x, y, z coordinates at each time step marginalized over all nodes in all systems in the predictions and the ground truth (testing set). Then we split the collection into 50 bins and compute the MAE in each bin, finally averaged across all time steps to obtain the score. Note that on MD17, instead of computing the pdf on coordinates, we compute the pdf on the length of the chemical bonds, which is a clearer signal that correlates to the validity of the generated MD trajectory, since during MD simulation the bond lengths are usually stable with very small vibrations. Marginal score gives a broad statistical measurement how each dimension of the generated samples align with the original data.

Classification score is computed as the cross-entropy loss of a sequence classification model that aims to distinguish whether the trajectory is generated by the model or from the testing set. To be specific, we construct a dataset mixed by the generated samples and the testing set, and randomly split it into 80% and 20% subsets for training and testing. Then the model is trained on the training set and the classification score is computed as the cross-entropy on the testing set. We use a 1-layer EqMotion with a classification head as the model. The classification score provided intuition on how difficult it is to distinguish the generated samples and the original data.

Prediction score is computed as the MSE loss of a train-on-synthetic-test-on-real sequence to sequence model. In detail, we train a 1-layer EqMotion on the sampled dataset with the task of predicting the second half of the trajectory given the first half. We then evaluate the model on the testing set and report the MSE as the prediction score. Prediction score provides intuition on the capability of the generative model on generating synthetic data that well aligns with the ground truth.

## 10 More Experiments and Discussions

## 10.1 Molecular

Additional experimental results on the Malonaldehyde and Naphthalene are shown below:

Table 11: Comparisons for Molecular Dynamics prediction on MD17 dataset (all results reported by × 10 -1 ). The best results are highlighted in bold. Results averaged over 5 runs

| Scenarios                                                                 | Malonaldehyde                                                                                                                   | Malonaldehyde                                                                                                                   | Malonaldehyde                                                                                                                        | Malonaldehyde                                                                                                                        | Naphthalene                                                                                                                     | Naphthalene                                                                                                                     | Naphthalene                                                                     | Naphthalene                                                                     |
|---------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| Task                                                                      | Downstream                                                                                                                      | Downstream                                                                                                                      | Pretrain                                                                                                                             | Pretrain                                                                                                                             | Downstream                                                                                                                      | Downstream                                                                                                                      | Pretrain                                                                        | Pretrain                                                                        |
| Metric                                                                    | ADE                                                                                                                             | FDE                                                                                                                             | ADE                                                                                                                                  | FDE                                                                                                                                  | ADE                                                                                                                             | FDE                                                                                                                             | ADE                                                                             | FDE                                                                             |
| Pretrain FT PARTIAL- k [12] MLP- k Prompt-Tem GPF [6] GPF-plus [6] GeoAda | 3.235 ± 0 . 012 0.897 ± 0 . 002 0.981 ± 0 . 004 0.997 ± 0 . 005 1.092 ± 0 . 019 1.176 ± 0 . 012 1.018 ± 0 . 003 0.862 ± 0 . 002 | 5.189 ± 0 . 023 1.511 ± 0 . 009 1.675 ± 0 . 015 1.694 ± 0 . 010 2.003 ± 0 . 056 1.931 ± 0 . 030 1.793 ± 0 . 010 1.414 ± 0 . 014 | 0.962 ± 0 . 007 1.405 ± 0 . 006 1.230 ± 0 . 003 1.291 ± 0 . 004 2.323 ± 0 . 024 282.1 ± 157 . 9 /inf 3.527 ± 0 . 501 0.963 ± 0 . 007 | 1.584 ± 0 . 021 2.237 ± 0 . 023 2.110 ± 0 . 006 2.051 ± 0 . 015 3.351 ± 0 . 081 24.97 ± 24 . 43 /inf 4.719 ± 1 . 397 1.573 ± 0 . 018 | 1.416 ± 0 . 003 0.555 ± 0 . 001 0.653 ± 0 . 002 0.718 ± 0 . 001 0.972 ± 0 . 006 0.758 ± 0 . 002 0.717 ± 0 . 003 0.581 ± 0 . 002 | 2.268 ± 0 . 005 0.867 ± 0 . 010 0.903 ± 0 . 003 0.969 ± 0 . 005 1.593 ± 0 . 021 1.005 ± 0 . 005 0.873 ± 0 . 006 0.822 ± 0 . 004 | 0.714 ± 0 . 002 nan 2.083 ± 0 . 009 nan nan nan 1.891 ± 0 . 056 0.714 ± 0 . 001 | 0.972 ± 0 . 006 nan 1.629 ± 0 . 007 nan nan nan 2.674 ± 0 . 156 0.969 ± 0 . 007 |

## 10.2 Human Motion

Additional experimental results on the jumping and soccer scenarios are presented below. We also report the standard deviations across all experiments.

Table 12: Short-term prediction on running from the CMU Mocap dataset.

| scenarios                                              | running                                                                                                  | running                                                                                                  | running                                                                                                   | running                                                                                                    | pretrain                                                     | pretrain                                                     | pretrain                                                     | pretrain                                                     |
|--------------------------------------------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------|
| millisecond (ms)                                       | 80                                                                                                       | 160                                                                                                      | 320                                                                                                       | 400                                                                                                        | 80                                                           | 160                                                          | 320                                                          | 400                                                          |
| Pretrain Full FT PARTIAL- k MLP- k GPF GPF-plus GeoAda | 28.74 ± 0 . 34 20.34 ± 0 . 32 20.47 ± 0 . 32 23.35 ± 0 . 39 19.17 ± 0 . 28 19.11 ± 0 . 54 18.70 ± 0 . 37 | 57.99 ± 0 . 33 35.26 ± 0 . 19 36.04 ± 0 . 40 44.44 ± 0 . 71 32.85 ± 0 . 66 31.93 ± 0 . 81 33.56 ± 0 . 25 | 126.20 ± 0 . 93 60.58 ± 1 . 14 68.15 ± 0 . 70 84.27 ± 1 . 58 52.83 ± 1 . 03 49.85 ± 1 . 21 50.26 ± 0 . 42 | 159.37 ± 1 . 15 70.20 ± 1 . 36 82.59 ± 0 . 86 101.42 ± 2 . 11 60.90 ± 1 . 54 56.67 ± 1 . 39 55.54 ± 0 . 36 | 7.941 ± 0 . 02 nan nan nan 21.38 ± 0 . 97 nan 7.972 ± 0 . 02 | 16.84 ± 0 . 04 nan nan nan 35.44 ± 1 . 05 nan 16.91 ± 0 . 05 | 39.91 ± 0 . 43 nan nan nan 65.23 ± 1 . 76 nan 40.06 ± 0 . 42 | 52.45 ± 0 . 07 nan nan nan 79.55 ± 1 . 11 nan 52.61 ± 0 . 07 |

Table 13: Short-term prediction on walking from the CMU Mocap dataset.

| scenarios                                              | walking                                                                                                 | walking                                                                                                  | walking                                                                                                  | walking                                                                                                  | pretrain                                                                | pretrain                                                                | pretrain                                                                | pretrain                                                                |
|--------------------------------------------------------|---------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------|
| millisecond (ms)                                       | 80                                                                                                      | 160                                                                                                      | 320                                                                                                      | 400                                                                                                      | 80                                                                      | 160                                                                     | 320                                                                     | 400                                                                     |
| Pretrain Full FT PARTIAL- k MLP- k GPF GPF-plus GeoAda | 15.49 ± 0 . 07 10.01 ± 0 . 17 10.47 ± 0 . 81 10.86 ± 1 . 07 10.79 ± 0 . 14 10.17 ± 0 . 16 8.92 ± 1 . 02 | 30.66 ± 0 . 16 15.12 ± 0 . 14 16.97 ± 0 . 43 18.74 ± 1 . 37 16.79 ± 0 . 16 16.23 ± 0 . 12 13.82 ± 1 . 26 | 68.71 ± 0 . 33 23.98 ± 0 . 13 30.97 ± 0 . 60 35.78 ± 1 . 92 28.32 ± 0 . 21 26.29 ± 0 . 22 22.99 ± 1 . 30 | 89.23 ± 0 . 43 28.49 ± 0 . 19 37.69 ± 0 . 60 44.86 ± 1 . 35 33.48 ± 0 . 22 31.54 ± 0 . 19 26.68 ± 1 . 31 | 7.941 ± 0 . 02 nan 17.26 ± 0 . 23 nan 22.04 ± 0 . 38 nan 7.932 ± 0 . 03 | 16.84 ± 0 . 04 nan 31.75 ± 0 . 36 nan 37.10 ± 0 . 40 nan 16.90 ± 0 . 04 | 39.91 ± 0 . 43 nan 66.29 ± 0 . 44 nan 71.24 ± 0 . 39 nan 38.96 ± 0 . 47 | 52.45 ± 0 . 07 nan 84.27 ± 0 . 79 nan 88.23 ± 0 . 97 nan 52.54 ± 0 . 10 |

Table 14: Short-term prediction comparison on the basketball action from the CMU Mocap dataset.

| scenarios        | basketball     | basketball     | basketball     | basketball      | pretrain       | pretrain       | pretrain       | pretrain       |
|------------------|----------------|----------------|----------------|-----------------|----------------|----------------|----------------|----------------|
| millisecond (ms) | 80             | 160            | 320            | 400             | 80             | 160            | 320            | 400            |
| Pretrain         | 17.92 ± 0 . 06 | 35.89 ± 0 . 12 | 78.48 ± 0 . 47 | 101.12 ± 0 . 69 | 7.941 ± 0 . 02 | 16.84 ± 0 . 04 | 39.91 ± 0 . 43 | 52.45 ± 0 . 07 |
| FT               | 16.95 ± 0 . 11 | 30.33 ± 0 . 17 | 58.28 ± 0 . 41 | 71.93 ± 0 . 54  | -              | -              | -              | -              |
| PARTIAL- k       | 17.54 ± 0 . 12 | 32.03 ± 0 . 43 | 62.96 ± 0 . 86 | 78.95 ± 0 . 98  | -              | -              | -              | -              |
| MLP- k           | 17.60 ± 0 . 50 | 34.76 ± 1 . 26 | 72.90 ± 2 . 36 | 86.34 ± 1 . 53  | -              | -              | -              | -              |
| GPF              | 18.48 ± 0 . 09 | 31.94 ± 0 . 14 | 58.80 ± 0 . 28 | 73.56 ± 0 . 29  | 21.38 ± 0 . 11 | 35.44 ± 0 . 27 | 65.23 ± 0 . 30 | 79.55 ± 0 . 51 |
| GPF-plus         | 17.17 ± 0 . 07 | 31.86 ± 0 . 15 | 58.48 ± 0 . 32 | 72.71 ± 0 . 24  | -              | -              | -              | -              |
| GeoAda           | 16.85 ± 0 . 24 | 29.71 ± 0 . 41 | 57.59 ± 0 . 39 | 71.19 ± 0 . 48  | 7.898 ± 0 . 05 | 16.79 ± 0 . 05 | 39.70 ± 0 . 04 | 52.50 ± 0 . 07 |

Table 15: Short-term prediction comparison on the jumping action from the CMU Mocap dataset.

| scenarios        | jumping        | jumping        | jumping        | jumping         | pretrain       | pretrain       | pretrain       | pretrain        |
|------------------|----------------|----------------|----------------|-----------------|----------------|----------------|----------------|-----------------|
| millisecond (ms) | 80             | 160            | 320            | 400             | 80             | 160            | 320            | 400             |
| Pretrain         | -              | -              | -              | -               | 7.941 ± 0 . 02 | 16.84 ± 0 . 04 | 39.91 ± 0 . 43 | 52.45 ± 0 . 07  |
| FT               | -              | -              | -              | -               | 26.67 ± 0 . 10 | 50.83 ± 0 . 29 | 94.13 ± 0 . 58 | 112.66 ± 0 . 71 |
| PARTIAL- k       | 26.01 ± 0 . 08 | 49.19 ± 0 . 09 | 95.84 ± 0 . 13 | 116.24 ± 0 . 23 | 19.08 ± 0 . 17 | 38.53 ± 0 . 34 | 79.77 ± 0 . 52 | 99.85 ± 1 . 07  |
| MLP- k           | 22.63/nan      | 44.68/nan      | 88.93/nan      | 108.43/nan      | 15.32 ± 0 . 30 | 31.92 ± 0 . 26 | 66.50 ± 0 . 42 | 82.55 ± 0 . 93  |
| GPF              | 28.74 ± 0 . 19 | 51.97 ± 0 . 19 | 97.80 ± 0 . 34 | 117.94 ± 0 . 37 | ± 0 . 08 -     | -              | -              | -               |
| GPF-plus         | -              | -              | -              | -               | -              | -              | -              | -               |
| GeoAda           | 25.91 ± 0 . 09 | 48.83 ± 0 . 83 | 91.51 ± 0 . 07 | 109.24 ± 0 . 60 | 7.956 ± 0 . 03 | 16.82 ± 0 . 04 | 39.55 ± 0 . 47 | 52.57 ± 0 . 09  |

Table 16: Short-term prediction comparison on the soccer action from the CMU Mocap dataset.

| scenarios        | soccer                                                                   | soccer                                                                      | soccer                                                                   | soccer                                                                | pretrain                                                                    | pretrain                          | pretrain                          | pretrain                        |
|------------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------|-----------------------------------------------------------------------------|-----------------------------------|-----------------------------------|---------------------------------|
| millisecond (ms) | 80                                                                       | 160                                                                         | 320                                                                      | 400                                                                   | 80                                                                          | 160                               | 320                               | 400                             |
| Pretrain         | - 17.65 ± 0 . 17 18.83 ± 0 . 10 - 19.28 ± 0 . 10 19.11 ± 0 . 18 ± 0 . 14 | - 31.43 ± 0 . 32.86 ± 0 . 07 - 32.18 ± 0 . 14 32.03 ± 0 . 31 30.03 ± 0 . 12 | - 59.76 ± 0 . 64.58 ± 0 . 37 - 69.58 ± 0 . 42 59.67 ± 0 . 46 53.51 ± 0 . | - 74.30 ± 0 . 81.53 ± 0 . - 73.63 ± 0 . 63 74.25 ± 0 . 64.78 ± 0 . 42 | 7.941 ± 0 . 02 - 14.14 ± 0 . 41 - 15.70 ± 0 . 30 15.22 ± 0 . 23 7.961 ± 0 . | 16.84 ± 0 . 04 - 24.95 ± 0 . 38 - | 39.91 ± 0 . 43 - 50.89 ± 0 . 40 - | 52.45 ± 0 . 07 - 64.24 ± 0 . 51 |
| FT               |                                                                          | 35                                                                          | 44                                                                       | 54                                                                    |                                                                             |                                   |                                   |                                 |
| PARTIAL- k       |                                                                          |                                                                             |                                                                          | 50                                                                    |                                                                             |                                   |                                   |                                 |
| MLP- k           |                                                                          |                                                                             |                                                                          |                                                                       |                                                                             |                                   |                                   | -                               |
| GPF              |                                                                          |                                                                             |                                                                          |                                                                       |                                                                             | 28.31 ± 0 . 28                    | 58.57 ± 0 . 42                    | 74.28 ± 0 . 61                  |
| GPF-plus         |                                                                          |                                                                             |                                                                          | 65                                                                    |                                                                             | 26.22 ± 0 . 29                    | 52.02 ± 0 . 47                    | 65.17 ± 0 . 53                  |
| GeoAda           | 17.04                                                                    |                                                                             | 25                                                                       |                                                                       | 02                                                                          | 16.90 ± 0 . 03                    | 39.46 ± 0 . 41                    | 52.43 ± 0 . 09                  |

Table 17: Long-term prediction on CMU Mocap: Running and Walking .

| scenarios                                                           | running                                                                                                     | running                                                                                                     | pretrain                                          | pretrain                                            | walking                                                                                                   | walking                                                                                                   | pretrain                                                                  | pretrain                                                                    |
|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|---------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| millisecond (ms)                                                    | 560                                                                                                         | 1000                                                                                                        | 560                                               | 1000                                                | 560                                                                                                       | 1000                                                                                                      | 560                                                                       | 1000                                                                        |
| Pretrain Full FT PARTIAL- k [12] MLP- k GPF [6] GPF-plus [6] GeoAda | 219.16 ± 2 . 18 85.14 ± 2 . 36 102.85 ± 1 . 68 127.67 ± 2 . 46 61.92 ± 1 . 02 63.56 ± 1 . 54 60.88 ± 0 . 82 | 314.85 ± 3 . 03 97.02 ± 1 . 45 108.47 ± 2 . 03 131.59 ± 2 . 90 71.42 ± 1 . 27 71.60 ± 0 . 95 70.22 ± 2 . 02 | 77.06 ± 0 . 47 nan nan nan nan nan 77.22 ± 0 . 45 | 130.51 ± 0 . 27 nan nan nan nan nan 130.17 ± 0 . 26 | 129.43 ± 0 . 77 36.92 ± 1 . 37 51.36 ± 1 . 93 62.97 ± 1 . 45 42.37 ± 0 . 31 41.31 ± 0 . 35 34.52 ± 2 . 26 | 212.94 ± 1 . 90 52.58 ± 0 . 62 84.72 ± 0 . 67 102.34 ± 0 . 86 52.24 ± 0 . 38 56.47 ± 0 . 4 50.49 ± 0 . 33 | 77.06 ± 0 . 47 nan 118.82 ± 0 . 58 nan 119.43 ± 0 . 60 nan 78.12 ± 0 . 49 | 130.51 ± 0 . 27 nan 182.88 ± 0 . 79 nan 171.74 ± 0 . 55 nan 129.97 ± 0 . 30 |

Table 18: Long-term prediction on CMU Mocap: Jumping and Soccer .

| scenarios                                                           | jumping                                                          | jumping                                                          | pretrain                                                                          | pretrain                                                                            | soccer                                                                             | soccer                                                                              | pretrain                                                                        | pretrain                                                                            |
|---------------------------------------------------------------------|------------------------------------------------------------------|------------------------------------------------------------------|-----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| millisecond (ms)                                                    | 560                                                              | 1000                                                             | 560                                                                               | 1000                                                                                | 560                                                                                | 1000                                                                                | 560                                                                             | 1000                                                                                |
| Pretrain Full FT PARTIAL- k [12] MLP- k GPF [6] GPF-plus [6] GeoAda | - - 149.06 ± 0 . 37 140.11/nan 151.50 ± 0 . 65 - 139.46 ± 0 . 29 | - - 181.52 ± 0 . 81 184.77/nan 194.94 ± 0 . 34 - 184.01 ± 0 . 60 | 77.06 ± 0 . 47 145.76 ± 1 . 23 135.25 ± 0 . 92 111.71 ± 0 . 74 - - 76.98 ± 0 . 51 | 130.51 ± 0 . 27 199.34 ± 2 . 01 186.16 ± 1 . 58 158.55 ± 1 . 37 - - 130.72 ± 0 . 26 | - 101.44 ± 0 . 51 113.88 ± 0 . 75 - 100.23 ± 0 . 88 101.15 ± 0 . 86 84.91 ± 0 . 46 | - 157.11 ± 0 . 96 170.50 ± 1 . 98 - 151.84 ± 1 . 23 153.43 ± 0 . 79 125.91 ± 0 . 75 | 77.06 ± 0 . 47 - 89.63 ± 1 . 62 - 104.03 ± 0 . 81 90.12 ± 1 . 10 77.19 ± 0 . 48 | 130.51 ± 0 . 27 - 142.41 ± 1 . 84 - 161.56 ± 1 . 31 142.30 ± 1 . 12 130.06 ± 0 . 31 |

Table 19: Long-term prediction on CMU Mocap: Basketball

.

| scenarios                                                           | basketball                                                                                                   | basketball                                                                                                      | pretrain                                                   | pretrain                                                 |
|---------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|----------------------------------------------------------|
| millisecond (ms)                                                    | 560                                                                                                          | 1000                                                                                                            | 560                                                        | 1000                                                     |
| Pretrain Full FT PARTIAL- k [12] MLP- k GPF [6] GPF-plus [6] GeoAda | 143.49 ± 1 . 19 94.59 ± 0 . 58 106.84 ± 1 . 00 107.30 ± 2 . 76 97.16 ± 0 . 35 104.54 ± 0 . 26 91.03 ± 0 . 33 | 223.99 ± 2 . 23 132.34 ± 1 . 30 146.27 ± 1 . 24 149.58 ± 2 . 24 128.29 ± 0 . 43 130.76 ± 1 . 28 120.35 ± 0 . 44 | 77.06 ± 0 . 47 nan nan nan nan 104.51 ± 0 . 76.94 ± 0 . 45 | 130.51 ± 0 . 27 nan nan nan nan 155.02 ± 0 . 51 ± 0 . 30 |
|                                                                     |                                                                                                              |                                                                                                                 | 57                                                         |                                                          |
|                                                                     |                                                                                                              |                                                                                                                 |                                                            | 129.81                                                   |

## 10.3 Ablations

## 10.3.1 Parameter efficiency analysis

As shown in Table 20, we explore the impact of varying the number of equivariant adapter blocks. Increasing the number of trainable copy layers generally improves performance, but introduces more parameters and computational cost, revealing a tradeoff between performance and efficiency. We have computed the number of tunable parameters for all baselines and GeoAda. The statistics are presented in Table 21. Except for Full Fine-Tuning, all methods

Table 20: Different numbers of adapter blocks

|   Number |   ADE |   FDE |
|----------|-------|-------|
|        1 | 1.321 | 3.088 |
|        2 | 1.291 | 2.968 |
|        3 | 1.108 | 2.621 |
|        4 | 1.104 | 2.588 |
|        5 | 1.106 | 2.686 |

have a comparable number of tunable parameters, ensuring a fair comparison.

Table 21: The number of tunable parameters for different tuning strategies.

| Dataset          | Tuning Strategy                                   | Total Parameters                                                                                                | Tunable Parameters                                                                                        |
|------------------|---------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Charged Particle | Full FT PARTIAL- k MLP- k Prompt-Tem GPF GPF-plus | 1418252 ~5.41MB 1418252 ~5.41MB 2125202 ~8.11MB 1418252 ~5.41MB 2125266 ~8.11MB 2125847 ~8.11MB 2125691 ~8.11MB | 1418252 ~5.41MB 711302 ~2.71MB 711302 ~2.71MB 711302 ~2.71MB 711366 ~2.71MB 711947 ~2.72MB 711791 ~2.72MB |
| Charged Particle | Full FT PARTIAL- k MLP- k Prompt-Tem              | 1424268 ~5.43MB 1424268 ~5.43MB 2132370 ~8.13MB 1424268 ~5.43MB                                                 |                                                                                                           |
| Charged Particle | GeoAda                                            |                                                                                                                 |                                                                                                           |
| MD17             |                                                   |                                                                                                                 | 1424268 ~5.43MB                                                                                           |
| MD17             |                                                   |                                                                                                                 | 716166 ~2.73MB                                                                                            |
| MD17             |                                                   |                                                                                                                 | 716166 ~2.73MB                                                                                            |
| MD17             |                                                   |                                                                                                                 | 716166 ~2.73MB                                                                                            |
| MD17             | GPF                                               | 2132434 ~8.13MB                                                                                                 | 716230 ~2.73MB                                                                                            |
| MD17             | GPF-plus                                          | 2135079 ~8.14MB                                                                                                 | 718875 ~2.74MB                                                                                            |
| MD17             | GeoAda                                            | 2132844 ~8.14MB                                                                                                 | 716640 ~2.73MB                                                                                            |
| CMUMocap         | Full FT                                           | 368012 ~1.40MB                                                                                                  | 368012~1.40MB                                                                                             |
| CMUMocap         | PARTIAL- k                                        | 368012 ~1.40MB                                                                                                  | 185990~0.71MB                                                                                             |
| CMUMocap         | MLP- k                                            | 550034 ~2.10MB                                                                                                  | 185990~0.71MB                                                                                             |
| CMUMocap         | GPF                                               | 550098 ~2.10MB                                                                                                  | 186054 ~0.71MB                                                                                            |
| CMUMocap         | GPF-plus                                          | 553259 ~2.11MB                                                                                                  | 189215 ~0.72MB                                                                                            |
| CMUMocap         | GeoAda                                            | 550075 ~2.10MB                                                                                                  | 186031~0.71MB                                                                                             |

## 10.3.2 Ablative Architectures

We study the following ablative architectures as shown in Figure 6, Figure 7, Figure 8:

Figure 6: GeoAda

<!-- image -->

Figure 7: w/o zero convolution

Proposed GeoAda. The proposed architecture in the main paper.

Without Zero Convolution. Replacing the zero convolutions with standard convolution layers initialized with Gaussian weights.

Lightweight Layers. This architecture does not use a trainable copy, and directly initializes single convolution layers.

Results We present the results of this ablative study in Table 22, Table 23, Table 24 and Table 25.

Table 22: Comparisons for Molecular Dynamics prediction on MD17 dataset (all results reported by × 10 -1 ). The best results are highlighted in bold. Results averaged over 5 runs

| Scenarios                               | Aspirin                                         | Aspirin                                         | Aspirin                                         | Aspirin                                         | Benzene                                         | Benzene                                         | Benzene                                         | Benzene                                         |
|-----------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|
| Task                                    | Downstream                                      | Downstream                                      | Pretrain                                        | Pretrain                                        | Downstream                                      | Downstream                                      | Pretrain                                        | Pretrain                                        |
| Metric                                  | ADE                                             | FDE                                             | ADE                                             | FDE                                             | ADE                                             | FDE                                             | ADE                                             | FDE                                             |
| w/o trainable copy w/o zero conv GeoAda | 2.232 ± 0 . 008 0.929 ± 0 . 002 0.891 ± 0 . 003 | 3.501 ± 0 . 022 1.619 ± 0 . 009 1.533 ± 0 . 008 | 2.197 ± 0 . 007 1.203 ± 0 . 009 1.060 ± 0 . 003 | 3.449 ± 0 . 010 1.975 ± 0 . 018 1.852 ± 0 . 012 | 0.607 ± 0 . 002 0.214 ± 0 . 001 0.191 ± 0 . 000 | 0.952 ± 0 . 010 0.359 ± 0 . 004 0.319 ± 0 . 002 | 0.584 ± 0 . 002 0.291 ± 0 . 001 0.240 ± 0 . 002 | 0.948 ± 0 . 006 0.469 ± 0 . 007 0.394 ± 0 . 005 |

Table 23: Ablation study of Short-term prediction on running from the CMU Mocap dataset.

| scenarios                               | running                                      | running                                      | running                                       | running                                       | pretrain                           | pretrain                           | pretrain                           | pretrain                           |
|-----------------------------------------|----------------------------------------------|----------------------------------------------|-----------------------------------------------|-----------------------------------------------|------------------------------------|------------------------------------|------------------------------------|------------------------------------|
| millisecond (ms)                        | 80                                           | 160                                          | 320                                           | 400                                           | 80                                 | 160                                | 320                                | 400                                |
| w/o trainable copy w/o zero conv GeoAda | 44.52 ± 0 . 48 19.07 ± 0 . 37 18.70 ± 0 . 37 | 76.98 ± 1 . 20 34.25 ± 0 . 82 33.56 ± 0 . 25 | 134.91 ± 2 . 04 51.75 ± 1 . 39 50.26 ± 0 . 42 | 159.75 ± 2 . 45 55.74 ± 1 . 53 55.54 ± 0 . 36 | 198.16 ± 2 . 97 nan 7.972 ± 0 . 02 | 139.24 ± 0 . 23 nan 16.91 ± 0 . 05 | 270.04 ± 0 . 56 nan 40.06 ± 0 . 42 | 314.89 ± 0 . 68 nan 52.61 ± 0 . 07 |

Table 24: Ablation study of Short-term prediction on walking from the CMU Mocap dataset.

Table 25: Ablation study of long-term prediction on running, walking from the CMU Mocap dataset.

| scenarios                               | walking                                     | walking                                   | walking                                      | walking                                       | pretrain                    | pretrain                    | pretrain                           | pretrain                           |
|-----------------------------------------|---------------------------------------------|-------------------------------------------|----------------------------------------------|-----------------------------------------------|-----------------------------|-----------------------------|------------------------------------|------------------------------------|
| millisecond (ms)                        | 80                                          | 160                                       | 320                                          | 400                                           | 80                          | 160                         | 320                                | 400                                |
| w/o trainable copy w/o zero conv GeoAda | 23.77 ± 0 . 15 12.62 ± 0 . 07 8.92 ± 1 . 02 | 43.43 ± 0 . 31 20.67 ± 0 . 13.82 ± 1 . 26 | 84.79 ± 0 . 77 36.75 ± 0 . 52 22.99 ± 1 . 30 | 105.08 ± 1 . 28 44.69 ± 0 . 45 26.68 ± 1 . 31 | nan 36.18 ± 0 . 7.932 ± 0 . | nan 58.18 ± 0 . 16.90 ± 0 . | nan 102.06 ± 0 . 05 38.96 ± 0 . 47 | nan 123.82 ± 0 . 04 52.54 ± 0 . 10 |
|                                         |                                             | 20                                        |                                              |                                               | 12                          | 08                          |                                    |                                    |
|                                         |                                             |                                           |                                              |                                               | 03                          | 04                          |                                    |                                    |

Figure 8: w/o trainable copy

| scenarios                               | running                                       | running                                       | pretrain                           | pretrain                            | walking                                       | walking                                       | pretrain                           | pretrain                            |
|-----------------------------------------|-----------------------------------------------|-----------------------------------------------|------------------------------------|-------------------------------------|-----------------------------------------------|-----------------------------------------------|------------------------------------|-------------------------------------|
| millisecond (ms)                        | 560                                           | 1000                                          | 560                                | 1000                                | 560                                           | 1000                                          | 560                                | 1000                                |
| w/o trainable copy w/o zero conv GeoAda | 198.16 ± 2 . 97 64.69 ± 0 . 97 60.88 ± 0 . 82 | 228.41 ± 2 . 96 74.64 ± 0 . 66 70.22 ± 2 . 02 | 365.41 ± 0 . 63 nan 77.22 ± 0 . 45 | 374.74 ± 0 . 48 nan 130.17 ± 0 . 26 | 143.35 ± 2 . 26 60.28 ± 1 . 07 34.52 ± 2 . 26 | 214.78 ± 2 . 82 93.99 ± 1 . 41 50.49 ± 0 . 33 | nan 165.86 ± 0 . 15 78.12 ± 0 . 49 | nan 249.90 ± 0 . 38 129.97 ± 0 . 30 |

## 10.4 Standard Deviations

We have already provided the standard deviations in App. 10.2.

## 11 Discussion

Limitation While GeoAda demonstrates strong empirical performance and theoretical grounding in preserving SE(3)-equivariance during fine-tuning, several limitations remain: The effectiveness of GeoAda hinges on the design of coupling and decoupling operators for control injection. While theoretically sound, these handcrafted designs may not generalize well to control signals with highdimensional or structured semantics. Moreover, although GeoAda is validated across multiple domains (e.g., particles, molecules, human motion), the evaluations are limited to medium-scale datasets and relatively small models. Assessing its scalability to larger systems-such as full proteins or macromolecular assemblies-remains an important direction for future work.