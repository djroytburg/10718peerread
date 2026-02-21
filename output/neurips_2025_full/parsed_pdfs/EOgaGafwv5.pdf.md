## GST-UNet: A Neural Framework for Spatiotemporal Causal Inference with Time-Varying Confounding

Miruna Oprescu Cornell University amo78@cornell.edu

## David K Park

Brookhaven National Laboratory dpark1@bnl.gov

## Shinjae Yoo

Brookhaven National Laboratory sjyoo@bnl.gov

## Abstract

Estimating causal effects from spatiotemporal observational data is essential in public health, environmental science, and policy evaluation, where randomized experiments are often infeasible. Existing approaches, however, either rely on strong structural assumptions or fail to handle key challenges such as interference, spatial confounding, temporal carryover, and time-varying confounding -where covariates are influenced by past treatments and, in turn, affect future ones. We introduce the GST-UNet ( G -computation S patioT emporal UNet ), a theoretically grounded neural framework that combines a U-Net-based spatiotemporal encoder with regression-based iterative G-computation to estimate location-specific potential outcomes under complex intervention sequences. GST-UNet explicitly adjusts for time-varying confounders and captures non-linear spatial and temporal dependencies, enabling valid causal inference from a single observed trajectory in data-scarce settings. We validate its effectiveness in synthetic experiments and in a real-world analysis of wildfire smoke exposure and respiratory hospitalizations during the 2018 California Camp Fire. Together, these results position GST-UNet as a principled and ready-to-use framework for spatiotemporal causal inference, advancing reliable estimation in policy-relevant and scientific domains.

## 1 Introduction

Environmental hazards, public health interventions, and socio-economic policies often require understanding complex cause-and-effect relationships across space and time [30, 34, 41]. For instance, evaluating the health impacts of air quality regulations requires assessing how interventions influence both immediate outcomes and downstream effects across regions. Such applications demand robust tools for estimating causal effects from observational spatiotemporal data.

However, causal inference in spatiotemporal settings poses unique challenges. Outcomes are influenced not only by local covariates and interventions but also by those of neighboring regions (spatial confounding and interference). Effects may persist and accumulate over time (temporal carryover), and covariates often evolve in response to past interventions while simultaneously affecting future ones (time-varying confounding). For example, air quality regulations are often implemented in reaction to recent pollution levels and hospitalizations, which themselves shape future exposures and health outcomes-creating feedback loops that violate standard independence assumptions. These complexities induce bias in naive estimators and are especially challenging in single-trajectory settings, where replication across units or time is infeasible.

## Xihaier Luo

Brookhaven National Laboratory xluo@bnl.gov

## Nathan Kallus

Cornell University &amp; Netflix kallus@cornell.edu

Existing approaches offer limited solutions: classical methods rely on rigid structural assumptions or user-defined exposure mappings, while recent neural models emphasize predictive accuracy over causal identification. Many assume independent time series or model only spatial correlations, leaving a gap in methods that can jointly address interference, temporal dependencies, and evolving confounding within a principled causal framework (see Section 2).

To bridge this gap, we introduce GST-UNet ( G -computation S patioT emporal UNet ), a theoretically grounded neural framework for estimating location-specific potential outcomes in spatiotemporal settings with time-varying confounding. GST-UNet builds on formal identification and consistency results derived under a representation-based time-invariance assumption, showing how causal effects can be recovered from a single observed trajectory. We then instantiate this theory in a practical neural architecture: a U-Net encoder with ConvLSTM and attention modules coupled to an iterative G-computation procedure that performs recursive causal adjustment over time. To ensure stable estimation over long horizons, we design a curriculum-based training strategy that gradually refines recursive pseudo-outcomes, enabling effective learning even in data-scarce regimes. Unlike existing approaches, GST-UNet requires no user-specified structural models and can be directly deployed in real-world spatiotemporal applications.

Our contributions are threefold: (1) We develop the first unified framework that couples theoretical identification and consistency guarantees with an end-to-end neural implementation for spatiotemporal causal inference; (2) We demonstrate through controlled simulations that GST-UNet robustly handles interference, temporal carryover, and time-varying confounding; and (3) We illustrate its practical value via a real-world analysis of wildfire smoke exposure and respiratory hospitalizations during the 2018 California Camp Fire.

In summary, GST-UNet provides a principled and ready-to-use framework for causal inference from spatiotemporal data, combining formal guarantees with a flexible neural implementation. By abstracting away model-specific assumptions, GST-UNet makes spatiotemporal causal estimation both accessible and reliable for applied scientific and policy domains.

## 2 Related Work

We summarize the most relevant prior work here, with a more detailed discussion in Appendix A.

Classical Spatiotemporal Causal Inference. Early approaches (e.g., spatial econometrics [2], difference-in-differences [20], synthetic controls [4]) rely on strong assumptions such as parallel trends and no interference. More recent methods incorporate time-varying confounding using inverse propensity weighting (IPW) and marginal structural models [31, 49], but cannot address interference unless via user-specified exposure mappings or hyper-local assumptions [11, 44, 48]. As noted by Zhou et al. [49], the literature remains sparse, particularly in settings with rich feedback dynamics.

Machine Learning for Spatiotemporal Modeling. Deep learning models for prediction-e.g., CNNs and RNNs [40, 47], graph-based methods [25, 46], and video transformers [6, 27]-capture complex spatial-temporal patterns but do not incorporate causal adjustments, and thus cannot estimate counterfactuals or adjust for time-varying confounders.

Time Series Causal Inference. Causal methods for longitudinal data include marginal structural models [36], iterative G-computation [35], and recent ML-based extensions using recurrent networks, transformers, or meta-learners [7, 16, 18, 24, 28, 39]. However, these assume access to independent time series ( e.g. across patients) and cannot model cross-unit interactions in spatiotemporal settings.

Neural-Based Spatiotemporal Causal Inference. Tec et al. [42] propose a UNet-based model that adjusts for non-local spatial confounding but focuses on static exposures and does not address interference or time-varying effects. Most similar to our work, [1] presents a climate-focused model that shares certain architectural similarities but emphasizes prediction rather than causal adjustment, leaving causal identification under time-varying confounding largely unaddressed.

Positioning of Our Work. Our work bridges these threads by uniting a theoretically grounded G-computation framework with a neural architecture for spatiotemporal data. Unlike prior time-series methods that assume independent units or spatial models that overlook confounding feedback, GSTUNet is the first end-to-end approach that (i) establishes identification and consistency under explicit assumptions for a single spatiotemporal trajectory, and (ii) implements this theory in a practical neural model capable of handling interference, spatial confounding, and time-varying dynamics.

Figure 1: Observational data (left) versus interventional data (right) for a horizon τ = 2 across multiple locations ( s, s ′ ) . Under the intervention (right), treatments are set independently of confounders, and the full history is not observed for the entire horizon.

<!-- image -->

## 3 Problem Formulation

Spatiotemporal Data. We model observed data as random variables on a discrete spatial domain represented by an N X × N Y lattice: S = { ( i, j ) | i ∈ [ N X ] , j ∈ [ N Y ] } , where [ N ] = { 1 , . . . , N } denotes the index set. Time is indexed by t ∈ [ T ] . At each spatial location s = ( i, j ) at time t , we observe a tuple ( X s,t , A s,t , Y s,t ) , where A s,t ∈ { 0 , 1 } represents a binary treatment (or intervention), Y s,t ∈ R is a continuous outcome of interest, and X s,t ∈ R d X is a vector of time-varying covariates ( e.g. local weather conditions, pollution levels, or socioeconomic indicators). Additionally, each location s is associated with static features V s ∈ R d v ( e.g. geographical characteristics and socioeconomic indicators). While we focus on binary interventions for clarity, the methods generalize to more complex treatments. Conceptually, each variable forms a 3D spatiotemporal tensor of size T × N X × N Y , though in practice, observations may be incomplete. Missing data can be accommodated using masking techniques during downstream modeling.

To streamline notation, we use boldface symbols for random variables defined over the entire spatial domain. For U ∈ { X,A,Y } , let U t denote its value at time t , and let U t : t + τ = ( U t , . . . , U t + τ ) denote its value over a time interval. For a specific location s , we write U s,t : t + τ = ( U s,t , . . . , U s,t + τ ) . The history up to time t is denoted by H 1: t = ( X 1: t , A 1: t -1 , Y 1: t , V ) for the entire spatial domain and H s, 1: t = ( X s, 1: t , A s, 1: t -1 , Y s, 1: t , V s ) for a specific location s . Specific instantiations of these random variables are denoted using lowercase letters (e.g., u ∈ { x, a, y, h } ).

Quantities of Interest. Our primary goal is to estimate location-specific Conditional Average Potential Outcomes (CAPOs) for a sequence of future spatiotemporal interventions, conditioned on observed history. Our approach builds on Rubin's potential outcomes framework [35, 36, 38], which we extend to accommodate spatiotemporal settings. More concretely, we consider a future time horizon of length τ ≥ 1 and a predetermined interventional sequence a t : t + τ -1 applied across the spatial domain starting at time t . Our goal is to estimate the potential outcomes at time t + τ , denoted as Y t + τ [ a t : t + τ -1 ] . In particular, we aim to compute:

<!-- formula-not-decoded -->

which represents the CAPOs at time t + τ under the given treatment sequence. Given two different interventional sequences a t : t + τ and a ′ t : t + τ , a related secondary goal is to estimate the location specific Conditional Average Treatment Effect (CATE), given by:

<!-- formula-not-decoded -->

Although we focus primarily on CAPOs, CATEs and other effect measures can be derived similarly.

Prefix Data in a Single Spatiotemporal Chain. The conditional expectations defining the CAPOs in Eq. (1) cannot be directly estimated from a single observed spatiotemporal realization, since the empirical averages would contain only one sample of each future outcome Y t + τ [ a t : t + τ -1 ] . To obtain a workable regression-based estimator, we therefore reorganize the single observed trajectory into overlapping "prefixes" of varying lengths. For each t ∈ { 1 , . . . , T -τ } , we define

<!-- formula-not-decoded -->

which represents the observed history up to time t + τ along with all covariates, treatments, and outcomes. When T ≫ τ , this construction yields T -τ segments that partially overlap in time, providing additional training samples in this intrinsically data-scarce, single-chain setting.

However, these prefixes are not independent: successive segments share overlapping histories, so standard i.i.d. assumptions do not apply. In the next section, we introduce conditions under which these prefixes can be treated as conditionally exchangeable given an appropriate learned embedding. This enables regression-based estimation of CAPOs by pooling information across time without violating the dependence structure of the original process.

## 4 Identification and Estimation of CAPOs in Spatiotemporal Settings

Identification of CAPOs from observational data relies on standard causal inference assumptions. In our setting, these must be complemented by additional structure to handle the fact that we observe only a single spatiotemporal trajectory. Building on the prefix construction introduced above, we impose conditions that render these overlapping segments conditionally exchangeable , enabling principled pooling of information across time.

Assumption 1 (Causal Inference Assumptions) . We assume: (Consistency) Y t + τ = Y t + τ [ a t : t + τ -1 ] whenever the observed sequence of treatments A t : t + τ -1 satisfies A t : t + τ -1 = a t : t + τ -1 ; (Positivity) P ( A s,t = a s,t | H 1: t = h 1: t ) &gt; 0 for any a s,t ∈ { 0 , 1 } and feasible realization of history h 1: t ; (Sequential Unconfoundedness) Y t +1: T [ a t +1: T ] ⊥ A t | H 1: t , ∀ a t +1: T ∈ { 0 , 1 } T -t , i.e. at each time step t , the treatment assignment is independent of future potential outcomes.

Assumption 2 (Representation-Based Time Invariance) . There exists a function (or embedding) ϕ : H×A→Z ⊆ R h that maps ( H 1: t , A t ) to a finite-dimensional representation such that once we condition on z = ϕ ( H 1: t , A t ) , the distribution ( X t +1 , Y t +1 ) does not explicitly depend on t . Formally, for any t, t ′ ∈ { 1 , . . . , T } and z ∈ Z , we have:

<!-- formula-not-decoded -->

Assumption 1 is a standard set of requirements in longitudinal causal inference settings (e.g., [7, 18, 24, 28, 35, 36]). Assumption 2 is specific to the single-time series setting, where pooling information across time is essential to enable estimation. We note that the single time-series setting frequently arises in causal inference, where assumptions such as stationarity or strict time homogeneity enable consistent estimation [8, 31, 49]. In contrast, our representation-based time invariance is weaker : rather than requiring X t , Y t themselves to have a time-invariant distribution, we only assume that, once the history is summarized by ϕ ( H 1: t , A t ) , the transition to ( X t +1 , Y t +1 ) follows a single shared mechanism. This approach aligns with modern time-series causal inference that learn timeinvariant latent embeddings to pool information across time steps [18, 24, 26], thus leveraging more data for a single, stable representation rather than time-dependent parameters.

Under Assumption 2, conditioning on ϕ ( H 1: t , A t ) removes explicit dependence on t , such that

<!-- formula-not-decoded -->

represents a shared conditional expectation across all prefix segments. In this view, t indexes the segment's position rather than a distinct distribution. Pooling over t thus yields T -τ approximately exchangeable segments from a single trajectory, enabling regression-based estimation of future outcomes from embedded histories.

## 4.1 Identification via Representation-Based G-Computation

Given P τ t , we next show how to identify CAPOs from observational data. For horizons τ ≥ 2 , future covariates and outcomes ( i.e. X t +1: t + τ -1 , Y t +1: t + τ -1 ) can influence subsequent treatments, inducing time-varying confounding [13]. Such feedback violates standard "condition-on-history" adjustments and leads to biased estimates. Figure 1 illustrates these dependencies by contrasting observational data (left) and hypothetical interventions (right) for τ = 2 . By contrast, when τ = 1 , conditioning on H 1: t is sufficient under standard assumptions, as no future confounders intervene between A t and Y t +1 . Formally, the following naive identification fails to hold for τ &gt; 1 :

̸

<!-- formula-not-decoded -->

To correct this bias, we adapt regression-based iterative G-computation [3, 35] to the spatiotemporal setting, yielding a principled adjustment procedure for evolving confounders and valid CAPO estimation. We formalize this connection in the following result:

Theorem 1 (Identification with G-Computation) . Assume that Assumption 1 and Assumption 2 hold. Further, let H a 1: t + k := ( X 1: t + k , [ A 1: t -1 , a t : t + k -1 ] , Y 1: t + k ) denote the history where observed treatments from time t onward are replaced by a t : t + k -1 . Define recursively:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We provide a proof of Theorem 1 in Appendix B. This result naturally motivates a recursive regression approach for spatiotemporal CAPO estimation, fitting each Q k ( · ) in reverse order and substituting interventional treatments where required.

## 4.2 Estimation via Iterative G-Computation

While Theorem 1 motivates a recursive regression algorithm for each Q k ( k = 1 , . . . , τ ), only Q τ can be directly estimated from the prefix data. At the next step, Q τ -1 depends on Q τ ( H a 1: t + τ -1 , a t + τ -1 ) -where the observed treatments A t : t + τ -1 are replaced by a t : t + τ -1 -but such substituted outcomes are not observed in the prefix data. Therefore, for k &lt; τ , we propose a procedure where we generate pseudo-outcomes by predicting with the previously learned ̂ Q k +1 . Going forward, we use ̂ F to denote any quantity F estimated from data. Formally, let ϕ ∈ Φ be an embedding satisfying Assumption 2, and let Q be our function class for Q k . We learn the sequence ̂ Q τ , . . . , ̂ Q 1 from prefix data { P τ t : t = 1 , . . . , T -τ } , via:

1. Initialization. Fit ̂ Q τ to predict Y t + τ from the prefix embedding ϕ ( H 1: t + τ -1 , A t + τ -1 ) .
2. Backward recursion. For k = τ -1 , . . . , 1 :
3. (a) Substitute interventions. For each prefix P τ t , replace A t + k by the interventional a t + k to form the modified history H a 1: t + k .
4. (b) Generate pseudo-outcomes. Let ˜ Y t + k +1 = ̂ Q k +1 ( H a 1: t + k , a t + k ) , where ̂ Q k +1 was learned in the previous step. These ˜ Y t + k +1 act as surrogates for Y t + k +1 in the prefix data.
5. (c) Fit ̂ Q k . Regress ˜ Y t + k +1 on the current embedding ϕ ( H 1: t + k -1 , A t + k -1 ) to learn ̂ Q k ∈ Q .
3. Final step. Given a new history h 1: t and an interventional path a t : t + τ -1 , we predict

<!-- formula-not-decoded -->

The iterative regression procedure yields consistent CAPO estimates provided each stage Q k is estimated consistently from data [22]. Informally, if the learned embedding ̂ ϕ converges to the true time-invariant representation ϕ , and small perturbations in ϕ or ̂ Q k lead to proportionally small changes in predictions, then the overall recursive estimator remains consistent. These regularity conditions-formalized through uniform stochastic equicontinuity-are detailed in Appendix C. Formally, we state the following theorem:

Theorem 2 (Consistency of Iterative G-Computation in Spatiotemporal Settings) . Assume Assumptions 1 and 2 and that (a) the learned embedding ̂ ϕ is L 2 -consistent for ϕ , and (b) each regression head ̂ Q k consistently estimates Q k and is uniformly well-behaved 1 on Im ϕ (intuitively, small input perturbations induce small output changes). Let Z k := ( H 1: t + k , A t + k ) denote the history-action pair at step k . Then

<!-- formula-not-decoded -->

so the recursive estimator ̂ Q 1 of the CAPO is probabilistically consistent.

We provide a proof of Theorem 2 in Appendix C. In the following section, we instantiate this procedure in our GST-UNet architecture, illustrating how to incorporate spatial dependencies and interference into ϕ and each Q k , and implement a streamlined, end-to-end training strategy that unifies history embeddings and outcome predictions.

1 We formalize "well-behaved" via uniform stochastic equicontinuity and continuity in Appendix C.

Figure 2: Overview of the GST-UNet architecture. The spatiotemporal learning module (left) is a U-Net augmented with a ConvLSTM layer and attention gates. Its final feature map is passed to a set of G-heads (right), where each G-head Q k implements iterative G-computation (see Algorithm 1).

<!-- image -->

## 5 GST-UNet Implementation

The theoretical results above establish how CAPOs can be identified and consistently estimated from a single spatiotemporal trajectory. We now provide a concrete neural implementation of this procedure. GST-UNet instantiates the iterative G-computation framework with a spatiotemporal deep architecture that embeds strong inductive biases-locality, translation invariance, and temporal smoothness-well suited to data-scarce settings. While alternative backbones could be employed, our U-Net with ConvLSTM and attention offers a natural choice for learning stable, history-invariant representations that satisfy Assumption 2. We now describe the architecture and the training procedure that realizes the GST-UNet (Algorithm 1).

## 5.1 Model Architecture

The GST-UNet consists of two main components:

1. Spatiotemporal Learning Module: a U-Net-based network augmented with ConvLSTM and attention gates for spatiotemporal processing.
2. Neural Causal Module: τ G-computation heads, each mapping the spatiotemporal features to the final outcome predictions in the iterative procedure.

We illustrate the GST-UNet architecture in Figure 2 and describe its main components below.

Spatiotemporal Learning Module. (1) Spatial Module. While our framework is agnostic to the choice of spatiotemporal learning module, we adopt a U-Net with ConvLSTM and attention due to its strong performance in data-scarce regimes. To efficiently process high-dimensional spatial data, we employ U-Net [37], a fully convolutional architecture originally developed for biomedical image segmentation. It employs an encoder-decoder design with skip connections: the encoder progressively downsamples the spatial grid through convolution and pooling, while the decoder upsamples it back to the original resolution, merging encoder features at each scale. (2) Temporal Module. U-Net has limitations in capturing temporal information. To address this, we integrate a Convolutional Long Short-Term Memory (ConvLSTM) layer [40] to the U-Net encoder. This module captures temporal dependencies by maintaining a hidden state across time steps while aggregating spatial information through convolutions. After computing the final ConvLSTM state, we append static (time-invariant) covariates V as additional feature channels, ensuring the subsequent U-Net encoder-decoder has direct access to both temporal dynamics and static location-specific information. In the decoder, we incorporate attention gates [29] to selectively highlight relevant spatial regions, refining skip connections and emphasizing critical global or local patterns. The embedding module ultimately produces a d h -dimensional feature map of size N X × N Y , capturing essential spatiotemporal contextincluding interference, spatial confounding, and static covariates-for downstream G-computation.

Neural Causal Module. We attach τ G-computation heads to the U-Net's final feature maps, corresponding to the Q k estimators in the iterative procedure (see Section 4.2). Each head can be a small convolutional module or a simple feed-forward network, depending on how much spatial structure

## Algorithm 1 GST-UNet Training and Inference

- 1: Input: Horizon τ , prefixes { P τ t } T -τ t =1 , interventions a t : t + τ -1 , curriculum α ( e ) k , total epochs E .
- 2: Initialize: parameters θ (U-Net embedding + G-heads).
- 3: for e = 1 . . . E do
- 4: for k = τ . . . 1 do
- 6: (Generation (detached)) For each prefix i , generate pseudo-outcomes:
- 5: (Supervision) For each prefix i , predict outcomes ̂ Y ( i ) t + k = Q k ( ϕ ( H ( i ) 1: t + k -1 , A ( i ) t + k -1 ); θ ) .

<!-- formula-not-decoded -->

where the observed A t : t + k -1 's were replaced with a t : t + k -1 in H a 1: t + k .

- 7: (Loss aggregation) Compute the MSE loss L ( θ ; e ) = 1 τ ∑ τ k =1 α ( e ) k ∑ i ( ̂ Y ( i ) t + k -˜ Y ( i ) t + k +1 ) 2 .
- 8: (Backward pass) Update θ by backpropagation.
- 9: (Inference) Given a h 1: t , return Q 1 ( ϕ ( h 1: t , a t ); ̂ θ ) .

remains to be captured. The information flow at the G-computation heads proceeds as follows: each head Q k ( k = 1 , . . . , τ ) receives the d h × N X × N Y U-Net embedding ̂ ϕ ( H 1: t + k -1 , A t + k -1 ) (encompassing spatiotemporal and static context) and outputs an N X × N Y prediction for that time step. We refer to this as the supervision step , since Q τ compares its predictions to the real observed outcomes Y t + τ , anchoring the model in genuine data, while each Q k&lt;τ compares its predictions to pseudo-outcomes ˜ Y t + k +1 provided by ̂ Q k +1 . These pseudo-outcomes arise in a subsequent generation step , wherein Q k +1 processes the intervened history ̂ ϕ ( H a 1: t + k , a t + k ) in a detached forward pass (so ̂ Q k +1 is not updated by Q k 's loss), thereby creating surrogate targets for Q k . This procedure realizes the iterative G-computation logic from Section 4.2, enabling GST-UNet to estimate future outcomes under various counterfactual treatments. By separating the spatiotemporal embedding from the G-heads, we maintain a common representation for all prefix data (see Assumption 2) and flexibly capture interference and spatial confounding. Each G-head enforces the proper temporal adjustments to yield bias-free counterfactual inference.

## 5.2 Training and Inference

While each G-head Q k could be trained sequentially-from Q τ down to Q 1 -by passing pseudooutcomes backward through time, this creates a conflict when all heads share the same U-Net embedding ϕ . Specifically, each Q k may push ϕ toward optimizing its own objective, resulting in misaligned training signals and unstable learning.

Joint Loss and Multi-Task Training. To address this issue, we employ a joint (or multi-task ) training approach [9, 15] by aggregating the loss terms from all G-heads into a single objective, then backpropagating once per batch. Concretely, for each head Q k , let ˜ Y t + k +1 be the real outcomes if k = τ or pseudo-outcomes (generated by ̂ Q k +1 ) if k &lt; τ . Our head-specific loss is a mean squared error (MSE) over all prefix samples:

<!-- formula-not-decoded -->

where θ encompasses all model parameters (the shared U-Net embedding ϕ and the G-heads Q k ). Let α ( e ) k denote a head-weight for epoch e . We then form the overall training objective at epoch e by

<!-- formula-not-decoded -->

By summing the losses and performing a single backward pass, we learn a common embedding ̂ ϕ that balances the needs of all G-heads, rather than fitting each head separately.

Curriculum Training. A naive implementation of Eq. (3)-where each G-head is given equal weightcan be suboptimal: early in training, Q τ (which sees real data) is inaccurate, and the pseudo-outcomes

generated for Q k&lt;τ are effectively noise. Consequently, Q 1 , . . . , Q τ -1 may overfit to poor targets before Q τ has converged, leading to suboptimal solutions. To mitigate this, we employ a curriculum training approach [5], gradually increasing the loss weight of earlier heads as Q τ improves.

While many curricula are possible, we adopt a simple scheme controlled by a single hyperparameter e c (the 'curriculum period') so we can readily tune it. Let p ( e ) = min { τ, ⌈ e/e c ⌉} , which indexes a 'phase' based on the current epoch e . We then define

<!-- formula-not-decoded -->

Hence, during epochs 1 ≤ e ≤ e c (phase p ( e ) = 1 ), only Q τ is active with α ( e ) τ = 1 ; in the next interval e c &lt; e ≤ 2 e c (phase p ( e ) = 2 ), Q τ and Q τ -1 each have weight 1 / 2 , and so on until all heads are active with uniform weight 1 /τ . For e &gt; τe c , training continues with α ( e ) k = 1 /τ for all heads. This schedule ensures Q τ becomes reasonably accurate before earlier heads rely on its pseudo-outcomes. The hyperparameter e c controls the pacing, helping prevent early training noise.

We also adopt standard neural network practices, including mini-batch optimization and early stopping, to stabilize training and mitigate overfitting. At inference time, given a new history h 1: t and an interventional sequence a t : t + τ -1 , we compute ̂ Q 1 ( ϕ ( h 1: t , a t ); θ ) as our target CAPO estimate. We sketch the overall training and inference procedure in Algorithm 1.

## 6 Experiments

We evaluate the proposed GST-UNet framework through two applications. First, we simulate synthetic data that incorporates key spatiotemporal causal inference challenges: interference, spatial confounding, temporal carryover, and time-varying confounding. Using this synthetic data generation process (DGP), we compare the GST-UNet algorithm against several baselines. Next, we demonstrate the utility of GST-UNet on a real-world dataset analyzing the impact of wildfire smoke on respiratory hospitalizations during the 2018 California Camp Fire.

Additional details-including exact simulation parameters, model architecture and execution setups, hyperparameter selection strategies, and validation procedures-can be found in Appendix D. Replication code is available at https://github.com/moprescu/GSTUNet .

## 6.1 Synthetic Data

We generate T = 200 time steps of a 64 × 64 ( N X × N Y ) grid of observational data using the following data generating process (DGP):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where d X = 1 , " ∗ " denotes a 3 × 3 spatial convolution over the N X × N Y grid, and ϵ X , ϵ Y ∼ N (0 , 1) are i.i.d. noise. Each kernel K X , K A , K Y A , K Y X encodes a local advection-diffusion process that mimics wind-driven pollutant transport, with interventions A t injecting additional emissions that propagate through the same kernel. This physically realistic setup produces interference , spatial confounding , and temporal carryover -the three challenges GST-UNet is designed to address. Each equation is evaluated at every spatial location, so X t , A t , and Y t are N X × N Y matrices. Here, X t acts as a time-varying confounder : its past influences both A t and Y t , while current interventions A t affect future X t +1 . For example, A t may represent regulatory actions, X t air quality, and Y t health outcomes-capturing feedback from policy to exposure to outcome, and back to future policy.

We vary β 1 to control time-varying confounding: when β 1 = 0 , X t does not affect A t , eliminating confounding; larger values increase its strength. For each β 1 , we generate 50 test trajectories from random initial states, fix their histories, and simulate 100 τ -step counterfactual futures to estimate true

Table 1: RMSE ± SD across test trajectories. Bold indicates lowest error per column; color shows improvement (RMSE decrease or increase ) over best baseline (excluding ablations).

|   τ | Model                                                     | β 1 = 0 . 0                                    | β 1 = 0 . 5                         | β 1 = 1 . 0                                  | β 1 = 1 . 5                                                             | β 1 = 2 . 0                                                 |
|-----|-----------------------------------------------------------|------------------------------------------------|-------------------------------------|----------------------------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------|
|   5 | UNet+ STCINet IPWUNet GST-UNet w/o Attention GST-UNet w/o | 0.28 ± 0.00 0.29 ± 0.60 ± 0.50 ± 0.69 ± 0.33 ± | 0.36 ± 0.00 0.38 ± 0.01 0.58 ± 0.01 | 0.54 ± 0.01 0.62 ± 0.58 ± 0.01 0.51 ± 0.63 ± | 0.71 ± 0.01 0.80 ± 0.01 0.59 ± 0.01 0.45 ± 0.01 0.61 ± 0.01 0.44 ± 0.00 | 0.81 ± 0.01 0.90 ± 0.01 0.59 ± 0.01 0.47 ± 0.01 0.61 ± 0.01 |
|   5 |                                                           | 0.00                                           |                                     | 0.01                                         |                                                                         |                                                             |
|   5 |                                                           | 0.01                                           |                                     |                                              |                                                                         |                                                             |
|   5 |                                                           | 0.00                                           | 0.46 ± 0.00                         | 0.00                                         |                                                                         |                                                             |
|   5 | Curriculum                                                | 0.00                                           | 0.64 ± 0.00                         | 0.00                                         |                                                                         |                                                             |
|   5 | GST-UNet                                                  | 0.00                                           | 0.35 ± 0.00                         | 0.40 ± 0.00                                  |                                                                         | 0.40 ± 0.01                                                 |
|   5 |                                                           | ( +17.9% )                                     | ( -2.7% )                           | ( -21.6% )                                   | ( -25.4% )                                                              | ( -32.2% )                                                  |
|  10 | UNet+                                                     | 0.28 ± 0.00                                    | 0.61 ± 0.00                         | 1.18 ± 0.00                                  | 1.45 ± 0.00                                                             | 1.71 ± 0.01                                                 |
|  10 | STCINet                                                   | 0.31 ± 0.00                                    | 0.68 ± 0.00                         | 1.25 ± 0.00                                  | 1.47 ± 0.01                                                             | 1.60 ± 0.01                                                 |
|  10 | IPWUNet                                                   | 0.78 ± 0.01                                    | 0.80 ± 0.01                         | 0.96 ± 0.01                                  | 1.19 ± 0.02                                                             | 1.08 ± 0.01                                                 |
|  10 | GST-UNet w/o Attention                                    | 0.42 ± 0.00                                    | 0.60 ± 0.00                         | 0.61 ± 0.00                                  | 0.79 ± 0.01                                                             | 1.07 ± 0.01                                                 |
|  10 | GST-UNet w/o Curriculum                                   | 0.62 ± 0.00                                    | 0.88 ± 0.00                         | 1.02 ± 0.00                                  | 1.08 ± 0.01                                                             | 1.12 ± 0.01                                                 |
|  10 | GST-UNet                                                  | 0.38 ± 0.00                                    | 0.55 ± 0.00                         | 0.68 ± 0.00                                  | 0.73 ± 0.01                                                             | 0.85 ± 0.01                                                 |
|  10 |                                                           | ( +35.7% )                                     | ( -9.8% )                           | ( -29.2% )                                   | ( -38.7% )                                                              | ( -21.3% )                                                  |

CAPOs, with τ ∈ { 5 , 10 } . We compare GST-UNet against three baselines: (i) UNet+ , which uses a U-Net + ConvLSTM + Attention backbone with A t as an input channel but performs no iterative adjustment; (ii) STCINet [1], which estimates direct and indirect effects without modeling timevarying confounding; and (iii) IPWUNet , an inverse-propensity-weighting variant that reweights pseudo-outcomes using a UNet-style propensity estimator but cannot correct for spatial interference (details in Appendix D). We also test ablations of GST-UNet without curriculum or attention. Table 1 shows that when β 1 = 0 , UNet+ performs best-G-computation is unnecessary and adds noise. As β 1 increases, UNet+ and STCINet degrade sharply, while GST-UNet remains stable. IPWUNet shows some benefit but is biased even at β 1 = 0 due to uncorrected interference. GST-UNet consistently outperforms all baselines, demonstrating the value of iterative G-computation. Curriculum training substantially improves performance across horizons, while attention yields modest gains-consistent with our predominantly local dynamics. Additional ablation analyses, including neighbor aggregation experiments, are reported in Appendix D.

## 6.2 Impact of Wildfires on Respiratory Health

Wildfire smoke has been linked to short-term respiratory harms [10, 12, 23, 33, 34], with older adults especially vulnerable [14]. At the time this work was conducted (January 2025), a series of 14 destructive wildfires affected the Los Angeles metropolitan area and San Diego County in California, underscoring the urgency of understanding the health impacts of such events. In this study, we focus on a previous large-scale episode: the 2018 California wildfire season [45], which included the Carr Fire (July-August) and the Camp Fire (November) and significantly worsened air quality.

We use daily, county-level data from Letellier et al. [23] (see Appendix D.2), including PM 2 . 5 , respiratory/cardiovascular hospitalizations, and weather variables (temperature, precipitation, humidity, radiation, wind), along with population estimates from the California Department of Finance. Each of the weather variables can be a time-varying confounder : weather conditions affect future smoke levels and health outcomes, while also being influenced by prior smoke levels.

We focus on weeks 20-48 (May 18-December 2, 2018), covering the Carr and Camp fires. Following standard practice, we label a county as 'treated' on days with mean PM 2 . 5 &gt; 10 µg/m 3 and use raw hospitalization counts (rather than per-10,000 incidence, which can be unstable for small counties). We interpolate daily county-level data (treatment, outcome, five covariates) onto a 40 × 44 latitude-longitude grid, discarding cells outside California, yielding a spatiotemporal tensor of size 203 × 7 × 40 × 44 . Interpolation ensures each grid cell approximates the region it overlaps (areaweighted), enabling the model to capture spatial gradients in PM 2 . 5 , weather, and hospitalizations. We train GST-UNet with horizon τ = 10 , using the Carr Fire period (June-July) for validation, and generate counterfactual predictions for the Camp Fire peak, November 8-17. See Appendix D.2 for preprocessing and masking details.

Figure 3: (Left) Daily PM2.5 levels across California from May to December 2018, with red lines marking major wildfires. (Center) Counties exposed to average PM2.5 &gt; 10 µg/m 3 during the Camp Fire (red), origin county in dark red. (Right) Factual minus CAPO-predicted daily respiratory admissions during peak Camp Fire. Hashed areas indicate small-population counties ( &lt; 30 , 000 ).

<!-- image -->

Figure 3 (left) shows the rise in PM 2 . 5 during the mid-late 2018 wildfire season; (center) highlights counties with daily PM 2 . 5 &gt; 10 , µg/m 3 . Using GST-UNet, we estimate daily CAPOs had the Camp Fire not occurred (i.e., setting PM 2 . 5 ≤ 10 , µg/m 3 statewide). Figure 3 (right) compares these to factual daily incidence (hospitalizations per 10 , 000 residents). To reduce small-sample variability, we exclude counties with population below 30,000 (vs. &gt; 70,000 for others), marking them with hatching (see Appendix D.2). Over November 8-17, GST-UNet predicts approximately 4,650 excess respiratory hospitalizations (465/day) attributable to the Camp Fire, with the highest incidence near the fire source. This aligns with a 95% bootstrap confidence interval of [1888 , 6535] . UNet+ yields a lower mean and higher uncertainty (3,981; [ -899 , 5202] ), STCINet produces highly variable near-zero estimates (88; [ -3077 , 3281] ), and IPWUNet gives implausibly high, near-constant values ( ∼ 20,500), reflecting limitations of weighting under rare-event support. These results underscore GSTUNet's improved stability and accuracy in counterfactual estimation. Our findings are qualitatively consistent with Letellier et al. [23], who report 259 excess daily cases averaged over a longer, lowerintensity window (Nov 8-Dec 5). Overall, the GST-UNet captures spatiotemporal variation in smoke exposure and health outcomes, illustrating its promise for real-world causal inference in domains such as environmental health and policy.

## 7 Conclusion

We presented GST-UNet , a neural framework for spatiotemporal causal inference that combines U-Net-based representation learning with iterative G-computation to adjust for time-varying confounders. GST-UNet addresses key challenges such as interference, spatial confounding, temporal carryover, and time-varying feedback. We establish theoretical identification and consistency guarantees, validate performance in synthetic settings with controlled confounding, and demonstrate practical utility in estimating the impact of wildfire smoke exposure during the 2018 Camp Fire. Together, these results position GST-UNet as a ready-to-use tool for practitioners , offering reliable, interpretable causal estimates in complex spatiotemporal environments. We discuss limitations and broader impacts in Appendix E.

## Acknowledgements

We thank the anonymous reviewers for their thoughtful feedback and the constructive dialogue during the review process, which greatly strengthened the final version of this work. Miruna Oprescu and BNL team (D. Park, X. Luo, S. Yoo) were supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research, under Awards DE-SC0023112 and DE-SC0012704, respectively. Nathan Kallus was supported by the U.S. National Science Foundation under Grant No. 1846210. Part of this work was conducted while Miruna Oprescu was a research intern at Brookhaven National Laboratory. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the U.S. Department of Energy or the U.S. National Science Foundation.

## References

- [1] S. Ali, O. Faruque, and J. Wang. Estimating direct and indirect causal effects of spatiotemporal interventions in presence of spatial interference. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases , pages 213-230. Springer, 2024.
- [2] L. Anselin. Spatial econometrics: methods and models , volume 4. Springer Science &amp; Business Media, 2013.
- [3] H. Bang and J. M. Robins. Doubly robust estimation in missing data and causal inference models. Biometrics , 61(4):962-973, 2005.
- [4] E. Ben-Michael, A. Feller, and J. Rothstein. Synthetic controls with staggered adoption. Journal of the Royal Statistical Society Series B: Statistical Methodology , 84(2):351-381, 2022.
- [5] Y. Bengio, J. Louradour, R. Collobert, and J. Weston. Curriculum learning. In Proceedings of the 26th annual international conference on machine learning , pages 41-48, 2009.
- [6] G. Bertasius, H. Wang, and L. Torresani. Is space-time attention all you need for video understanding? In ICML , volume 2, page 4, 2021.
- [7] I. Bica, A. M. Alaa, J. Jordon, and M. van der Schaar. Estimating counterfactual treatment outcomes over time through adversarially balanced representations. arXiv preprint arXiv:2002.04083 , 2020.
- [8] I. Bojinov and N. Shephard. Time series experiments and causal estimands: exact randomization tests and trading. Journal of the American Statistical Association , 2019.
- [9] R. Caruana. Multitask learning. Machine learning , 28:41-75, 1997.
- [10] W. E. Cascio. Wildland fire smoke and human health. Science of the total environment , 624: 586-595, 2018.
- [11] R. Christiansen, M. Baumann, T. Kuemmerle, M. D. Mahecha, and J. Peters. Toward causal inference for spatio-temporal data: Conflict and forest loss in Colombia. Journal of the American Statistical Association , 117(538):591-601, 2022.
- [12] S. E. Cleland, M. L. Serre, A. G. Rappold, and J. J. West. Estimating the acute health impacts of fire-originated PM2.5 exposure during the 2017 California wildfires: Sensitivity to choices of inputs. Geohealth , 5(7):e2021GH000414, 2021.
- [13] A. Coston, E. Kennedy, and A. Chouldechova. Counterfactual predictions under runtime confounding. Advances in neural information processing systems , 33:4150-4162, 2020.
- [14] S. DeFlorio-Barker, J. Crooks, J. Reyes, and A. G. Rappold. Cardiopulmonary effects of fine particulate matter exposure among older adults, during wildfire and non-wildfire periods, in the United States 2008-2010. Environmental health perspectives , 127(3):037006, 2019.
- [15] T. Evgeniou and M. Pontil. Regularized multi-task learning. In Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining , pages 109-117, 2004.
- [16] D. Frauen, K. Hess, and S. Feuerriegel. Model-agnostic meta-learners for estimating heterogeneous treatment effects over time. arXiv preprint arXiv:2407.05287 , 2024.
- [17] X. Glorot and Y. Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS) , pages 249-256. JMLR Workshop and Conference Proceedings, 2010.
- [18] K. Hess, D. Frauen, V. Melnychuk, and S. Feuerriegel. G-transformer for conditional average potential outcome estimation over time. arXiv preprint arXiv:2405.21012 , 2024.

- [19] K. Jordahl, J. V . den Bossche, M. Fleischmann, J. Wasserman, J. McBride, J. Gerard, J. Tratner, M. Perry, A. G. Badaracco, C. Farmer, G. A. Hjelle, A. D. Snow, M. Cochran, S. Gillies, L. Culbertson, M. Bartos, N. Eubank, maxalbert, A. Bilogur, S. Rey, C. Ren, D. ArribasBel, L. Wasser, L. J. Wolf, M. Journois, J. Wilson, A. Greenhall, C. Holdgraf, Filipe, and F. Leblanc. geopandas/geopandas: v0.8.1, July 2020. URL https://doi.org/10.5281/ zenodo.3946761 .
- [20] L. J. Keele and R. Titiunik. Geographic boundaries as regression discontinuities. Political Analysis , 23(1):127-155, 2015.
- [21] D. P. Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 , 2014.
- [22] M. J. Laan and J. M. Robins. Unified methods for censored longitudinal data and causality . Springer, 2003.
- [23] N. Letellier, M. Hale, K. U. Salim, Y . Ma, F. Rerolle, L. Schwarz, and T. Benmarhnia. Applying a two-stage generalized synthetic control approach to quantify the heterogeneous health effects of extreme weather events: A 2018 large wildfire in California event as a case study. Environmental Epidemiology , 9(1):e362, 2025.
- [24] R. Li, S. Hu, M. Lu, Y. Utsumi, P. Chakraborty, D. M. Sow, P. Madan, J. Li, M. Ghalwash, Z. Shahn, et al. G-net: a recurrent network approach to g-computation for counterfactual prediction under a dynamic treatment regime. In Machine Learning for Health , pages 282-299. PMLR, 2021.
- [25] Y. Li, R. Yu, C. Shahabi, and Y. Liu. Diffusion convolutional recurrent neural network: Datadriven traffic forecasting. arXiv preprint arXiv:1707.01926 , 2017.
- [26] B. Lim. Forecasting treatment responses over time using recurrent marginal structural networks. Advances in neural information processing systems , 31, 2018.
- [27] Z. Liu, J. Ning, Y. Cao, Y. Wei, Z. Zhang, S. Lin, and H. Hu. Video swin transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 3202-3211, 2022.
- [28] V. Melnychuk, D. Frauen, and S. Feuerriegel. Causal transformer for estimating counterfactual outcomes. In International Conference on Machine Learning , pages 15293-15329. PMLR, 2022.
- [29] O. Oktay, J. Schlemper, L. L. Folgoc, M. Lee, M. Heinrich, K. Misawa, K. Mori, S. McDonagh, N. Y. Hammerla, B. Kainz, et al. Attention u-net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999 , 2018.
- [30] G. Papadogeorgou, F. Mealli, and C. M. Zigler. Causal inference with interfering units for cluster and population level treatment allocation programs. Biometrics , 75(3):778-787, 2019.
- [31] G. Papadogeorgou, K. Imai, J. Lyall, and F. Li. Causal inference with spatio-temporal data: estimating the effects of airstrikes on insurgent violence in iraq. Journal of the Royal Statistical Society Series B: Statistical Methodology , 84(5):1969-1999, 2022.
- [32] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala. Pytorch: An imperative style, high-performance deep learning library. NeurIPS, 2019.
- [33] C. E. Reid, M. Brauer, F. H. Johnston, M. Jerrett, J. R. Balmes, and C. T. Elliott. Critical review of health impacts of wildfire smoke exposure. Environmental health perspectives , 124 (9):1334-1343, 2016.
- [34] C. E. Reid, M. Jerrett, I. B. Tager, M. L. Petersen, J. K. Mann, and J. R. Balmes. Differential respiratory health effects from the 2008 northern California wildfires: A spatiotemporal approach. Environmental research , 150:227-235, 2016.

- [35] J. Robins and M. Hernan. Estimation of the causal effects of time-varying exposures. Chapman &amp;Hall/CRC Handbooks of Modern Statistical Methods , pages 553-599, 2008.
- [36] J. M. Robins, M. A. Hernan, and B. Brumback. Marginal structural models and causal inference in epidemiology, 2000.
- [37] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention-MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18 , pages 234-241. Springer, 2015.
- [38] D. B. Rubin. Bayesian inference for causal effects: The role of randomization. The Annals of statistics , pages 34-58, 1978.
- [39] N. Seedat, F. Imrie, A. Bellot, Z. Qian, and M. van der Schaar. Continuous-time modeling of counterfactual outcomes using neural controlled differential equations. arXiv preprint arXiv:2206.08311 , 2022.
- [40] X. Shi, Z. Chen, H. Wang, D.-Y. Yeung, W.-K. Wong, and W.-c. Woo. Convolutional lstm network: A machine learning approach for precipitation nowcasting. Advances in neural information processing systems , 28, 2015.
- [41] C. Song, Y. Wang, X. Yang, Y. Yang, Z. Tang, X. Wang, and J. Pan. Spatial and temporal impacts of socioeconomic and environmental factors on healthcare resources: a county-level bayesian local spatiotemporal regression modeling study of hospital beds in southwest china. International Journal of Environmental Research and Public Health , 17(16):5890, 2020.
- [42] M. Tec, J. G. Scott, and C. M. Zigler. Weather2vec: Representation learning for causal inference with non-local confounding in air pollution and climate studies. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 37, pages 14504-14513, 2023.
- [43] A. W. Van Der Vaart, J. A. Wellner, A. W. van der Vaart, and J. A. Wellner. Weak convergence . Springer, 1996.
- [44] Y. Wang. Causal inference under temporal and spatial interference. arXiv e-prints , pages arXiv-2106, 2021.
- [45] Wikipedia. Camp Fire (2018) - Wikipedia, the free encyclopedia. http://en.wikipedia. org/w/index.php?title=Camp%20Fire%20(2018)&amp;oldid=1271689743 , 2025. [Online; accessed 29-January-2025].
- [46] Z. Wu, S. Pan, G. Long, J. Jiang, and C. Zhang. Graph wavenet for deep spatial-temporal graph modeling. arXiv preprint arXiv:1906.00121 , 2019.
- [47] J. Zhang, Y. Zheng, and D. Qi. Deep spatio-temporal residual networks for citywide crowd flows prediction. In Proceedings of the AAAI conference on artificial intelligence , volume 31, 2017.
- [48] W. Zhang and K. Ning. Spatiotemporal heterogeneities in the causal effects of mobility intervention policies during the covid-19 outbreak: A spatially interrupted time-series (sits) analysis. Annals of the American Association of Geographers , 113(5):1112-1134, 2023.
- [49] L. Zhou, K. Imai, J. Lyall, and G. Papadogeorgou. Estimating heterogeneous treatment effects for spatio-temporal causal inference: How economic assistance moderates the effects of airstrikes on insurgent violence. arXiv preprint arXiv:2412.15128 , 2024.

Table 2: Key differences between prior neural G-computation methods and GST-UNet.

| Aspect         | Prior Neural G-Computation                                                                                                  | GST-UNet (ours)                                                                                                                                       |
|----------------|-----------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| Data structure | Many independent temporal trajectories (e.g., patient sequences); no inter-unit interactions such as spatial dependence.    | Single spatiotemporal chain where out- comes, covariates, and treatments evolve jointly across a lattice; strong spatial cou- pling and interference. |
| Encoder        | RNN/Transformer over time only.                                                                                             | ConvLSTM-UNet encoder aggregates neighbour covariates/treatments before G-heads, capturing interference and spa- tial confounding.                    |
| Training       | Standard end-to-end due to i.i.d trajec- tories; stability arises from large data rather than curriculum or spatial priors. | Curriculum-stabilized multi-head train- ing for accurate pseudo-outcome genera- tion under limited samples.                                           |
| Theory         | Classical G-formula under i.i.d. trajecto- ries; no single-chain guarantees.                                                | Identification (Theorem 1) and consis- tency (Theorem 3) under representation- based time-invariance for a single chain.                              |

## A Extended Literature Review

Classical Spatiotemporal Causal Inference. Early spatiotemporal causal inference methodsincluding spatial econometrics [2], difference-in-differences [20], and synthetic controls [4]-provide useful frameworks for estimating treatment effects across regions but rely on strong assumptions such as parallel trends or stable treatment assignment. These approaches struggle with interference, nonlinear dependencies, and time-varying confounders, limiting their applicability in complex settings. More recent approaches for spatiotemporal causal inference handle time-varying confounding through inverse propensity weighting (IPW), typically by extending marginal structural models to the spatial or spatiotemporal domain. For instance, Papadogeorgou et al. [31] and Zhou et al. [49] employ IPWstyle adjustments to estimate regional average treatment effects across space and time. However, these approaches cannot accommodate interference unless strong assumptions are made-e.g., defining a user-specified exposure mapping or restricting attention to hyper-local interactions (see also [11, 31, 44, 48]). Such simplifications may be ill-suited for real-world systems with rich spatial dependencies. Moreover, even recent advances in this space remain limited; as noted by Zhou et al. [49], the literature on spatiotemporal causal inference remains sparse, especially in settings with feedback loops or time-varying confounding.

Machine Learning for Spatiotemporal Modeling. Spatiotemporal predictive modeling has seen rapid progress with the rise of deep learning. Convolutional and recurrent neural networks are widely used for forecasting spatially indexed time series (e.g., weather or traffic) [40, 47], while graph-based methods (e.g., Graph WaveNet [46], Diffusion Convolutional RNN [25]) capture non-Euclidean spatial dependencies. Vision transformer variants, including Video Swin Transformers [27] and TimeSformer [6], extend attention-based models to spatiotemporal video data. These architectures can learn complex non-local interactions over space and time. However, such models are typically optimized for prediction tasks and do not include causal adjustments. Without mechanisms like propensity modeling or G-computation, they remain ill-equipped to estimate counterfactual outcomes or adjust for time-varying confounding. Some recent work integrates spatial representations for causal inference-e.g., Tec et al. [42] incorporate non-local confounders using a UNet-based model-but these methods do not explicitly model dependencies over time or adjust for time-varying confounders.

Time-Series Causal Inference. In the longitudinal domain, time-series causal inference has developed tools for handling temporal confounding using models such as marginal structural models [36], IPW-style estimation [26], and iterative G-computation [35]. Recent ML-based extensions include recurrent networks [7, 24, 39], Transformers [18, 28] and meta-learners [16]. However, all these methods assume access to independent time series-e.g., across units or patients-which allows for pooling across trajectories. These methods do not consider spatial dependencies, interference, or scenarios with a single observed spatiotemporal realization. As such, while they may handle time-varying confounding, they do not generalize to our setting. Table 2 summarizes the key methodological differences between GST-UNet and prior neural G-computation frameworks.

Neural-Based Spatiotemporal Causal Inference. There has been limited work on neural models that explicitly address spatiotemporal causal inference. Tec et al. [42] use a U-Net backbone to learn spatial representations for causal inference in air pollution studies but do not address time-varying confounding or feedback loops. Ali et al. [1] present a U-Net-based architecture for predicting direct and indirect effects in climate contexts, but primarily focus on forecasting rather than causal identification. While these works highlight growing interest in neural approaches to causal inference in spatiotemporal domains, none incorporate an iterative adjustment procedure like G-computation that handles time-varying confounders, leaving identification in these settings largely unaddressed.

Our Contribution. GST-UNet bridges these gaps by combining flexible spatiotemporal neural architectures with a theoretically grounded iterative G-computation framework. This allows valid estimation of potential outcomes in the presence of interference, spatial confounding, and time-varying confounding-without requiring practitioners to specify structural models or exposure mappings. To our knowledge, this is the first end-to-end framework to implement G-computation for causal inference over a single spatiotemporal trajectory. We integrate spatiotemporal processing via U-Nets and ConvLSTMs with a principled multi-head neural causal module, and we design a curriculum-based training strategy to stabilize learning of recursive pseudo-outcomes. Together, these components yield a ready-to-use tool for practitioners, with consistent identification guarantees and robust empirical performance. By abstracting away the modeling choices typically required in structural spatiotemporal methods, GST-UNet makes spatiotemporal causal estimation more accessible, interpretable, and reliable for real-world applications.

## B Proof of Theorem 1

We aim to show that under Assumption 1 and Assumption 2, the CAPOs in Equation (1) can be identified recursively from a single time series via a sequence of conditional expectations.

Step 1: Recursive decomposition for the intractable expectation We first demonstrate the recursive decomposition of the intractable expectation in the CAPO definition (Equation (1)). While this expectation is theoretically well-defined, it cannot be directly estimated in practice due to the limited availability of data. Specifically, we only observe a single time series, meaning we have just one sample of the history at time t + τ for each t . Nevertheless, as we will show, we can convert these expectations into expectations over prefix-based segments that allow us to estimate these quantities from the data.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, if we had multiple spatiotemporal time-series samples, we could directly estimate this nested expression from data, since the right-hand side depends solely on observed quantities, ensuring identifiability.

Step 2: From intractable to prefix-based expectations We now show how to estimate the nested expectations using the prefix data. First, by Assumption 2, we can rewrite the inner-most expectation as

<!-- formula-not-decoded -->

Thus, by using Assumption 1, we can write this expectation over the prefix data which we have many samples of. Now consider the next nested expectation:

<!-- formula-not-decoded -->

Tracing this argument recursively through the nested expectation in Step 1, we obtain:

<!-- formula-not-decoded -->

as desired. Thus, Q 1 - which can be estimated from the prefix data - recovers the CAPOs, under our assumptions, even from a single chain.

## C Consistency of the Iterative G-Computation Estimator

In this section, we state the conditions under which the iterative G-computation procedure in Section 4.2 yields a consistent estimator, and show that our implementation of the Q k estimators satisfies these conditions.

Notation: We denote the L 2 norm of a function f as ∥ f ∥ 2 := E P [ f ( X ) 2 ] 1 / 2 , where the expectation is over the probability distribution P . The notation ̂ f n represents the estimated value of a parameter or function learned on n data points, where f is the true value. For a sequence of random variables { Z n } n ≥ 1 we write Z n = o p (1) if Pr( | Z n | &gt; ε ) → 0 for every ε &gt; 0 , i.e. Z n p - → 0 .

To begin, we introduce the following stochastic equicontinuity condition from [43]:

Definition 1 (Stochastic equicontinuity [43, Def. 1.5.7]) . Let ( Z , d ) be a semi-metric space and { ̂ f n } n ≥ 1 ⊂ ℓ ∞ ( Z ) a sequence of random functions. It is uniformly stochastically equi-continuous if, for every ϵ &gt; 0 , η &gt; 0 , there exists a δ &gt; 0 such that

<!-- formula-not-decoded -->

Stochastic equicontinuity ensures that, with high probability, each estimator changes only slightly when its input is perturbed by a small amount. It is strictly weaker than global Lipschitz continuity any family that is Lipschitz on a bounded domain with constants bounded in probability automatically satisfies Definition 1. We impose this condition in Theorem 3 so that the o p (1) error in the learned embedding propagates to only o p (1) errors in the G-heads, making the recursive estimator consistent.

The following theorem restates Theorem 2 from the main text in full detail and provides its proof.

Theorem 3 (Consistency under Uniform Stochastic Equicontinuity) . Suppose the conditions of Theorem 1 hold, and let ̂ ϕ be a learned embedding. Define Z k := ( H 1: t + k , A t + k ) , and recursively define the learned estimators ̂ Q k ( Z k -1 ; ̂ ϕ ) := ̂ E P [ ̂ Q k +1 ( Z k ; ̂ ϕ ) | ̂ ϕ ( Z k )] for k = 1 , . . . , τ , with terminal condition ̂ Q τ +1 ( Z τ ; ̂ ϕ ) = Y t + τ . Assume that { ̂ Q k } τ k =1 are obtained via the iterative G-computation algorithm. If:

- (i) ∥ ̂ ϕ -ϕ ∥ 2 = o p (1) ;
- (ii) ∥ ̂ Q k ( Z k -1 ; ϕ ) -Q k ( Z k -1 ; ϕ ) ) ∥ 2 = o p (1) for all k ;
- (iii) for every k the random maps z ↦→ ̂ Q k ( h, a ; z ) are stochastically equicontinuous on Im ϕ (Definition 1), and Q k ( · ) is uniformly continuous there,

then

We now consider

<!-- formula-not-decoded -->

Again, decompose:

<!-- formula-not-decoded -->

The second term is o p (1) by assumption (ii). The first term is also o p (1) because ̂ ϕ → ϕ in L 2 and the stochastic equicontinuity of ̂ Q k ensures that perturbations in ϕ yield small changes in predictions uniformly over Im ϕ . Thus,

<!-- formula-not-decoded -->

By induction, the result holds for all k = τ, τ -1 , . . . , 1 , and in particular:

<!-- formula-not-decoded -->

Thus, the proof is now complete.

<!-- formula-not-decoded -->

Thus the recursive G-computation estimator is (probabilistically) consistent.

Proof. We proceed by reverse induction on k , starting from k = τ and working backward to k = 1 . For each k , we aim to show:

<!-- formula-not-decoded -->

Base case ( k = τ ). By definition, ̂ Q τ +1 ( Z τ ; ̂ ϕ ) = Y t + τ , which is observed. Thus,

<!-- formula-not-decoded -->

We decompose the difference:

<!-- formula-not-decoded -->

Term Λ 2 is o p (1) by assumption (ii). Term Λ 1 converges to zero in probability due to (i) ∥ ̂ ϕ -ϕ ∥ 2 = o p (1) and (iii) stochastic equicontinuity of ̂ Q τ . Therefore,

<!-- formula-not-decoded -->

Inductive step. Suppose for some k +1 ≤ τ that

<!-- formula-not-decoded -->

Example 1 (Feed-forward or convolutional heads) . Suppose each G-computation head Q k ( · ; z ) is implemented as a depthd neural network

<!-- formula-not-decoded -->

where the activations σ ℓ are Lipschitz continuous (e.g., ReLU, Leaky ReLU, SoftPlus, Tanh, Sigmoid, or ArcTan). If each layer weight satisfies a spectral norm bound ∥ W ℓ ∥ 2 ≤ ρ ℓ &lt; ∞ , then Ψ is globally Lipschitz on R h with constant L = ∏ ℓ ρ ℓ , and thus uniformly continuous on any compact subset. This implies the stochastic equicontinuity condition in Definition 1.

In practice, norm control can be enforced via weight decay, spectral normalization, or weight clipping during training. Similarly, the encoder output ̂ ϕ ( H,A ) can be bounded-e.g., through normalization or clipping-so its image lies in a compact subset of R h . Together, these ensure the continuity and equicontinuity conditions required by Theorem 3.

The same argument applies to convolutional networks, since 2-D convolutions are linear operators whose induced matrix representations also admit spectral norm bounds controlled via spectral normalization.

## D Experimental Details

In this appendix, we provide further information on the simulation experiments (Section 6.1) and the real-world wildfire application (Section 6.2), including exact parameter settings, model architecture and execution details, hyperparameter selection strategies, and validation procedures. All code for generating, preprocessing, and analyzing both the synthetic and real-world datasets-and for training and evaluating GST-UNet-is available at https://github.com/moprescu/GSTUNet , with step-by-step replication instructions in the repository's README.md .

For both applications, GST-UNet employs a U-Net backbone with a single ConvLSTM layer (hidden dimension 32) and a contracting-expanding path of channel sizes 16 → 32 → 64 → 128 → 256 . The G-computation heads are implemented as shallow feed-forward neural networks that operate on the U-Net feature maps at each grid cell for G-computation. In practice, to ensure stable ConvLSTM training and reduce computational overhead, we truncate the input history to a fixed length. All neural networks are implemented via the nn module in PyTorch [32]. Experiments were conducted on an NVIDIA A100 (Ampere) GPU using the Perlmutter system at the National Energy Research Scientific Computing Center (NERSC). The synthetic experiments required roughly 55 minutes per hyperparameter set, while the wildfire experiment completed in about 5 minutes.

## D.1 Synthetic Experiments

Data Simulation Process. For our primary simulation experiments, we generate T = 200 time steps on a 64 × 64 grid. The simulation parameters in the generating equations

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

are given by:

· X t :

<!-- formula-not-decoded -->

where K X influences how X diffuses across neighboring cells, with an asymmetry due to advection.

Figure 4: Samples from the DGP at t = 100 , comparing feature X 100 (left), intervention A 100 (center), and outcome Y 101 (right) for varying β 1 ∈ { 0 . 0 , 1 . 0 , 2 . 0 } .

<!-- image -->

Table 3: Hyperparameters and their ranges. We boldface the values that provided the best validation performance.

| Hyperparameter                                                                                                                                                      | Model(s)                                                                                 | Value Range                                                                                                                                        |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| Batch size Learning rate Scheduler patience Early stopping patience Curriculum period Curriculum learning rate UNet output dim d h G-head hidden size G-head layers | All models All models All models All models GST-UNet GST-UNet GST-UNet GST-UNet GST-UNet | {2, 4 , 8} { 10 - 4 , 5 × 10 - 4 , 10 - 3 } {3, 5 , 10} {5, 10 } {1, 3 , 5, 7} { 10 - 4 , 5 × 10 - 4 , 10 - 3 } {8, 16 , 32} { 8 , 16} { 1 , 2, 3} |

## · A t :

## · Y t :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We use L = 5 temporal lags for X and Y , a seed of 42 for reproducibility. The parameter values were chosen such that the simulation remains stable ( i.e. , the process does not diverge). See Figure 4 for representative t = 100 snapshots of X 100 , A 100 , and Y 101 under varying β 1 .

For each β 1 , we first generate a factual dataset of length T = 200 ( i.e. , { ( X t , A t , Y t ) } 200 t =1 ). We then create n test = 50 test histories of length l H = 10 . For each test history, we simulate 100 trajectories

<!-- formula-not-decoded -->

Table 4: Ablation on spatial kernel size ( τ = 5 ). Removing neighbor aggregation ( 1 × 1 kernel) degrades performance, confirming the need to model spatial spill-overs.

| Kernel size   | β 1 =0 . 0   | 0 . 5        | 1 . 0        | 1 . 5        | 2 . 0        |
|---------------|--------------|--------------|--------------|--------------|--------------|
| 3 × 3         | 0.33 ± 0.004 | 0.35 ± 0.004 | 0.40 ± 0.005 | 0.44 ± 0.004 | 0.40 ± 0.005 |
| 1 × 1         | 0.53 ± 0.004 | 0.55 ± 0.005 | 0.54 ± 0.005 | 0.60 ± 0.007 | 0.64 ± 0.006 |

Table 5: Effect of increasing T on RMSE for β 1 = 2 . 0 . GST-UNet improves with more data, while baselines remain biased.

| Model    |   T=100 |   T=200 |   T=400 |   T=600 |   T=800 |
|----------|---------|---------|---------|---------|---------|
| UNet+    |    0.78 |    0.81 |    0.82 |    0.95 |    0.87 |
| STCINet  |    0.8  |    0.9  |    1.04 |    1.02 |    0.91 |
| GST-UNet |    0.69 |    0.4  |    0.32 |    0.32 |    0.36 |

under a randomly chosen (yet fixed over the test data) counterfactual intervention of length τ = 10 , and average the outcomes at each step to approximate the true CAPOs. This procedure yields a final test set of shape n test × ( l H + τ +1) × 64 × 64 , i.e. , 50 × 21 × 64 × 64 .

Neural Architectures. The GST-UNet comprises a single ConvLSTM layer (hidden dimension 32 ), followed by a U-Net with channel sizes 16 → 32 → 64 → 128 → 256 . Its G-computation heads are shallow feed-forward networks operating on the final U-Net feature maps at each grid cell; both the U-Net's output dimension ( d h ) and the G-head architecture (number of layers, hidden size) are treated as hyperparameters. The UNet+ baseline uses the same ConvLSTM+U-Net backbone as GST-UNet but outputs a single channel ( d h = 1 ), omitting any G-computation. For direct comparison, we also implement STCINet [1] with an identical ConvLSTM+U-Net backbone, and retaining their original Latent Factor Model (LFM) details.

IPWUNet Baseline. We adapt the Inverse Propensity Weighting (IPW) estimator from [49] to the spatiotemporal setting. Given estimated propensities ˆ π ( a l | H 1: l ) , the estimator is defined as:

<!-- formula-not-decoded -->

We implement the IPWUNet baseline by reusing the UNet+ architecture (U-Net + ConvLSTM + Attention) for both propensity estimation and outcome prediction. Specifically, we first train the propensity model with a binary cross-entropy loss to estimate ˆ π ( A t | H t ) at each time t . We then freeze this model and use the estimated weights to train a second instance of the same architecture with a weighted MSE loss, where pseudo-outcomes are reweighted by the estimated inverse propensities along the counterfactual treatment path. While this allows partial adjustment for time-varying confounding, the method does not correct for spatial interference and is sensitive to small propensity values, which can lead to high variance.

Training Details. We randomly initialize all model parameters (GST-UNet and baselines) with Xavier uniform weights [17]. We use the Adam optimizer [21] with an initial learning rate, halving it whenever the validation loss plateaus for a specified scheduler patience. To mitigate overfitting, we adopt early stopping when the validation loss fails to improve for a specified early stopping patience epochs. Validation uses 40 of the 190 training prefixes, and the total training is capped at 100 epochs. We tune the following hyperparameters: (i) batch size, learning rate, scheduler patience, and early stopping patience (common to all models); (ii) for GST-UNet, the curriculum period and learning rate for curriculum phases, the U-Net output dimension d h , and the number and width of hidden layers in the feed-forward G-heads. Table 3 lists the hyperparameter ranges considered, with the values yielding the best validation performance in bold .

Evaluation Procedure. We evaluate each model by averaging the root mean square error (RMSE) of the estimated CAPOs against ground truth across 50 test trajectories. Table 1 in the main text reports RMSE ± standard deviation for horizon lengths τ ∈ { 5 , 10 } and β 1 ∈ { 0 , 0 . 5 , 1 . 0 , 1 . 5 , 2 . 0 } .

Effect of Varying T . We ran additional simulations varying the trajectory length T ∈ { 100 , 200 , 400 , 600 , 800 } and β 1 = 2 . 0 while keeping the grid size fixed ( d x = d y = 64 ). Results are shown Appendix D.1. GST-UNet consistently improves with more data, while the baselines

Figure 5: (Left) Daily respiratory illness incidence (cases per 10,000). (Center) Weekly aggregated incidence. (Right) Average daily PM2.5 during the Camp Fire.

<!-- image -->

Figure 6: An example of county-level ( left ) vs. grid-interpolated ( right ) PM2.5 levels on November 18 (during the Camp Fire). The grid interpolation produces a 40 × 44 lattice of area-weighted estimates aligned with our spatiotemporal framework.

<!-- image -->

remain biased-even as T increases. This highlights the importance of adjusting for time-varying confounding: without it, there is a persistent asymptotic bias.

Effect of Neighbor Aggregation. To evaluate the importance of spatial spill-over modeling, we ablate the convolutional kernel used in the ConvLSTM encoder. Table 4 compares GST-UNet with a standard 3 × 3 kernel against a variant that removes neighbor aggregation by using a 1 × 1 kernel. Across all levels of confounding strength ( β 1 ), performance deteriorates markedly when neighbor information is excluded, with RMSE increasing by 30-40%. This confirms that explicitly aggregating information from nearby locations is essential for capturing spatial interference and achieving unbiased counterfactual estimates.

## D.2 Wildfire Application

Data Preprocessing and Interpolation. We analyze daily, county-level data from Letellier et al. [23] that include PM 2 . 5 (particulate matter &lt; 2 . 5 µm ), hospitalization counts for respiratory and cardiovascular conditions, and weather variables (temperature, precipitation, humidity, radiation, wind), plus population estimates from the California Department of Finance. Our study period spans weeks 20-48 (May 18-December 2, 2018), covering both the Carr and Camp fires. As illustrated in Figure 5, daily and weekly aggregated respiratory illness rates rise around these events, while PM2.5 levels also surge during the Camp Fire.

To align with our spatiotemporal framework, we use geopandas [19] to interpolate county-level covariates, PM 2 . 5 , and hospitalizations onto a latitude-longitude grid from 32 ◦ N to 42 ◦ N latitude and -125 ◦ to -114 ◦ longitude, at a resolution of 0 . 25 ◦ . Each grid cell's values are an area-weighted average of the counties it intersects, yielding a 40 × 44 spatial lattice. We mask out non-California cells by setting them to zero, thus obtaining a consistent dataset for further analysis. Figure 6 illustrates how the raw county-level data compare to the interpolated grid for PM2.5 on November 18.

Model Training and Validation. We train GST-UNet, UNet+, STCINet, and IPWUNet with a prediction horizon of τ = 10 days. All models use a shared set of hyperparameters: batch size = 4 ,

Table 6: Estimated county-level increases in respiratory ED visits attributable to the wildfire event, with 95% bootstrap confidence intervals. Population is reported in units of 10,000. Counties marked with * have smaller populations, which may lead to greater uncertainty.

| County     |   Mean |   2.5% |   97.5% |   Population ( × 10 4 ) |   Interval Width / Population |
|------------|--------|--------|---------|-------------------------|-------------------------------|
| Tehama     |     37 |   -126 |     158 |                     6.4 |                          44.4 |
| Butte      |    168 |     30 |     325 |                    23   |                          12.8 |
| Glenn*     |    -52 |   -262 |      39 |                     2.8 |                         107.6 |
| Colusa*    |     13 |   -158 |     107 |                     2.1 |                         124   |
| Sutter     |    -18 |   -170 |      70 |                     9.6 |                          24.9 |
| Napa       |     81 |    -41 |     192 |                    13.9 |                          16.8 |
| Lake       |    103 |    -66 |     203 |                     6.4 |                          41.8 |
| Solano     |     38 |    -79 |     173 |                    44.6 |                           5.6 |
| Sacramento |    202 |   -107 |     484 |                   153.9 |                           3.8 |

learning rate = 5 × 10 -4 , scheduler patience = 5 , early stopping patience = 10 , and a curriculum period = 5 (with curriculum learning rate = 5 × 10 -4 ). For GST-UNet, we additionally set the U-Net output dimension to d h = 16 , the G-head hidden layer size to 8 , and the number of G-head layers to 1 . We optimize a mean squared error (MSE) loss with two adjustments: (1) we mask grid cells outside California to exclude them from the loss computation, and (2) we apply cell-specific weights proportional to the number of grid cells per county to avoid bias toward geographically larger counties. Hyperparameter tuning and validation are performed using data from the first 50 days of the wildfire season. Using the selected configuration, we generate counterfactual predictions for the Camp Fire peak period (November 8-17) by iteratively applying each trained model with increasing history lengths. We note that counties with populations below 20 , 000 -30 , 000 can yield unreliable incidence rate estimates (baseline daily rates of approximately 4 cases per 10 , 000 individuals); in Figure 3, we denote these high-uncertainty counties with hatched markings.

Bootstrap Confidence Intervals. We compute 95% bootstrap confidence intervals for all models using n = 40 bootstrap samples, balancing statistical rigor with computational load. Counties with populations below 20 , 000 -30 , 000 tend to yield unstable incidence rate estimates, driven by low baseline daily counts (approximately 4 cases per 10,000), and are excluded from the analysis. These counties are indicated with hatching in Figure 3, a choice further supported by the bootstrap results. In Table 6, we report bootstrap intervals for the counties closest to the Camp Fire. Glenn and Colusa exhibit disproportionately wide intervals-reflecting the uncertainty introduced by their small population sizes-and this further justifies their exclusion from the final analysis.

## E Limitations and Broader Impacts

Limitations. While GST-UNet demonstrates strong empirical performance and theoretical grounding, several limitations should be acknowledged. First, our method relies on standard causal identification assumptions, including no unobserved confounding (Assumption 1), which is inherently untestable and may not hold in all real-world settings. Second, our framework assumes the existence of a time-invariant representation of the spatiotemporal process (Assumption 2)-a useful but idealized condition that may be violated in domains with highly non-stationary or regime-shifting dynamics. Finally, GST-UNet is designed for gridded spatiotemporal data and assumes a regular spatial lattice; while this is common in environmental and health applications, adapting the framework to irregular spatial structures (e.g., graphs or administrative boundaries) is an important direction for future work.

Broader Impacts. This work advances machine learning by introducing a spatiotemporal causal inference framework for estimating treatment effects in complex real-world settings. The GST-UNet has broad applications in public health, environmental science, and social policy, where understanding interventions supports evidence-based decisions. For example, it can inform pollution control, wildfire response, or health resource allocation. However, like all observational methods, GST-UNet depends on the quality and completeness of the data, as well as the assumptions stated in this work. We caution against uncritical use in high-stakes settings, as violations of model assumptions or data biases can lead to misleading conclusions. We encourage responsible deployment-especially in contexts affecting vulnerable populations-and recommend pairing our method with domain expertise, sensitivity checks, and uncertainty quantification.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction state that the paper introduces GST-UNet, a neural framework for estimating causal effects in spatiotemporal settings with timevarying confounding. They accurately summarize the paper's three core contributions: the development of the framework with theoretical guarantees, empirical validation in synthetic settings, and a real-world application to wildfire smoke exposure. The claims align with both the theoretical results and empirical findings presented in the main text.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We present the limitations of our work in Appendix E.

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

Justification: We present two theorems, Theorem 1 and Theorem 3 which cover the identification and consistency guarantees of our estimator. The (complete and correct) proofs are

included in Appendix B and Appendix C, respectively. The assumptions are incorporated in Assumption 1, Assumption 2, as well as the text of the theorems.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes]

Justification: Section 6 and Appendix D provide the information necessary (including data generation processes, model choices, training and validation procedures, hyperparameter choices, etc.) to reproduce the main experimental results.

## Guidelines:

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

Justification: We provide the replication data and code at https://github.com/ moprescu/GSTUNet , along with instructions for reproducibility (see README.md document).

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

Justification: Section 6 and Appendix D.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We report the standard deviation for the synthetic experiments in Table 1 and boostrap CIs for the real-world case study in Section 6.2

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

Justification: See Appendix D.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have reviewed the ethics guidelines at https://neurips.cc/public/ EthicsGuidelines and confirm that our work adheres to them.

## Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: See Appendix E.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks,

mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The models and datasets used in this work do not pose significant risks for misuse. GST-UNet is a causal inference framework for scientific and policy applications using structured spatiotemporal data.

## Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All datasets and code used in this work are publicly available and appropriately cited (e.g., Letellier et al. [23] for wildfire data). License and usage terms are respected as per the original sources.

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

Justification: We release code and simulation scripts to reproduce all experiments, along with documentation and usage instructions (see Appendix D). All assets are anonymized for review and available at the provided URL.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.

- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing or research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve crowdsourcing or research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: LLMs were used solely for writing assistance and code debugging; they were not involved in the development or implementation of the core methodology.

## Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.