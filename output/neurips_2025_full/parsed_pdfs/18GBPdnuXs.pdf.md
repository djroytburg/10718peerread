## Adaptive Quantization in Generative Flow Networks for Probabilistic Sequential Prediction

Nadhir Hassen ∗

University of Adelaide, AIML and Mila - Quebec AI Institute

Johan Verjans

University of Adelaide, AIML

## Abstract

Probabilistic time series forecasting, essential in domains like healthcare and neuroscience, requires models capable of capturing uncertainty and intricate temporal dependencies. While deep learning has advanced forecasting, generating calibrated probability distributions over continuous future values remains challenging. We introduce Temporal Generative Flow Networks (Temporal GFNs), adapting Generative Flow Networks (GFNs) - a powerful framework for generating compositional objects - to this sequential prediction task. GFNs learn policies to construct objects (eg. forecast trajectories) step-by-step, sampling final objects proportionally to a reward signal. However, applying GFNs directly to continuous time series necessitates addressing their inherently discrete action spaces and ensuring differentiability. Our framework tackles this by representing time series segments as states and sequentially generating future values via quantized actions chosen by a forward policy. We introduce two key innovations: (1) An adaptive, curriculum-based quantization strategy that dynamically adjusts the number of discretization bins based on reward improvement and policy entropy, balancing precision and exploration throughout training. (2) A straight-through estimator mechanism enabling the forward policy to output both discrete (hard) samples for trajectory construction and continuous (soft) samples for stable gradient propagation. Training utilizes a trajectory balance loss objective, ensuring flow consistency, augmented by an entropy regularizer. We provide rigorous theoretical bounds on the quantization error's impact and the adaptive factor's range. We demonstrate how Temporal GFNs offer a principled way to leverage the structured generation capabilities of GFNs for probabilistic forecasting in continuous domains.

## 1 Introduction

Time series forecasting - predicting future values based on historical observations - is fundamental to informed decision-making across countless applications, from managing energy grids and financial portfolios to monitoring patient health and understanding brain activity [14, 7]. In many critical domains, particularly healthcare (e.g., forecasting vital signs from Electronic Health Records - EHRs) and neuroscience (e.g., predicting Electroencephalogram - EEG signals), obtaining not just a single point prediction but a full probabilistic forecast is paramount. Probabilistic forecasts quantify the inherent uncertainty in predictions, providing prediction intervals or complete density estimations, crucial for risk assessment and reliable decision support [10]. The advent of deep learning has revolutionized time series forecasting. Models based on Recurrent Neural Networks (RNNs) [30, 27]

∗ Correspondence to nadhir.hasseng@mila.quebec, nadhir.hassen@adelaide.edu.au

Zhen Zhang

University of Adelaide, AIML

and Transformers [20, 24, 32] can capture complex temporal patterns from large datasets. These models often achieve probabilistic forecasts by parameterizing a specific output distribution (e.g., Gaussian [30]) or predicting specific quantiles [34]. While effective, these approaches can impose potentially restrictive assumptions on the shape of the predictive distribution or suffer from issues like quantile crossing. More flexible generative models like Normalizing Flows [28] and Diffusion Models [19] have also been adapted, offering richer distributional modeling at potentially higher computational costs. Recently, Large Language Models (LLMs) have demonstrated remarkable zero-shot capabilities in various domains, including time series forecasting [12]. Frameworks like Chronos [1] show that standard transformer architectures, trained on time series tokenized via simple scaling and quantization, can achieve state-of-the-art results. This highlights the power of sequence modeling and the potential of mapping continuous time series to a discrete vocabulary. However, these methods typically rely on a fixed quantization scheme decided a priori. An alternative paradigm for structured generation is offered by Generative Flow Networks (GFNs) [3, 6]. GFNs learn policies to construct complex objects (like molecules or computational graphs) through a sequence of discrete actions in a state space. The key idea is to sample terminal objects (completed structures) with a probability proportional to a given reward function, effectively learning to navigate the state space towards high-reward configurations. This focus on sampling from a reward-modulated distribution makes GFNs naturally suited for tasks where we want to generate diverse, high-quality samples, rather than just finding a single optimal solution (as in reinforcement learning). This generative process seems well-aligned with time series forecasting: we can view a forecast trajectory as a compositional object built step-by-step, where the reward reflects the accuracy of the final forecast. However, GFNs traditionally operate in discrete state and action spaces. How can we adapt them to generate sequences of continuous time series values? Directly applying GFNs faces two primary hurdles: 1. Continuous Action Space: Time series values are continuous, but GFN policies typically output probabilities over discrete actions. 2. Differentiability: Selecting a discrete action (even if representing a continuous value via quantization) breaks the gradient flow required for training neural network policies.

In this work, we propose Temporal Generative Flow Networks (Temporal GFNs) , a framework designed to overcome these challenges and harness GFNs for probabilistic continuous time series forecasting. Our approach leverages the strengths of GFNs - structured sequential generation and reward-driven sampling - while introducing mechanisms specifically tailored for the continuous nature of time series data.

## Our core contributions are:

1. GFN Formulation for Probabilistic Forecasting: We frame forecasting as constructing a trajectory in a GFN state space, where states are time series windows and actions append the next predicted (quantized) value. The GFN learns to sample trajectories whose associated rewards (forecast accuracy) match the target distribution.
2. Adaptive Curriculum-Based Quantization: We move beyond fixed quantization by introducing a dynamic scheme. The number of quantization bins K is adjusted during training based on reward improvement and policy entropy, enabling the model to learn effectively with adaptive precision.
3. Differentiable Discrete Actions via STE: We employ a straight-through estimator (STE) [5] to reconcile discrete action selection with gradient-based optimization. The forward policy generates both discrete samples for state transitions and continuous approximations for gradient propagation.
4. Trajectory Balance Learning: We train the framework using the Trajectory Balance (TB) loss [22], enforcing flow consistency across entire forecast trajectories, and include an entropy bonus to encourage exploration.
5. Theoretical Guarantees: We provide theoretical bounds on the policy gradient error introduced by our quantization and STE approach, and on the range of the adaptive quantization factor, ensuring controlled behavior.

The paper is structured as follows: Section 2 reviews related work in forecasting and Generative Flow Networks (GFNs). Section 3 introduces the Temporal GFN framework, detailing its components like adaptive quantization, STE, and the training objective. Section 4 provides empirical validation through mechanism-focused ablation studies and performance comparisons against benchmarks.

Section 5 discusses the implications of our approach, and 6 concludes with a summary and future directions.

## 2 Background and Related Work

Our work builds upon concepts from probabilistic time series forecasting, Generative Flow Networks, and methods for handling continuous data in discrete frameworks.

## 2.1 Time Series Forecasting

Time series forecasting aims to predict future values x C +1: C + H = [ x C +1 , . . . , x C + H ] given historical context x 1: C = [ x 1 , . . . , x C ] . We focus on probabilistic forecasting, which involves predicting the conditional probability distribution p ( x C +1: C + H | x 1: C ) . Classical methods like ARIMA and ETS [16] model time series components (trend, seasonality, noise) and often provide prediction intervals under specific statistical assumptions. Deep learning methods leverage architectures like RNNs [30, 27] or Transformers [20, 24] to learn complex dependencies. Probabilistic outputs are commonly achieved by parameterizing output distributions [30], predicting quantiles [34], or using generative models like Flows [28] or Diffusion Models [19]. LLM-based forecasters [12, 1] have shown strong performance, often by tokenizing time series values into a discrete vocabulary similar to natural language. Chronos [1], for instance, uses simple scaling and uniform quantization, demonstrating the effectiveness of applying standard language model architectures and training objectives (cross-entropy) to tokenized time series. While powerful, these often rely on fixed quantization and may require very large models. Our Temporal GFN differs by using a GFN objective and introducing adaptive quantization.

## 2.2 Generative Flow Networks (GFNs)

GFNs [3, 6] are a class of probabilistic generative models designed to learn policies for constructing objects x in a compositional manner, such that the probability of sampling x is proportional to a given reward R ( x ) &gt; 0 . They operate on a directed acyclic graph (DAG) where nodes represent states s (partial or complete objects) and edges represent actions a leading from one state to another. GFNs excel at exploring complex, high-dimensional discrete spaces and sampling diverse, high-reward objects [17, 37]. Applying them to continuous domains, especially for sequential generation, remains an open challenge [26, 36].

## 2.3 Discretization and Continuous Actions in Time Series Forecasting

Bridging continuous time series data with models often designed for discrete inputs is a central challenge in modern forecasting. Quantization , or tokenization, which maps continuous values to a finite set of discrete symbols, is a widely adopted strategy. This has enabled the successful application of powerful sequence models, like Transformers, to time series by treating them as "sentences" of tokens [1] or like amino acids in proteins navigating a continuous space to model discrete sequences [2]. Recent advancements also explore more sophisticated tokenization schemes, such as wavelet-based approaches, to better capture multi-scale temporal features before feeding them to foundation models [23]. While these methods highlight the power of discretization, a commonality is often the reliance on a fixed quantization scheme (e.g., a predetermined number of bins or a fixed tokenization vocabulary). As discussed by Rabanser et al. [25], such fixed schemes can be suboptimal: too few bins (coarse quantization) limit precision and lead to information loss, whereas too many bins (fine quantization) can create an overly sparse action space, hindering exploration and slowing down the learning process, particularly in early training phases. Our proposed adaptive quantization mechanism (Section 3) is designed to directly mitigate this by dynamically adjusting the quantization granularity. Furthermore, when discrete actions (like selecting a quantization bin) are part of the model's generative process, as in our Temporal GFN framework, ensuring differentiability for gradient-based training of neural policies is critical. The Straight-Through Estimator (STE) [5, 35] serves as a standard technique to enable gradient propagation through such discrete steps. We employ STE to allow our GFN's forward policy to make discrete action selections (choosing quantized values) while facilitating effective gradient-based learning of the policy network. Our work, therefore, uniquely combines the GFN paradigm with both adaptive quantization for optimized

precision and STE for differentiability, specifically targeting the challenges of probabilistic continuous time series forecasting.

## 3 Methodology: Temporal Generative Flow Networks

In this section, we detail the proposed Temporal Generative Flow Network framework. We begin by mapping the probabilistic time series forecasting problem onto the Generative Flow Network paradigm. Subsequently, we describe the core components designed to handle continuous data within this framework: state representation, the adaptive quantization scheme coupled with the forward policy network, the mechanism for ensuring differentiability, and finally, the training objective based on Trajectory Balance.

## 3.1 Forecasting as Trajectory Generation in a GFN

Generative Flow Networks (GFNs) provide a powerful lens for learning to generate complex, structured objects [6]. They learn a policy to navigate a state space through a sequence of actions, ultimately sampling terminal objects (complete structures) with probabilities proportional to a predefined reward signal. This compositional generation process, guided by reward, makes GFNs an appealing candidate for sequential prediction tasks like time series forecasting.

We conceptualize the generation of a forecast as constructing a trajectory τ in a GFN state space.

- States ( s ): A state s t encapsulates the information available at step t of the forecast generation. We choose to represent states as fixed-length sliding windows containing the most recent history and any forecast values generated so far. The process begins from an initial state s 0 , which corresponds to the observed historical context window, s 0 = ( x obs , . . . , x obs + T -1 ) .
- Actions ( a ): An action a t taken from state s t corresponds to selecting the predicted value for the next time step, x obs + T + t . As GFNs fundamentally operate over discrete action spaces, a crucial step is mapping the inherently continuous space of possible time series values to a finite set of discrete actions. This is achieved through quantization, detailed in Section 3.2.
- State Transitions: Applying a chosen (discrete) action a hard t to state s t results in the next state s t +1 . This is typically achieved by appending a hard t to the window represented by s t and removing the oldest value, effectively sliding the window forward: s t +1 = ( s t [1] , . . . , s t [ T -1] , a hard t ) .
- Trajectories ( τ ): A complete forecast trajectory τ = ( s 0 , s 1 , . . . , s T ′ ) is formed by sequentially applying actions a 0 , . . . , a T ′ -1 according to a learned policy, spanning the desired forecast horizon T ′ .
- Reward ( R ( τ ) ): A reward function assigns a scalar value to each completed trajectory, quantifying the desirability or quality of the generated forecast (e.g., based on its accuracy against the true future values). The GFN's objective is to learn a policy that samples trajectories τ such that the sampling probability P ( τ ) ∝ exp( R ( τ )) .

The main technical challenges lie in adapting the discrete action mechanism of GFNs to the continuous nature of time series values and ensuring the entire process remains differentiable for effective training using gradient-based optimization.

## 3.2 Forward Policy, Adaptive Quantization, and Differentiability

The core generative component is the forward policy P F ( a t | s t ) , which learns the probability distribution over possible next actions given the current state. To manage the continuous action space and enable learning, we combine quantization, a Transformer-based policy network, an adaptive mechanism for quantization granularity, and the Straight-Through Estimator (STE).

## 3.2.1 Quantization and the Forward Policy Network

To bridge the gap between continuous time series values and the GFN's discrete action space, we employ quantization. We map continuous values within a range [ v min , v max ] to a set of K discrete

bins, represented by their centers: q k = v min + k -1 K -1 ( v max -v min ) , k = 1 , . . . , K. These K bin centers { q 1 , . . . , q K } constitute the discrete action space A available to the GFN at any given state s t . The forward policy network P F ( a t = q k | s t ) computes the probability of selecting action q k given the current state window s t . Recognizing the success of Transformer architectures [32] in capturing longrange dependencies in sequential data, we use a Transformer encoder as the backbone of our policy network. The state window s t , a sequence of T real values, is processed by the Transformer encoder, yielding a fixed-dimensional representation h t that summarizes the relevant historical context. This representation is then projected by a linear output layer to produce logits over the K possible discrete actions: logits t = W F h t + b F , where logits t ∈ R K . Applying the softmax function to these logits yields the policy distribution: P F ( a t = q k | s t ) = exp( logits t,k ) ∑ K j =1 exp( logits t,j ) . This network learns to predict the probability of the next value falling into each of the K quantization bins, based on the preceding time series pattern encoded in s t .

## 3.2.2 Adaptive Quantization: A Curriculum for Precision

Using a fixed number of bins K throughout training presents a dilemma common in discretization tasks [25]. A small K leads to high quantization error, limiting forecast precision, while a large K creates a vast, sparse action space that can significantly slow down learning and hinder exploration, especially in the initial phases when the policy is still poorly defined. To overcome this, we introduce an adaptive quantization scheme that treats the number of bins K as a parameter that evolves during training, akin to a curriculum learning strategy [4]. The model starts with a relatively small K (coarse granularity), making the initial exploration task simpler. As training progresses, K is dynamically adjusted based on the model's performance and confidence. Specifically, after a warmup period E warmup, we compute an adaptive update factor η e at each epoch e . This factor depends on the recent reward improvement ∆ R e = R e -R e -δ (average reward difference over δ epochs) and the average normalized forward policy entropy H e ∈ [0 , 1] :

<!-- formula-not-decoded -->

Here, λ &gt; 0 controls the sensitivity of adaptation, and ϵ &gt; 0 is a target reward improvement threshold.

- The Improvement Signal becomes positive if the model's reward gain is below the threshold ϵ . Slow improvement suggests the current quantization might be too coarse, thus pushing η e &gt; 1 to encourage increasing K .
- The Confidence Signal is large when entropy H e is low (i.e., the policy is highly confident or "peaked"). High confidence might indicate premature convergence or insufficient exploration. Increasing K in this case refines the action space, potentially revealing finer structures and encouraging further exploration.

The number of bins for the next epoch K e is updated multiplicatively, bounded by a maximum K max :

<!-- formula-not-decoded -->

If K e changes value, the size of the final linear layer ( W F , b F ) in the forward policy network (and the backward network, if learned) must be adjusted to match the new number of output bins.

To ensure training stability when the number of bins K e is increased, a simple random re-initialization of the output layer would be catastrophic, erasing previously learned knowledge. Instead, we employ a weight-reuse strategy. The weights and biases in the linear layer corresponding to the pre-existing bins are preserved. The weights and biases for the newly added bins are initialized to near-zero values. This method allows the policy to retain its learned distribution over the existing action space while cautiously exploring the new, finer-grained actions, thus preventing catastrophic forgetting and ensuring a smooth curriculum.

## 3.2.3 Ensuring Differentiability: The Straight-Through Estimator

A critical issue arises when selecting the discrete action a hard t = q k from the policy P F ( ·| s t ) to update the state. Operations like sampling or taking the argmax are non-differentiable, which would

prevent gradients from flowing back from the final loss calculation to the parameters of the policy network ( W F , b F and the Transformer encoder). To circumvent this, we employ the Straight-Through Estimator (STE) technique [5, 35]. The core idea is to use the discrete value in the forward pass for computations that require it (like state transitions) but to use a continuous approximation for the backward pass (gradient computation). Our forward policy generates two outputs at each step t :

- Hard Sample ( a hard t ): The discrete quantized value selected for constructing the trajectory. This is typically the most likely action: a hard t = q argmax k P F ( a t = q k | s t ) . This a hard t value is used to update the state s t → s t +1 .
- Soft Sample ( a soft t ): A continuous, differentiable proxy for the chosen action, computed as the expectation of the action under the current policy distribution: a soft t = ∑ K k =1 q k P F ( a t = q k | s t ) . During backpropagation, the gradient is computed with respect to a soft t . This gradient is then passed straight through the non-differentiable argmax/sampling operation that produced a hard t , effectively using the gradient of the continuous expectation a soft t to update the parameters that generated the policy P F ( ·| s t ) .

This STE mechanism allows us to maintain the integrity of the discrete state transitions inherent to the GFN formulation while enabling end-to-end training with gradient descent.

## 3.3 Learning Objective: Trajectory Balance and Exploration

GFNs are trained by enforcing flow consistency conditions. We adopt the Trajectory Balance (TB) loss [22], which offers robust credit assignment, particularly for long sequences.

## 3.3.1 Backward Policy

The TB objective necessitates a backward policy P B ( s t -1 | s t ) , modeling the probability of transitioning backward from state s t to a potential predecessor state s t -1 . We consider two implementations:

1. Uniform Policy: Assumes all valid predecessor states are equally likely. Given the fixed action space size K , this simplifies to log P B ( s t -1 | s t ) = -log K . This is parameter-free but ignores temporal structure.
2. Learned Policy: Parameterizes P B using a neural network that takes the representations of the involved states ( h t -1 , h t ) as input, e.g., log P B ( s t -1 | s t ) = log softmax ( W B concat ( h t -1 , h t ) + b B ) . This allows P B to adapt but increases model size.

## 3.3.2 Reward Function

The reward function R ( τ ) guides the GFN towards generating desirable trajectories. For forecasting, desirability equates to accuracy. We define the reward based on the discrepancy between the generated forecast sequence z = ( z 1 , . . . , z T ′ ) (derived from the sequence of soft samples a soft t ) and the ground truth future sequence y = ( y 1 , . . . , y T ′ ) . We use an exponential function of the negative normalized Mean Squared Error (MSE):

<!-- formula-not-decoded -->

where β &gt; 0 controls the reward scaling. Higher values of β create a sharper reward landscape, more strongly penalizing deviations from the ground truth.

## 3.3.3 Trajectory Balance Loss with Entropy Regularization

The TB loss [22] equates the flow along a trajectory based on forward probabilities and the initial flow Z (a learnable parameter representing the partition function or total flow) with the flow based on backward probabilities and the terminal reward R ( τ ) . The squared difference forms the loss for a

single trajectory:

<!-- formula-not-decoded -->

The overall TB loss is the expectation over trajectories sampled from the forward policy: L TB = E τ ∼ P F [ L TB ( τ )] .

To mitigate the risk of the policy becoming prematurely deterministic and to encourage broader exploration of the action space, we incorporate an entropy bonus into the learning objective (equivalent to subtracting an entropy penalty from the loss). The entropy of the forward policy at state s t is: H ( P F ( ·| s t )) = -∑ K k =1 P F ( a t = q k | s t ) log P F ( a t = q k | s t ) . The final loss function balances the TB objective with the average entropy across the trajectory: L = L TB -λ entropy E τ ∼ P F [ 1 T ′ ∑ T ′ -1 t =0 H ( P F ( ·| s t )) ] , where λ entropy ≥ 0 is a hyperparameter controlling the weight of the entropy regularization. Optimizing this final loss L trains the forward policy P F , the backward policy P B (if learned), and the partition function Z , we summarize this section with Algorithm A and we provide theorical analysis of our methodology in Appendix B.

## 4 Experiments

In this section, we present a comprehensive empirical evaluation of the Temporal Generative Flow Network (Temporal GFN) framework. Our experimental investigation is twofold. First (Section 4.1), we assess the performance of optimized Temporal GFN configurations against a broad range of established baselines on standard forecasting benchmarks and challenging related datasets. Throughout this section, we explicitly link empirical observations to the mathematical concepts introduced in Appendix B. Detailed experimental setup common to both studies is described in Appendix E. Second (Section 4.2), we conduct mechanism-focused analyses and ablations using synthetic data and standard benchmarks to validate the theoretical underpinnings of our approach, particularly concerning adaptive quantization and the Straight-Through Estimator (STE).

## 4.1 Benchmark Performance Comparison

Datasets, Training Strategy and Evaluation Metrics. We train and evaluate Temporal GFN on a large and diverse collection of publicly available time series datasets, comprehensively gathered and curated, mirroring the scale used in recent foundation model studies [1], the dataset, the training strategy, evaluation tasks and metrics are detailed in Appendix J. We evaluate optimized Temporal GFN configurations against relevant baselines across the defined benchmarks.

## 4.1.1 Comparison with RL and MCMC Baselines (Benchmark I)

Table 1 summarizes the aggregated relative performance on Benchmark I against PPO, SAC, and MCMCbaselines. Temporal GFN (Adaptive K20) shows substantial improvements (25-35%) across all metrics compared to these methods. The Learned Policy variant further improves probabilistic metrics (CRPS/WQL). This highlights the benefits of the GFN framework tailored for forecasting over generic sequential decision-making approaches shown in table 1.

## 4.1.2 Comparison with SOTA Time Series Forecasters (Benchmark I &amp; II)

Table 2 presents aggregated performance against SOTA deep learning models, evaluated on both Benchmark I (In-Domain) and Benchmark II (Zero-Shot) where applicable. Temporal GFN demonstrates highly competitive performance, often leading in probabilistic metrics (CRPS, WQL) and particularly excelling in zero-shot scenarios and on metrics sensitive to distribution shape like Calibration Error and Multimodality Score (evaluated on extended/healthcare benchmarks, details in Appendix I). Qualitative examples in Appendix K (Figures 20) visually support Temporal GFN's ability to capture complex dynamics and multimodality.

Table 1: Aggregated Relative Performance vs. RL/MCMC on Benchmark I.

| Method                                |   CRPS ( ↓ ) |   WQL( ↓ ) |   MASE ( ↓ ) | Avg. Improv. (%)   |
|---------------------------------------|--------------|------------|--------------|--------------------|
| Temporal GFN (Adaptive K20)           |       0.1674 |     0.2273 |       0.8938 | -                  |
| Temporal GFN (Fixed K20)              |       0.1781 |     0.2401 |       0.9363 | -5.9%              |
| Temporal GFN (Learned Pol. Adapt K10) |       0.1453 |     0.2137 |       1.0345 | +6.0%*             |
| PPO                                   |       0.2299 |     0.3013 |       1.1759 | -25.1%             |
| SAC                                   |       0.2423 |     0.3227 |       1.2491 | -29.5%             |
| MCMC                                  |       0.2673 |     0.3522 |       1.3523 | -35.5%             |

Table 2: Aggregated Relative Performance vs. SOTA Models (In-Domain / Zero-Shot).

| Model                       | Eval Setting           |   CRPS ( ↓ ) |   WQL( ↓ ) |   MASE ( ↓ ) | Calib. Err. ( ↓ )   | Multimod. ( ↑ )   |
|-----------------------------|------------------------|--------------|------------|--------------|---------------------|-------------------|
| Temporal GFN                | Zero-Shot / In-Domain  |       0.1542 |     0.2158 |       0.9378 | 0.0348              | 0.8762            |
| Lag-Llama                   | Zero-Shot / Pretrained |       0.1675 |     0.2401 |       0.9532 | 0.0623              | 0.7234            |
| Chronos (Base)              | Zero-Shot / Pretrained |       0.1731 |     0.2534 |       1.0273 | 0.0571              | 0.6957            |
| MOREI (Base)                | Task-Specific Trained  |       0.1859 |     0.2678 |       1.0871 | 0.0745              | 0.6182            |
| TimeFM                      | Task-Specific Trained  |       0.2076 |     0.2846 |       1.1258 | 0.1183              | 0.3012            |
| Temporal GFN (Learned Pol.) | In-Domain              |       0.1453 |     0.2137 |       1.0345 | -                   | -                 |

Main Temporal GFN results reflect strong performance across settings. Calib. Err. &amp; Multimod. primarily from healthcare/extended benchmarks. Task-Specific models trained per dataset. Learned Policy results from Table 3.

Table 3: Ablation Study Configurations and Main Results.

| Experiment Config          | Quantization   |   Start K | Policy Type   |   CRPS |    WQL |   MASE |
|----------------------------|----------------|-----------|---------------|--------|--------|--------|
| Fixed K=10                 | fixed          |        10 | uniform       | 0.1546 | 0.2346 | 1.1946 |
| Fixed K=20                 | fixed          |        20 | uniform       | 0.1921 | 0.2521 | 0.9721 |
| Adaptive K=10              | adaptive       |        10 | uniform       | 0.1688 | 0.2408 | 1.1048 |
| Adaptive K=20              | adaptive       |        20 | uniform       | 0.1845 | 0.2385 | 0.9532 |
| Learned Policy (Adapt K10) | adaptive       |        10 | learned       | 0.1453 | 0.2137 | 1.0345 |

## 4.2 Analysis of Quantization Mechanism

We investigate the quantitative relationship between quantization resolution (number of bins, K ), the resulting approximation error, and the implications for policy learning and performance. Table 3 summarizes the configurations and main results for this ablation study.

Quantization Error Bound Validation: Theorem B.1 established that the policy gradient error depends linearly on the quantization error ϵ . We empirically verify the relationship between K and ϵ . Figure 1 (bottom left) plots the Mean Squared Error (MSE), serving as a proxy for ϵ 2 , against K . As predicted, increasing K significantly reduces the quantization error, consistent with the premise that finer discretization leads to smaller ϵ . The relationship between error reduction and reward gain is explored in Figure 2. The left panel shows a strong negative correlation between quantization error and achievable reward, while the right panel shows that increasing K yields diminishing returns in reward improvement, suggesting a trade-off that motivates adaptive approaches.. We visually confirms this reduction in approximation error across the data distribution for increasing K in Appendix K Figure 3.

Adaptive Quantization Dynamics and Bounded Updates: The adaptive mechanism (Section 3.2.2) aims to optimize K based on reward improvement ( ∆ R e ) and policy entropy ( H e ), as governed by Eq. 1. Theorem B.2 guarantees the update factor η e is bounded: 1 ≤ η e ≤ 1 + 2 λ . This prevents uncontrolled explosion or collapse of K . In Appendix K, figure 10 (left panel, purple/blue lines for Adaptive Quantization) shows the coupled evolution of reward and entropy, which drive η e . The resulting non-uniform bin distribution learned by adaptive quantization (Figure 4, Adaptive panel) demonstrates its ability to allocate bins effectively compared to fixed uniform quantization (Uniform panel). This optimized allocation minimizes the effective ϵ in relevant regions, leading to superior performance observed in benchmark comparisons (Tables 1, 3).

<!-- image -->

Number of Bins (k)

Parameter 1

Parameter 1

Parameter 1

Figure 1: Gradient flow fields: Continuous (left) vs. Quantized with STE (right, K=16), validating STE.

Quantization and Policy Representation: In Figure 5 Appendix K visualizes how the choice of K impacts the policy's representation of the underlying continuous energy landscape. A low K provides a very coarse approximation, while higher K values capture more detail but remain discrete. The adaptive mechanism, by optimizing bin placement, aims to create a quantized landscape that best preserves the essential features (modes, relative probabilities) of the continuous landscape within the constraints of K bins, facilitating more accurate policy learning.

Entropy-Error Trade-off Quantification: In Figure 6 Appendix K quantifies the trade-off inherent in selecting K . It plots Normalized MSE (proxy for quantization error ϵ ) against Normalized Entropy (uniformity of bin usage). As K increases (indicated by color/size), error decreases, but entropy might slightly decrease from its maximum if adaptation leads to concentrated bins. This plot empirically illustrates the optimization landscape navigated by the adaptive mechanism.

## 5 Discussion

While our results demonstrate the effectiveness of Temporal GFNs, particularly in probabilistic accuracy and multimodality capture, it is important to contextualize its limitations. The sequential, step-by-step generation of trajectories, fundamental to the GFN paradigm, can be more computationally intensive during training compared to models like standard Transformers or LLMs that generate forecasts in a single forward pass. Furthermore, the performance is contingent on a welldesigned reward function R ( τ ) . Although the exponential MSE reward used in our work is effective for accuracy, designing rewards for more complex or domain-specific objectives (e.g., penalizing physiologically implausible trajectories in healthcare) remains a critical and potentially challenging modeling decision. 1) .Our adaptive quantization (Section 3) robustly applies GFNs to continuous data by dynamically balancing precision and exploration. Governed by reward and entropy signals (Eq. 1) and stabilized by theoretical bounds (Theorem B.2 Appendix B), it acts as an effective learning curriculum. While alternative data-driven quantization strategies (e.g., quantile-based bins) were explored in [9], particularly for skewed data by reducing quantization error ϵ more aggressively in dense regions (Figure 8, leading to CRPS improvements in Figure 9), it introduces significant implementation complexity (e.g., estimating quantiles dynamically or learning boundaries) and potentially higher training overhead (15% reported increase). Our adaptive uniform spacing approach provides a strong balance, achieving competitive performance (Tables 1, 2) and adapting the number

of bins effectively based on learning dynamics (Figure 10) without the full complexity of learning non-uniform boundaries, representing a practical and performant design choice. The reduction in ϵ achieved through adaptation directly contributes to improved gradient accuracy (Theorem C.1, Eq. 7). 2) .For enabling gradient flow through discrete quantization choices, we employ the Straight-Through Estimator (STE) [5]. While STE is a widely adopted and practically effective method, alternatives like Gumbel-Softmax (GS) relaxation [18] offer a different approach to managing differentiability in discrete action spaces via z k = softmax (( ℓ k + g k ) /τ temp ) . Our comparative analysis highlights the nuanced trade-offs. Figure 11 shows that GS can lead to faster initial convergence in terms of training loss, potentially due to its inherently smoother gradient signals early in training (high τ temp ). However, this initial advantage can sometimes plateau, with STE achieving a slightly lower final loss in some configurations. A key factor is gradient stability; Figure 7 illustrates that while STE maintains relatively stable gradient variance throughout training, GS, despite generally lower average variance, can exhibit occasional spikes, particularly if temperature annealing is not perfectly tuned. Furthermore, as detailed in Figure 12, GS introduces notable overhead: approximately 35% increased compute time, 41% higher memory usage, and a significant 70% rise in perceived implementation complexity due to temperature annealing and managing the continuous relaxation. Considering these factors, Figure 13 summarizes our practical guidance: STE is preferred for its simplicity, resource efficiency, and stability, especially for smaller models or complex distributions where its biased but consistent gradient proves effective. 3) .Finally, the combination of the Trajectory Balance objective with entropy regularization ( L ) proves highly effective. TB loss facilitates credit assignment, while the entropy bonus explicitly manages the exploration-exploitation trade-off, crucial for learning diverse and well-calibrated distributions. Removing the entropy bonus leads to significantly poorer final rewards (Figure 14), highlighting its role in preventing mode collapse and ensuring sufficient exploration (Figures 15, 16).

## 6 Conclusion

Temporal Generative Flow Networks (Temporal GFNs) offer a novel approach to probabilistic time series forecasting by adapting GFN principles for continuous data. Key to our framework is the framing of forecasting as reward-driven trajectory sampling, which facilitates exploration and learning distributions over diverse futures, potentially yielding more robust probabilistic forecasts than standard methods. We successfully address the challenges of continuous data through adaptive quantization-providing dynamic precision control as a curriculum and overcoming fixed discretization limitations-and the (STE), which enables learning by bridging discrete state transitions with differentiable gradient updates. Trained via Trajectory Balance for effective credit assignment and entropy regularization for managed exploration, Temporal GFNs learn to sample accurate and diverse forecast trajectories. Theoretical bounds affirm the stability of our adaptive mechanism and quantify the quantization error impact. Compared to alternatives like LLM-based forecasters [1, 23, 29], Temporal GFN provides adaptive precision and a distinct flow-matching learning paradigm, while its unique sequential decision-making generative process offers advantages for complex dependencies and multimodality over standard probabilistic models. Future work will focus on multivariate extensions, extensive empirical validation, and exploring alternative GFN objectives.

Multivariate Extension : A key direction for future work is the extension of Temporal GFNs to the multivariate setting. While this paper focused on the univariate case to rigorously establish the core framework, the approach is not fundamentally limited. The state representation s t can be expanded from a vector to a matrix representing a window of multivariate observations, and the action a t can become a vector of quantized values for each dimension. The primary challenge lies in designing a policy network P F that efficiently models the joint distribution over the multivariate action space. This could be achieved through an autoregressive factorization of the output dimensions or by learning a parameterized covariance structure, offering a rich area for future research.

Irregular Sampling : Furthermore, the current implementation assumes regularly sampled time series due to its use of a fixed-size sliding window. To handle the irregularly sampled data common in domains like healthcare (EHRs) or finance, the framework can be adapted. The state representation could be augmented to include not just observation values but also the time elapsed between them, for instance, by feeding a sequence of ( value , time\_delta ) tuples into the Transformer encoder. This would allow the policy network to explicitly learn time-dependent dynamics, making the model robust to irregular sampling and significantly broadening its applicability.

## References

- [1] Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, et al. Chronos: Learning the language of time series. arXiv preprint arXiv:2403.07815 , 2024.
- [2] Timothy Atkinson, Thomas D. Barrett, Scott Cameron, Bora Guloglu, Matthew Greenig, Louis Robinson, Alex Graves, Liviu Copoiu, and Alexandre Laterre. Protein sequence modelling with bayesian flow networks. bioRxiv , 2024. doi: 10.1101/2024.09.24.614734. URL https: //www.biorxiv.org/content/early/2024/09/26/2024.09.24.614734 .
- [3] Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Chris Link Cremer, Tristan Deleu Lau, and Yoshua Bengio. Flow network based generative models for non-iterative diverse candidate generation. In Advances in Neural Information Processing Systems , volume 34, pages 2738127394, 2021.
- [4] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th Annual International Conference on Machine Learning , ICML '09, page 41-48, New York, NY, USA, 2009. Association for Computing Machinery. ISBN 9781605585161. doi: 10.1145/1553374.1553380. URL https://doi.org/10.1145/ 1553374.1553380 .
- [5] Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. 2013.
- [6] Yoshua Bengio, Salem Lahlou, Tristan Deleu, Edward J Hu, Mo Tiwari, and Moksh Jain. Gflownet foundations. arXiv preprint arXiv:2111.09266 , 2021.
- [7] Konstantinos Benidis, Syama Sundar Rangapuram, Valentin Flunkert, Yuyang Wang, Danielle C Maddix, Caner Turkmen, Jan Gasthaus, Michael Bohlke-Schneider, David Salinas, Lorenzo Stella, et al. Deep learning for time series forecasting: Tutorial and literature survey. ACM Computing Surveys (CSUR) , 55(6):1-36, 2022.
- [8] Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for time-series forecasting, 2024. URL https://arxiv.org/abs/2310.10688 .
- [9] Sebastian Espinosa, Jorge F. Silva, and Pablo Piantanida. A data-driven quantization design for distributed testing against independence with communication constraints. In ICASSP 2022 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 5238-5242, 2022. doi: 10.1109/ICASSP43922.2022.9746197.
- [10] Jan Gasthaus, Konstantinos Benidis, Yuyang Wang, Syama Sundar Rangapuram, David Salinas, Valentin Flunkert, and Tim Januschowski. Probabilistic forecasting with spline quantile function rnns. In Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics , pages 1901-1910. PMLR, 2019.
- [11] Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation. Journal of the American statistical Association , 102(477):359-378, 2007.
- [12] Nate Gruver, Marc Finzi, Shikai Qiu, and Andrew Gordon Wilson. Large language models are zero-shot time series forecasters. Advances in Neural Information Processing Systems , 36, 2023.
- [13] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor, 2018. URL https://arxiv.org/abs/1801.01290 .
- [14] Rob J Hyndman and George Athanasopoulos. Forecasting: principles and practice. In OTexts , 2018.
- [15] Rob J Hyndman and Anne B Koehler. Another look at measures of forecast accuracy. International journal of forecasting , 22(4):679-688, 2006.

- [16] Rob J Hyndman, Anne B Koehler, J Keith Ord, and Ralph D Snyder. Forecasting with exponential smoothing: the state space approach. In Springer Science &amp; Business Media , 2008.
- [17] Moksh Jain, Maksym Korablyov, Hugo Larochelle, Tristan Deleu, and Yoshua Bengio. Biological sequence design with generative flow networks. arXiv preprint arXiv:2203.04115 , 2022.
- [18] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144 , 2016.
- [19] Marcel Kollovieh, Abdul Fatir Ansari, Michael Bohlke-Schneider, Jasper Zschiegner, Hao Wang, and Yuyang Wang. Predict, refine, synthesize: Self-guiding diffusion models for probabilistic time series forecasting. In Advances in Neural Information Processing Systems , volume 36, pages 28341-28364, 2023.
- [20] Bryan Lim, Sercan Ö Arık, Nicolas Loeff, and Tomas Pfister. Temporal fusion transformers for interpretable multi-horizon time series forecasting. International Journal of Forecasting , 37(4): 1748-1764, 2021.
- [21] Xu Liu, Juncheng Liu, Gerald Woo, Taha Aksu, Yuxuan Liang, Roger Zimmermann, Chenghao Liu, Silvio Savarese, Caiming Xiong, and Doyen Sahoo. Moirai-moe: Empowering time series foundation models with sparse mixture of experts, 2024. URL https://arxiv.org/abs/ 2410.10469 .
- [22] Nikolay Malkin, Kan Miao, Zhendong Li, Guangyao Zhang, Emmanuel Bengio, Chenghao Zhang, Tristan Deleu, Dinghuai Shen, Meng Liu, Hugo Larochelle, et al. Trajectory balance: Improved credit assignment in generative flow networks. In International Conference on Machine Learning , pages 14817-14836. PMLR, 2022.
- [23] Luca Masserano, Abdul Fatir Ansari, Boran Han, Xiyuan Zhang, Christos Faloutsos, Michael W. Mahoney, Andrew Gordon Wilson, Youngsuk Park, Syama Rangapuram, Danielle C. Maddix, and Yuyang Wang. Enhancing foundation models for time series forecasting via wavelet-based tokenization, 2024. URL https://arxiv.org/abs/2412.05244 .
- [24] Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations , 2023.
- [25] Stephan Rabanser, Tim Januschowski, Valentin Flunkert, David Salinas, and Jan Gasthaus. The effectiveness of discretization in forecasting: An empirical study on neural time series models. arXiv preprint arXiv:2005.10111 , 2020.
- [26] Francisco Ramalho, Meng Liu, Zihan Liu, and Etienne Mathieu. Towards gflownets for continuous control. arXiv preprint arXiv:2310.18664 , 2023.
- [27] Syama Sundar Rangapuram, Matthias W Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang, and Tim Januschowski. Deep state space models for time series forecasting. In Advances in neural information processing systems , volume 31, 2018.
- [28] Kashif Rasul, Calvin Seward, Ingmar Schuster, and Roland Vollgraf. Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting. In International Conference on Machine Learning , pages 8857-8868. PMLR, 2021.
- [29] Kashif Rasul, Arjun Ashok, Andrew Robert Williams, Hena Ghonia, Rishika Bhagwatkar, Arian Khorasani, Mohammad Javad Darvishi Bayazi, George Adamopoulos, Roland Riachi, Nadhir Hassen, Marin Biloš, Sahil Garg, Anderson Schneider, Nicolas Chapados, Alexandre Drouin, Valentina Zantedeschi, Yuriy Nevmyvaka, and Irina Rish. Lag-llama: Towards foundation models for probabilistic time series forecasting, 2024. URL https://arxiv.org/abs/2310. 08278 .
- [30] David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. Deepar: Probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting , 36(3): 1181-1191, 2020.

- [31] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347 .
- [32] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems , volume 30, 2017.
- [33] Vladimir Vovk, Alex Gammerman, and Glenn Shafer. Algorithmic learning in a random world . Springer Science &amp; Business Media, 2005.
- [34] Ruofeng Wen, Kari Torkkola, Balakrishnan Narayanaswamy, and Dhruv Madeka. A multihorizon quantile recurrent forecaster. arXiv preprint arXiv:1711.11053 , 2017.
- [35] Peiqian Yin, Subhankar S K, Shuai Zhang, Yingyong Xie, Jun Liu, Jian Tu, Siyu Wang, X M Leo Zhang, Qi Chen, and Ping Luo. Understanding straight-through estimator in training activation quantized nets. arXiv preprint arXiv:1903.05662 , 2019.
- [36] Chenghao Zhang, Kan Miao, and Yoshua Bengio. Generating continuous objects with discrete gflownets. arXiv preprint arXiv:2306.01764 , 2023.
- [37] Dinghuai Zhang and Nikolay Malkin Kan Miao Zhendong Li Guangyao Zhang Chenghao Liu Jian Tang Yoshua Bengio. Generative flow networks for discrete probabilistic modeling. In International Conference on Machine Learning , 2022.
- [38] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization, 2018. URL https://arxiv.org/abs/1710.09412 .

## A Training Algorithm

## Algorithm 1 Temporal GFN Training Procedure

̸

```
Require: Time series dataset D , initial K 0 , max K max , adaptive params λ, ϵ, δ , E warmup, E , hyperparameters β, λ entropy . 1: Initialize θ ( P F , opt. P B , Z ). K ← K 0 . Init past rewards. 2: for epoch e = 1 to E do 3: if e ≥ E warmup then ▷ Adaptive Quantization Update 4: Compute avg. reward R e , avg. entropy H e . 5: ∆ R e ← R e -R e -δ . 6: η e ← 1 + λ (max(0 , ϵ -∆ R e ) /ϵ +(1 -H e )) (Eq. 1). 7: K new ← min( K max , ⌊ K · η e ⌋ ) (Eq. 2). 8: if K new = K then 9: K ← K new. Adjust policy network output layers for K bins. 10: end if 11: end if 12: for each batch ( x, y ) in D do ▷ Trajectory Sampling and Loss Computation 13: s 0 ← Extract context window from x . 14: τ ←{ s 0 } , log P sum F ← 0 , H sum ← 0 , generated_seq ← [] . s t ← s 0 . 15: for t = 0 to T ′ -1 do ▷ Forward Pass 16: h t ← Transformer ( s t ) . 17: P F ( ·| s t ) ← softmax ( W F h t + b F ) . 18: a hard t ← q argmax k P F ( a t = q k | s t ) . ▷ Hard sample for state transition 19: a soft t ← ∑ K k =1 q k P F ( a t = q k | s t ) . ▷ Soft sample for reward/gradient 20: log P sum F ← log P sum F +log( P F ( a hard t | s t )) . 21: H sum ←H sum + H ( P F ( ·| s t )) . 22: s t +1 ← concat ( s t [1 :] , a hard t ) . ▷ Update state 23: Append s t +1 to τ . Append a soft t to generated_seq. s t ← s t +1 . 24: end for 25: Compute log P sum B ← ∑ T ′ t =1 log P B ( s t -1 | s t ) . ▷ Backward Pass (or use uniform) 26: Compute R ( τ ) using generated_seq and target y (Eq. 3). 27: Compute L TB ( τ ) ← (log Z +log P sum F -log P sum B -log R ( τ )) 2 . ▷ TB Loss 28: Compute batch loss L batch ←L TB ( τ ) -λ entropy ( H sum /T ′ ) . ▷ Final Loss 29: Compute gradients ∇ θ L batch . ▷ Gradients flow via a soft t 30: Update θ using gradients. ▷ Parameter Update 31: end for 32: end for
```

## B Theoretical Analysis

To better understand the behavior and stability of our Temporal GFN framework, we provide theoretical analysis on two crucial aspects: the impact of our quantization and STE approximation on the learning signal, and the bounds governing the adaptive quantization mechanism.

## B.1 Quantization Error Bound on Policy Gradient

The core learning process relies on policy gradients. However, our framework uses discrete actions ( a hard t ) for state transitions while calculating gradients via continuous approximations ( a soft t ) using the STE. This introduces a potential discrepancy between the true gradient of the expected reward and the gradient actually used for updates. It is crucial to understand if this approximation significantly corrupts the learning signal. Theorem B.1 provides a bound on this error, showing that it depends on the inherent quantization error and the STE approximation quality.

Theorem B.1 (Quantization Error Bound) . Assume the reward function R is L R -Lipschitz w.r.t. its input sequence and ∇ log ˜ P F has bounded variance. Let ϵ = max ∥ actiontrue -actionquantized ∥ be the max quantization error. Let δ = ∇ log P F -∇ log ˜ P F be the difference in score functions due to STE.

The error between the true policy gradient ∇ E [ R ] and the estimated gradient ∇ E [ ˜ R ] is bounded by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. (The proof is given in the appendix C, following the decomposition in Eq. 11 and applying Lipschitz continuity and Cauchy-Schwarz). The bound demonstrates that the gradient error is controlled by the quantization resolution ( ϵ ) and the fidelity of the STE approximation (measured by δ ). If quantization is fine ( ϵ small) and STE provides a reasonable gradient proxy ( δ small in expectation), the learning signal remains informative.

## B.2 Adaptive Update Factor Bound

The adaptive quantization mechanism (Eq. 1, 2) modifies the number of bins K multiplicatively based on performance and entropy. Unbounded updates could lead to instability - either K growing uncontrollably large or collapsing. Theorem B.2 establishes that the multiplicative factor η e is strictly bounded within a predictable range determined by the hyperparameter λ . This ensures the adaptation process is controlled and prevents drastic, potentially harmful, changes to the action space in a single step.

Theorem B.2 (Adaptive Update Factor Bound) . Let η e be defined as in Eq. 1, with ∆ R e ∈ [0 , ϵ ] and H e ∈ [0 , 1] . Then η e is bounded as:

<!-- formula-not-decoded -->

Proof. (The proof is given in Appendix section D, analyzing the bounds of the terms T 1 = max(0 , ϵ -∆ R e ) /ϵ and T 2 = (1 -H e ) and their sum S ). This bound guarantees that K can only increase (since η e ≥ 1 ) and the maximum multiplicative increase per step is limited by (1 + 2 λ ) , ensuring a degree of stability in the adaptation process.

## C Quantization Error Bound

The use of discrete hard samples ( a hard t ) in the state transition and trajectory construction, while using soft samples ( a soft t ) for gradient computation via STE, introduces a discrepancy in the policy gradient estimates. We aim to bound the error in the policy gradient due to this approximation.

Let R ( x ) be the true reward based on a continuous action x , and ˜ R ( q hard ) be the reward obtained when using the discrete action q hard . Let P F be the policy producing continuous outputs (before quantization) and ˜ P F be the policy over discrete bins. Let ∇ E [ R ] be the true policy gradient and ∇ E [ ˜ R ] be the gradient obtained using our STE approach (gradient through q soft, reward based on q hard or q soft ).

Theorem C.1 (Quantization Error Bound) . Assume the reward function R is L R -Lipschitz with respect to its input sequence and that the score function ∇ log ˜ P F has bounded variance. Let ϵ denote the maximum quantization error, i.e., ∥ actioncontinuous -actionquantized ∥ ≤ ϵ . Let δ = ∇ log P F - ∇ log ˜ P F represent the difference in score functions due to the STE approximation (gradient computed via q soft vs. actual discrete choice q hard). The error in the policy gradient estimate is bounded by:

<!-- formula-not-decoded -->

where L = max { L R √ E [ ∥∇ log ˜ P F ∥ 2 ] , √ E [ ˜ R 2 ] } is a constant depending on the Lipschitz constant of the reward and the moments of the reward and score function.

Proof. We decompose the gradient difference:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Bounding Term A: By the Lipschitz property of R , we have | R -˜ R | ≤ L R ∥ x -˜ x ∥ ≤ L R ϵ , where x represents the sequence generated with continuous actions and ˜ x with quantized actions. Using Cauchy-Schwarz:

<!-- formula-not-decoded -->

Bounding Term B: Using Cauchy-Schwarz again:

<!-- formula-not-decoded -->

(We use ˜ R in the definition of L as R might not be directly computable in the context of ˜ P F ).

Combining the bounds for Term A and Term B using the triangle inequality on Eq. 11:

<!-- formula-not-decoded -->

This completes the proof. The bound shows that the gradient error depends linearly on the quantization error ϵ and the square root of the expected squared norm of the score function difference δ induced by the STE.

## D Adaptive Update Factor Bound

The adaptive update factor η e (Eq. 1) controls the change in the number of bins K . It is important to ensure this factor remains within reasonable bounds to guarantee stability.

Theorem D.1 (Adaptive Update Factor Bound) . Let the adaptive update factor η e be defined as in Eq. 1, where the reward improvement ∆ R e ∈ [0 , ϵ ] (assuming reward is non-decreasing or improvement is capped by ϵ ) and the normalized entropy H e ∈ [0 , 1] . Then, η e is bounded as:

<!-- formula-not-decoded -->

Specifically, η e = 1 when ∆ R e = ϵ (maximum improvement considered) and H e = 1 (maximum entropy). Conversely, η e = 1 + 2 λ when ∆ R e = 0 (no improvement) and H e = 0 (minimum entropy/maximum confidence).

<!-- formula-not-decoded -->

Since ∆ R e ∈ [0 , ϵ ] :

- If ∆ R e = ϵ , then ϵ -∆ R e = 0 , so T 1 = max(0 , 0) ϵ = 0 .
- If ∆ R e = 0 , then ϵ -∆ R e = ϵ , so T 1 = max(0 ,ϵ ) ϵ = 1 .
- For intermediate values 0 &lt; ∆ R e &lt; ϵ , we have 0 &lt; ϵ -∆ R e &lt; ϵ , thus 0 &lt; T 1 &lt; 1 .

Therefore, T 1 ∈ [0 , 1] .

Since the normalized entropy H e ∈ [0 , 1] :

- If H e = 1 , then T 2 = 1 -1 = 0 .
- If H e = 0 , then T 2 = 1 -0 = 1 .
- For intermediate values 0 &lt; H e &lt; 1 , we have 0 &lt; T 2 &lt; 1 .

Therefore, T 2 ∈ [0 , 1] .

The sum inside the parenthesis is S = T 1 + T 2 . Since T 1 ∈ [0 , 1] and T 2 ∈ [0 , 1] , the sum S ∈ [0 , 2] . The adaptive factor is η e = 1 + λS . Since S ∈ [0 , 2] and λ &gt; 0 , we have λS ∈ [0 , 2 λ ] . Therefore, η e = 1 + λS ∈ [1 , 1 + 2 λ ] .

The minimum value η e = 1 is achieved when S = 0 , which requires T 1 = 0 and T 2 = 0 . This happens when ∆ R e = ϵ and H e = 1 . The maximum value η e = 1 + 2 λ is achieved when S = 2 , which requires T 1 = 1 and T 2 = 1 . This happens when ∆ R e = 0 and H e = 0 . This completes the proof. The theorem shows that the multiplicative update factor is bounded, preventing excessively large or small changes in K in a single step.

## E Experimental Setup for Benchmark Comparaison

Datasets: We utilize a range of datasets for evaluation. Core ablation studies are performed on benchmark datasets commonly used in time series forecasting (e.g., Electricity, Traffic, ETTm1, ETTh1. We also present results on healthcare-specific datasets (MIMIC-III Vital Signs) to evaluate performance in our target application domains.

Training Strategy. Temporal GFN models, particularly those intended for zero-shot evaluation, undergo a large-scale training phase on the combined "Pre-training only" and "In-domain (Benchmark I)" datasets. The context length T (input window size) for sequences sampled during training is set to 512, matching common Transformer setups, and the prediction length T ′ generated during training rollouts is set to 64, ensuring coverage for various downstream evaluation horizons H . Training employs the Trajectory Balance objective (Section 3) with default entropy regularization ( λ entropy = 0 . 01 ) and adaptive quantization enabled (typically starting K = 20 , max K = 128 ) unless evaluating fixed-K ablations. To enhance robustness and data diversity, we incorporate data augmentation: each training sequence is generated either from the original datasets or with probability 0.9 from a TSMixup set (convex combinations of different time series, adapted from 38) and with probability 0.1 from a synthetic dataset generated via Gaussian Processes with randomly combined kernels (KernelSynth, similar to 1). Training typically proceeds for 10,000 optimization steps on multi-GPU hardware.

Evaluation Tasks and Metrics. For both in-domain (Benchmark I) and zero-shot (Benchmark II) evaluations, we use the final H observations of each time series as a held-out test set ( H is task-specific, see Appendix J). We compute the primary metrics: WQL [10] on 9 levels { 0 . 1 , ..., 0 . 9 } , MASE[15] using the median prediction, and CRPS [11]. To aggregate metrics across diverse datasets and provide fair comparisons, we compute each model's score divided by the score of a baseline model (here, Seasonal Naive), yielding relative scores. These relative scores are then aggregated across all datasets within the benchmark using the geometric mean , which provides a robust aggregation insensitive to outlier datasets and the choice of baseline.

Following standard practices in probabilistic forecasting [10, 1], we evaluate models using:

- CRPS (Continuous Ranked Probability Score): Assesses the overall accuracy and calibration of the full predictive distribution. Lower is better.

- WQL (Weighted Quantile Loss, τ ∈ { 0 . 1 , ..., 0 . 9 } ): Assesses calibration across different quantiles. Lower is better.
- MASE (Mean Absolute Scaled Error): Evaluates point forecast accuracy (median) relative to a naive seasonal baseline [15]. Lower is better.
- Calibration Error &amp; Multimodality Score: Used specifically for healthcare/multimodal analysis (lower/higher is better, respectively).
- Trajectory / Action Diversity: Quantifies the variety in generated forecast trajectories or chosen actions.

Baselines: We compare Temporal GFN configurations against:

- GFN Ablations: Fixed K (K=10, K=20), Adaptive K (start K=10, start K=20), Learned Backward Policy vs. Uniform.
- RL/Sampling Methods: PPO [31], SAC [13], and a generic MCMC approach.
- SOTA Forecasters: Lag-Llama [29], Chronos [1], MOREI [21], and a decoder-only Transformer (TimeFM) [8].

Implementation Details: The core policy network is a Transformer encoder [32]. Default hyperparameters include λ entropy = 0 . 01 . Adaptive quantization parameters are tuned. We train typically for 10,000 steps. Baselines are run using publicly available implementations or standard libraries where possible.

Training Startegy The adaptive quantization mechanism's sensitivity to training dynamics is primarily managed by the hyperparameter δ , the number of epochs over which reward improvement is averaged (Eq. 1). This parameter's role is to smooth the reward signal ∆ R e , preventing volatile updates to the number of bins K that might arise from high-variance, batch-level rewards. We observed during development that settings with smaller batch sizes or higher learning rates, which typically yield noisier gradient and reward signals, benefit from a larger δ (e.g., δ = 10 ) to ensure stability. For the experiments reported in this paper, we found δ = 5 to provide a good balance between responsiveness and stability across our primary configurations. The adaptation is governed by λ = 0 . 1 and ϵ = 0 . 02 , which we found to be robust across datasets.

## F Ablation study of Temporal GFN

This section presents empirical results designed to validate the theoretical underpinnings and analyze the behavior of the proposed Temporal Generative Flow Network (Temporal GFN) framework. Through controlled experiments and ablations, we investigate the quantitative impact of the core mechanisms-adaptive quantization and the Straight-Through Estimator (STE)-on learning dynamics, representation accuracy, and optimization efficacy. We explicitly connect these empirical findings to the mathematical concepts developed in Section B, including the Quantization Error Bound, Adaptive Update Factor Bound, and the role of entropy in the Trajectory Balance objective.

## F.1 Validation of the Straight-Through Estimator (STE)

STE is crucial for enabling gradient-based optimization with discrete actions. We validate its effectiveness empirically.

Gradient Flow Approximation: Theorem B.1 bounded the gradient error based on ϵ and the STE approximation quality δ . Figure 21 provides a visual validation by comparing the continuous gradient field with the STE-computed gradient field on the quantized landscape. The preservation of gradient directions and relative magnitudes, particularly the flow towards energy minima, demonstrates that STE provides a sufficiently accurate approximation for effective optimization ( δ is functionally small).

Optimization Trajectory: Figure 17 shows that optimization using STE gradients on the quantized landscape successfully converges to the same low-energy regions as optimization on the continuous landscape. The energy plots (right panel) confirm convergence, although dynamics differ slightly. This provides empirical proof that the gradient signal provided by STE, despite being an approximation bounded by Theorem B.1, is effective for guiding the policy towards optimal (high-reward) states.

Necessity of STE (Ablation): Removing STE entirely breaks the gradient path. Figure 4 (Without STE panel) shows the policy completely fails to learn the target distribution without gradient information. Figure 14 confirms this, with the "No STE" configuration achieving minimal reward, demonstrating its critical role.

## G Analysis of GFN Learning Objective and Dynamics

We examine the role of the Trajectory Balance loss and entropy regularization.

Reward Components and TB Loss: Figure 19 illustrates the components contributing to the overall reward signal driving learning, implicitly linked to minimizing the TB loss (Eq. 4). Successful training maximizes the total reward, which requires balancing prediction accuracy, achieving flow consistency (implicitly rewarded), and benefiting from the entropy bonus.

Entropy Regularization and Exploration: The entropy term H ( P F ( ·| s t )) in L section 3.3.3 encourages exploration. Figure 10 shows the typical inverse relationship between reward and entropy during convergence. Critically, comparing the "Adaptive Quantization" curves (with λ entropy = 0 . 01 ) to "No Entropy Regularization" reveals that regularization helps maintain higher entropy, particularly later in training. Figure 14 confirms that removing this regularization ("No Entropy Regularization", teal line) leads to suboptimal final reward (0.55 vs 0.74 for full model), empirically demonstrating the benefit of the entropy term for balancing exploration and exploitation, leading to better overall solutions. This is further supported by sustained high levels of unique actions ratio (Figure 16) and trajectory diversity (Figure 15) throughout training when entropy regularization is present.

## H GFN Calibration

A key advantage of Temporal GFN is its inherent capacity for strong calibration . Unlike methods relying solely on quantile regression or implicit uncertainty mechanisms (like dropout), which often exhibit systematic over- or under-confidence as seen in Figure 18 for baselines like LagLlama, Chronos, and MOREI, Temporal GFN's calibration curve aligns closely with the ideal diagonal. This superior intrinsic calibration arises from the GFN objective itself (learning P ( τ ) ∝ R ( τ ) via Trajectory Balance, Eq. 4), which naturally encourages modeling the full distribution of high-reward outcomes, leading to more reliable uncertainty estimates directly from sampled trajectories. While post-hoc techniques like Conformal Prediction (CP) [33] can further enhance calibration and provide formal coverage guarantees for any model (Figure 22, 23), Temporal GFN provides a significantly better-calibrated starting point, making subsequent recalibration potentially more effective or less necessary compared to inherently miscalibrated models.

## I Multimodality Metric and GFN Advantage

In time series forecasting, particularly in domains like healthcare or complex system modeling, future outcomes may not follow a single path but could diverge into multiple plausible scenarios (modes). Effectively capturing this inherent multimodality is crucial for robust decision-making. This appendix details the composite metric used to quantify a model's ability to represent multimodal predictive distributions and analyzes how the GFlowNet framework is inherently suited to excel at this task.

## I.1 Definition of the Multimodality Score

The Multimodality Score used in our evaluations is a composite measure designed to assess several facets of how well a predicted distribution P (derived from forecast samples) matches a true (potentially multimodal) distribution Q . It incorporates the following components, typically averaged over relevant forecast horizons:

## I.1.1 Mode Count Accuracy

This measures whether the model identifies the correct number of distinct modes (peaks) in the distribution. It penalizes both missing true modes and detecting spurious ones.

<!-- formula-not-decoded -->

where N detected and N true are the number of detected and true modes respectively, and λ (e.g., 0.5) controls the penalty strength for detecting extra modes. A value of 1 indicates perfect mode count agreement ( N detected = N true ). N true = 0 is handled as a special case.

## I.1.2 Jensen-Shannon (JS) Divergence

JS divergence quantifies the similarity between the overall shapes of the predicted distribution P and the true distribution Q . It is a symmetrized and smoothed version of the Kullback-Leibler (KL) divergence.

<!-- formula-not-decoded -->

where M = 1 2 ( P + Q ) is the mixture distribution, and D KL ( P || Q ) = ∑ i P ( i ) log P ( i ) Q ( i ) . Lower JS divergence indicates higher similarity between the distributions. P and Q are typically derived from binned densities or empirical distributions from samples.

## I.1.3 Mode Proportion Error

This assesses whether the model assigns the correct probability mass to each identified mode.

<!-- formula-not-decoded -->

where prop ( i ) is the probability mass associated with the i -th true mode (and its corresponding predicted mode, if found). Calculation requires identifying modes and estimating the probability mass associated with each (e.g., by integrating density between valleys). Lower error indicates better allocation of probability mass.

## I.1.4 Wasserstein Distance

The 1-Wasserstein distance (Earth Mover's Distance) measures the cost of transforming one distribution into another. For 1D distributions, it can be efficiently computed using the inverse CDFs F -1 P , F -1 Q or sorted samples:

<!-- formula-not-decoded -->

where x ( i ) , y ( i ) are the i -th sorted samples from P and Q . It captures differences in location and shape. Lower distance is better.

These components are often combined into a single score or reported individually to provide a nuanced assessment of multimodality capture.

## I.2 Multimodality Algorithm

The practical calculation of the multimodality score involves several steps, summarized in Algorithm 2. This typically requires estimating probability density functions from samples, identifying modes within these PDFs, and comparing properties of the predicted modes and distribution against the true ones.

## Algorithm 2 Pseudo-Code of Multimodality Score Calculation

- Require: Forecast samples S P = { x 1 , ..., x N } , True distribution samples S Q = { y 1 , ..., y M } (or analytical form Q ), Horizon indices H interest, Density estimation parameters (bandwidth h , range X range, resolution N grid), Peak detection threshold θ density .
- 1: Initialize aggregate metrics M agg ←{ Mode Acc.: 0, JS Div.: 0, Mode Prop. Err.: 0, Wass. Dist.: 0 }
- 2: for each horizon index h ∈ H interest do
- 3: Extract horizon samples S P,h ←{ x i,h } N i =1 , S Q,h ←{ y j,h } j M =1 ▷ Isolate data for current horizon
- 4: Estimate predicted PDF P h ( x ) using KDE on S P,h over X range with N grid points. ▷ Density from forecast
- 5: Estimate true PDF Q h ( x ) using KDE on S Q,h (or use analytical Q ) over X range . ▷ Density from ground truth
- 6: Detect predicted modes Modes P ←{ ( v k , d k ) } by finding peaks in P h ( x ) &gt; θ density . ▷ Find peaks in predicted PDF
- 7: N detected ←| Modes P |
- 8: Detect true modes Modes Q ←{ ( v ′ k , d ′ k ) } by finding peaks in Q h ( x ) &gt; θ density . ▷ Find peaks in true PDF
- 9: N true ←| Modes Q |
- 10: Calculate Mode Count Accuracy m acc using Eq. 13. ▷ Compare number of modes
- 11: Normalize estimated densities ˆ P h ← P h / ∑ P h , ˆ Q h ← Q h / ∑ Q h . ▷ Ensure densities sum to 1
- 12: Calculate JS Divergence m js ← JS ( ˆ P h || ˆ Q h ) using Eq. 14. ▷ Similarity of overall shapes
- 13: Estimate predicted mode proportions { prop ( k ) pred } from Modes P , P h ( x ) . ▷ Requires robust area calculation
- 14: Estimate true mode proportions { prop ( k ) true } from Modes Q , Q h ( x ) . ▷ Mass under each true mode
- 15: Calculate Mode Proportion Error m prop using Eq. 15. ▷ Accuracy of probability mass per mode
- 16: Calculate Wasserstein Distance m wass ← W 1 ( S P,h , S Q,h ) using Eq. 16. ▷ Distance between sample sets
- 17: Store horizon metrics M h ←{ m acc , m js , m prop , m wass }
- 18: Update aggregate metrics M agg ← M agg + M h
- 19: end for
- 20: Average aggregate metrics M final ← M agg / | H interest | ▷ Overall multimodality score components
- 21: return M final

## I.3 How GFlowNets deal with Multimodality

The GFlowNet framework possesses inherent properties that make it particularly well-suited for capturing multimodal distributions, explaining the superior performance observed empirically (e.g., Table 2).

1. Sampling Proportional to Reward: The fundamental objective of a GFlowNet is to learn a forward policy P F such that the probability of sampling a complete trajectory τ terminating in state

s f (representing a full forecast) is proportional to its reward R ( s f ) :

<!-- formula-not-decoded -->

If the true underlying process generating the time series can lead to multiple distinct future scenarios (modes), and our reward function R ( τ ) correctly assigns high values to forecasts matching any of these true scenarios, the GFN will naturally learn to sample trajectories corresponding to all highreward modes. Unlike maximum likelihood estimation which might focus on the single most likely mode, or simple regression which might average modes, the GFN objective explicitly encourages exploration and sampling from the entire distribution defined by R ( τ ) .

2. Trajectory Balance and Flow Consistency: The Trajectory Balance loss (Eq. 4) enforces consistency across the entire state space. It ensures that the flow generated by P F matches the flow dictated by the reward R and the backward policy P B . For a distribution R with multiple modes (peaks), the TB objective forces the learned flow F ( s ) and policy P F to correctly distribute probability mass across the different pathways leading to these modes. If the policy were to ignore a significant mode, the flow equations would be violated, leading to a high TB loss. Training implicitly pushes the policy to allocate appropriate probability flow towards all high-reward regions of the state space.

3. Exploration via Entropy Regularization: The entropy bonus term -λ entropy H ( P F ) in the loss function L (section. 3.3.3) explicitly encourages the forward policy to maintain diversity and avoid premature convergence to a single mode (mode collapse). By penalizing overly confident (lowentropy) policies, the training process incentivizes the GFN to keep exploring different trajectories, making it more likely to discover and represent multiple distinct modes present in the reward landscape (and thus, the true data distribution if the reward is well-specified). The empirical results in Figure 10 and 14 support this, showing that entropy regularization leads to better overall reward, likely by preventing mode collapse.

4. Adaptive Quantization's Role: Our adaptive quantization mechanism further enhances multimodality capture. By dynamically allocating more bins (higher resolution) around regions corresponding to emerging or established modes (as suggested by Figure 4), it allows the GFN policy P F to represent the shape and separation of these modes with greater fidelity than a fixed, uniform quantization scheme might permit. This synergy between the GFN's distributional learning objective and the adaptive representation capacity allows Temporal GFN to effectively model complex, multimodal predictive distributions.

That is, the combination of reward-proportional sampling, flow consistency constraints enforced by the TB loss, explicit exploration encouraged by entropy regularization, and enhanced representational capacity from adaptive quantization provides a strong theoretical and practical foundation for Temporal GFN's superior performance in capturing multimodality compared to traditional forecasting approaches.

## J Datasets

These datasets span multiple domains (e.g., energy, finance, healthcare, web traffic, transport, nature) and exhibit varied properties regarding time series length, sampling frequencies (ranging from minutes to yearly), seasonality, trend complexity, noise levels, and potential multimodality. For structured evaluation and analysis of generalization capabilities, these datasets are divided into three distinct categories:

- Pre-training only: A large set of 13 diverse datasets (approx. 795k series) used exclusively for the initial large-scale training phase of Temporal GFN models intended for zero-shot evaluation. These datasets are not used during evaluation.
- In-domain (Benchmark I): A collection of 15 datasets (approx. 97k series) that are included in the training corpus. The final segment (forecast horizon H ) of the time series in this benchmark is held out for evaluating the model's performance on familiar data distributions and tasks.
- Zero-shot (Benchmark II): Aset of 27 datasets (approx. 190k series) that are never seen by the model during any training phase. These are used solely for evaluating the model's ability to generalize its learned forecasting capabilities to entirely new time series and domains without any task-specific fine-tuning.

Table 4: Details of datasets used for experiments, adapted from Ansari et al. [1], partitioned according to their use in training and evaluation for Temporal GFN models.

| Dataset                                          | Domain                                           | Freq.                                            | Num. Series                                      | Series Length                                    | Series Length                                    | Series Length                                    | Prediction                                       |
|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|
|                                                  |                                                  |                                                  |                                                  | min                                              | avg                                              | max                                              | Length ( H )                                     |
| Pretraining-only                                 |                                                  |                                                  |                                                  |                                                  |                                                  |                                                  |                                                  |
| Solar (5 Min.)                                   | energy                                           | 5min                                             | 5166                                             | 105 , 120                                        | 105 , 120                                        | 105 , 120                                        | -                                                |
| Solar (Hourly)                                   | energy                                           | 1h                                               | 5166                                             | 8760                                             | 8760                                             | 8760                                             | -                                                |
| Spanish Energy and Weather                       | energy                                           | 1h                                               | 66                                               | 35 , 064                                         | 35 , 064                                         | 35 , 064                                         | -                                                |
| Taxi (Hourly)                                    | transport                                        | 1h                                               | 2428                                             | 734                                              | 739                                              | 744                                              | -                                                |
| USHCN                                            | nature                                           | 1D                                               | 6090                                             | 5906                                             | 38 , 653                                         | 59 , 283                                         | -                                                |
| Weatherbench (Daily)                             | nature                                           | 1D                                               | 225 , 280                                        | 14 , 609                                         | 14 , 609                                         | 14 , 610                                         | -                                                |
| Weatherbench (Hourly)                            | nature                                           | 1h                                               | 225 , 280                                        | 350 , 633                                        | 350 , 639                                        | 350 , 640                                        | -                                                |
| Weatherbench (Weekly)                            | nature                                           | 1W                                               | 225 , 280                                        | 2087                                             | 2087                                             | 2087                                             | -                                                |
| Wiki Daily (100k)                                | web                                              | 1D                                               | 100 , 000                                        | 2741                                             | 2741                                             | 2741                                             | -                                                |
| Wind Farms (Hourly)                              | energy                                           | 1h                                               | 337                                              | 1715                                             | 8514                                             | 8784                                             | -                                                |
| In-domain evaluation (Benchmark I)               | In-domain evaluation (Benchmark I)               |                                                  |                                                  |                                                  |                                                  |                                                  |                                                  |
| Electricity (15 Min.)                            | energy                                           | 15min                                            | 370                                              | 16 , 032                                         | 113 , 341                                        | 140 , 256                                        | 24                                               |
| Electricity (Hourly)                             | energy                                           | 1h                                               | 321                                              | 26 , 304                                         | 26 , 304                                         | 26 , 304                                         | 24                                               |
| Electricity (Weekly)                             | energy                                           | 1W                                               | 321                                              | 156                                              | 156                                              | 156                                              | 8                                                |
| KDD Cup 2018                                     | nature                                           | 1h                                               | 270                                              | 9504                                             | 10 , 897                                         | 10 , 920                                         | 48                                               |
| London Smart Meters                              | energy                                           | 30min                                            | 5560                                             | 288                                              | 29 , 951                                         | 39 , 648                                         | 48                                               |
| M4 (Daily)                                       | various                                          | 1D                                               | 4227                                             | 107                                              | 2371                                             | 9933                                             | 14                                               |
| M4 (Hourly)                                      | various                                          | 1h                                               | 414                                              | 748                                              | 901                                              | 1008                                             | 48                                               |
| Pedestrian Counts                                | transport                                        | 1h                                               | 66                                               | 576                                              | 47 , 459                                         | 96 , 424                                         | 48                                               |
| Rideshare                                        | transport                                        | 1h                                               | 2340                                             | 541                                              | 541                                              | 541                                              | 24                                               |
| Taxi (30 Min.)                                   | transport                                        | 30min                                            | 2428                                             | 1469                                             | 1478                                             | 1488                                             | 48                                               |
| Temperature-Rain                                 | nature                                           | 1D                                               | 32 , 072                                         | 725                                              | 725                                              | 725                                              | 30                                               |
| Uber TLC (Daily)                                 | transport                                        | 1D                                               | 262                                              | 181                                              | 181                                              | 181                                              | 7                                                |
| Uber TLC (Hourly)                                | transport                                        | 1h                                               | 262                                              | 4344                                             | 4344                                             | 4344                                             | 24                                               |
| Zero-shot evaluation (Benchmark II)              | Zero-shot evaluation (Benchmark II)              | Zero-shot evaluation (Benchmark II)              | Zero-shot evaluation (Benchmark II)              |                                                  |                                                  |                                                  |                                                  |
| Australian Electricity                           | energy                                           | 30min                                            | 5                                                | 230 , 736                                        | 231 , 052                                        | 232 , 272                                        | 48                                               |
| CIF 2016                                         | banking                                          | 1M                                               | 72                                               | 28                                               | 98                                               | 120                                              | 12                                               |
| Car Parts                                        | retail                                           | 1M                                               | 2674                                             | 51                                               | 51                                               | 51                                               | 12                                               |
| Covid Deaths                                     | healthcare                                       | 1D                                               | 266                                              | 212                                              | 212                                              | 212                                              | 30                                               |
| Dominick                                         | retail                                           | 1D                                               | 100 , 014                                        | 201                                              | 296                                              | 399                                              | 8                                                |
| ERCOT Load                                       | energy                                           | 1h                                               | 8                                                | 154 , 854                                        | 154 , 854                                        | 154 , 854                                        | 24                                               |
| ETT (15 Min.)                                    | energy                                           | 15min                                            | 14                                               | 69 , 680                                         | 69 , 680                                         | 69 , 680                                         | 24                                               |
| ETT (Hourly)                                     | energy                                           | 1h                                               | 14                                               | 17 , 420                                         | 17 , 420                                         | 17 , 420                                         | 24                                               |
| Exchange Rate                                    | finance                                          | 1B                                               | 8                                                | 7588                                             | 7588                                             | 7588                                             | 30                                               |
| FRED-MD                                          | economic                                         | 1M                                               | 107                                              | 728                                              | 728                                              | 728                                              | 12                                               |
| Hospital                                         | healthcare                                       | 1M                                               | 767                                              | 84                                               | 84                                               | 84                                               | 12                                               |
| M1 (Monthly)                                     | various                                          | 1M                                               | 617                                              | 48                                               | 90                                               | 150                                              | 18                                               |
| M3 (Monthly)                                     | various                                          | 1M                                               | 1428                                             | 66                                               | 117                                              | 144                                              | 18                                               |
| M4 (Quarterly)                                   | various                                          | 3M                                               | 24 , 000                                         | 24                                               | 100                                              | 874                                              | 8                                                |
| M5                                               | retail                                           | 1D                                               | 30 , 490                                         | 124                                              | 1562                                             | 1969                                             | 28                                               |
| NN5 (Daily)                                      | finance                                          | 1D                                               | 111                                              | 791                                              | 791                                              | 791                                              | 56                                               |
| NN5 (Weekly)                                     | finance                                          | 1W                                               | 111                                              | 113                                              | 113                                              | 113                                              | 8                                                |
| Tourism (Monthly) Tourism (Quarterly)            | various various                                  | 1M                                               | 366                                              | 91                                               | 298                                              | 333                                              | 24                                               |
|                                                  |                                                  | 1Q                                               | 427                                              | 30                                               | 99                                               | 130                                              | 8                                                |
| Tourism (Yearly)                                 | various                                          | 1Y                                               | 518                                              | 11                                               | 24                                               | 47                                               | 4                                                |
| Traffic Weather                                  | transport nature                                 | 1h 1D                                            | 862 3010                                         | 17 , 544 1332                                    | 17 , 544 14 , 296                                | 17 , 544 65 , 981                                | 24 30                                            |
| Zero-shot evaluation (Benchmark II - Healthcare) | Zero-shot evaluation (Benchmark II - Healthcare) | Zero-shot evaluation (Benchmark II - Healthcare) | Zero-shot evaluation (Benchmark II - Healthcare) | Zero-shot evaluation (Benchmark II - Healthcare) | Zero-shot evaluation (Benchmark II - Healthcare) | Zero-shot evaluation (Benchmark II - Healthcare) | Zero-shot evaluation (Benchmark II - Healthcare) |
| MIMIC-III Heart Rate                             | healthcare                                       | 1h                                               | 500                                              | 100                                              | 240                                              | 720                                              | 48                                               |
| MIMIC-III SpO2                                   | healthcare                                       | 1h                                               | 500                                              | 100                                              | 240                                              | 720                                              | 48                                               |
| MIMIC-III Resp. Rate                             | healthcare                                       | 1h                                               | 450                                              | 100                                              | 230                                              | 680                                              | 24                                               |

## K List of Figures

Figure 2: Reward and Entropy dynamics over training and their relationship for different configurations.

<!-- image -->

Figure 3: Quantization of sorted continuous data points. Visualizes the reduction in MSE (red area) as K increases from 5 to 50.

<!-- image -->

<!-- image -->

Figure 4: Comparison of learned bin probability distributions (K=10). Adaptive Quantization learns a non-uniform distribution, contrasting with Fixed Uniform. Without STE, the policy fails to learn the target (Ideal). Entropy values reflect uniformity.

<!-- image -->

Parameter Dimension 1

Parameter Dimension1

Figure 5: Continuous policy energy landscape (left) vs. its quantized approximation (right, example for fixed K) on synthetic data. Adaptive quantization dynamically optimizes bin placement to better capture the modes (A, B, C).

Figure 6: Trade-off between bin distribution entropy (uniformity) and quantization accuracy (Normalized MSE 3 ) for different numbers of bins (K). Color/size indicates K. Increasing K generally improves accuracy (reduces error) but may slightly decrease entropy if adaptation concentrates bins.

<!-- image -->

Note on Normalized MSE axis: The displayed MSE values were transformed for visualization via 1 -( MSE / max( MSE )) . Therefore, higher values on this axis correspond to lower original MSE (better accuracy), facilitating comparison where higher values are generally desirable across axes.

Figure 7: Gradient variance: STE (green) vs. Gumbel-Softmax (orange). STE is more stable.

<!-- image -->

Figure 8: Visualization comparing uniform vs. data-driven binning strategies.

<!-- image -->

Figure 9: Performance improvements across metrics when implementing enhancements.

<!-- image -->

Figure 10: Left: Coupled evolution of Reward and Entropy over training epochs for Adaptive (purple/blue) vs. Fixed (teal) Quantization and No Entropy Regularization (green/lime). Right: Reward vs. Entropy scatter plot, illustrating the operating regimes.

<!-- image -->

Figure 11: Training loss: STE (green) vs. Gumbel-Softmax (red). GS shows faster initial convergence.

<!-- image -->

Figure 12: Relative resource requirements: STE (baseline) vs. Gumbel-Softmax.

<!-- image -->

## Practical Approach Selection Guide

| Model Type                  | Better Approach   |
|-----------------------------|-------------------|
| Small models (< 10M params) | STE               |
| Complex distributions       | STE               |
| Very largemodels            | Gumbel-Softmax    |
| Resource-constrained        | STE               |
| Implementationsimplicity    | STE               |

STEispreferredformostpracticalapplications

Figure 13: Practical guide for selecting STE vs. Gumbel-Softmax.

<!-- image -->

Figure 14: Average reward evolution ablating key components. "No STE" (lime green) shows near-zero reward, highlighting its necessity for learning.

<!-- image -->

TrainingSteps

Figure 15: Trajectory diversity metric over training steps, showing sustained diversity.

Figure 16: Ratio of unique actions utilized during training, indicating exploration breadth.

<!-- image -->

Figure 17: Optimization paths: Continuous (left) vs. Quantized with STE (middle). Energy during optimization (right) shows convergence for both continuous (solid) and quantized (dashed) paths.

<!-- image -->

Figure 18: Calibration curves for different forecasting models.

<!-- image -->

Figure 19: Breakdown and evolution of reward components during training.

<!-- image -->

<!-- image -->

Score (Higher is Better)

Figure 20: Multimodality performance metrics comparing forecasting methods. The figure presents three visualizations of comparative metrics. Left: Radar chart showing five key multimodality metrics across all methods, with Temporal GFN (green) consistently outperforming other approaches across all dimensions. Top right: Bar chart of distributional accuracy metrics (CRPS and Energy Score) where lower values indicate better performance; Temporal GFN achieves the lowest scores. Bottom right: Horizontal bar chart of Mode Coverage showing Temporal GFN's superior ability (92%) to identify and assign appropriate probability to true modes compared to other methods..

Figure 21: Gradient flow fields: Continuous (left) vs. Quantized with STE (right, K=16). STE effectively approximates the true gradient directions.

<!-- image -->

Figure 22: Effect of applying recalibration methods.

<!-- image -->

## Conformal Prediction Calibration Improvement 1.0

Figure 23: Temporal GFN calibration improvement with Conformal Prediction.

<!-- image -->

Figure 24: Comparison of key metrics across different model configurations.

<!-- image -->

## L Broader Impacts

The development of Temporal Generative Flow Networks (Temporal GFNs) for probabilistic time series forecasting, while primarily an advancement in machine learning methodology, carries a spectrum of potential societal impacts that merit careful consideration. The most significant positive impacts stem from the potential to enhance decision-making in critical domains where understanding uncertainty is paramount. In healthcare, for instance, the ability of Temporal GFNs to provide more accurate and well-calibrated probabilistic forecasts for patient vital signs or physiological signals could lead to earlier detection of adverse events, optimized resource allocation in intensive care units, and more personalized treatment strategies. The framework's capacity to capture multimodality is particularly valuable here, as patients may exhibit distinct future health trajectories. Similar benefits extend to finance, where reliable probabilistic forecasts can improve risk management and portfolio optimization; to energy systems, where better predictions of demand and renewable generation can enhance grid stability and efficiency; and to climate science, where nuanced forecasts with uncertainty can better inform policy. Beyond specific applications, by offering a new way to learn distributions over complex sequential behaviors, Temporal GFNs might also contribute to a deeper scientific understanding of the underlying dynamics in diverse systems.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction (Section 1) clearly state the core contributions regarding the Temporal GFN framework, adaptive quantization, STE for differentiability, TB loss, and theoretical bounds, which are then supported by the methodology, theory, and experimental sections.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The Discussion (5) and Conclusion (6) touches upon trade-offs (e.g., STE bias, complexity of data-driven quantization vs. adaptive uniform). A dedicated "Limitations" subsection within the Discussion or Conclusion are elaborated on assumptions (e.g., univariate focus, specific GFN objective) and potential areas for improvement.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: Theoretical results (Theorems C.1 and D.1) are presented in Section B. Assumptions (e.g., Lipschitz reward, bounded variance) are stated, and full proofs are provided in Appendix B.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Section 4 and Appendix J detail the datasets, evaluation strategy, baseline implementations, and Temporal GFN hyperparameters (context length, prediction length, training steps, adaptive parameters, optimizer details). The core methodology, including the algorithm, is described in Appendix A.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We plan to release the source code for Temporal GFNs and scripts for experiments upon publication. Publicly available benchmark datasets are used and cited. Instructions for data access and preprocessing will be provided in the supplementary material/repository README.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: Section E provide details on data splits (Pre-training, Benchmark I, Benchmark II), evaluation horizons, model hyperparameters (e.g., Transformer architecture, GFN parameters, adaptive quantization settings), optimizer (Adam), learning rates, and training duration.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Main benchmark results for Temporal GFN variants are reported as averages over 3 random seeds (Section E). While explicit error bars or statistical tests are not present in all tables/figures in the main text, the multi-seed averaging addresses variability. Appendix figures (e.g., on gradient variance) sometimes show error bars.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: Appendix E mentions training on multi-GPU hardware and typical training duration (e.g., 10,000 steps). The (previously discussed but removed) computational efficiency table provided typical training times per configuration. This information will be consolidated in the appendix.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research aims to advance probabilistic forecasting with a novel method. Standard benchmarks and synthetic data are primarily used. For any healthcare data, ethical considerations regarding data privacy and responsible use would be paramount (to be detailed further if specific private datasets were used and not just MIMIC-III which has its own usage agreements).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: A Broader Impacts section has been included in Appendix L. Positive impacts include improved decision-making in healthcare and other critical domains via better uncertainty quantification. Potential negative impacts could relate to misuse if applied to sensitive personal data without safeguards, or over-reliance on model predictions. Mitigation strategies like emphasizing model calibration and responsible deployment are discussed.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The proposed Temporal GFN model itself, while a generative model for time series, does not inherently pose the same high risk for direct misuse as large-scale image/text generators trained on vast uncurated web data. Benchmark datasets used are standard public datasets. If applied to sensitive healthcare data, access would be governed by existing data use agreements.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Publicly available benchmark datasets (e.g., ETT, M-series, MIMIC-III) are cited with their original sources. Baselines like Chronos, Lag-Llama are cited. Any opensource code leveraged (e.g., for GFN components or evaluation metrics) will be credited in the implementation details or acknowledgments, and licenses respected.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: The primary contribution is a new modeling framework and methodology. If code is released, it will be accompanied by a README and usage instructions. No new large-scale datasets or foundational models are being introduced as assets themselves.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This research does not involve crowdsourcing or direct research with human subjects in terms of data collection for this study. Any healthcare data used (e.g., MIMIC-III) is pre-existing, de-identified, and accessed under established ethical protocols and data use agreements.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The research uses pre-existing, de-identified datasets (like MIMIC-III, public benchmarks) or synthetic data. No new human subject data was collected for this study, thus direct IRB approval for this specific work was not required beyond the original approvals for the datasets themselves.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [No]

Justification: LLMs were not used as a core component of the proposed Temporal GFN methodology or experimental research.