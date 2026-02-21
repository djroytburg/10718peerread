## Neural SDEs as a Unified Approach to Continuous-Domain Sequence Modeling

## Anonymous authors

## Abstract

Inspired by the ubiquitous use of differential equations to model continuous dynamics across diverse scientific and engineering domains, we propose a novel and intuitive approach to continuous sequence modeling. Our method interprets time-series data as discrete samples from an underlying continuous dynamical system , and models its time evolution using Neural Stochastic Differential Equation (Neural SDE), where both the flow (drift) and diffusion terms are parameterized by neural networks. We derive a principled maximum likelihood objective and a simulation-free scheme for efficient training of our Neural SDE model. We demonstrate the versatility of our approach through experiments on sequence modeling tasks across both embodied and generative AI 1 . Notably, to the best of our knowledge, this is the first work to show that SDE-based continuous-time modeling also excels in such complex scenarios, and we hope that our work opens up new avenues for research of SDE models in high-dimensional and temporally intricate domains.

## 1 Introduction

Sequence modeling is a fundamental task in artificial intelligence, underpinning a wide range of applications from natural language processing to time-series analysis [Sutskever, 2014, Graves and Graves, 2012, Graves, 2013]. In recent years, advances in sequence modeling have led to breakthroughs across multiple domains. Models like Generative Pre-trained Transformers (GPT) [Radford et al., 2019, Brown, 2020] have revolutionized language generation, while diffusion models have achieved state-of-the-art results in areas such as image and video generation [Ho et al., 2020, Song and Ermon, 2019, Song et al., 2021, Ho et al., 2022]. These successes underscore the importance of effective sequence modeling techniques in the development of advanced AI systems, both for discrete and continuous data.

Auto-regressive models have been the dominant approach for sequence modeling of discrete variables. This method has proven highly effective for tasks like language generation, where data is naturally discrete and sequential. However, extending auto-regressive models to continuous data is less straightforward. Continuous data often represent

1 Codes and demos are accessible from our project website: https://neuralsde.github.io/ .

Figure 1: A new paradigm for continuousdomain sequence modeling using SDEs. Rather than modeling conditional densities directly, our approach represents dynamics using SDEs. The Fokker-Planck equation provides the theoretical link by describing the evolution of the probability density. This framework unifies embodied and generative AI under the same modeling paradigm.

<!-- image -->

smooth, time-evolving dynamics that are not naturally segmented into discrete tokens. Tokenizing continuous data can introduce quantization errors and obscure important information such as the closeness between states induced by the distance metric of the continuous state space [Argall et al., 2009, Li et al., 2024]. To model such data effectively, it's essential to preserve their inherent continuity and stochasticity.

Recent methods like diffusion models [Ho et al., 2020, Song et al., 2021] and flow matching frameworks [Lipman et al., 2022] have made significant progress in handling continuous variables by leveraging stochastic processes. These approaches transform a simple initial distribution into a complex target distribution through a series of learned transformations, effectively capturing the data distribution in continuous spaces. However, in the context of sequence modeling, this iterative approach can be less natural and computationally intensive, which often involves a high transport cost and requires many iterative steps to produce satisfactory results [Tong et al., 2023, Kornilov et al., 2024]. In contrast, a more intuitive and efficient strategy is to model the sequence by directly learning the transitions between consecutive states, reflecting the inherent temporal continuity of the data.

These limitations highlight the need for sequence models designed specifically for continuous data, capable of capturing both the deterministic trends and stochastic fluctuations of the underlying dynamics without relying on extensive iterative transformations.

Differential Equations as a Unifying Framework. In many real-world applications, continuous dynamical systems are naturally described by ordinary or stochastic differential equations. From physical processes like fluid flow and orbital mechanics [Gianfelici, 2008, Harier et al., 2000] to robotic control and autonomous systems [Zhu, 2023], the underlying dynamics are often given-or well approximated-by state-space differential equations that evolve continuously in time.

By leveraging Neural Stochastic Differential Equations (Neural SDEs) [Kong et al., 2020, Kidger et al., 2021a], we propose a method that models the time evolution of continuous systems directly. This approach maintains the intrinsic continuity of the data and provides a more natural representation of continuous-time processes compared to traditional discrete-time or tokenization-based methods. In summary, the contributions of this paper include:

- A novel Neural SDE framework for continuous sequence modeling utilizing a simulationfree maximum-likelihood method with a decoupled two-stage optimizer, eliminating the costly forward simulations required in traditional gradient estimation.
- Significant improvements in inference efficiency due to directly modeling continuous-time dynamics-rather than discretely transitioning from Gaussian noise as in diffusion and flow matching approaches-leading to substantially fewer function evaluations (NFEs).
- Empirical validations demonstrating our method's unique advantages, including modeling of multi-modal distributions, robustness to irregular dynamics and natural temporal interpolation, while achieving performance comparable to baselines across diverse tasks such as imitation learning and video prediction.

## 2 Backgrounds

In this section, we provide an overview of foundational concepts relevant to our approach. We begin with diffusion models and their extension to continuous stochastic differential equations (SDEs). We then discuss the flow matching and stochastic interpolant frameworks, which are instrumental in formulating our method. Finally, we introduce Neural SDEs and highlight the challenges associated with their training.

## 2.1 Diffusion Models and Their SDE Formulation

Diffusion models have become a cornerstone in generative modeling, achieving remarkable success in generating high-fidelity images and other data types [Ho et al., 2020, Song et al., 2021]. At their core, these models involve a forward process that gradually adds noise to the data and a reverse process that reconstructs the data by denoising.

## 2.1.1 Discrete Diffusion Models

In the discrete setting, the forward diffusion process is defined over discrete time steps, where Gaussian noise is incrementally added to the data. The reverse process involves learning to denoise the corrupted data to recover the original data distribution. This is typically achieved by training a neural network to predict the noise added at each step, using a mean squared error loss between the predicted and true noise.

## 2.1.2 Continuous SDE Formulation

The continuous-time formulation of diffusion models represents the forward and reverse processes as solutions to SDEs.

The forward diffusion process is defined by the SDE:

<!-- formula-not-decoded -->

where x t ∈ R d is the data at time t ∈ [0 , T ] , f ( x t , t ) is the flow coefficient, g ( t ) is the diffusion coefficient, and w t is a standard Wiener process.

In the case of the Variance Preserving (VP) SDE [Song et al., 2021], which corresponds to the discrete diffusion model with a variance schedule β t , the flow and diffusion coefficients are predefined functions given by:

<!-- formula-not-decoded -->

Here, β ( t ) is a continuous-time version of the noise schedule from the discrete model.

## 2.1.3 Reverse SDE and Score Function

To generate data samples, we consider the reverse-time SDE that has the same marginal distribution as the forward SDE:

<!-- formula-not-decoded -->

where q t ( x t ) is the marginal distribution of x t at time t , and ¯ w t is a reverse-time Wiener process. The term ∇ x t log q t ( x t ) is the score function , representing the gradient of the log probability density at time t .

In practice, since q t ( x t ) is unknown, the score function is approximated by a neural network s θ ( x t , t ) , trained to minimize the score-matching objective which corresponds to the denoising objective in the discrete formulation:

<!-- formula-not-decoded -->

where x 0 is the original data sample, x t is obtained by solving the forward SDE starting from x 0 , λ ( t ) is a weighting function, and q ( x t | x 0 ) is the transition kernel of the forward SDE. In the case of linear Gaussian transition as in Eq. (2), ∇ x t log q ( x t | x 0 ) has a simple closed-form solution which is proportional to the sampled noise.

## 2.2 Flow Matching and Stochastic Interpolants

Instead of learning SDEs, Flow matching and Stochastic Interpolants provide frameworks for learning a diffusion-free ordinary differential equation (ODE) for generative modeling.

## 2.2.1 Interpolative Approach

Both flow matching [Lipman et al., 2022] and stochastic interpolants [Albergo et al., 2022] rely on constructing an interpolant between a simple base distribution p 0 ( x ) and a target distribution p 1 ( x ) . The interpolant is defined as a time-dependent process x t that smoothly transitions from x 0 ∼ p 0 to x 1 ∼ p 1 as t goes from 0 to 1 .

An example of a stochastic interpolant is:

<!-- formula-not-decoded -->

where α ( t ) and β ( t ) are deterministic scalar functions satisfying α (0) = 1 , α (1) = 0 , β (0) = 0 , β (1) = 1 , σ ( t ) controls the magnitude of the stochastic component, and ξ ∼ N ( 0 , I ) is standard Gaussian noise.

## 2.2.2 Simulation-Free Training Objective

The key idea is to learn a vector field v θ ( x t , t ) such that the time derivative of the interpolant matches the vector field:

<!-- formula-not-decoded -->

The training objective minimizes the expected squared difference between the model's vector field and the true time derivative of the interpolant:

<!-- formula-not-decoded -->

Because x t and d x t dt can be computed analytically from (5), training does not require simulating the dynamics of x t .

## 2.3 Neural Stochastic Differential Equations

Neural Stochastic Differential Equations (Neural SDEs) [Li et al., 2020a, Kidger et al., 2021b] extend Neural ODEs [Chen et al., 2018] by incorporating stochastic components into the system dynamics. A Neural SDE models the evolution of a stochastic process as:

<!-- formula-not-decoded -->

where f θ and G θ are neural networks parameterized by θ .

Training Neural SDEs involves computing gradients of a loss function with respect to the parameters θ . A common approach is the adjoint sensitivity method [Li et al., 2020b], which computes these gradients by solving an adjoint SDE backward in time alongside the forward simulation. While this method is memory-efficient, it poses computational challenges:

- Computational Overhead : Simulating both the forward and backward SDEs increases computational cost.
- Numerical Stability : Backward integration through stochasticity can introduce numerical instability, as stochastic differential equations are less straightforward to reverse due to their inherent randomness.

These challenges motivate the exploration of alternative training methods that can efficiently handle Neural SDEs without the need for backward simulation through stochastic processes.

## 2.4 Connections and Motivations

The continuous SDE formulation of diffusion models and the flow matching framework provide powerful tools for modeling data distributions through stochastic processes. However, these methods often assume fixed diffusion coefficients or rely on extensive iterative computations. While Neural SDEs have been employed for continuous sequence modeling [Oh et al., 2024, Tzen and Raginsky, 2019], gradient computation under these formulations typically incurs a cost that scales linearly with the number of simulation steps. This results in significant overhead for high-dimensional data or models with large parameter counts, as most existing differentiation methods for Neural SDEs-including pathwise automatic differentiation [Tzen and Raginsky, 2019, Li et al., 2020b], adjoint sensitivity analysis[Li et al., 2020b, Blasingame and Liu, 2024], and likelihood-ratio-based estimators[Glynn, 1990]-require full forward (and sometimes backward) integration of the SDE per training sample.

Our approach leverages Neural SDEs' ability to model both drift and diffusion-faithfully capturing intrinsic data uncertainty and stochasticity-while adopting the simulation-free training paradigm pioneered by flow matching (which itself is derived from an optimal-transport control formulation, not likelihood). Crucially, we re-derive this flow regression from a maximum-likelihood (ML) objective and extend it with a decoupled two-stage optimizer: first fitting the flow via our ML-driven flowmatching analogue, then separately estimating diffusion coefficients through a residual log-likelihood objective. This two-stage scheme entirely avoids forward SDE unrolling.

By formulating the problem as learning the parameters of an SDE that describes the evolution of continuous sequences, we directly model the time dynamics without the need for discretization or high transport costs associated with iterative methods.

## 3 Our Approach

We introduce a Neural SDE framework to model continuous-time sequences, capturing both deterministic trends and stochastic fluctuations inherent in the data. Our model learns a time-invariant SDE of the form:

<!-- formula-not-decoded -->

Our goal is to learn f and g from data such that the SDE accurately represents the underlying continuous-time dynamics observed in the sequences.

Compared with the general SDE (Eq. 8), we are making two key modeling choices:

1. Time-Invariant System: The flow and the diffusion function are implicit function of time through the dependency on the state vector, because many real-world systems exhibit dynamics that are consistent over time [Zhu, 2023, Slotine et al., 1991]. Modeling these systems with a time-invariant SDE simplifies the learning task and captures the essential state-dependent dynamics without unnecessary complexity.
2. Diagonal Diffusion Matrix: For computational tractability, we use a diagonal diffusion matrix:

<!-- formula-not-decoded -->

instead of a full matrix. This implies that the stochastic components affecting each state dimension are independent. The diagonal assumption simplifies calculations, particularly the inversion and determinant of the covariance matrix during training, leading to more efficient optimization. Moreover, in many practical applications, the noise affecting different state variables can be reasonably considered uncorrelated [Gardiner, 2009, Oksendal, 2013].

## 3.1 Negative Log-Likelihood Derivation

Using Euler-Maruyama discretization (detailed in Appendix A) and the resulting Gaussian transition probability (Appendix B), the negative log-likelihood (NLL) for a single transition segment simplifies to:

<!-- formula-not-decoded -->

where ∆ x = x t +∆ t -x t , and we have omitted constant terms that do not depend on the modeling parameters.

This loss function consists of:

- Prediction Error Term (first): Measures how well the flow function f ( x t ) matches the observed state changes ∆ x . Note that this term is scaled by inverse of diffusion term g ( x t ) , which is absent in Flow-Matching methods. The intuition is that a mismatch between the flow and the observed state change can be attributed to either an inaccurate flow prediction or the intrinsic uncertainty of the process measured by the diffusion term.
- Complexity Penalty Term (second): The logarithmic determinant term regularizes overly large diffusion coefficients, preventing the model from assigning high uncertainty indiscriminately.

Given a trajectory of observed data points { x t k } N k =0 at times { t k } N k =0 , we can aggregate the singlesegment losses to obtain the total loss:

<!-- formula-not-decoded -->

Unlike Auto-regressive models where x t k +1 needs to be conditioned on all the previous states, here in our model the conditional distribution is simply between consecutive steps due to the the markovian property of the SDE.

## 3.2 Training Strategy: Decoupled Optimization of Flow and Diffusion

The first term in Eq. (11) depends on both f and g , we empirically observe that joint training is susceptible to getting stuck at local minimum. To enhance training stability and interpretability, we derive a decoupled optimization scheme for the flow f and the diffusion term g .

We notice that Eq. (11) can be analytically minimized w.r.t. g by setting

<!-- formula-not-decoded -->

Plugging Eq. (13) back to Eq. (11) and discarding constant terms, we get the following simplified objective for the flow term

<!-- formula-not-decoded -->

Likewise, the constraint Eq. (13) suggests the following objective for the diffusion term

<!-- formula-not-decoded -->

This objective suggests that the diffusion coefficients are matching the residual error of the flow prediction. Similarly, the training objective for the whole trajectory can be obtained by aggregating all the segments as in Eq. (12).

## 3.3 Implications of the Simplified Flow Objective

The logarithmic-squared flow loss (Eq. 14) provides two primary benefits: (i) scale invariance across different dimensions, which eliminates the need for manually tuning per-dimension loss weights-a property particularly valuable in multi-modal settings where each modality may have distinct units or dynamic ranges; and (ii) robustness to large errors due to sub-linear growth, allowing the model to handle high-variance data by naturally increasing the learned diffusion term rather than incurring large penalties. A detailed analysis and discussion appear in Appendix C.

## 3.4 Implementation Details

Our main derivation thus far focuses on the theoretical formulation. In practice, we make several additional choices to improve training stability and performance. For example, we employ data interpolation between observed time steps, noise injection for regularization, and an optional denoiser network for improving inference-time trajectory likelihood. We also discuss how we handle the numerical stability of the logarithmic loss and datasets without explicit time. Full details about these implementation aspects are provided in Appendix D.

## 4 Experiments

In this section, we empirically evaluate the capabilities of Neural SDE across three distinct sequence modeling tasks: (1) a 2D Bifurcation task designed to assess multi-modal trajectory generation, (2) the Push-T imitation learning task, and (3) video prediction on standard benchmark datasets. Details about the setup are provided in the Appendix E.

## 4.1 2D Branching Trajectories

Accurately capturing multi-modal distributions is crucial for sequence modeling tasks. To evaluate the ability of our model to capture multi-modal distributions, we designed a simple 2D trajectory generation task, where the ground truth trajectories exhibit a Y-shaped bifurcation pattern.

In Appendix E.1, Figure 4 shows the generated trajectories for our approach and diffusion/flowmatching approaches at both low and high densities (number of steps per trajectory). At the low density, all three models successfully capture the bimodal distribution, producing trajectories that

Figure 2: Inference Efficiency . The plots show the performance of Neural SDE, Flow Matching, and PFI on the KTH and CLEVRER datasets, measured by the metrics FVD, JEDI, SSIM, and PSNR, with respect to the number of function evaluations (NFE). Lower FVD and JEDI and higher SSIM and PSNR indicate better performance. To control the NFEs of PFI and Neural SDE, we use fixed step sizes. All metrics are estimated by sampling 256 test video sequences (with replacement), and the evaluation procedure is repeated 4 times. We report the mean performance across the 4 runs, along with the 95% confidence interval bands.

<!-- image -->

branch as expected. However, a significant difference occurs at the high density. While Neural SDE still captures the two branches, DDIM and Rectified Flow fail. We speculate that this degradation in performance for DDIM and Rectified Flow at higher density is due to covariate-shift.

To verify, we show the contribution of different components within our model in Figure 5. Comparing with the Flow and Flow+Diffusion plots, we see that the addition of the diffusion term introduces stochasticity into the trajectories, leading to branching. The Flow+Denoiser+Diffusion plot shows that incorporating the denoiser effectively mitigates covariate-shift.

## 4.2 Push-T

To evaluate the effectiveness of our Neural SDE in imitation learning, we employ the Push-T task. Details about the setup are provided in appendix E.2.

Results Table 1 shows the performance of our Neural SDE compared to the Diffusion Policy baseline. Our model achieves competitive performance, demonstrating its effective-

Table 1: Target area coverage (TAC) comparison on PushT. Baseline results (trained on a Transformer backbone) are from Chi et al. [2023]. Neural SDE uses an MLP composed of four residual blocks, each containing two linear layers with a skip connection.

| Method   |   LSTM-GMM |   IBC |   BET |   DP |   NSDE |
|----------|------------|-------|-------|------|--------|
| TAC ↑    |       0.67 |   0.9 |  0.79 | 0.95 |   0.92 |

ness in policy learning. This is notable given that SDEs often assume some level of smoothness in the underlying dynamics, while imitation learning often involves non-smooth dynamics. To further investigate the ability of our model to handle trajectories with sharp transitions, we visualize a representative push-T trajectory in Figure 6. This figure shows that Neural SDE can effectively model such non-smooth behavior, demonstrating its robust sequence-modeling capability.

## 4.3 Video Prediction

To assess our model's ability to learn complex temporal dynamics, we evaluate its performance on two video prediction benchmarks: the KTH Schuldt et al. [2004] dataset and the CLEVRER [Yi et al., 2019] dataset. Details about the setup are provided in appendix E.3.

Quantitative Comparison and Inference Efficiency Our Neural SDE model achieves comparable results to the Flow Matching [Lipman et al., 2022] and Probabilistic Forecasting with Interpolants (PFI) [Chen et al., 2024] baselines across multiple metrics, including FVD Unterthiner et al. [2019], JEDI [Bicsi et al., 2023] , SSIM [Wang et al., 2004], and PSNR [Hore and Ziou, 2010], on the KTH and CLEVRER datasets. One of the most compelling advantages of our method is its inference efficiency. Figure 2 shows the relationship between the number of function evaluations (NFE) and performance metrics. Our approach significantly outperforms the baselines in efficiency, requiring only 2 steps to achieve results comparable to Flow Matching and PFI, which typically need 5 to 20 steps to generate future frames of reasonable quality. This substantial reduction in NFE highlights the

intrinsic efficiency of directly modeling continuous-time dynamics, making it an appealing solution to sequence modeling tasks of high computational complexity.

Temporal Resolution and Implicit Interpolation Our Neural SDE approach also offers an appealing capability of improving temporal resolution without incurring additional training cost. As demonstrated in Appendix F, Neural SDEs can generate coherent intermediate frames between two consecutive frames in the original dataset. In contrast, flow-based methods like Flow Matching are limited to generating frames at the specific temporal resolution defined by the training data, requiring retraining or ad hoc interpolation techniques to achieve higher frame rates. This "free" interpolation capability underscores the power of modeling video as a continuous process.

Scalability Analysis We further investigated the scalability of our Neural SDE model by analyzing the relationship between model size and validation loss. Following the methodology in [Kaplan et al., 2020, Tian et al., 2024], we varied the depth and hidden dimensions of the U-ViT

Figure 3: Scaling law of Neural SDE with U-ViT backbone on the CLEVRER dataset. The Pearson correlation coefficient ( -0 . 9988 ) indicates a strong power-law relationship, suggesting that increasing model size leads to improved performance.

<!-- image -->

backbone used in our model, resulting in models ranging from 0 . 13 million to 113 . 44 million parameters, see Appendix H for detail.

Figure 3 shows the scaling behavior of our model. We observe a clear power-law relationship between the number of model parameters and the validation loss. This scaling behavior suggests that our Neural SDE architecture can leverage larger model capacities to achieve better video prediction performance.

## 5 Related Works

A large class of related works are bridging-based generative models, including diffusion approaches [Ho et al., 2020, Song et al., 2021], flow-based or flow-matching frameworks [Dinh et al., 2016, Ho et al., 2019, Kobyzev et al., 2020, Lipman et al., 2022, Hu et al., 2024], stochastic interpolants [Albergo et al., 2022, Chen et al., 2024, Albergo et al., 2023], and Schrödinger bridges [De Bortoli et al., 2021, Liu et al., 2023]. Despite varying details, these methods commonly start from a simple (often Gaussian) prior distribution and transport it to the target data distribution via learned transformations. This 'from-noise-to-data' perspective has driven state-of-the-art results in static, unpaired data domains (e.g., image generation). However, two main issues arise when directly applying such frameworks to continuous time-series : (1) they generally unroll samples all the way from an uninformative prior, incurring large transport costs for dense or high-dimensional sequences; and (2) they often treat time as a 'pseudo-time' schedule (e.g., noise levels) rather than the actual chronological axis of the system. As a result, bridging-based methods can be less intuitive and potentially suboptimal for tasks where the true temporal progression-and possibly irregular sampling-is essential.

Meanwhile, Neural SDE approaches [Kong et al., 2020, Li et al., 2020a, Kidger et al., 2021a, Liu et al., 2020, Park et al., 2021, Djeumou et al., 2023, Tzen and Raginsky, 2019, Oh et al., 2024] have emerged to incorporate Brownian noise into Neural ODEs, targeting time-series forecasting or sporadic data interpolation. For example, SDE-Net [Kong et al., 2020] and Neural SDE [Li et al., 2020a] inject noise into network layers for uncertainty estimation or robustness, while others develop specialized SDE solvers or variational methods to handle irregular temporal observations [Kidger et al., 2021a, Liu et al., 2020, Oh et al., 2024]. Although conceptually related, these works often employ complex training objectives (e.g., forward-backward SDE solves or Bayesian evidence bounds) and focus on low-dimensional time-series datasets rather than more complex tasks as imitation learning or video prediction.

By contrast, our Neural SDE framework adopts a direct maximum-likelihood approach for learning continuous-time dynamics from observed consecutive states. Rather than bridging data from a simple noise prior, each pair of adjacent observations is treated as a sample from an underlying SDE, bypassing the overhead of unrolling from isotropic noise. This design both respects real time evolution -via Flow and diffusion that capture deterministic trends and random fluctuations-and streamlines conditional modeling , since every transition likelihood is straightforwardly evaluated in a Markovian manner. Consequently, our method is more natural for embodied and generative AI , where continuous trajectories must be modeled reliably, yet classical bridging-based methods may be unnecessarily complex and less aligned with genuine temporal structure.

## 6 Limitations and Future Work

Our formulation rests on a time invariant Markov assumption: the transition kernel p ( x t +∆ t | x t ) is time-independent, so past influences are encoded only through the current state. It can miss longrange non-Markovian dependencies; on KTH, for instance, visually similar clips occasionally trigger unintended action switches. Moving forward, relaxing this assumption by introducing time-varying dynamics or slower-evolving latent variables could provide the model with a memory mechanism to better capture long-horizon dependencies. Second, we model diffusion coefficients with a diagonal covariance for tractability; this excludes cross-dimensional noise correlations and may under-represent uncertainty in coupled systems. Exploring richer noise models such as full or low-rank covariance structures could allow the model to represent inter-variable stochastic coupling without sacrificing efficiency. Third, the weighting coefficient α in the denoiser network is manually selected per dataset, adding another hyperparameter. Automating the selection of α either by learning it directly from data or adapting it dynamically-could eliminate manual tuning and improve generalization across tasks.

## 7 Conclusions

We have presented a novel Neural SDE framework for continuous-domain sequence modeling, offering an alternative to existing bridging-based approaches such as diffusion and flow-matching methods. By learning both drift and diffusion terms via a direct maximum likelihood objective, our method naturally captures the stochastic and deterministic components of time-evolving data. Moreover, the Markovian formulation circumvents the need for iterative unrolling from a simple noise prior, thus reducing transport costs and simplifying inference.

Through experiments on multiple domains, we demonstrated that Neural SDEs (1) faithfully model multi-modal distributions, (2) handle sharp or irregular dynamics, (3) generate high-quality predictions with few inference steps, and (4) offer 'free' temporal interpolation beyond the training schedule. Our analysis also highlights key properties such as the scale invariance of the log-flow loss and the interpretability gained by explicitly modeling diffusion.

Looking ahead, an interesting extension would be to incorporate additional conditional variables for capturing longer histories, rather than the current approach of augmenting only the latest state. This could improve performance in tasks requiring extended temporal memory. Another promising direction involves tackling noisy actions in real-world embodied AI by focusing on state or observation transitions paired with a learned inverse-dynamics model-alleviating the need for accurately recorded actions. We hope this work stimulates further study into Neural SDEs as a unified, principled approach to continuous-domain sequence modeling, bridging the gap between traditional differential equation frameworks and modern machine learning techniques.

## References

- Michael S. Albergo, Gurtej Kanwar, and Phiala E. Shanahan. Building normalizing flows with stochastic interpolants. In International Conference on Machine Learning , pages 213-224. PMLR, 2022.
- Michael S Albergo, Mark Goldstein, Nicholas M Boffi, Rajesh Ranganath, and Eric Vanden-Eijnden. Stochastic interpolants with data-dependent couplings. arXiv preprint arXiv:2310.03725 , 2023.
- Brenna D. Argall, Sonia Chernova, Manuela Veloso, and Brett Browning. A survey of robot learning from demonstration. Robotics and Autonomous Systems , 57(5):469-483, 2009.

- Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 22669-22679, 2023.
- Mustafa Bayram, Tugcem Partal, and Gulsen Orucova Buyukoz. Numerical methods for simulation of stochastic differential equations. Advances in Difference Equations , 2018:1-10, 2018.
- Lucian Bicsi, Bogdan Alexe, Radu Tudor Ionescu, and Marius Leordeanu. Jedi: Joint expert distillation in a semi-supervised multi-dataset student-teacher scenario for video action recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 953-962, 2023.
- Zander W. Blasingame and Chen Liu. Adjointdeis: Efficient gradients for diffusion models. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems , volume 37, pages 2449-2483. Curran Associates, Inc., 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ file/04badd3b048315c8c3a0ca17eff723d7-Paper-Conference.pdf .
- Tom B. et al. Brown. Language models are few-shot learners. In Advances in Neural Information Processing Systems , volume 33, pages 1877-1901, 2020.
- Tian Qi Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary differential equations. In Advances in Neural Information Processing Systems , volume 31, pages 6571-6583, 2018.
- Yifan Chen, Mark Goldstein, Mengjian Hua, Michael S Albergo, Nicholas M Boffi, and Eric VandenEijnden. Probabilistic forecasting with stochastic interpolants and follmer processes. arXiv preprint arXiv:2403.13724 , 2024.
- Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research , page 02783649241273668, 2023.
- Aram Davtyan, Sepehr Sameni, and Paolo Favaro. Efficient video prediction via sparsely conditioned flow matching. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 23263-23274, 2023.
- Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion schrödinger bridge with applications to score-based generative modeling. Advances in Neural Information Processing Systems , 34:17695-17709, 2021.
- Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp. arXiv preprint arXiv:1605.08803 , 2016.
- Franck Djeumou, Cyrus Neary, and Ufuk Topcu. How to learn and generalize from three minutes of data: Physics-constrained and uncertainty-aware neural stochastic differential equations. 7th Conference on Robot Learning , 2023.
- Crispin Gardiner. Stochastic methods , volume 4. Springer Berlin Heidelberg, 2009.
- Francesco Gianfelici. Numerical solutions of stochastic differential equations (kloeden, pk and platen, e.; 2008)[book reviews]. IEEE Transactions on Neural Networks , 19(11):1990-1991, 2008.
- Peter W Glynn. Likelihood ratio gradient estimation for stochastic systems. Communications of the ACM , 33(10):75-84, 1990.
- Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850 , 2013.
- Alex Graves and Alex Graves. Long short-term memory. Supervised sequence labelling with recurrent neural networks , pages 37-45, 2012.
- Ernst Harier, Christian Lubich, and Gerhard Wanner. Geometric numerical integration. StructurePreserving Algorithms for Ordinary , 2000.

- Jonathan Ho, Xi Chen, Aravind Srinivas, Yan Duan, and Pieter Abbeel. Flow++: Improving flowbased generative models with variational dequantization and architecture design. In International conference on machine learning , pages 2722-2730. PMLR, 2019.
- Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems , volume 33, pages 6840-6851, 2020.
- Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems , 35:8633-8646, 2022.
- Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In 2010 20th international conference on pattern recognition , pages 2366-2369. IEEE, 2010.
- Xixi Hu, Bo Liu, Xingchao Liu, and Qiang Liu. Adaflow: Imitation learning with variance-adaptive flow-based policies. arXiv preprint arXiv:2402.04292 , 2024.
- Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361 , 2020.
- Patrick Kidger, James Foster, Xuechen Chen Li, and Terry Lyons. Efficient and accurate gradients for neural sdes. Advances in Neural Information Processing Systems , 34:18747-18761, 2021a.
- Patrick Kidger, James Morrill, James Foster, and Terry Lyons. Neural sdes as infinite-dimensional gans. In International Conference on Machine Learning , pages 5453-5463. PMLR, 2021b.
- Ivan Kobyzev, Simon JD Prince, and Marcus A Brubaker. Normalizing flows: An introduction and review of current methods. IEEE transactions on pattern analysis and machine intelligence , 43 (11):3964-3979, 2020.
- Lingkai Kong, Jimeng Sun, and Chao Zhang. Sde-net: Equipping deep neural networks with uncertainty estimates. arXiv preprint arXiv:2008.10546 , 2020.
- Nikita Kornilov, Petr Mokrov, Alexander Gasnikov, and Alexander Korotin. Optimal flow matching: Learning straight trajectories in just one step. arXiv preprint arXiv:2403.13117 , 2024.
- Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. In Advances in Neural Information Processing Systems (NeurIPS) , 2024. Spotlight Presentation.
- Xuechen Li, Ting-Kam Leonard Wong, Ricky T. Q. Chen, and David Duvenaud. Scalable gradients for stochastic differential equations. In International Conference on Artificial Intelligence and Statistics , pages 3870-3882. PMLR, 2020a.
- Xuechen Li, Ting-Kam Leonard Wong, Ricky TQ Chen, and David Duvenaud. Scalable gradients for stochastic differential equations. In International Conference on Artificial Intelligence and Statistics , pages 3870-3882. PMLR, 2020b.
- Yaron Lipman, Matthew Tancik, and Jiajun Lu. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 , 2022.
- Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A Theodorou, Weili Nie, and Anima Anandkumar. I 2 sb: Image-to-image schrödinger bridge. arXiv preprint arXiv:2302.05872 , 2023.
- Yingru Liu, Yucheng Xing, Xuewen Yang, Xin Wang, Jing Shi, Di Jin, and Zhaoyue Chen. Learning continuous-time dynamics by stochastic differential networks. arXiv preprint arXiv:2006.06145 , 2020.
- YongKyung Oh, Dong-Young Lim, and Sungil Kim. Stable neural stochastic differential equations in analyzing irregular time series data. International Conference on Learning Representations , 2024.
- Bernt Oksendal. Stochastic differential equations: an introduction with applications . Springer Science &amp; Business Media, 2013.

- Sung Woo Park, Kyungjae Lee, and Junseok Kwon. Neural markov controlled sde: Stochastic optimization for continuous-time data. In International Conference on Learning Representations , 2021.
- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners, 2019. URL https://cdn.openai.com/better-language-models/language\_models\_are\_ unsupervised\_multitask\_learners.pdf . Technical report, OpenAI.
- Christian Schuldt, Ivan Laptev, and Barbara Caputo. Recognizing human actions: a local svm approach. In Proceedings of the 17th International Conference on Pattern Recognition, 2004. ICPR 2004. , volume 3, pages 32-36. IEEE, 2004.
- Jean-Jacques E Slotine, Weiping Li, et al. Applied nonlinear control , volume 199. Prentice hall Englewood Cliffs, NJ, 1991.
- Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems , volume 32, pages 1195-1205, 2019.
- Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations , 2021.
- I Sutskever. Sequence to sequence learning with neural networks. arXiv preprint arXiv:1409.3215 , 2014.
- Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905 , 2024.
- Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid RectorBrooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. arXiv preprint arXiv:2302.00482 , 2023.
- Belinda Tzen and Maxim Raginsky. Neural stochastic differential equations: Deep latent gaussian models in the diffusion limit. arXiv preprint arXiv:1905.09883 , 2019.
- Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. 2019.
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems , 30, 2017.
- Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing , 13(4):600-612, 2004.
- Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. Clevrer: Collision events for video representation and reasoning. International Conference on Learning Representations , 2019.

Quanxin Zhu. Nonlinear systems . MDPI-Multidisciplinary Digital Publishing Institute, 2023.

## A Euler-Maruyama Discretization

To work with discrete data, we discretize the continuous SDE using the Euler-Maruyama method [Bayram et al., 2018]. Over a small time interval ∆ t , the discretized SDE approximates the evolution of x t as:

<!-- formula-not-decoded -->

where ∆ w t = w t +∆ t -w t is the increment of the Wiener process over ∆ t . Since ∆ w t ∼ N ( 0 , ∆ t I d ) , the stochastic term introduces Gaussian noise with covariance proportional to ∆ t .

## B Transition Probability

Under this discretization, the conditional distribution of the next state x t +∆ t given the current state x t is Gaussian:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where:

Under the diagonal form of g ( x t ) , the covariance matrix Σ t simplifies to:

<!-- formula-not-decoded -->

This probabilistic description allows us to compute the likelihood of observed data under our model.

## C Implications of the Simplified Flow Objective

The simplified flow objective in Eq. (14) introduces two key advantages:

Scale-Invariance: The logarithmic squared loss imparts a scale-invariant property to the objective function. Specifically, scaling each dimension of the flow function f i ( x t ) and the observed rate ∆ x i ∆ t i by independent positive constants c i adds only a constant term to the loss:

<!-- formula-not-decoded -->

The additive constants log c 2 i do not affect optimization with respect to model parameters. This property is particularly beneficial when dealing with multi-modal data where different dimensions have varying units or scales-as in embodied AI scenarios involving diverse physical quantities. It eliminates the need for manual weighting or scaling of loss terms across dimensions.

Robustness to Large Errors The logarithmic function grows sub-linearly, making the loss function more tolerant of large errors compared to a standard squared loss. This aligns with the model's treatment of uncertainty via the diffusion term g ( x t ) . Larger discrepancies between the predicted flow and observed changes are accommodated as increased stochasticity rather than penalized heavily. This characteristic enables the model to handle systems with inherent variability more effectively, without being unduly influenced by outliers or noise.

## D Implementation Details

Algorithm Overview Our Neural SDE learning algorithm consists of the following high-level steps:

## Algorithm 1 Neural SDE Learning Algorithm

Require: Dataset of transition tuples D = { x t i , x t i +1 } N i =1

Require: Initialized neural networks: flow f ( x ; θ f ) , diffusion σ 2 ( x ; θ g ) , and optional denoiser d ( x ; θ d )

Ensure: Trained parameters θ f , θ g , θ d

- 1: for each mini-batch transitions in D do
- 2: Interpolate between observed states to obtain x τ
- 3: Add noise to interpolated states: ˜ x τ = x τ + η
- 4: Update flow network parameters θ f using the flow loss (Eq.14)
- 5: Update diffusion network parameters θ g using the diffusion loss (Eq.15)
- 6: (Optional) Update denoiser network parameters θ d using the denoising score matching loss (Eq.22)
- 7: end for

State Interpolation for Training To augment the training data and improve generalization, we interpolate between observed discrete states, similar to techniques used in flow matching and diffusion models. For each pair of consecutive observed states x t k and x t k +1 , we sample intermediate times τ ∈ [ t k , t k +1 ] uniformly and generate linearly interpolated states x τ . This strategy can be interpreted as using a modified discretization scheme that introduces additional evaluation points without altering the order of discretization error inherent in the Euler-Maruyama method. By incorporating interpolated states, we effectively increase the diversity of training samples and encourage the model to capture the underlying continuous dynamics more accurately.

Noise Injection Inspired by the stochastic interpolants framework [Albergo et al., 2022], we add random noise to the interpolated states during training. Specifically, we perturb the states with Gaussian noise: ˜ x τ = x τ + η , where η ∼ N ( 0 , σ 2 I ) . This can be seen as modifying the discretization scheme to account for stochastic variability, while maintaining the same order of discretization error. Noise injection acts as an implicit regularizer, enhancing the model's robustness to data perturbations and encouraging the diffusion function g ( x t ) to account for intrinsic uncertainty in the data.

Incorporating a Denoiser Network To ensure that the inferred trajectories remain close to highprobability regions of the data distribution, we integrate a denoiser network into the model. Specifically, we augment the flow function with an estimate of the score function trained via denoising score matching, leading to the modified SDE:

<!-- formula-not-decoded -->

Here, we learn a denoiser network d ( x t ) to approximate ∇ x t log p ( x t ) by training on the denoising score matching (DSM) objective:

<!-- formula-not-decoded -->

where ϵ is an isotropic Gaussian perturbation of variance σ 2 . The denoiser thus learns to predict the direction pointing back toward the unperturbed sample, which acts as an approximation to the true score ∇ x log p ( x ) . During inference, adding α d ( x t ) to the Flow term helps guide trajectories toward regions of higher data density.

Remark on α : Note that α need not be a fixed constant; it can be a learnable function of the current state x . In principle, one might choose α ( x t ) = 1 2 g ( x t ) g ( x t ) ⊤ (or a scaled variant thereof) to partially or fully cancel the diffusion in the Fokker-Planck equation, balancing the expansive effect of diffusion with contraction toward high-density regions. We discuss this balance further in Section I.

Desingularization of the Flow Loss The logarithmic squared loss in the flow objective (Eq. 14) has a singularity when the residual approaches zero. This singularity corresponds to a degenerate Gaussian distribution and can lead to overfitting of the flow network to the observed state changes. To mitigate this issue and enhance numerical stability, we introduce a positive regularizer ϵ into the loss function:

<!-- formula-not-decoded -->

This desingularization constant δ can be interpreted as a smooth transition parameter between the logarithmic square loss ( δ = 0) and the standard square loss ( δ = + ∞ ).

Handling Datasets Without Explicit Time When dealing with datasets that lack explicit time information, we introduce a uniform, manually selected time step, ∆ t . In our experiments, we set ∆ t = 1 , effectively scaling the observed changes to a suitable range where the machine learning models can converge more efficiently. This choice of ∆ t influences the magnitude of the learned flow ( f ) and diffusion ( g ) terms. Specifically, for a scaling ∆ t → λ ∆ t , it is straightforward to derive from Eq. 14 and 15 that the optimal flow and diffusion follow the corresponding scaling f → f λ , g → g √ λ . This scaling, however, does not affect the underlying dynamics captured by the Neural SDE. As proven in the subsequent section, the model's inference results are invariant to the manually chosen ∆ t . In this sense, ∆ t can be regarded as a virtual time unit which is not crucial for the learned dynamics. This property allows for training models on data without precise temporal labels.

## Proof: Temporal Scale Invariance of SDE Numerical Simulation

Consider a data-driven Stochastic Differential Equation (SDE):

<!-- formula-not-decoded -->

- X ∈ R D is the state variable
- f θ : R D → R D is the parameterized drift term
- g γ : R D → R D × D is the parameterized diffusion term
- dW is a D -dimensional Wiener process increment

Theorem 1 (Numerical Simulation Temporal Scale Invariance) . For any scaling factor λ &gt; 0 , let:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then the scaled SDE:

has Euler-Maruyama discretization statistically equivalent to the original SDE.

Proof. Original discretization ( ∆ t n,k ):

<!-- formula-not-decoded -->

Scaled discretization ( λ ∆ t n,k ):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This recovers the original discretization exactly, proving statistical equivalence. ■

## E Additional Experimental Details.

Our experiments were performed using a server on Ubuntu 22.04 LTS and 8 NVIDIA A800 80GB GPUs. The source code can be accessed at https://github.com/NeuralSDE/NeuralSDE .

## E.1 2D Branching Trajectories

## Multi-modal Distribution

Figure 4: Trajectory generation on a Y-shape Bifurcation (multi-modal Distribution). We compare our proposed Neural SDE approach with DDIM and Rectified Flow at two different densities (number of steps per trajectory). At a lower density, all models successfully generate bi-modal trajectories. At a higher density, DDIM and Rectified Flow fail due to covariate-shift, while Neural SDEs still accurately captures both branches.

<!-- image -->

where:

## Ablation Study

Figure 5: Ablation Study of the Neural SDE Components on the Y-shape Bifurcation Task (high density) . We visualize the learned vector fields with different combinations of the Flow , Diffusion , and Denoiser terms. The scale of vector fields is scaled for visual clarity. The Flow term alone captures the general direction but lacks stochasticity. Adding Diffusion introduces stochasticity but fails to reach the bifurcation point accurately due to covariate-shift. The Denoiser effectively mitigates covariate-shift. As a result, the full model ( Flow+Denoiser+Diffusion ) accurately models the multi-modal distribution.

<!-- image -->

## Experiment Setup

To illustrate our method, we construct a simple synthetic dataset where the trajectory follows a piecewise constant velocity field: For time t &lt; 4 . 5 , the velocity is defined by a magnitude u and an angle θ 1 = 0 ◦ :

<!-- formula-not-decoded -->

After t ≥ 4 . 5 , the trajectories diverge, following one of two paths with angle θ 2 = 30 ◦ :

<!-- formula-not-decoded -->

We generated datasets with two different densities. For the low density (visualized with density = 1 ), we sampled 100 data points. For the high density ( density = 10 ), we sampled 1000 points.We trained Neural SDE, along with DDIM and Rectified Flow baselines, using the same MLP architecture to ensure a fair comparison. For each model and density configuration, we generated 10 independent trajectories.

## Network Architecture and Training Details

For the flow, diffusion, and denoiser networks, we employ three identical MLPs. The input state x t is first transformed into a latent representation via a linear embedding layer. The main body of each network consists of five pre-activation residual blocks: each block applies LayerNorm and ReLU between two linear layers (hidden dimension 128), then adds its input to the block output before the final nonlinearity. After the residual stack, a final linear layer projects features to the output dimension; for the diffusion network, this output is then passed through a Tanh nonlinearity-separating rare nonzero responses from the mass of zeros-linearly rescaled to the target logit interval, and exponentiated to ensure positive, sharply peaked local values while preserving near-zero predictions elsewhere.

All models are trained for 200 epochs using the Adam optimizer. The initial learning rate is set to 2 × 10 -4 for the flow and denoiser networks, and 4 × 10 -5 for the diffusion network. No weight decay is applied. A ReduceLROnPlateau scheduler is employed for all networks, with a patience of 10 epochs and a decay factor of 0.5. The learning rate is reduced down to a minimum of 1 × 10 -5 for the flow and denoiser networks, and 1 × 10 -6 for the diffusion network. During denoiser training, Gaussian noise with a standard deviation of 0.1 is added to the input.

## E.2 Push-T

## Non-Smooth Trajectory

Figure 6: Non-Smooth Trajectory Generation. A Push-T trajectory generated by our Neural SDE, showcasing its ability to handle drastic changes in direction.

<!-- image -->

Experiment Setup This benchmark, initially proposed by Florence et al. and subsequently utilized by Chen et al. [Chi et al., 2023], requires controlling a circular end-effector to push a T-shaped block to a target location. The task inherently involves non-smooth trajectories due to intermittent contact and changes in contact dynamics, providing a challenging scenario for sequence modeling. For our evaluation, the agent receives as input the pose of the T-shaped block (represented by 1 keypoint and 1 angle) and the end-effector's location.

To cast imitation learning as sequence generation, we treat the expert demonstrations as sequences of states and actions. For the Diffusion Policy (DP) baseline [Chi et al., 2023], the input consists of T o = 2 observations and T p noisy actions, predicting a sequence of T p = 16 actions, from which T a = 8 actions are used for execution. Our Neural SDE (NSDE) model concatenates the current observation and action as the state and predicts a sequence of future states. From this predicted sequence, T ′ a = 4 actions are extracted and executed. We report results from the average of 200 environment initislization.

## Network Architecture

In this experiment, we use the same residual MLP architecture for the flow, diffusion, and denoiser modules as the 2-D case. Unless otherwise specified, we set the number of residual blocks to 4, the hidden dimension to 1024.

All networks are trained for 500 epochs using the Adam optimizer with a batch size of 512 and an initial learning rate of 2 . 5 × 10 -4 . Distinct L 2 regularization values are applied to each network: 2 . 4 × 10 -6 for the flow network, 3 . 5 × 10 -6 for the denoiser, and 1 . 7 × 10 -4 for the diffusion network. A ReduceLROnPlateau scheduler is used for all models, with a patience of 20 epochs, a decay factor of 0.5, and a minimum learning rate of 1 × 10 -6 . During the training of denoiser Gaussian noise with a standard deviation of 0.02 is added to the input.

## E.3 Video Prediction

Table 2: Performance Comparison of Video Prediction on KTH and CLEVRER. Lower FVD and JEDi are better, while higher SSIM and PSNR are better. All metrics are computed by evaluating on 20 independent test sets, each containing 256 video sequences randomly sampled with replacement. We report the mean and standard deviation across the 20 evaluations. For PFI, Flow Matching, and RIVER, we use 100 steps, while for Neural SDE, we use adaptive step sizes with approximately 17 steps on average.

| Method        | FVD ↓              | JEDi ↓          | SSIM ↑          | PSNR ↑           |
|---------------|--------------------|-----------------|-----------------|------------------|
| KTH           | KTH                | KTH             | KTH             | KTH              |
| Neural SDE    | 377 . 51 ± 20 . 57 | 1 . 31 ± 0 . 08 | 0 . 81 ± 0 . 15 | 27 . 89 ± 8 . 62 |
| Flow Matching | 313 . 82 ± 17 . 16 | 1 . 14 ± 0 . 05 | 0 . 82 ± 0 . 14 | 28 . 75 ± 6 . 96 |
| PFI           | 312 . 49 ± 12 . 64 | 1 . 16 ± 0 . 06 | 0 . 82 ± 0 . 14 | 28 . 49 ± 6 . 88 |
| RIVER         | 198 . 13 ± 10 . 76 | 0 . 54 ± 0 . 05 | 0 . 88 ± 0 . 11 | 31 . 73 ± 6 . 42 |
| CLEVRER       | CLEVRER            | CLEVRER         | CLEVRER         | CLEVRER          |
| Neural SDE    | 126 . 73 ± 11 . 29 | 1 . 06 ± 0 . 06 | 0 . 95 ± 0 . 04 | 34 . 29 ± 8 . 49 |
| Flow Matching | 99 . 63 ± 3 . 78   | 0 . 78 ± 0 . 07 | 0 . 95 ± 0 . 04 | 35 . 18 ± 8 . 58 |
| PFI           | 100 . 31 ± 5 . 51  | 0 . 84 ± 0 . 06 | 0 . 95 ± 0 . 04 | 34 . 95 ± 8 . 53 |
| RIVER         | 134 . 89 ± 14 . 93 | 1 . 09 ± 0 . 07 | 0 . 92 ± 0 . 05 | 30 . 49 ± 7 . 71 |

## Experiment Setup and Evaluation Protocol

We evaluate our method on two datasets. The KTH dataset encompasses human actions performed in various real-world settings, while the CLEVRER dataset features complex interactions between simple 3D shapes governed by physical laws. For both datasets, we condition on the first 4 frames and generate 36 future frames for KTH and 12 for CLEVRER.

To ensure a fair comparison and address inconsistencies in prior work, we adopt a unified evaluation protocol as recommended by Unterthiner et al. [Unterthiner et al., 2019], due to the sensitivity of FVD to sample size. Specifically, for each dataset, we sample 256 test videos, generate one completion per video, compute the Fréchet Video Distance (FVD) against a resampled set of 256 real videos, repeat this process 20 times, and report the mean. Prior studies have reported widely varying FVD scores for the same model on the same dataset-for example, Chen et al. report 41.88 for RIVER on KTH[Chen et al., 2024], while Davtyan et al. report 180 [Davtyan et al., 2023]-despite using the same codebase and evaluation size. For consistency, we also re-evaluate RIVER using our protocol to provide a direct point of comparison.

## Network Architecture

For the autoencoder, we use the pretrained VQGAN model provided in the repository 2 , which encodes each video frame into a latent space with 4 channels. For the flow, diffusion, and denoiser networks, we adopt a U-ViT architecture based on the RIVER codebase. All three models share the same core design, which we now detail for the vector field regressor with implicit time.

We implement the network as a U-ViT [Bao et al., 2023], whose full architecture is identical across datasets. Unlike RIVER, we omit both the explicit time input t , the time distance k -τ and its associated context frame x τ , retaining only the state input x t . Although including t , k -τ , and x τ can improve empirical performance, these modifications do not follow directly from a maximumlikelihood formulation, so we consider them nonfundamental and exclude them.

The state x t is first mapped into the model's latent space via a linear embedding layer. After this projection, the core of the network consists of 13 standard ViT blocks. We introduce four 'long' skip connections that bridge the outputs of the first four blocks directly into the inputs of the final four blocks: at each connection point, we concatenate the feature maps channel-wise and then reproject them into the ViT hidden size. Inside each block, a LayerNorm precedes both the multi-head

2 https://github.com/araachie/river

self-attention module [Vaswani et al., 2017] and the following feed-forward network. Every block employs a hidden dimension of 768 and uses eight attention heads.

All networks are trained for 300k steps with AdamW, using a learning rate of 1 × 10 -4 , weight decay of 5 × 10 -6 , a 5000 -step linear warm-up, and a square-root decay schedule thereafter. During denoiser training, Gaussian noise (std = 0.1) is added to the input.

## Implementation Details

We implemented Flow Matching [Lipman et al., 2022] and Probabilistic Forecasting with Interpolants (PFI) [Chen et al., 2024]. 3 In the original PFI paper, the sparse conditioning mechanism from RIVER is retained-yielding performance close to RIVER. To ensure an fair comparison, we remove this sparse condition in our PFI implementation. In table 2, we provide a detailed comparison. For readers interested in the effect of sparse conditioning, we also report RIVER's results under our unified evaluation protocol; comparing those to Flow Matching highlights the effect of sparse conditioning.

In the following, we describe how each variant modifies the original RIVER inputs:

- RIVER: takes the reference frame x s (an interpolation of the frame x k immediately preceding the predicted frame x k +1 and Gaussian noise), time s , a context frame x τ (sampled from the past), and the interval k -τ , with s ∈ [0 , 1] , as input.
- Flow Matching: removes the context inputs τ and k -τ , yielding inputs ( x s , s ) .
- NSDE: removes k -τ , x τ , and s , replacing the discrete reference with a continuous state interpolation

<!-- formula-not-decoded -->

- PFI: also removes k -τ and x τ , but employs a stochastic interpolant

<!-- formula-not-decoded -->

All other architectural choices (U-ViT blocks, training hyperparameters, etc.) remain as described above.

In order to provide sufficient motion information for each object, all methods concatenate 4 frames to construct the state X t and generate the next frame autoregressively. If only a single frame is used, we find that the motion direction of objects cannot be determined, and as a result, the objects tend to remain stationary.

<!-- formula-not-decoded -->

At each training step, the model sees a pair of consecutive states ( X t , X t +∆ t ) . Under the Neural SDE framework, we aim to estimate

<!-- formula-not-decoded -->

However, since X t itself encapsulates multiple consecutive frames, a naive method would be to predict all four finite differences:

<!-- formula-not-decoded -->

Unfortunately, this can encourage a 'shortcut' solution: the last three differences are merely trivial subtractions of already-visible inputs, allowing the network to ignore the first (crucial) term and simply copy known differences.

To avoid this pitfall and foster better generalization, we only predict the first difference,

<!-- formula-not-decoded -->

and rely on simple algebraic subtraction for the remaining terms:

<!-- formula-not-decoded -->

3 All methods were adapted from the code of RIVER [Davtyan et al., 2023]. The official video-prediction code for PFI has not been released, so we re-implemented it from the paper's descriptions.

In other words, our flow network f ( · ) directly learns only the ˙ x (1) t component. By doing so, the model cannot simply 'memorize' those last three differences; it must genuinely learn to predict how the next frame ( x t +∆ t ) evolves from the current frame ( x t ) . This design choice improves generalization, as it enforces non-trivial predictions of future data rather than letting the network exploit short-term historical redundancy.

## Inference-Time Construction (Euler-Maruyama Step).

At inference time, we approximate each SDE step from t to t +∆ t using a single Euler-Maruyama update:

<!-- formula-not-decoded -->

where ζ ∼ N (0 , I ) . Once x t +∆ t is computed, the state X t +∆ t is updated by shifting:

<!-- formula-not-decoded -->

This procedure respects the continuous-time Markov property , ensuring that newly generated frames emerge directly from the learned SDE dynamics rather than from extraneous predictions of alreadyknown differences. In doing so, the model achieves more robust performance for trajectory or video frame generation.

## F High Temporal Resolution Video

Figure 7: High Temporal Resolution Video Generation. This figure compares ground truth video frames with predictions from Flow Matching and our Neural SDE (NSDE) model. The ground truth frames are subsampled by a factor of 5, in order to reduce the computational cost of training. The top half of the figure shows a sequence where, in the ground truth, a yellow cylinder and a blue sphere move towards each other, appear to make contact, then move apart. Flow Matching's prediction shows the blue sphere moving towards the yellow cylinder but moving away before contact appears to occur, indicating a potential skipped frame. NSDE accurately captures the approaching, contacting, and separating motion of the two objects. The bottom half of the figure shows a sequence where, in the ground truth, a blue cylinder and a silver sphere move toward each other. Both the Ground Truth and Flow Matching show the objects moving towards each other and then separating, but missing the contact frame because of the subsampling. NSDE is able to generate intermediate frames that display the contact, demonstrating its capacity for generating videos at high temporal resolution even with relatively sparse training data. For more examples, please check out our project website https://neuralsde.github.io/.

<!-- image -->

Obtaining and training on data at high temporal resolution can be expensive. In this section, we investigate the ability of our Neural SDE model to generate videos at a high temporal resolution, even when trained on sparse, subsampled data. Figure 7 presents a visual comparison of our model's performance against a baseline, demonstrating our model's capacity to generate intermediate frames.

## G Ablation Study

| Method              | FVD ↓              | JEDi ↓            | SSIM ↑            | PSNR ↑           |
|---------------------|--------------------|-------------------|-------------------|------------------|
|                     |                    | KTH               |                   |                  |
| Neural SDE          | 395.75 ± 23 . 03   | 1 . 325 ± 0 . 102 | 0 . 807 ± 0 . 148 | 27 . 55 ± 8 . 44 |
| w/o Denoiser        | 408 . 91 ± 23 . 90 | 1.155 ± 0 . 080   | 0.828 ± 0 . 137   | 28.80 ± 7 . 37   |
| w/o Noise Injection | 470 . 46 ± 20 . 65 | 1 . 722 ± 0 . 090 | 0 . 787 ± 0 . 158 | 26 . 30 ± 8 . 98 |
| CLEVRER             | CLEVRER            | CLEVRER           | CLEVRER           | CLEVRER          |
| Neural SDE          | 128.07 ± 14 . 30   | 1.172 ± 0 . 101   | 0.944 ± 0 . 043   | 33.78 ± 9 . 19   |
| w/o Denoiser        | 139 . 98 ± 12 . 76 | 1 . 437 ± 0 . 068 | 0 . 936 ± 0 . 048 | 32 . 63 ± 9 . 19 |
| w/o Noise Injection | 153 . 72 ± 20 . 11 | 1 . 334 ± 0 . 105 | 0 . 936 ± 0 . 046 | 32 . 82 ± 9 . 01 |

Table 3: Ablation Study on Denoiser Network and Noise Injection . Lower FVD and JEDi are better, while higher SSIM and PSNR are better. All metrics are computed by evaluating on 20 independent test sets, each containing 256 video sequences randomly sampled with replacement. We report the mean and standard deviation across the 20 evaluations.

In Table 3, we present the performance of the Neural SDE model when removing the denoiser network and the noise injection, evaluated on two video prediction datasets. The results demonstrate that both components are crucial for achieving better performance.

## H Validation Loss for Scaling Experiment

We evaluated the validation performance on the CLEVRER dataset using the reduced loss function 4 defined in Equation (31), with the desingularization constant δ = 0 . 001 factored out:

<!-- formula-not-decoded -->

## I Guiding Diffusion with a Score Function: Balancing Expansion and Contraction

In many bridging-based generative frameworks, one introduces a score function term to the drift of an SDE, effectively 'counteracting' some portion of the diffusion. Concretely, consider modifying our baseline SDE

<!-- formula-not-decoded -->

to

<!-- formula-not-decoded -->

where ∇ ln p ( t, X t ) is the score function , and α ( X t ) can be chosen as a state-dependent matrix (e.g. a diagonal function of G ). In particular, one may set

<!-- formula-not-decoded -->

to fully cancel the diffusion operator in the corresponding Fokker-Planck equation (under idealized conditions). However, this extreme choice is generally not mandatory; partial or approximate cancellation can also be beneficial.

Balancing Expansion and Contraction. From the perspective of the probability density p ( t, x ) , each step in (32) comprises:

- Diffusion , via G , which tends to spread or 'expand' the distribution.

4 See [Kaplan et al., 2020] for detailed explanation.

- Score-based drift , via α ( x ) ∇ ln p , which pulls trajectories toward higher-density regions, acting as an 'anti-diffusion' force.

By tuning α ( x ) appropriately, we can strike a balance between these opposing effects, preventing the distribution from diverging (if diffusion is too large) or collapsing into a narrow region (if anti-diffusion is too strong).

Implications for Model Design. This idea underlies many score-based generative models:

- Adaptive Anti-Diffusion : Rather than using a constant α , we can learn a network that predicts α ( x ) based on local properties of G ( x ) and the data distribution.
- Control Over Sampling Dynamics : By partially canceling diffusion, the model can sample more efficiently-avoiding many small, iterative denoising steps-yet still capture multimodal or uncertain behaviors where non-zero stochasticity is essential.

Hence, the controlled interplay between diffusion and a learned score function can yield flexible, stable, and computationally efficient continuous-time modeling-particularly useful for complex or high-dimensional tasks in embodied AI and generative pipelines.

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1-2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

## IMPORTANT, please:

- Delete this instruction block, but keep the section heading 'NeurIPS Paper Checklist" ,
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers .

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly state our main contributions.

## Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: In section 6

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

Justification: In Appendix A, B, C, D and Section 3

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

Justification: In Appendix D, E and Section 4

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

Justification: In Appendix E, we provide the github link of code

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

Justification: In Appendix D, E, F and Section 4

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: In Appendix E and Section 4

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

Justification: In Appendix E

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have carefully reviewed the NeurIPS Code of Ethics and confirm that our research fully complies with its guidelines.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: Our work focuses on theoretical and algorithmic contributions to continuoustime sequence modeling and does not discuss societal impact in the paper. We acknowledge that even foundational research can have downstream applications with societal implications, such as synthetic media generation or decision-making systems. We plan to include a discussion of these aspects in future versions.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

## Answer: [NA]

Justification: Our paper does not release any model or dataset that poses a risk for misuse. The work focuses on theoretical and empirical aspects of continuous-time sequence modeling.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All assets used in this paper are publicly available and cited with their original sources. The terms of use and licenses are respected.

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

Justification: [TODO]

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

Justification:

[TODO]

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.