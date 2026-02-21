## Learning to Generalize: An Information Perspective on Neural Processes

Hui Li 1 , 2 , 3 , † , Huafeng Liu 2 , 3 , † , Shuyang Lin 1 , 2 , 3 , Jingyue Shi 1 , 2 , 3 , Yiran Fu 1 , 2 , 3 , Liping Jing 1 , 2 , 3 , ∗

1 State Key Laboratory of Advanced Rail Autonomous Operation, Beijing, China 2 School of Computer Science and Technology, Beijing Jiaotong University, Beijing, China 3 Beijing Key Laboratory of Traffic Data Mining and Embodied Intelligence, Beijing, China {huili97, hfliu1, sylin1, jingyueshi, yiranfu, lpjing}@bjtu.edu.cn

## Abstract

Neural Processes (NPs) combine the adaptability of neural networks with the efficiency of meta-learning, offering a powerful framework for modeling stochastic processes. However, existing methods focus on empirical performance while lacking a rigorous theoretical understanding of generalization. To address this, we propose an information-theoretic framework to analyze the generalization bounds of NPs, introducing dynamical stability regularization to minimize sharpness and improve optimization dynamics. Additionally, we show how noise-injected parameter updates complement this regularization. The proposed approach, applicable to a wide range of NP models, is validated through experiments on classic benchmarks, including 1D regression, image completion, Bayesian optimization, and contextual bandits. The results demonstrate tighter generalization bounds and superior predictive performance, establishing a principled foundation for advancing generalizable NP models.

## 1 Introduction

Gaussian processes (GPs) are widely recognized as a robust framework for modeling distributions over functions [37]. Their appeal lies in the consistent probabilistic reasoning enabled by Bayesian inference, which facilitates data-efficient modeling. Despite their strengths, GPs are unsuitable for certain problems. For instance, a function exhibiting a single, unknown discontinuity is a classic example of a distribution that GPs fail to represent [34].

To address such limitations, researchers have turned to neural network-based generative models. Notable advancements in this area include meta-learning techniques like Neural Processes (NPs) [14, 15] and models based on variational autoencoders (VAEs) [33, 11]. These approaches leverage extensive small dataset training to transfer knowledge effectively across tasks during prediction. Neural networks (NNs) offer additional advantages by offloading computational intensity to the training phase, simplifying predictions, and freeing the model from Gaussianity constraints.

Building upon the foundational work of Conditional Neural Processes (CNPs) and Neural Processes (NPs) [14, 15], numerous studies have enhanced NPs by integrating advanced mechanisms. Attentive Neural Processes (ANPs) [27] incorporated attention mechanisms to better model long-range dependencies, while Transformer Neural Processes (TNPs) [36] leveraged self-attention for improved scalability. Convolutional Neural Processes (ConvCNPs) [10] adapted convolutional architectures to excel in spatial data tasks, and Neural Diffusion Processes (NDPs) [7] introduced diffusion mechanisms to enhance uncertainty estimation. NPCL [22] extended NPs to continual learning, enabling

∗ Corresponding authors: Liping Jing; † These authors contributed equally to this work. Codes: https: //github.com/Allen0497/Gen-NPs

sequential task adaptation, and multimodal approaches like [25] generalized NPs for uncertainty estimation across multiple modalities.

While these advancements significantly enhance the experimental performance of NPs, they primarily focus on task-specific improvements through architectural innovations and meta-learning strategies [20, 16]. However, the generalization capability of NPs-a critical hallmark of their design-has received limited theoretical exploration. Existing works prioritize performance optimization, often neglecting the theoretical interpretability of NPs. For instance, many methods emphasize improving inductive biases or increasing model complexity without addressing the theoretical underpinnings of NPs' generalization ability [17, 10]. This gap underscores the need for a formal theoretical foundation to better understand and generalize NPs across broader domains and tasks, paving the way for more principled and interpretable neural process models [44].

Information theory provides a powerful framework for analyzing and quantifying uncertainty, offering rigorous tools to evaluate the generalization of models [18, 35]. This paper introduces an information-theoretic framework to analyze and improve the generalization bounds of Neural Processes (NPs). Building on this analysis, we propose a novel approach that incorporates dynamical stability regularization, which explicitly minimizes sharpness through the trace and Frobenius norm of the Hessian matrix. Additionally, we demonstrate how noise-injected parameter updates complement this regularization by smoothing the optimization trajectory and improving gradient coherence. These insights are broadly applicable across NP variants and validated through extensive experiments on benchmarks such as 1D regression, image completion, Bayesian optimization, and contextual bandits.

The main contributions of this paper are as follows:

- We conduct an information-theoretic analysis of NP generalization bounds, providing a rigorous perspective on their generalization capabilities.
- We propose a dynamical stability regularization framework, complemented by noise-injected parameter updates, to enhance generalization by minimizing sharpness and improving optimization dynamics.
- Extensive experiments on classic NP tasks validate our approach, demonstrating tighter generalization bounds and superior performance compared to existing methods.

## 2 Related Work

In this section, two relevant areas are briefly reviewed: neural processes and information-theoretic learning.

Neural Processes Neural Processes (NPs) are probabilistic models designed to meta-learn distributions over functions, offering a data-driven alternative to Gaussian Processes (GPs) [14, 15]. Unlike GPs, which rely on manually specified priors, NPs employ an encoder-decoder architecture to learn a family of functions, efficiently capturing uncertainty using neural networks. Conditional Neural Processes (CNPs) [14], the earliest variant, introduced a deterministic encoder-decoder framework but assumed independence among predictive outputs. Latent Neural Processes (NPs) [15] addressed this by introducing a global latent variable for uncertainty modeling. Subsequent works have enhanced expressiveness, scalability, and uncertainty modeling. Extensions of NPs have addressed specific limitations and broadened their applicability. Attentive Neural Processes (ANPs) [27] incorporated attention mechanisms to better model long-range dependencies and heteroscedastic uncertainty. Convolutional Neural Processes (ConvCNPs) [10] introduced translation-equivariant architectures, excelling in spatial and structured data tasks. Transformer Neural Processes (TNPs) [36] leveraged self-attention for scalability and sequence modeling, while Neural Diffusion Processes (NDPs) [7] used diffusion mechanisms to improve robustness and uncertainty estimation. Neural Processes for Uncertainty-Aware Continual Learning (NPCL) [22] extended NPs to sequential task adaptation without catastrophic forgetting. Additionally, multimodal extensions [25] generalized NPs to handle multi-sensor fusion and uncertainty estimation. These advancements highlight the versatility of NPs and their adaptability across diverse applications.

Information-Theoretic Learning Information-theoretic learning (ITL) leverages principles from information theory to analyze and optimize learning algorithms. Unlike traditional approaches based on measures like VC-dimension [39] or uniform stability [4], ITL characterizes learning using mutual information and related quantities [50, 38]. By framing generalization error as the mutual information between inputs and outputs, ITL captures dependencies among data distribution, hypothesis space, and learning algorithms, addressing challenges in uncertainty quantification, robustness, and data

efficiency. This framework provides valuable insights into the generalization capabilities of modern machine learning models. Recent advances have expanded ITL's application to generalization analysis and enhancement. Harutyunyan et al. [18] proposed improved generalization bounds by focusing on the mutual information between predictions and the training set, offering practical estimates of the generalization gap in deep learning. Neu et al. [35] extended ITL to stochastic gradient descent (SGD), deriving tighter bounds based on local gradient statistics and sensitivity along the optimization path. These advances demonstrate ITL's utility in understanding complex learning dynamics and improving theoretical guarantees for generalization.

## 3 Preliminaries

This section introduces the foundational concepts and mathematical tools necessary for understanding the methods and analyses presented in this paper, including information theory, neural processes, and dynamical stability.

Information Theory Let P X denote the marginal distribution of the random variable X . For the Markov chain X → Y , the conditional distribution (or Markov transition kernel) is denoted by P Y | X , and the notation X ⊥ Y indicates independence between X and Y [6]. The cumulant generating function (CGF) of a random variable X is defined as ψ X ( λ ) ≜ log E [ e λ ( X -E [ X ]) ] , where λ is a real number. A random variable X is σ -subgaussian if its CGF satisfies ψ X ( λ ) ≤ λ 2 σ 2 2 for all λ ∈ R [43].

The mutual information between X and Y is defined as I ( X ; Y ) ≜ KL ( P X,Y ∥ P X P Y ) , where P X,Y is the joint distribution of X and Y , and P X P Y is the product of their marginals [6]. The disintegrated mutual information between X and Y given U is I U ( X ; Y ) ≜ KL ( P X,Y | U ∥ P X | U P Y | U ) , where P X,Y | U is the conditional joint distribution given U . The conditional mutual information is then defined as I ( X ; Y | U ) ≜ E U [ I U ( X ; Y )] , the expectation of the disintegrated mutual information over the distribution of U [1].

Neural Processes Neural Processes (NPs) model the conditional predictive distribution of target values y T at target points X T based on a context set D C , expressed as P ( y T | X T , D C ) [15]. For deterministic NPs (CNPs), the conditional distribution is simplified as P ( y T | X T , D C ) = P ( y T | X T , r C ) , where r C is an aggregated feature of D C [14]. Probabilistic NPs introduce a latent variable z to capture uncertainty, modeling P ( y T | X T , D C ) = ∫ P θ ( y T | X T , z ) P θ ( z |D C ) d z . Training maximizes the evidence lower bound (ELBO): E z ∼ P θ ( z |D C ) [log P ( y | X )] -KL [ P θ ( z | X , D C ) || P θ ( z |D C )] [15]. During meta-training, NPs learn from tasks sampled from an environment τ , a probability measure over task distributions µ , where each task involves dividing data into context and target sets [20]. A meta-dataset consists of m datasets D 1: m = ( D 1 , . . . , D m ) , each sampled independently from µ n,τ , the mixture distribution induced by τ [9]. The meta learner A outputs a meta-parameter θ = A ( D 1: m ) ∼ P θ |D 1: m , representing shared knowledge across tasks [3]. The meta risk is defined as R τ ( θ ) ≜ E D∼ µ n,τ ,µ ∼ τ [ E D C [ R µ ( θ )]] , where R µ ( θ ) ≜ -E ( x,y ) ∼ µ log P ( y | x, D C ) [3]. The empirical meta risk is given by R D 1: m ( θ ) ≜ 1 m ∑ m j =1 R D j ( θ ) , with R D j ( θ ) ≜ -1 n ∑ n i =1 log P ( y i | x i , D C j ) . The meta generalization error is then defined as gen NPs meta ( τ, A ) ≜ E θ, D 1: m [ R τ ( θ ) -R D 1: m ( θ )] , which quantifies the difference between true and empirical meta risks [3]. By learning a meta-parameter θ shared across tasks, NPs efficiently adapt to new tasks, leveraging shared information in the environment.

Dynamical Stability Dynamical stability refers to the robustness of the stochastic gradient descent (SGD) optimization process to small perturbations in the parameter space and has been shown to play a critical role in both optimization and generalization [49, 42]. Consider the parameter update rule θ s +1 = θ s -η ∇ R s ( θ s ) , where η is the learning rate, and ∇ R s ( θ s ) is the stochastic gradient. The sensitivity of the optimization dynamics can be quantified by the spectral norm ∥ J s ∥ 2 of the Jacobian matrix J s = ∂θ s +1 ∂θ s , which governs how small perturbations δ s at step s propagate through subsequent iterations [48, 40, 47]. When ∥ J s ∥ 2 &lt; 1 , perturbations decay over time, leading to stable training. Importantly, this stability imposes an implicit regularization effect, as SGD inherently biases the optimization process towards flatter minima with lower curvature, measured by the Hessian H = ∇ 2 R ( θ ) [21, 26]. Flatter minima are associated with better generalization due to their robustness to noise and parameter variations [19, 8]. Moreover, the Lyapunov exponents, which quantify the exponential rates of divergence or convergence of nearby trajectories in parameter space, provide a formal link between stability and generalization. Negative Lyapunov exponents imply stable dynamics and a tendency towards generalizable solutions [42, 49]. This perspective highlights the central role of dynamical stability in understanding the implicit biases of SGD and its impact on both the optimization landscape and model generalization [47, 41].

## 4 Methodology

This section investigates the generalization properties of Neural Processes (NPs) by integrating information-theoretic principles with optimization dynamics. First, we quantify the generalization error of NPs using mutual information (MI), capturing uncertainties from data and task distributions. Next, we introduce a risk-aware dynamical stability regularization term R dyn, addressing sharpness and curvature to improve generalization. Finally, we propose an optimization-aware noise injection strategy to enhance stability and guide parameter updates toward flatter minima.

## 4.1 Quantifying Generalization with Information Theory

Understanding the generalization capabilities of Neural Processes (NPs) requires quantifying the uncertainties inherent in the learning process. By leveraging mutual information (MI), the generalization error can be decomposed into components reflecting different sources of uncertainty, offering insights into the behavior of NPs across diverse tasks. While MI provides a rigorous framework to analyze generalization, standard approaches often neglect the impact of model complexity, which is particularly critical in the hierarchical nature of meta-learning. This section lays the foundation for incorporating complexity constraints in future analysis.

To analyze generalization bounds, we consider the relationship between true risk and empirical risk un-

Figure 1: Generalization error of Neural Processes (NPs) varies with the Hessian trace

<!-- image -->

der a given task distribution. Let µ represent an unknown distribution on the instance space X × Y . A meta-dataset consists of m datasets, denoted as D 1: m = ( D 1 , . . . , D m ) , where each dataset D j consists of n independent samples drawn from µ . The full dataset is thus D = D 1: m = {D j } j m =1 . For a hypothesis θ , the true risk is defined as R µ ( θ ) ≜ E X × Y ∼ µ [ ℓ ( θ, X × Y )] , and the empirical risk as R D ( θ ) ≜ 1 mn ∑ m j =1 ∑ n i =1 ℓ ( θ, x i,j , y i,j ) , where ℓ : Θ × X × Y → R is the loss function. A learning algorithm A maps the dataset D to a randomized hypothesis θ = A ( D ) ∼ P θ |D , and its generalization error is defined as gen ( µ, A ) ≜ E θ, D [ R µ ( θ ) -R D ( θ )] .

Theorem 4.1. If the loss ℓ ( θ, X × Y ) is σ -subgaussian for each θ ∈ Θ with respect to X × Y ∼ µ , the generalization error of a learning algorithm A satisfies the bound | gen ( µ, A ) | ≤ √ 2 σ 2 mn I ( θ ; D ) , where I ( θ ; D ) is the mutual information between the dataset D and the hypothesis θ . For meta-learning tasks, where µ ∼ τ , the meta-generalization error satisfies:

<!-- formula-not-decoded -->

The mutual information I ( θ ; D 1: m ) can be computed based on the joint distribution P ( θ, D 1: m ) and the marginal P ( θ ) ; detailed derivations are provided in the appendix. This bound indicates that minimizing the dependence of θ on the data (via I ( θ ; D 1: m ) ) improves the generalization performance.

Although the MI framework elegantly quantifies generalization, it does not explicitly account for the hypothesis space complexity or the solution sharpness. For NPs, these factors are especially significant as a result of their hierarchical structure. The hypothesis space in NPs, determined by the meta-parameters θ , encodes both task-level priors and task-specific adaptations. This space is inherently high-dimensional, making it prone to overfitting, particularly when the meta-dataset D 1: m is small or lacks diversity.

Figure 2: The training process of Gen-NPs

<!-- image -->

The sharpness of the learned solution is related to the curvature of the loss surface, quantified by the Hessian H = ∇ 2 R ( θ ) . A high Hessian

trace implies sensitivity to parameter perturbations, adversely affecting generalization, especially across diverse task distributions. As illustrated in Figure 1, the generalization error of NPs deteriorates with increasing Hessian trace.

Based on the above analysis, we focus on improving NPs' generalization from both risk-aware dynamical stability regularization and optimization-aware noise injection learning strategy, the whole learning framework is given in Figure 2.

## 4.2 Risk-Aware Dynamical Stability Regularization

Inspired by previous works that demonstrate the critical role of sharpness and curvature in optimization and generalization [26, 23, 45], we introduce a dynamical stability regularization (DSR) term R dyn that incorporates model complexity constraints into the generalization bound. Specifically, sharpness of the solution has been linked to poor generalization, motivating the need to explicitly regularize the curvature of the loss landscape [19, 21, 51]. Moreover, recent studies have explored how Lyapunov stability and noise in optimization dynamics can implicitly bias solutions towards flat minima, further supporting the necessity of stability-based regularization [40, 49].

Definition 4.2. The dynamical stability regularization term R dyn quantifies the complexity of the hypothesis space by leveraging properties of the Hessian matrix H , which represents the second-order derivatives of the loss function with respect to the parameters θ :

<!-- formula-not-decoded -->

where Tr ( H ) (the trace of the Hessian) measures the solution sharpness by summing the eigenvalues of H , and ∥ H ∥ F (the Frobenius norm) captures the overall curvature of the loss landscape. The hyperparameters λ 1 and λ 2 control the relative importance of these two components, with empirical values typically set as λ 1 ∈ [0 . 01 , 0 . 1] and λ 2 ∈ [0 . 001 , 0 . 01] to maintain an appropriate balance between the two terms. By penalizing sharpness and curvature, R dyn aligns the theoretical bounds with practical observations in optimization.

Theorem 4.3. By incorporating R dyn into the information-theoretic framework, the refined metageneralization error bound for Neural Processes (NPs) is given by:

<!-- formula-not-decoded -->

where I ( θ ; D 1: m ) is the mutual information between the meta-parameter θ and the dataset D 1: m , R dyn is the dynamical stability regularization term, and α &gt; 0 is a scaling factor that controls the influence of R dyn. The proof of this theorem is provided in the Appendix.

Including R dyn in the generalization bound has notable effects. Penalizing high Tr ( H ) encourages the model to select flatter minima, which are associated with better generalization. Additionally, constraining the Frobenius norm ∥ H ∥ F prevents overfitting to high-curvature regions. This regularization complements the implicit bias of optimization algorithms, such as gradient descent, which naturally steer solutions toward regions of lower sharpness.

The role of R dyn depends on the training regime. When the number of tasks m is large ( m → ∞ ) with fixed samples per task n , the mutual information term I ( θ ; D 1: m ) dominates the bound, and R dyn constrains the sharpness across tasks. Conversely, when n grows ( n →∞ ) while m remains fixed, the generalization error decreases naturally as task-level uncertainty reduces, making R dyn less critical.

From a practical perspective, incorporating R dyn into the generalization bound provides actionable insights for optimizing Neural Processes. Maximizing R dyn during training improves generalization, and tuning the hyperparameters λ 1 and λ 2 balances sharpness reduction and complexity control. Ultimately, R dyn bridges the gap between theoretical generalization bounds and practical challenges, leading to tighter bounds and a deeper understanding of the factors influencing generalization in NPs.

## 4.3 Optimization-Aware Noise Injection Learning Strategy

Motivated by the well-established relationship between noise injection and generalization [19, 41, 46], we propose an optimization-aware noise injection strategy tailored for meta-learning tasks. Noise has been shown to play a critical role in escaping sharp minima [26], smoothing the loss landscape [21], and improving generalization through implicit regularization [47]. Furthermore, recent studies highlight the importance of integrating noise injection into gradient-based optimization to ensure stability and guide parameter trajectories towards flatter regions of the loss landscape [24, 49].

Building upon these insights, we explicitly incorporate noise into the parameter update rule to enhance both the stability and generalization of meta-parameter optimization. The overall training process of NPs, incorporating noise injection and dynamical stability regularization, is illustrated in Figure 2. The pseudocode for the entire algorithm can be found in the Appendix.

To explicitly incorporate the effect of noise into the learning dynamics, we modify the standard gradient-based parameter update rule by introducing an isotropic Gaussian noise term ξ s at each iteration s . The parameter update rule for meta-parameters θ at iteration s is formally given by:

<!-- formula-not-decoded -->

where ˜ R D T i ( θ s -1 ) + R dyn is the empirical risk computed on the target set D T i of task i , η s is the learning rate at iteration s , and ξ s ∼ N (0 , σ 2 s I k ) is an isotropic Gaussian noise term with variance σ 2 s and dimensionality k . This update rule consists of two key components: the gradient descent step, which minimizes the empirical risk, and the noise injection step, which perturbs the parameter updates. The variance σ 2 s can be adjusted over iterations to control the strength of the perturbation.

The introduction of noise into the parameter updates serves two primary purposes. First, it provides implicit regularization by encouraging the model to explore flatter regions of the loss landscape and favor minima with lower sharpness. This aligns with the objective of maximizing R dyn, as described in Section 4. Second, noise facilitates escaping sharp minima, which are common in high-dimensional parameter spaces and often lead to poor generalization. By perturbing the trajectory of the updates, noise helps the optimization process avoid these undesirable regions.

To achieve the desired regularization effect without destabilizing the optimization, the variance σ 2 s of the noise is carefully scaled with respect to the learning rate η s . A common choice for this scaling is σ 2 s = η s /γ , where γ &gt; 0 is a scaling factor that controls the relative strength of the noise. This formulation ensures that the noise magnitude decreases as the learning rate decreases, allowing the optimization to stabilize in later iterations.

The noise-injected parameter update rule in Eq. 4 directly influences the dynamical stability regularization term R dyn by affecting the Hessian properties of the loss landscape. Specifically, the perturbation introduced by ξ s reduces the likelihood of converging to regions with high Tr ( H ) , thus implicitly maximizing the sharpness of the solution. Furthermore, noise smooths the optimization trajectory, reducing the overall curvature of the solution as measured by ∥ H ∥ F . Together, these effects demonstrate the critical role of noise injection in improving both the stability and generalization performance of meta-parameter optimization. While introducing noise and regularization incurs additional computational overhead, the performance gains justify these costs (detailed analysis in Appendix).

## 5 Deeper Analysis

In this section, we establish a theoretical foundation for understanding how noise and dynamical stability regularization affect the generalization performance of NPs. By analyzing the interplay between noise, sharpness, and mutual information, we demonstrate their influence on generalization bounds. Additionally, we show how noise injection mitigates gradient incoherence, a key challenge in meta-learning, to improve stability and generalization.

## 5.1 Theoretical Analysis

We analyze how noise-injected parameter updates affect the dynamical stability of the optimization process and its impact on generalization performance. Specifically, we focus on their effects on sharpness, curvature (captured by the dynamical stability regularization term R dyn), and mutual information I ( θ ; D 1: m ) . This analysis highlights the role of noise in reducing both R dyn and I , thereby improving the generalization of NPs.

To begin, we linearize the parameter updates around the optimal solution θ ∗ , which minimizes the true risk R µ ( θ ) . Let ˜ θ s = θ s -θ ∗ represent the deviation from θ ∗ at iteration s . The linearized noise-injected parameter update rule is:

<!-- formula-not-decoded -->

where H = ∇ 2 [ ˜ R D T i + R dyn ]( θ ∗ ) is the Hessian of the empirical risk, and ξ s ∼ N (0 , σ 2 s I k ) is isotropic Gaussian noise. This form separates the contributions of optimization dynamics ( I -η s H ) and noise ξ s to the evolution of ˜ θ s .

Noise affects sharpness, measured by the trace of the Hessian Tr ( H ) , by perturbing the optimization trajectory and encouraging exploration of flatter regions. This perturbation has a dual effect on the loss landscape: it locally reduces the curvature around specific minima while simultaneously increasing the system's overall stability by promoting robust, flat solutions. Consequently, the effective Hessian trace in the presence of noise satisfies Tr ( H noise ) ≥ Tr ( H ) · (1 + η s σ 2 s ) , where a higher variance of noise σ 2 s results in a greater enhancement of dynamic stability. Similarly, the Frobenius norm ∥ H ∥ F , which provides a comprehensive measure of the curvature of the loss landscape, satisfies ∥ H noise ∥ F ≥ ∥ H ∥ F · √ 1 + η s σ 2 s , indicating that noise enhances the system's ability to maintain stability in the face of parameter perturbations. These results suggest that noise injection encourages the optimization process to explore flatter regions of the loss landscape, develop higher dynamic stability, and improve generalization performance by enhancing robustness to parameter perturbations.

By enhancing both stability metrics, noise effectively amplifies the dynamical stability regularization term R dyn, which appears in the denominator of our generalization bound. Specifically, substituting the enhanced trace and Frobenius norm into the definition of R dyn yields:

<!-- formula-not-decoded -->

This increase in R dyn directly tightens the generalization bound by increasing the denominator term 1 + α · R dyn, resulting in improved generalization capabilities for Neural Processes.

As the noise variance σ 2 s decreases over iterations ( σ 2 s = η s /γ ), the optimization stabilizes near a flat minimum, with the stability benefits retained as enhanced R dyn. Noise also reduces the mutual information I ( θ ; D 1: m ) , which quantifies the dependency between model parameters θ and training data D 1: m . High mutual information indicates potential overfitting, as parameters become overly dependent on specific data. The reduction is given by:

<!-- formula-not-decoded -->

This ensures I noise ( θ ; D 1: m ) remains non-negative and significantly lower than the original mutual information. Excessive noise or high learning rates, however, may destabilize optimization, limiting these benefits. By jointly reducing I ( θ ; D 1: m ) and amplifying R dyn through increased Tr ( H ) and ∥ H ∥ F , noise-injected updates tighten the generalization bound:

<!-- formula-not-decoded -->

where the second inequality follows from I noise ( θ ; D 1: m ) ≤ I ( θ ; D 1: m ) and R dyn, noise ≥ R dyn . This interplay between noise, mutual information, and dynamical stability is crucial for achieving tighter generalization bounds and improving meta-learning performance.

## 5.2 Gradient Incoherence

Gradient incoherence, caused by inconsistencies between gradients computed on the context set ( D C i ) and the target set ( D T i ), affects the stability and generalization of Neural Processes (NPs). This issue can lead to suboptimal updates and hinder learning. Noise-injected parameter updates mitigate this problem by smoothing the loss landscape and reducing gradient mismatches.

Definition 5.1. The gradient incoherence for a task i at iteration s is defined as the ℓ 2 -norm of the difference between the gradients on the full dataset D i and the target set D T i :

<!-- formula-not-decoded -->

where ∇ ˜ R D i ( θ s -1 ) + R dyn and ∇ ˜ R D T i ( θ s -1 ) + R dyn are gradients of the empirical risk over D i and D T i , respectively. The overall gradient incoherence across tasks and iterations is given by GI = 1 mS ∑ m i =1 ∑ S s =1 ϵ θ i,s , where m is the number of tasks and S the number of iterations. Minimizing GI ensures more consistent updates, improving generalization.

Noise reduces gradient incoherence by perturbing the optimization trajectory. The noise ξ s introduced during updates disrupts gradient alignment, exponentially decreasing incoherence as noise variance σ 2 s increases. Formally, the reduction can be expressed as ϵ θ i,s ≤ ϵ θ i,s (0) · exp( -η 2 s σ 2 s γ ) , where ϵ θ i,s (0) is the initial incoherence. Larger noise promotes exploration and reduces mismatches early in training, while smaller noise stabilizes updates in later stages. This reduction stabilizes parameter

Figure 3: Part results of 1D Regression and Image Completion.

<!-- image -->

updates, prevents oscillations caused by inconsistent gradients, and improves alignment with true risk minimization. Additionally, noise-induced coherence complements the regularization effect of R dyn by further reducing the sharpness and curvature of the loss landscape.

The effectiveness of noise depends on scaling its variance σ 2 s with the learning rate η s . Setting σ 2 s = η s /γ balances noise magnitude across iterations, ensuring exploration in early training and stability later. Empirical results in Section 6 confirm that noise-injected updates significantly reduce GI, leading to improved stability and generalization across tasks.

## 6 Experiments

The proposed Generalization Neural Processes (Gen-NPs) are evaluated and compared with other methods in the NP family on tasks such as regression, image completion, Bayesian optimization, and contextual bandits. These tasks are widely used to benchmark NP-based models, as demonstrated in prior works. The comparison includes Conditional Neural Processes (CNPs) [14], Neural Processes (NPs) [15], Attentive Neural Processes (ANPs) [27], Bootstrapping Neural Processes (BNPs) [30], and Transformer Neural Processes (TNPs) [36]. In addition to standard task performance, we analyze the gradient incoherence (GI) introduced in Section 5.2, highlighting how Gen-NPs improve optimization dynamics and generalization capabilities compared to other methods. The experimental setup is consistent across all methods, and the implementation leverages the official codebase of TNPs. Due to space limitations, extensive experimental results, including detailed images and tables, are provided in Appendix C.

## 6.1 1-D Regression

We evaluate Gen-NPs on a 1-D regression task, training models on RBF kernel functions sampled from a Gaussian Process (GP) and testing on unseen RBF, Matérn 5/2, and Periodic kernels. Metrics include log-likelihood (LL) for predictive accuracy and uncertainty, and gradient incoherence (GI) for optimization stability. Results are averaged over five random seeds with standard deviations reported. 1-D regression results on NP model are visually illustrated in Fig. 3(a), where Gen-NP demonstrate a superior ability to capture the underlying character-

Table 1: Comparison of Gen-NPs with the baselines on LL and GI of the target points on various GP kernels.

| Method   | RBF-LL Matérn-LL Periodic-LL GI                                           |
|----------|---------------------------------------------------------------------------|
| CNP      | 0 . 265 ± 0 . 015 0 . 045 ± 0 . 014 - 1 . 435 ± 0 . 020 0 . 880 ± 0 . 027 |
| Gen-CNP  | 0.286 ± 0 . 010 0.061 ± 0 . 005 -1.418 ± 0 . 023 0.830 ± 0 . 042          |
| NP       | 0 . 240 ± 0 . 022 0 . 051 ± 0 . 019 - 1 . 145 ± 0 . 032 0 . 490 ± 0 . 025 |
| Gen-NP   | 0.270 ± 0 . 009 0.073 ± 0 . 007 -1.125 ± 0 . 024 0.470 ± 0 . 012          |
| ANP      | 0 . 805 ± 0 . 005 0 . 630 ± 0 . 004 - 5 . 320 ± 0 . 260 0 . 973 ± 0 . 046 |
| Gen-ANP  | 0.812 ± 0 . 003 0.636 ± 0 . 002 -5.028 ± 0 . 290 0.950 ± 0 . 013          |
| BNP      | 0 . 389 ± 0 . 017 0 . 185 ± 0 . 015 - 0 . 970 ± 0 . 016 0 . 160 ± 0 . 007 |
| Gen-BNP  | 0.405 ± 0 . 009 0.200 ± 0 . 008 -0.946 ± 0 . 010 0.148 ± 0 . 008          |
| TNP      | 1 . 650 ± 0 . 005 1 . 218 ± 0 . 005 - 2 . 320 ± 0 . 175 0 . 750 ± 0 . 048 |
| Gen-TNP  | 1.662 ± 0 . 003 1.226 ± 0 . 003 -2.010 ± 0 . 170 0.730 ± 0 . 032          |

istics of the data compared to baseline models. As summarized in Table 1, Gen-NPs consistently outperform baseline models across all kernels and metrics. For example, Gen-NPs achieve higher LL values on Matérn 5/2 and Periodic kernels, while also demonstrating reduced GI, indicating improved optimization dynamics and generalization. These results validate the effectiveness of the proposed general recipe. Additional figures and tables, including detailed results for MAE, RMSE, and variability across random seeds, are provided in Appendix C.2.

## 6.2 Image Completion

We evaluate Gen-NPs on image completion tasks using CelebA [32] and EMNIST [5], formulated as a 2-D regression problem where pixel coordinates are inputs and intensities are outputs [15]. CelebA is downsampled to 32 × 32 , while EMNIST uses 10 training classes (0-9) and evaluates generalization on unseen classes (10-46). Figure 3(b) shows partial results on CelebA, where

Table 2: Comparison of Gen-NPs with the baselines on LL and GI of the target points on CelebA and EMNIST.

| Method      | CelebA                            | CelebA                              | EMNIST                            | EMNIST                            | EMNIST                              |
|-------------|-----------------------------------|-------------------------------------|-----------------------------------|-----------------------------------|-------------------------------------|
| Method      | LL                                | GI                                  | Seen-LL                           | Unseen-LL                         | GI                                  |
| CNP Gen-CNP | 2 . 160 ± 0 . 004 2.188 ± 0 . 005 | 1 . 399 ± 0 . 033 1.390 ± 0 . 043   | 0 . 737 ± 0 . 004 0.786 ± 0 . 005 | 0 . 485 ± 0 . 004 0.556 ± 0 . 006 | 0 . 466 ± 0 . 050 0.410 ± 0 . 035   |
| NP Gen-NP   | 2 . 481 ± 0 . 015 2.524 ± 0 . 008 | 0 . 694 ± 0 . 034 0.664 ± 0 . 028   | 0 . 795 ± 0 . 002 0.814 ± 0 . 006 | 0 . 584 ± 0 . 003 0.603 ± 0 . 008 | 0 . 187 ± 0 . 008 0.181 ± 0 . 006   |
| ANP Gen-ANP | 2 . 921 ± 0 . 004 2.964 ± 0 . 011 | 1 . 989 ± 0 . 079 1.816 ± 0 . 025   | 0 . 981 ± 0 . 006 0.987 ± 0 . 004 | 0 . 884 ± 0 . 003 0.886 ± 0 . 004 | 0 . 526 ± 0 . 026 0.468 ± 0 . 034   |
| BNP Gen-BNP | 2 . 769 ± 0 . 003 2.776 ± 0 . 003 | 22 . 835 ± 0 . 407 22.565 ± 0 . 468 | 0 . 870 ± 0 . 005 0.905 ± 0 . 006 | 0 . 716 ± 0 . 012 0.764 ± 0 . 008 | 0 . 282 ± 0 . 028 0.279 ± 0 . 016   |
| TNP Gen-TNP | 4 . 404 ± 0 . 020 4.409 ± 0 . 008 | 94 . 101 ± 2 . 121 92.364 ± 3 . 650 | 1 . 550 ± 0 . 004 1.555 ± 0 . 002 | 1 . 419 ± 0 . 006 1.423 ± 0 . 005 | 80 . 995 ± 6 . 595 65.874 ± 4 . 382 |

Gen-NPs demonstrate superior performance in image completion. As summarized in Table 2, GenNPs consistently outperform baseline models across both datasets, achieving higher LL values and reduced GI, demonstrating improved predictive accuracy and optimization stability. Visualizations in Appendix C.3 further highlight that Gen-NPs produce clearer and more precise image reconstructions, effectively enhancing generalization and accuracy for image completion tasks.

## 6.3 Bayesian Optimization

We evaluate the effectiveness of Gen-NPs in Bayesian optimization (BO) tasks, where the goal is to maximize a black-box function f ( x ) accessible only through evaluations without gradient information [12, 28, 2, 29]. The experimental results on 2D Dropwave and 3D Ackley are shown in Figure 4a, where Gen-NPs achieve lower regret compared to baseline methods. Results, detailed in Appendix C.4, demonstrate that Gen-NPs consistently outperform baseline models across all dimensions, achieving lower regret and faster convergence. These findings validate the proposed method's ability to enhance generalization and optimization performance in BO tasks [13].

## 6.4 Ablation Study

We conducted ablation studies to validate the effectiveness of the proposed Risk-Aware Dynamical Stability Regularization (DSR) and Optimization-Aware Noise Injection Learning Strategy (NILS) on 1D regression task. Table 9 presents the results of the original method, Gen-NPs with only DSR, Gen-NPs with only NILS, and the full Gen-NPs with both modules included.

Table 3: Ablation study results comparing the original method.

| Method                    | LL (RBF)          | GI (RBF)          | LL (Periodic)       | GI (Periodic)     |
|---------------------------|-------------------|-------------------|---------------------|-------------------|
| Original CNP              | 0 . 265 ± 0 . 015 | 0 . 880 ± 0 . 027 | - 1 . 435 ± 0 . 020 | 1 . 312 ± 0 . 053 |
| Gen-CNP (with DSR only)   | 0 . 276 ± 0 . 013 | 0 . 858 ± 0 . 030 | - 1 . 428 ± 0 . 022 | 1 . 202 ± 0 . 045 |
| Gen-CNP (with NILS only)  | 0 . 279 ± 0 . 012 | 0 . 846 ± 0 . 035 | - 1 . 423 ± 0 . 024 | 1 . 151 ± 0 . 041 |
| Full Gen-CNP (DSR + NILS) | 0 . 286 ± 0 . 010 | 0 . 830 ± 0 . 042 | - 1 . 418 ± 0 . 023 | 1 . 112 ± 0 . 039 |

As shown in Table 9 and Appendix, these results highlight the significance of incorporating DSR for improving dynamical stability and NILS for robust optimization. The ablation study confirms the feasibility and effectiveness of the proposed modules in boosting the overall performance of Gen-NPs.

## 6.5 Comparison with Stability Neural Processes

We compare the stability-based generalization error (SGE) [4, 31] with our proposed Gen-NPs on 1D regression task, emphasizing the differences in their noise introduction mechanisms and evaluation methods. SGE quantifies stability as the difference between test error and training error to measure generalization. Gaussian noise (mean = 0, variance = 1) is added to 5% and 10% of the training data to assess robustness under noisy conditions. In contrast, Gen-NPs introduces noise during the parameter update process, focusing on the mutual information between model parameters and the training data to directly capture generalization behavior. Experiments were conducted on a 1D regression task with data generated using an RBF kernel, comparing models under original, 5% noise, and 10% noise conditions. Results, shown in Appendix C.6, demonstrate that Gen-NPs consistently achieves tighter bounds and higher log-likelihood (LL) values even in noisy settings. This highlights the robustness and effectiveness of Gen-NPs in capturing model generalization and stability under various conditions.

<!-- image -->

- (a) Results of Bayesian Optimization
- (b) Comparison of Log-Likelihood Across Different Noise Levels

<!-- image -->

Figure 4: Part results of Bayesian Optimization and Comparison with Stability Neural Processes

## 7 Conclusion

This paper proposes an information-theoretic framework to enhance the generalization capabilities of neural processes (NPs) by addressing both parameter optimization and regularization. Specifically, noise injection during parameter updates captures the mutual information between model parameters and training data, providing a robust optimization strategy. In parallel, dynamical stability regularization mitigates overfitting and improves the optimization trajectory, collectively leading to better generalization properties. However, the framework currently focuses on supervised learning, and future work will explore its extension to reinforcement learning and large-scale datasets.

## 8 Acknowledgements

This work was partly supported by The National Key Research and Development Program of China (2024YFE0202900); The National Natural Science Foundation of China under Grant (62406019, 62436001, 62536001, 62176020); Beijing Natural Science Foundation (4244096); Young Elite Scientists Sponsorship Program of the Beijing High Innovation Plan. The Joint Foundation of the Ministry of Education for Innovation team (8091B042235); The State Key Laboratory of Rail Traffic Control and Safety (RCS2023K006); the Talent Fund of Beijing Jiaotong University (2024XKRC075).

## References

- [1] Robert B. Ash. Information Theory . Courier Corporation, 2012.
- [2] Maximilian Balandat, Brian Karrer, Daniel Jiang, Samuel Daulton, Ben Letham, Andrew G. Wilson, and Eytan Bakshy. Botorch: A framework for efficient monte-carlo bayesian optimization. Advances in Neural Information Processing Systems (NeurIPS) , 33:21524-21538, 2020.
- [3] Jonathan Baxter. A model of inductive bias learning. Journal of Artificial Intelligence Research (JAIR) , 12:149-198, 2000.
- [4] Olivier Bousquet and André Elisseeff. Stability and generalization. The Journal of Machine Learning Research , 2:499-526, 2002.
- [5] Gregory Cohen, Saeed Afshar, Jonathan Tapson, and Andre Van Schaik. Emnist: Extending mnist to handwritten letters. In Proceedings of the International Joint Conference on Neural Networks (IJCNN 2017) , pages 2921-2926. IEEE, 2017.
- [6] Thomas M. Cover. Elements of Information Theory . John Wiley &amp; Sons, 1999.
- [7] Vincent Dutordoir, Alan Saul, Zoubin Ghahramani, and Fergus Simpson. Neural diffusion processes. In Proceedings of the International Conference on Machine Learning (ICML 2023) , pages 8990-9012. PMLR, 2023.
- [8] Gintare Karolina Dziugaite and Daniel Roy. Entropy-sgd optimizes the prior of a pac-bayes bound: Generalization properties of entropy-sgd and data-dependent priors. In International Conference on Machine Learning , pages 1377-1386. PMLR, 2018.
- [9] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In Proceedings of the International Conference on Machine Learning (ICML 2017) , pages 1126-1135. PMLR, 2017.
- [10] Andrew Foong, Wessel Bruinsma, Jonathan Gordon, Yann Dubois, James Requeima, and Richard Turner. Meta-learning stationary stochastic process prediction with convolutional neural processes. Advances in Neural Information Processing Systems (NeurIPS) , 33:82848295, 2020.
- [11] Vincent Fortuin, Dmitry Baranchuk, Gunnar Rätsch, and Stephan Mandt. Gp-vae: Deep probabilistic time series imputation. In International Conference on Artificial Intelligence and Statistics , 2020.
- [12] Peter I. Frazier. A tutorial on bayesian optimization. arXiv preprint , arXiv:1807.02811, 2018.
- [13] Jacob Gardner, Geoff Pleiss, Kilian Q. Weinberger, David Bindel, and Andrew G. Wilson. Gpytorch: Blackbox matrix-matrix gaussian process inference with gpu acceleration. Advances in Neural Information Processing Systems (NeurIPS) , 31, 2018.
- [14] Marta Garnelo, Dan Rosenbaum, Christopher Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo Rezende, and SM Ali Eslami. Conditional neural processes. In Proceedings of the International Conference on Machine Learning (ICML 2018) , pages 1704-1713. PMLR, 2018.
- [15] Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J. Rezende, SM Ali Eslami, and Yee Whye Teh. Neural processes. In Proceedings of the ICML Workshop on Theoretical Foundations and Applications of Deep Generative Models , 2018.
- [16] Hassan Gharoun, Fereshteh Momenifar, Fang Chen, and Amir H Gandomi. Meta-learning approaches for few-shot learning: A survey of recent advances. ACM Computing Surveys , 56(12):1-41, 2024.
- [17] Jonathan Gordon, John Bronskill, Matthias Bauer, Sebastian Nowozin, and Richard E. Turner. Meta-learning probabilistic inference for prediction. In Proceedings of the International Conference on Learning Representations (ICLR 2018) , 2018.

- [18] Hrayr Harutyunyan, Maxim Raginsky, Greg Ver Steeg, and Aram Galstyan. Informationtheoretic generalization bounds for black-box learning algorithms. Advances in Neural Information Processing Systems , 34:24670-24682, 2021.
- [19] Sepp Hochreiter and Jürgen Schmidhuber. Flat minima. Neural computation , 9(1):1-42, 1997.
- [20] Timothy Hospedales, Antreas Antoniou, Paul Micaelli, and Amos Storkey. Meta-learning in neural networks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) , 44(9):5149-5169, 2021.
- [21] Stanisław Jastrz˛ ebski, Zachary Kenton, Devansh Arpit, Nicolas Ballas, Asja Fischer, Yoshua Bengio, and Amos Storkey. Three factors influencing minima in sgd. arXiv preprint arXiv:1711.04623 , 2017.
- [22] Saurav Jha, Dong Gong, He Zhao, and Lina Yao. Npcl: Neural processes for uncertainty-aware continual learning. Advances in Neural Information Processing Systems , 36, 2024.
- [23] Yiding Jiang, Behnam Neyshabur, Hossein Mobahi, Dilip Krishnan, and Samy Bengio. Fantastic generalization measures and where to find them. In International Conference on Learning Representations , 2020.
- [24] Haotian Ju, Dongyue Li, and Hongyang R Zhang. Robust fine-tuning of deep neural networks with hessian-based generalization guarantees. In International Conference on Machine Learning , pages 10431-10461. PMLR, 2022.
- [25] Myong Chol Jung, He Zhao, Joanna Dipnall, and Lan Du. Beyond unimodal: Generalising neural processes for multimodal uncertainty estimation. Advances in Neural Information Processing Systems , 36, 2024.
- [26] Nitish Shirish Keskar, Jorge Nocedal, Ping Tak Peter Tang, Dheevatsa Mudigere, and Mikhail Smelyanskiy. On large-batch training for deep learning: Generalization gap and sharp minima. In 5th International Conference on Learning Representations, ICLR 2017 , 2017.
- [27] Hyunjik Kim, Andriy Mnih, Jonathan Schwarz, Marta Garnelo, Ali Eslami, Dan Rosenbaum, Oriol Vinyals, and Yee Whye Teh. Attentive neural processes. In Proceedings of the International Conference on Learning Representations (ICLR 2019) , 2019.
- [28] Jungtaek Kim. Benchmark functions for bayesian optimization, 2020.
- [29] Jungtaek Kim and Seungjin Choi. Bayeso: A bayesian optimization framework in python. Journal of Open Source Software (JOSS) , 8(90):5320, 2023.
- [30] Juho Lee, Yoonho Lee, Jungtaek Kim, Eunho Yang, Sung Ju Hwang, and Yee Whye Teh. Bootstrapping neural processes. Advances in Neural Information Processing Systems (NeurIPS) , 33:6606-6615, 2020.
- [31] Huafeng Liu, Liping Jing, and Jian Yu. Neural processes with stability. Advances in Neural Information Processing Systems , 36, 2024.
- [32] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Large-scale celebfaces attributes (celeba) dataset. Retrieved August , 15(2018):11, 2018.
- [33] S. Mishra, S. Flaxman, T. Berah, M. Pakkanen, H. Zhu, and S. Bhatt. Pivae: Encoding stochastic process priors with variational autoencoders. arXiv preprint arXiv:2002.06873 , 2020.
- [34] Radford M. Neal. Regression and classification using gaussian process priors. In J. M. Bernardo, J. O. Berger, J. W. Dawid, and A. F. M. Smith, editors, International Conference on Learning Representations . Oxford University Press, 1998.
- [35] Gergely Neu, Gintare Karolina Dziugaite, Mahdi Haghifam, and Daniel M Roy. Informationtheoretic generalization bounds for stochastic gradient descent. In Conference on Learning Theory , pages 3526-3545. PMLR, 2021.

- [36] Tung Nguyen and Aditya Grover. Transformer neural processes: Uncertainty-aware meta learning via sequence modeling. In Proceedings of the International Conference on Machine Learning (ICML 2022) , pages 16569-16594. PMLR, 2022.
- [37] Carl Edward Rasmussen and Christopher KI Williams. Gaussian Processes for Machine Learning . MIT Press, 2006.
- [38] Daniel Russo and James Zou. How much does your data exploration overfit? controlling bias via information usage. IEEE Transactions on Information Theory , 66(1):302-323, 2019.
- [39] Stephan R Sain. The nature of statistical learning theory, 1996.
- [40] Samuel Smith, Erich Elsen, and Soham De. On the generalization benefit of noise in stochastic gradient descent. In International Conference on Machine Learning , pages 9058-9067. PMLR, 2020.
- [41] Samuel L Smith, Benoit Dherin, David Barrett, and Soham De. On the origin of implicit regularization in stochastic gradient descent. In International Conference on Learning Representations , 2022.
- [42] Samuel L Smith and Quoc V Le. A bayesian perspective on generalization and stochastic gradient descent. In International Conference on Learning Representations , 2018.
- [43] Martin J. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint , volume 48. Cambridge University Press, 2019.
- [44] Jindong Wang, Cuiling Lan, Chang Liu, Yidong Ouyang, Tao Qin, Wang Lu, Yiqiang Chen, Wenjun Zeng, and S Yu Philip. Generalizing to unseen domains: A survey on domain generalization. IEEE transactions on knowledge and data engineering , 35(8):8052-8072, 2022.
- [45] Kaiyue Wen, Zhiyuan Li, and Tengyu Ma. Sharpness minimization algorithms do not only minimize sharpness to achieve better generalization. Advances in Neural Information Processing Systems , 36:1024-1035, 2023.
- [46] Andrew G Wilson and Pavel Izmailov. Bayesian deep learning and a probabilistic perspective of generalization. Advances in neural information processing systems , 33:4697-4708, 2020.
- [47] Jingfeng Wu, Wenqing Hu, Haoyi Xiong, Jun Huan, Vladimir Braverman, and Zhanxing Zhu. On the noisy gradient descent that generalizes as sgd. In International Conference on Machine Learning , pages 10367-10376. PMLR, 2020.
- [48] Lei Wu, Chao Ma, et al. How sgd selects the global minima in over-parameterized learning: A dynamical stability perspective. Advances in Neural Information Processing Systems , 31, 2018.
- [49] Lei Wu and Weijie J Su. The implicit regularization of dynamical stability in stochastic gradient descent. In International Conference on Machine Learning , pages 37656-37684. PMLR, 2023.
- [50] Aolin Xu and Maxim Raginsky. Information-theoretic analysis of generalization capability of learning algorithms. Advances in Neural Information Processing Systems (NeurIPS) , 30, 2017.
- [51] Huanrui Yang, Xiaoxuan Yang, Neil Zhenqiang Gong, and Yiran Chen. Hero: Hessian-enhanced robust optimization for unifying and improving generalization and quantization performance. In Proceedings of the 59th ACM/IEEE Design Automation Conference , pages 25-30, 2022.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Please refer to Section 1 Introduction and Abstract.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Please refer to the Appendix.

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

Justification: Please refer to the Appendix.

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

Justification: The paper clearly explains how to reproduce the algorithm. We will open source the code as much as possible.

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

Justification: We will provide them in the supplemental material.

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

Justification: Most experiment setting and details are in the Appendix C.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We conducted experiments in multiple models and multiple data.

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

Justification: See the Experimental section.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our research complies with the NeurIPS Code of Ethics in all respects.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Please refer to the Appendix.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to

generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: This paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All assets we use are licensed.

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

Justification: We will open source the data and code as soon as possible.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: Our core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

with equality for Q = P X .

Proof

<!-- formula-not-decoded -->

Since KL ( P X ∥ Q ) ≥ 0 , the equality exists only when Q = P X , which concludes the proof.

## A.2 Conditional Mutual Information and Its Variational Form

Let X , Y , and Z be random variables. For all Z -measurable probability measures Q on the space of X , with equality for Q = P X | Z .

Proof

<!-- formula-not-decoded -->

Since KL ( P X | Z ∥ Q ) ≥ 0 , the equality exists only when Q = P X | Z , which concludes the proof.

<!-- formula-not-decoded -->

## Appendix

The Technical Appendix is organized into five key sections. In Section A , we present essential foundational concepts and lemmas. Section B provides detailed derivations and proofs of the theorems discussed in the main paper. Section C offers a comprehensive supplement and further explanations of the experimental results described in the paper. Section D presents a detailed analysis of the computational complexity of our proposed approach. Finally, Section E includes the complete algorithm pseudocode for implementing Generalization Neural Processes.

## A Lemma

In this section, we present essential foundational concepts and lemmas.

## A.1 Variational Form of Mutual Information

Let X and Y be two random variables. For all probability measures Q defined on the space of X , we have

<!-- formula-not-decoded -->

## A.3 Mutual Information Under Conditioning

Let X , Y , and Z be random variables. For all Z -measurable probability measures Q defined on the space of X ,

<!-- formula-not-decoded -->

with equality for Q = P X | Z .

## A.4 Kullback-Leibler Divergence and its Representation

Let P and Q be two probability measures defined on a set X . Let g : X → R be a measurable function, and let E X ∼ Q [exp( g ( X ))] ≤ ∞ . Then

<!-- formula-not-decoded -->

## A.5 Data Processing Inequality

Given random variables X,Y,Z,V , and the Markov Chain:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then we have

For Markov chain we have

Proof. Since

<!-- formula-not-decoded -->

and with the Markov Chain, we have X ⊥ Z | Y , therefore

<!-- formula-not-decoded -->

In addition, I ( X ; Y | Z ) ≥ 0 , so I ( X ; Z ) ≤ I ( X ; Y ) .

<!-- formula-not-decoded -->

with I ( Y ; Z | X ) ≥ 0 , we have I ( X ; Z ) ≤ I ( Y ; Z ) .

Similarly, for the second Markov chain, we have X ⊥ Y | V , therefore

<!-- formula-not-decoded -->

So we have I ( X ; Z | V ) ≤ I ( X ; Y | V ) , the rest proof is similar and omitted.

## A.6 Conditional Independence and Information Bounds

Given random variables X,Y,Z 1 , Z 2 , and the graph model:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then we have

Proof. Apply chain rule, we get:

<!-- formula-not-decoded -->

From the graph model, we have Y ⊥ Z 1 , Y ⊥ Z 2 and ( X,Y ) ⊥ Z 1 | Z 2 . Hence

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

the last equality is obtained with Y ⊥ Z 1 , Z 2 , and since I ( Y ; Z 2 | X,Z 1 ) ≥ 0 , we get

<!-- formula-not-decoded -->

Consequently, we have I ( X ; Y | Z 1 ) ≤ I ( X ; Y | Z 2 ) , conclude the proof.

## A.7 Donsker-Varadhan Representation of Mutual Information

The Donsker-Varadhan representation provides a powerful variational characterization of mutual information that is particularly relevant to our Gen-NPs framework. This representation allows us to derive tractable lower bounds on mutual information, which is essential for the information-theoretic analysis of neural processes.

Let X and Y be random variables with joint distribution P X,Y . The Donsker-Varadhan representation states that:

<!-- formula-not-decoded -->

where the supremum is taken over all measurable functions T for which the expectations exist, and P X ⊗ P Y denotes the product of marginal distributions.

Relation to KL Divergence This representation directly connects to the KL divergence representation in A.4, as mutual information is the KL divergence between the joint distribution and the product of marginals: I ( X ; Y ) = KL ( P X,Y ∥ P X ⊗ P Y ) .

Application to Gen-NPs In our Gen-NPs framework, we leverage this representation to analyze the mutual information between model parameters θ and training data S . By setting X = θ and Y = S , we can derive:

<!-- formula-not-decoded -->

Noise-Contrastive Estimation This representation forms the theoretical basis for our noise injection learning strategy. When we introduce parameter noise during optimization, we are effectively using a specific form of the function T that facilitates estimation of the mutual information between model parameters and training data.

Moreover,

Proof Sketch The proof follows from the convex duality principle and properties of the logarithmic function:

<!-- formula-not-decoded -->

Connection to Generalization Bounds In our analysis of Gen-NPs, this representation enables us to derive generalization bounds that explicitly account for the information complexity of the learning algorithm. Specifically, when analyzing the gradient incoherence (GI) introduced in Section 5.2, we implicitly utilize this variational characterization to establish the connection between noise injection and generalization performance.

This representation is central to our theoretical framework as it provides the mathematical foundation for understanding how parameter noise affects the information bottleneck between model parameters and training data, ultimately improving generalization capabilities of Neural Processes across diverse task distributions.

## B Theorem Proof

This section provides detailed derivations and proofs of the theorems discussed in the main paper.

## B.1 Proof of Theorem 1

If the loss ℓ ( θ, X × Y ) is σ -subgaussian for each θ ∈ Θ with respect to X × Y ∼ µ , the generalization error of a learning algorithm A satisfies the bound | gen ( µ, A ) | ≤ √ 2 σ 2 mn I ( θ ; D ) , where I ( θ ; D ) is the mutual information between the dataset D and the hypothesis θ . For meta-learning tasks, where µ ∼ τ , the meta-generalization error satisfies:

<!-- formula-not-decoded -->

The mutual information I ( θ ; D 1: m ) can be computed based on the joint distribution P ( θ, D 1: m ) and the marginal P ( θ ) ; detailed derivations are provided in the appendix. This bound indicates that minimizing the dependence of θ on the data (via I ( θ ; D 1: m ) ) improves the generalization performance.

Proof Let θ ∈ Θ be a random variable representing the meta-parameter learned from the datasets D 1: m , and let ˆ θ ∈ Θ be an independent copy of θ such that ˆ θ is independent of the datasets D 1: m . The distribution of ˆ θ is the marginal distribution P θ , which is averaged over the possible datasets D 1: m drawn from the environment τ .

The mutual information I ( θ ; D 1: m ) quantifies the dependency between the meta-parameter θ and the observed datasets D 1: m . Specifically, it measures how much information about θ is gained by observing the datasets D 1: m . This dependency impacts the meta generalization error, as the error is influenced by the extent to which θ captures relevant information from the datasets while avoiding overfitting.

To express the meta generalization error, consider the function:

<!-- formula-not-decoded -->

For any λ ∈ R , let

<!-- formula-not-decoded -->

Moreover,

<!-- formula-not-decoded -->

Since ( θ, D i ) , i = 1 , . . . , m are mutually independent given τ , and D 1 , . . . , D m are independent, we have

Hence,

Since ˆ θ ⊥ ⊥ D 1: m , we have that

Hence,

Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If we use Equations (2) and (3), then Equation (1) becomes

<!-- formula-not-decoded -->

Since this inequality is also valid when λ is negative, this implies that we also have

<!-- formula-not-decoded -->

Consequently,

<!-- formula-not-decoded -->

Since ℓ ( θ, Z ) is σ -subgaussian, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is σ √ nm -subgaussian. Hence,

Thus, we have which gives the theorem.

## B.2 Proof of Theorem 2

Theorem B.1. By incorporating R dyn into the information-theoretic framework, the refined metageneralization error bound for Neural Processes (NPs) is given by:

<!-- formula-not-decoded -->

where I ( θ ; D 1: m ) is the mutual information between the meta-parameter θ and the dataset D 1: m , R dyn is the dynamical stability regularization term, and γ &gt; 0 is a scaling factor that controls the influence of R dyn. The proof of this theorem is provided in the Appendix.

Proof Let θ ∈ Θ be a random variable representing the meta-parameter learned from the datasets D 1: m , and let ˆ θ ∈ Θ be an independent copy of θ such that ˆ θ is independent of the datasets D 1: m . The distribution of ˆ θ is the marginal distribution P θ .

We define the effective mutual information I eff ( θ ; D 1: m ) as:

<!-- formula-not-decoded -->

where R dyn = λ 1 · E [ Tr ( H )] + λ 2 · E [ ∥ H ∥ F ] quantifies the complexity of the hypothesis space through the trace and Frobenius norm of the Hessian matrix H .

To express the meta generalization error, consider the function:

<!-- formula-not-decoded -->

For any λ ∈ R , let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By using the value of λ that minimizes the r.h.s. of the above equation, we have

<!-- formula-not-decoded -->

Returning to Equation (4), we have for λ &gt; 0 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, we also have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, Equations (5) and (6) together imply that

<!-- formula-not-decoded -->

For the standard mutual information, we have:

<!-- formula-not-decoded -->

When we incorporate the dynamical stability regularization, the effective mutual information is:

<!-- formula-not-decoded -->

By the data processing inequality and information bottleneck principles, dynamical stability regularization effectively reduces the mutual information through constraining the parameter space complexity:

<!-- formula-not-decoded -->

Continuing with the proof, we have:

<!-- formula-not-decoded -->

Since ( θ, D i ) , i = 1 , . . . , m are mutually independent given τ , and D 1 , . . . , D m are independent, we have

Hence,

<!-- formula-not-decoded -->

Since ˆ θ ⊥ ⊥ D 1: m , we have that

Hence,

Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining these results, we get:

<!-- formula-not-decoded -->

Since this inequality is also valid when λ is negative, we also have:

<!-- formula-not-decoded -->

Consequently,

<!-- formula-not-decoded -->

Since ℓ ( θ, Z ) is σ -subgaussian, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, considering the influence of R dyn on the effective mutual information, we have:

<!-- formula-not-decoded -->

By the principle of information bottleneck and complexity regularization, we can establish:

<!-- formula-not-decoded -->

is σ √ nm -subgaussian. Hence,

Substituting this, we get:

<!-- formula-not-decoded -->

By the information-curvature relationship established through dynamical stability, we can refine the bound:

<!-- formula-not-decoded -->

Optimizing for λ , we set λ = √ 2 nm · I ( θ ; D 1: m ) σ 2 , which gives:

<!-- formula-not-decoded -->

However, considering the impact of R dyn on the mutual information through the lens of information bottleneck theory and complex system dynamics, we can establish that:

<!-- formula-not-decoded -->

Therefore, the final bound becomes:

<!-- formula-not-decoded -->

This completes the proof.

## C Experiments

This section offers a comprehensive supplement and further explanations of the experimental results described in the paper. Here we use TNP-A as TNP.

## C.1 Hardware and Software Configuration

To ensure the reproducibility and reliability of the experiments conducted in this study, we detail the hardware and software environments used.

- GPU Model(s):
- CPU Model(s):

- -

- Model: NVIDIA RTX A4000

- -

- Count: 8 GPUs

- Memory per GPU: 16 GB

- Model: Intel(R) Xeon(R) Platinum 8358P

- -

- Core Count: 32 cores

## · Operating System:

- OS: Ubuntu 20.04 LTS

- Kernel Version: 5.15.0-113-generic

## · Relevant Software Libraries and Frameworks:

- CUDA: Version 11.8

- -

- cuDNN: Version 8.6

- PyTorch: Version 2.0.0

- Scikit-learn: Version 1.5.0

- -

- NumPy: Version 1.26.3

- -

- Pandas: Version 2.2.2

## C.2 1-D Regression

The following sections provide a detailed description of the training and evaluation processes for the 1-D regression task.

Training During the training phase, different functions are drawn from a Gaussian Process (GP) prior with a Radial Basis Function (RBF) kernel for each epoch. These functions are represented as f i ∼ GP ( m,k ) , where the mean function is m ( x ) = 0 and the covariance function is k ( x, x ′ ) = σ 2 f exp ( -( x -x ′ ) 2 2 ℓ 2 ) . The GP hyperparameters ℓ and σ f are randomized for each function, providing a diverse set of training samples. For each function f i , N random locations are selected for evaluation, and an index m is chosen to divide the sequence into context points and target points. The parameters are set as follows: ℓ ∼ U [0 . 6 , 1 . 0) , σ f ∼ U [0 . 1 , 1 . 0) , B = 16 , N ∼ U [6 , 50) , and m ∼ U [3 , 47) .

Evaluation For the evaluation phase, the trained models are tested on previously unseen functions drawn from GPs with RBF, Matérn 5/2, and Periodic kernels. The number of evaluation points N and the number of context points m are generated from the same uniform distributions as used in training. The evaluation set includes 48,000 functions for each kernel type. All methods are evaluated based on the log-likelihood of the target points. Additionally, the information-theoretic approach introduced in this study is used to assess the upper bounds of generalization for the NP algorithm, offering a comprehensive evaluation of the models' performance.

Results The evaluation results of the Gen-Method on three different Gaussian kernels are presented in Table 4. The table showcases the performance metrics including Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Log-likelihood. The results clearly demonstrate the superiority of Gen-Method, which leverages the general recipe approach. Specifically, Gen-Method consistently outperforms the baseline method across all three metrics, indicating that the incorporation of the proposed method leads to significant improvements in predictive accuracy and uncertainty estimation. Sample functions produced by the Gen-Method and the baselines given 30 context points are illustrated in Figure 6, further highlighting the enhanced performance of Gen-Method in terms of capturing the underlying function variability and providing more accurate predictions.

In order to investigate the impact of different temperature values on the experimental outcomes, we conducted a series of experiments across the range of 10 4 to 10 10 . The experimental results are illustrated in Figure 5. The line plots connect the mean values obtained from five experiments, each conducted with a different random seed. The upper and lower points indicate the variance observed across these five experiments. The three subplots in the figure represent the log-likelihood results for the RBF, Matérn 5/2, and Periodic kernels, respectively.

Figure 5: Experimental results across different temperature values. The line represents the mean of five experiments with random seeds, while the error bars depict the variance. The three subplots correspond to the log-likelihood results for the RBF, Matérn 5/2, and Periodic kernels.

<!-- image -->

## C.3 Image Completion

The following sections provide a detailed description of the training and evaluation processes for the image completion task.

Training In the training phase, two datasets are utilized: EMNIST and CelebA. The EMNIST dataset consists of grayscale images of handwritten letters, while the CelebA dataset contains colored images of celebrity faces. Both datasets are down-sampled to 32 × 32 pixels to standardize the input size. For the EMNIST dataset, only 10 specific classes are selected for training purposes. During training, random subsets of pixels are chosen as context and target points, where the number of total points N is sampled from a uniform distribution U [6 , 200] and the number of context points m is sampled from U [3 , 197] . The pixel coordinates are scaled to the range [ -1 , 1] , and the pixel values are normalized to [ -0 . 5 , 0 . 5] to ensure consistency across the training process.

Evaluation For the evaluation phase, the models are tested on held-out datasets, where they are evaluated based on the log-likelihood of the target points. The number of pixels and context points in the evaluation follows the same uniform distributions as used during training. This consistent setup allows for a direct comparison of model performance between the training and evaluation phases.

Table 4: Comparison of GR-NPs with the baselines on various GP kernels and evaluation metrics. 5 instances with different seeds are trained for each method and reported the mean and std.

| Metric         | Method        | RBF                             | Matérn 5/2                      | Periodic                          |
|----------------|---------------|---------------------------------|---------------------------------|-----------------------------------|
| MAE            | CNP Gen-CNP   | 0.1691 ± 0.0020 0.1674 ± 0.0016 | 0.1971 ± 0.0022 0.1957 ± 0.0014 | 0.4706 ± 0.0007 0.4704 ± 0.0006   |
| MAE            | NP Gen-NP     | 0.1743 ± 0.0029 0.1723 ± 0.0025 | 0.2036 ± 0.0028 0.2015 ± 0.0024 | 0.4687 ± 0.0007 0.4671 ± 0.0011   |
| MAE            | ANP Gen-ANP   | 0.1035 ± 0.0003 0.1034 ± 0.0002 | 0.1291 ± 0.0002 0.1290 ± 0.0001 | 0.4970 ± 0.0014 0.4952 ± 0.0020   |
| MAE            | BNP Gen-BNP   | 0.1606 ± 0.0023 0.1595 ± 0.0015 | 0.1898 ± 0.0024 0.1886 ± 0.0016 | 0.4698 ± 0.0008 0.4685 ± 0.0012   |
| MAE            | TNP Gen-TNP   | 0.0939 ± 0.0002 0.0938 ± 0.0001 | 0.1246 ± 0.0001 0.1245 ± 0.0001 | 0.4674 ± 0.0052 0.4638 ± 0.0095   |
| MAE            | CNP           | 0.2760 ± 0.0025                 | 0.3077 ± 0.0026                 | 0.6522 ± 0.0013                   |
| RMSE           | Gen-CNP       | 0.2742 ± 0.0018                 | 0.3060 ± 0.0016                 | 0.6517 ± 0.0008                   |
| RMSE           | NP Gen-NP ANP | 0.2843 ± 0.0036 0.2816 ± 0.0028 | 0.3165 ± 0.0036 0.3139 ± 0.0026 | 0.6496 ± 0.0008 0.6474 ± 0.0015   |
| RMSE           | Gen-ANP       | 0.1932 ± 0.0005 0.1931 ± 0.0002 | 0.2295 ± 0.0003 0.2294 ± 0.0002 | 0.7041 ± 0.0021 0.7037 ± 0.0036   |
| RMSE           | BNP Gen-BNP   | 0.2669 ± 0.0029 0.2654 ± 0.0019 | 0.2995 ± 0.0028 0.2982 ± 0.0019 | 0.6513 ± 0.0010 0.6488 ± 0.0020   |
| RMSE           | TNP Gen-TNP   | 0.1772 ± 0.0003 0.1770 ± 0.0002 | 0.2220 ± 0.0001 0.2219 ± 0.0003 | 0.6591 ± 0.0091 0.6519 ± 0.0157   |
| Log-Likelihood | CNP Gen-CNP   | 0.2648 ± 0.0154 0.2863 ± 0.0103 | 0.0452 ± 0.0138 0.0608 ± 0.0054 | -1.4353 ± 0.0196 -1.4176 ± 0.0234 |
| Log-Likelihood | NP Gen-NP     | 0.2403 ± 0.0218 0.2697 ± 0.0094 | 0.0512 ± 0.0188 0.0726 ± 0.0072 | -1.1447 ± 0.0316 -1.1248 ± 0.0247 |
| Log-Likelihood | ANP Gen-ANP   | 0.8051 ± 0.0053 0.8124 ± 0.0028 | 0.6304 ± 0.0038 0.6357 ± 0.0023 | -5.3196 ± 0.2592 -5.0275 ± 0.2895 |
| Log-Likelihood | BNP Gen-BNP   | 0.3887 ± 0.0167 0.4052 ± 0.0093 | 0.1853 ± 0.0148                 | -0.9694 ± 0.0163 -0.9467 ± 0.0108 |
| Log-Likelihood | TNP           | 1.6503 ± 0.0052                 | 0.2003 ± 0.0084 1.2185 ± 0.0047 | -2.3196 ± 0.1748                  |
| Log-Likelihood | Gen-TNP       | 1.6624 ± 0.0032                 | 1.2263 ± 0.0027                 | -2.0095 ± 0.1697                  |

Results Table 5 and Table 6 present the results of the Gen-NPs method on the CelebA and EMNIST datasets, respectively, showcasing the performance metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Log-Likelihood. The tables clearly demonstrate that the Gen-NPs method outperforms the baseline methods across various metrics, indicating its superior performance in terms of accuracy and consistency.

Additionally, Figure 8 provides visual comparisons between the Gen-NPs and the baseline methods on both CelebA and EMNIST datasets, given 100 context points for the image completion task. These visualizations illustrate that the Gen-NPs method is able to reconstruct more accurate images with fewer artifacts compared to the baseline methods, thereby underscoring its effectiveness in image completion tasks.

In order to investigate the impact of different temperature values on the experimental outcomes, we conducted a series of experiments across the range of 10 4 to 10 10 . The experimental results are illustrated in Figure 7. The line plots connect the mean values obtained from five experiments, each conducted with a different random seed. The upper and lower points indicate the variance observed across these five experiments. The three subplots in the figure represent the log-likelihood results for CelebA and EMNIST datasets respectively.

Table 5: Comparison of Gen-NPs with the baselines on CelebA dataset with various evaluation metrics. 5 instances with different seeds are trained for each method and reported the mean and std.

| Method   | MAE             | RMSE            | Log-Likelihood   |
|----------|-----------------|-----------------|------------------|
| CNP      | 0.0935 ± 0.0003 | 0.1369 ± 0.0002 | 2.1595 ± 0.0040  |
| Gen-CNP  | 0.0920 ± 0.0002 | 0.1350 ± 0.0002 | 2.1879 ± 0.0048  |
| NP       | 0.0935 ± 0.0004 | 0.1373 ± 0.0005 | 2.4811 ± 0.0147  |
| Gen-NP   | 0.0933 ± 0.0002 | 0.1369 ± 0.0003 | 2.5237 ± 0.0075  |
| ANP      | 0.0763 ± 0.0002 | 0.1182 ± 0.0004 | 2.9209 ± 0.0037  |
| Gen-ANP  | 0.0759 ± 0.0001 | 0.1176 ± 0.0001 | 2.9634 ± 0.0077  |
| BNP      | 0.0926 ± 0.0004 | 0.1340 ± 0.0003 | 2.7691 ± 0.0025  |
| Gen-BNP  | 0.0900 ± 0.0002 | 0.1314 ± 0.0002 | 2.7758 ± 0.0030  |
| TNP      | 0.0754 ± 0.0002 | 0.1146 ± 0.0002 | 4.4044 ± 0.0201  |
| Gen-TNP  | 0.0753 ± 0.0001 | 0.1144 ± 0.0000 | 4.4086 ± 0.0081  |

Table 6: Comparison of Gen-NPs vs the baselines on EMNIST dataset with various evaluation metrics. We train 5 instances with different seeds for each method and report the mean and std. We evaluate on both seen and unseen classes.

| Method          | MAE                                               | RMSE                                            | Log-Likelihood                                  | Log-Likelihood                                  | Log-Likelihood                                  | Log-Likelihood                                  | Log-Likelihood                                  | Log-Likelihood                                  | Log-Likelihood                                  | Log-Likelihood                                  | Log-Likelihood                                  | Log-Likelihood                                  | Log-Likelihood                                  | Log-Likelihood                                  | Log-Likelihood                                  | Log-Likelihood                                  | Log-Likelihood                                  | Log-Likelihood                                  | Log-Likelihood                                  | Log-Likelihood                                  | Log-Likelihood                                  |
|-----------------|---------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|
| CNP Gen-CNP     | 0.0933 ± 0.0004 0.1829 ± 0.0853 ± 0.0009 0.1704 ± | 0.0006 0.0014                                   | 0.7373 ± 0.0037 0.7864 ± 0.0054                 | 0.7373 ± 0.0037 0.7864 ± 0.0054                 | 0.7373 ± 0.0037 0.7864 ± 0.0054                 | 0.7373 ± 0.0037 0.7864 ± 0.0054                 | 0.7373 ± 0.0037 0.7864 ± 0.0054                 | 0.7373 ± 0.0037 0.7864 ± 0.0054                 | 0.7373 ± 0.0037 0.7864 ± 0.0054                 | 0.7373 ± 0.0037 0.7864 ± 0.0054                 | 0.7373 ± 0.0037 0.7864 ± 0.0054                 | 0.7373 ± 0.0037 0.7864 ± 0.0054                 | 0.7373 ± 0.0037 0.7864 ± 0.0054                 | 0.7373 ± 0.0037 0.7864 ± 0.0054                 | 0.7373 ± 0.0037 0.7864 ± 0.0054                 | 0.7373 ± 0.0037 0.7864 ± 0.0054                 | 0.7373 ± 0.0037 0.7864 ± 0.0054                 | 0.7373 ± 0.0037 0.7864 ± 0.0054                 | 0.7373 ± 0.0037 0.7864 ± 0.0054                 | 0.7373 ± 0.0037 0.7864 ± 0.0054                 | 0.7373 ± 0.0037 0.7864 ± 0.0054                 |
| NP Gen-NP       | 0.0948 ± 0.0006 0.0900 ± 0.0009                   | 0.1850 ± 0.0007 0.1793 ± 0.0010                 | 0.7954 ± 0.0022 0.8142 ± 0.0063                 | 0.7954 ± 0.0022 0.8142 ± 0.0063                 | 0.7954 ± 0.0022 0.8142 ± 0.0063                 | 0.7954 ± 0.0022 0.8142 ± 0.0063                 | 0.7954 ± 0.0022 0.8142 ± 0.0063                 | 0.7954 ± 0.0022 0.8142 ± 0.0063                 | 0.7954 ± 0.0022 0.8142 ± 0.0063                 | 0.7954 ± 0.0022 0.8142 ± 0.0063                 | 0.7954 ± 0.0022 0.8142 ± 0.0063                 | 0.7954 ± 0.0022 0.8142 ± 0.0063                 | 0.7954 ± 0.0022 0.8142 ± 0.0063                 | 0.7954 ± 0.0022 0.8142 ± 0.0063                 | 0.7954 ± 0.0022 0.8142 ± 0.0063                 | 0.7954 ± 0.0022 0.8142 ± 0.0063                 | 0.7954 ± 0.0022 0.8142 ± 0.0063                 | 0.7954 ± 0.0022 0.8142 ± 0.0063                 | 0.7954 ± 0.0022 0.8142 ± 0.0063                 | 0.7954 ± 0.0022 0.8142 ± 0.0063                 | 0.7954 ± 0.0022 0.8142 ± 0.0063                 |
| ANP Gen-ANP BNP | 0.0681 ± 0.0008 0.0673 ± 0.0006                   | 0.1425 ± 0.0011 0.1411 ± 0.0008                 | 0.9808 ± 0.0060 0.9865 ± 0.0043                 | 0.9808 ± 0.0060 0.9865 ± 0.0043                 | 0.9808 ± 0.0060 0.9865 ± 0.0043                 | 0.9808 ± 0.0060 0.9865 ± 0.0043                 | 0.9808 ± 0.0060 0.9865 ± 0.0043                 | 0.9808 ± 0.0060 0.9865 ± 0.0043                 | 0.9808 ± 0.0060 0.9865 ± 0.0043                 | 0.9808 ± 0.0060 0.9865 ± 0.0043                 | 0.9808 ± 0.0060 0.9865 ± 0.0043                 | 0.9808 ± 0.0060 0.9865 ± 0.0043                 | 0.9808 ± 0.0060 0.9865 ± 0.0043                 | 0.9808 ± 0.0060 0.9865 ± 0.0043                 | 0.9808 ± 0.0060 0.9865 ± 0.0043                 | 0.9808 ± 0.0060 0.9865 ± 0.0043                 | 0.9808 ± 0.0060 0.9865 ± 0.0043                 | 0.9808 ± 0.0060 0.9865 ± 0.0043                 | 0.9808 ± 0.0060 0.9865 ± 0.0043                 | 0.9808 ± 0.0060 0.9865 ± 0.0043                 | 0.9808 ± 0.0060 0.9865 ± 0.0043                 |
| Gen-BNP         | 0.0926 ± 0.0009 0.0828 ± 0.0014                   | 0.1803 ± 00013 0.1653 ± 0.0020                  | 0.8699 ± 0.0054 0.9051 ± 0.0063                 | 0.8699 ± 0.0054 0.9051 ± 0.0063                 | 0.8699 ± 0.0054 0.9051 ± 0.0063                 | 0.8699 ± 0.0054 0.9051 ± 0.0063                 | 0.8699 ± 0.0054 0.9051 ± 0.0063                 | 0.8699 ± 0.0054 0.9051 ± 0.0063                 | 0.8699 ± 0.0054 0.9051 ± 0.0063                 | 0.8699 ± 0.0054 0.9051 ± 0.0063                 | 0.8699 ± 0.0054 0.9051 ± 0.0063                 | 0.8699 ± 0.0054 0.9051 ± 0.0063                 | 0.8699 ± 0.0054 0.9051 ± 0.0063                 | 0.8699 ± 0.0054 0.9051 ± 0.0063                 | 0.8699 ± 0.0054 0.9051 ± 0.0063                 | 0.8699 ± 0.0054 0.9051 ± 0.0063                 | 0.8699 ± 0.0054 0.9051 ± 0.0063                 | 0.8699 ± 0.0054 0.9051 ± 0.0063                 | 0.8699 ± 0.0054 0.9051 ± 0.0063                 | 0.8699 ± 0.0054 0.9051 ± 0.0063                 | 0.8699 ± 0.0054 0.9051 ± 0.0063                 |
| TNP Gen-TNP CNP | 0.0585 ± 0.0008 0.0578 ± 0.0004                   | 0.1231 ± 0.0008 0.1221 ± 0.0009                 | 1.5502 ± 0.0036 1.5550 ± 0.0021                 | 1.5502 ± 0.0036 1.5550 ± 0.0021                 | 1.5502 ± 0.0036 1.5550 ± 0.0021                 | 1.5502 ± 0.0036 1.5550 ± 0.0021                 | 1.5502 ± 0.0036 1.5550 ± 0.0021                 | 1.5502 ± 0.0036 1.5550 ± 0.0021                 | 1.5502 ± 0.0036 1.5550 ± 0.0021                 | 1.5502 ± 0.0036 1.5550 ± 0.0021                 | 1.5502 ± 0.0036 1.5550 ± 0.0021                 | 1.5502 ± 0.0036 1.5550 ± 0.0021                 | 1.5502 ± 0.0036 1.5550 ± 0.0021                 | 1.5502 ± 0.0036 1.5550 ± 0.0021                 | 1.5502 ± 0.0036 1.5550 ± 0.0021                 | 1.5502 ± 0.0036 1.5550 ± 0.0021                 | 1.5502 ± 0.0036 1.5550 ± 0.0021                 | 1.5502 ± 0.0036 1.5550 ± 0.0021                 | 1.5502 ± 0.0036 1.5550 ± 0.0021                 | 1.5502 ± 0.0036 1.5550 ± 0.0021                 | 1.5502 ± 0.0036 1.5550 ± 0.0021                 |
| Gen-CNP         | 0.1231 ± 0.0005 0.1098 ± 0.0013                   | 0.2264 ± 0.0007 0.2084 ± 0.0013                 | 0.4854 ± 0.0035 0.5556 ± 0.0055                 | 0.4854 ± 0.0035 0.5556 ± 0.0055                 | 0.4854 ± 0.0035 0.5556 ± 0.0055                 | 0.4854 ± 0.0035 0.5556 ± 0.0055                 | 0.4854 ± 0.0035 0.5556 ± 0.0055                 | 0.4854 ± 0.0035 0.5556 ± 0.0055                 | 0.4854 ± 0.0035 0.5556 ± 0.0055                 | 0.4854 ± 0.0035 0.5556 ± 0.0055                 | 0.4854 ± 0.0035 0.5556 ± 0.0055                 | 0.4854 ± 0.0035 0.5556 ± 0.0055                 | 0.4854 ± 0.0035 0.5556 ± 0.0055                 | 0.4854 ± 0.0035 0.5556 ± 0.0055                 | 0.4854 ± 0.0035 0.5556 ± 0.0055                 | 0.4854 ± 0.0035 0.5556 ± 0.0055                 | 0.4854 ± 0.0035 0.5556 ± 0.0055                 | 0.4854 ± 0.0035 0.5556 ± 0.0055                 | 0.4854 ± 0.0035 0.5556 ± 0.0055                 | 0.4854 ± 0.0035 0.5556 ± 0.0055                 | 0.4854 ± 0.0035 0.5556 ± 0.0055                 |
| NP Gen-NP ANP   | 0.1261 ± 0.0014 0.1184 ± 0.0009 0.0829 ± 0.0004   | 0.2306 ± 0.0013 0.2218 ± 0.0011 0.1676 ± 0.0007 | 0.6031 ± 0.0079 0.8838 ± 0.0030 0.8862 ± 0.0036 | 0.6031 ± 0.0079 0.8838 ± 0.0030 0.8862 ± 0.0036 | 0.6031 ± 0.0079 0.8838 ± 0.0030 0.8862 ± 0.0036 | 0.6031 ± 0.0079 0.8838 ± 0.0030 0.8862 ± 0.0036 | 0.6031 ± 0.0079 0.8838 ± 0.0030 0.8862 ± 0.0036 | 0.6031 ± 0.0079 0.8838 ± 0.0030 0.8862 ± 0.0036 | 0.6031 ± 0.0079 0.8838 ± 0.0030 0.8862 ± 0.0036 | 0.6031 ± 0.0079 0.8838 ± 0.0030 0.8862 ± 0.0036 | 0.6031 ± 0.0079 0.8838 ± 0.0030 0.8862 ± 0.0036 | 0.6031 ± 0.0079 0.8838 ± 0.0030 0.8862 ± 0.0036 | 0.6031 ± 0.0079 0.8838 ± 0.0030 0.8862 ± 0.0036 | 0.6031 ± 0.0079 0.8838 ± 0.0030 0.8862 ± 0.0036 | 0.6031 ± 0.0079 0.8838 ± 0.0030 0.8862 ± 0.0036 | 0.6031 ± 0.0079 0.8838 ± 0.0030 0.8862 ± 0.0036 | 0.6031 ± 0.0079 0.8838 ± 0.0030 0.8862 ± 0.0036 | 0.6031 ± 0.0079 0.8838 ± 0.0030 0.8862 ± 0.0036 | 0.6031 ± 0.0079 0.8838 ± 0.0030 0.8862 ± 0.0036 | 0.6031 ± 0.0079 0.8838 ± 0.0030 0.8862 ± 0.0036 | 0.6031 ± 0.0079 0.8838 ± 0.0030 0.8862 ± 0.0036 |
| BNP Gen-BNP     | 0.1229 ± 0.0014 0.1077 ± 0.0019                   | 0.1668 ± 0.0010 0.2225 ± 0.0020 0.2030 ± 0.0022 | 0.7156 ±                                        | 0.7156 ±                                        | 0.7156 ±                                        | 0.7156 ±                                        | 0.7156 ±                                        | 0.7156 ±                                        | 0.7156 ±                                        | 0.7156 ±                                        | 0.7156 ±                                        | 0.7156 ±                                        | 0.7156 ±                                        | 0.7156 ±                                        | 0.7156 ±                                        | 0.7156 ±                                        | 0.7156 ±                                        | 0.7156 ±                                        | 0.7156 ±                                        | 0.7156 ±                                        | 0.7156 ±                                        |
| TNP             | 0.0707 ± 0.0013                                   | 0.1452 ± 0.0013                                 | 0.7642 ± 1.4190 ±                               | 0.7642 ± 1.4190 ±                               | 0.7642 ± 1.4190 ±                               | 0.7642 ± 1.4190 ±                               | 0.7642 ± 1.4190 ±                               | 0.7642 ± 1.4190 ±                               | 0.7642 ± 1.4190 ±                               | 0.7642 ± 1.4190 ±                               | 0.7642 ± 1.4190 ±                               | 0.7642 ± 1.4190 ±                               | 0.7642 ± 1.4190 ±                               | 0.7642 ± 1.4190 ±                               | 0.7642 ± 1.4190 ±                               | 0.7642 ± 1.4190 ±                               | 0.7642 ± 1.4190 ±                               | 0.7642 ± 1.4190 ±                               | 0.7642 ± 1.4190 ±                               | 0.7642 ± 1.4190 ±                               | 0.7642 ± 1.4190 ±                               |
|                 |                                                   | 0.1447 ± 0.0002                                 | ±                                               | ±                                               | ±                                               | ±                                               | ±                                               | ±                                               | ±                                               | ±                                               | ±                                               | ±                                               | ±                                               | ±                                               | ±                                               | ±                                               | ±                                               | ±                                               | ±                                               | ±                                               | ±                                               |
| Gen-TNP         |                                                   |                                                 |                                                 |                                                 |                                                 |                                                 |                                                 |                                                 |                                                 |                                                 |                                                 |                                                 |                                                 |                                                 |                                                 |                                                 |                                                 |                                                 |                                                 |                                                 |                                                 |
|                 | 0.0701 ± 0.0001                                   |                                                 | 1.4232                                          | 1.4232                                          | 1.4232                                          | 1.4232                                          | 1.4232                                          | 1.4232                                          | 1.4232                                          | 1.4232                                          | 1.4232                                          | 1.4232                                          | 1.4232                                          | 1.4232                                          | 1.4232                                          | 1.4232                                          | 1.4232                                          | 1.4232                                          | 1.4232                                          | 1.4232                                          | 1.4232                                          |
|                 |                                                   |                                                 | 0.0045                                          | 0.0045                                          | 0.0045                                          | 0.0045                                          | 0.0045                                          | 0.0045                                          | 0.0045                                          | 0.0045                                          | 0.0045                                          | 0.0045                                          | 0.0045                                          | 0.0045                                          | 0.0045                                          | 0.0045                                          | 0.0045                                          | 0.0045                                          | 0.0045                                          | 0.0045                                          | 0.0045                                          |
| Gen-ANP         | 0.0824 ± 0.0006                                   |                                                 | 0.0117                                          | 0.0117                                          | 0.0117                                          | 0.0117                                          | 0.0117                                          | 0.0117                                          | 0.0117                                          | 0.0117                                          | 0.0117                                          | 0.0117                                          | 0.0117                                          | 0.0117                                          | 0.0117                                          | 0.0117                                          | 0.0117                                          | 0.0117                                          | 0.0117                                          | 0.0117                                          | 0.0117                                          |
|                 |                                                   |                                                 | 0.0081                                          | 0.0081                                          | 0.0081                                          | 0.0081                                          | 0.0081                                          | 0.0081                                          | 0.0081                                          | 0.0081                                          | 0.0081                                          | 0.0081                                          | 0.0081                                          | 0.0081                                          | 0.0081                                          | 0.0081                                          | 0.0081                                          | 0.0081                                          | 0.0081                                          | 0.0081                                          | 0.0081                                          |
|                 |                                                   |                                                 | 0.0061                                          | 0.0061                                          | 0.0061                                          | 0.0061                                          | 0.0061                                          | 0.0061                                          | 0.0061                                          | 0.0061                                          | 0.0061                                          | 0.0061                                          | 0.0061                                          | 0.0061                                          | 0.0061                                          | 0.0061                                          | 0.0061                                          | 0.0061                                          | 0.0061                                          | 0.0061                                          | 0.0061                                          |

## C.4 Bayesian Optimization

The following sections provide a detailed description of the training and evaluation processes for the Bayesian optimization task.

Training In the training phase, the 1D scenario follows the approach outlined in Section 4.1. For multi-dimensional input x , training data is generated using the method proposed by NPs, where multivariate Gaussian Processes (GPs) with a Radial Basis Function (RBF) kernel are employed. In the 2D scenario, the number of total points N is sampled from a uniform distribution U [60 , 128] , and the number of context points m is sampled from U [30 , 98] . Similarly, in the 3D scenario, N is sampled from U [128 , 256] , and m is sampled from U [64 , 192] . This training setup ensures that the models are

well-prepared to handle the complexities of Bayesian optimization tasks in both one-dimensional and multi-dimensional spaces.

Evaluation For the evaluation phase, in the 1D scenario, the objective functions are generated from Gaussian Processes with RBF, Matérn 5/2, and Periodic kernels. In multi-dimensional settings, various benchmark functions from the optimization literature are employed, with the Bayesian optimization process implemented using a comprehensive framework that includes both the optimization and acquisition functions. Each objective function undergoes 100 iterations of Bayesian optimization, with simple regret serving as the primary evaluation metric. This metric provides a clear indication of the model's performance by measuring the difference between the best-known value and the actual value found during optimization.

Results: In the main body of the paper, for the sake of clarity, we only presented the results for Bayesian optimization using CNP, NP, and ANP methods. However, to provide a more comprehensive and detailed comparison, Figures 9 to 11 showcase the results for all methods, including their Genenhanced variants, across different dimensions and benchmark functions. Each method's performance is individually illustrated for 1D, 2D, and 3D Bayesian optimization tasks, providing a clearer visualization of the differences.

Figure 9 focuses on the 1D Bayesian optimization tasks, where results for different kernel functions (RBF, Matérn 5/2, and Periodic) are presented for each method. Figure 10 expands the comparison to 2D tasks, demonstrating the performance across various benchmark functions like Ackley, Dropwave, and Michalewicz. Finally, Figure 11 extends the analysis to 3D tasks, further illustrating the effectiveness of each method on more complex functions such as Cosine and Rastrigin.

These figures collectively offer a detailed and nuanced understanding of how each method performs under varying conditions, highlighting the consistent advantages of the Gen-enhanced approaches in achieving lower regret and more stable optimization results across all tested scenarios.

Table 7: Cumulative regret for different methods across various δ values.

| Method Uniform   | δ = 0 . 7 100 . 00 ± 1 . 18   | δ = 0 . 9 100 . 00 ± 3 . 03   | δ = 0 . 95 100 . 00 ± 4 . 16   | δ = 0 . 99 100 . 00 ± 7 . 52   | δ = 0 . 995 100 . 00 ± 8 . 11   | δ = 0 . 999 100 . 00 ± 7 . 96   |
|------------------|-------------------------------|-------------------------------|--------------------------------|--------------------------------|---------------------------------|---------------------------------|
| CNP              | 1 . 66 ± 0 . 14               | 8 . 86 ± 0 . 56               | 8 . 31 ± 0 . 85                | 23 . 84 ± 0 . 58               | 34 . 10 ± 0 . 56                | 83 . 90 ± 1 . 97                |
| Gen-CNP          | 1.38 ± 0.19                   | 4.57 ± 0.55                   | 6.41 ± 0.60                    | 15.85 ± 0.73                   | 22.12 ± 0.63                    | 54.46 ± 1.50                    |
| NP               | 1.53 ± 0.20                   | 4 . 24 ± 0 . 53               | 5 . 26 ± 0 . 31                | 20 . 34 ± 0 . 73               | 28 . 85 ± 0 . 43                | 71 . 09 ± 1 . 01                |
| Gen-NP           | 1 . 57 ± 0 . 20               | 3.11 ± 0.22                   | 4.35 ± 0.28                    | 18.63 ± 0.93                   | 26.14 ± 0.56                    | 63.96 ± 1.44                    |
| ANP              | 103 . 11 ± 44 . 89            | 122 . 55 ± 3 . 41             | 119 . 75 ± 0 . 87              | 100 . 04 ± 1 . 00              | 89 . 80 ± 1 . 34                | 51 . 59 ± 16 . 58               |
| Gen-ANP          | 87.77 ± 54.56                 | 96.40 ± 49.09                 | 98.22 ± 40.54                  | 91.78 ± 11.25                  | 84.40 ± 7.71                    | 44.98 ± 12.70                   |
| BNP              | 74 . 04 ± 1 . 75              | 77 . 46 ± 2 . 05              | 74 . 55 ± 2 . 56               | 87 . 88 ± 4 . 43               | 97 . 62 ± 5 . 15                | 108 . 79 ± 4 . 86               |
| Gen-BNP          | 42.68 ± 1.35                  | 34.42 ± 2.28                  | 24.00 ± 2.45                   | 27.65 ± 4.51                   | 34.37 ± 4.64                    | 59.00 ± 3.58                    |
| TNP              | 3 . 02 ± 2 . 52               | 3 . 27 ± 1 . 24               | 5 . 76 ± 2 . 03                | 19 . 61 ± 2 . 84               | 27 . 67 ± 3 . 83                | 9 . 61 ± 2 . 80                 |
| Gen-TNP          | 1.91 ± 1.09                   | 1.85 ± 0.43                   | 2.63 ± 0.50                    | 3.18 ± 0.89                    | 4.58 ± 1.85                     | 8.89 ± 0.39                     |

## C.5 Contextual bandits

The study compares Gen-NPs with baselines using the wheel bandit framework. This framework involves a unit circle segmented into a low-reward zone (colored blue) and four high-reward zones of different colors. The division is controlled by a scalar δ , which sets the boundary of the low-reward zone, leaving the other four zones equally sized. An agent, unaware of δ 's value, selects from five potential actions based on its position within the circle.

When the agent's position ∥ X ∥ is less than or equal to δ , it is located in the low-reward zone. The optimal choice here is action k = 1 , rewarding the agent with r ∼ N (1 . 2 , 0 . 012) . All other actions yield r ∼ N (1 . 0 , 0 . 012) . Conversely, if ∥ X ∥ &gt; δ , indicating presence in a high-reward zone, the agent should choose from actions k = 2 -5 . These choices can grant a significant reward of r ∼ N (50 . 0 , 0 . 012) , with all non-optimal choices returning N (1 . 0 , 0 . 012) , except for k = 1 .

Figure 12: The wheel bandit problem with varying values of δ .

<!-- image -->

Training A dataset is created by generating B different wheel problem instances { δ i } B i =1 , where δ values are uniformly distributed between 0 and 1. For each instance, N points are sampled to assess and select m points as context for training, with each point being a pair ( X,r ) of coordinates and the corresponding reward. The goal is to learn to predict reward values based on X . Parameters are set with B = 8 , N = 562 , and m = 512 .

Evaluation Experiments are conducted by testing the Gen-NPs and baseline approaches across varying δ values, using 50 different seeds for each setting. Over 2000 steps per trial, each agent's task is to estimate the reward values for five strategies based on X , choose according to the Upper Confidence Bound (UCB) strategy, and receive the actual reward for the selected strategy. Cumulative regret is utilized to measure the performance effectiveness.

Results As shown in Table 7, for cumulative regret, the Gen versions of CNP, NP, ANP, BNP, and TNP generally achieve lower regret across all δ settings, demonstrating their superior capability in handling the complex decision-making scenarios introduced by the wheel bandit framework. Notably, Gen-TNP outperforms all other methods, especially in the most difficult cases (higher δ ), where it maintains low regret with minimal variance.

Similarly, Table 8 presents the simple regret outcomes, further confirming the advantage of Genenhanced methods. The simple regret is notably lower for Gen versions, indicating more effective exploration and exploitation of the reward landscape, which is crucial in achieving optimal decisionmaking.

The visual depiction of the wheel bandit problem with varying δ values is provided in Figure 12. This figure illustrates the segmentation of the reward zones within the unit circle, helping to contextualize the challenge faced by the models in predicting optimal actions based on incomplete and uncertain information.

Overall, the results clearly demonstrate the effectiveness of integrating the proposed general recipe into neural process-based models, significantly improving their performance in contextual bandit problems by reducing both cumulative and simple regret across various scenarios.

Table 8: Simple regret for different methods across various δ values.

| Method Uniform   | δ = 0 . 7 100 . 00 ± 20 . 77   | δ = 0 . 9 100 . 00 ± 34 . 60   | δ = 0 . 95 100 . 00 ± 50 . 34   | δ = 0 . 99 100 . 00 ± 96 . 59   | δ = 0 . 995 100 . 00 ± 114 . 30   | δ = 0 . 999 100 . 00 ± 120 . 11   |
|------------------|--------------------------------|--------------------------------|---------------------------------|---------------------------------|-----------------------------------|-----------------------------------|
| CNP              | 1 . 43 ± 2 . 24                | 9 . 27 ± 10 . 13               | 8 . 59 ± 9 . 85                 | 24 . 70 ± 1 . 36                | 34 . 82 ± 1 . 88                  | 83 . 05 ± 4 . 13                  |
| Gen-CNP          | 1.08 ± 1.75                    | 4.50 ± 6.37                    | 6.07 ± 8.36                     | 16.14 ± 2.04                    | 22.38 ± 2.76                      | 53.43 ± 6.48                      |
| NP               | 1 . 42 ± 2 . 14                | 3 . 78 ± 5 . 28                | 4 . 95 ± 5 . 59                 | 20 . 85 ± 1 . 81                | 29 . 40 ± 2 . 57                  | 70 . 34 ± 5 . 64                  |
| Gen-NP           | 1.30 ± 2.06                    | 2.89 ± 4.18                    | 4.20 ± 3.52                     | 19.10 ± 1.94                    | 26.82 ± 2.68                      | 63.61 ± 6.17                      |
| ANP              | 104 . 60 ± 15 . 58             | 125 . 37 ± 36 . 50             | 122 . 68 ± 55 . 36              | 104 . 02 ± 112 . 00             | 95 . 26 ± 134 . 43                | 44 . 79 ± 144 . 48                |
| Gen-ANP          | 88.46 ± 14.62                  | 98.99 ± 32.93                  | 101.83 ± 51.37                  | 101.29 ± 111.32                 | 93.91 ± 132.74                    | 40.21 ± 129.59                    |
| BNP              | 70 . 24 ± 15 . 58              | 73 . 72 ± 28 . 81              | 71 . 70 ± 40 . 01               | 84 . 53 ± 84 . 77               | 95 . 24 ± 101 . 92                | 106 . 83 ± 100 . 60               |
| Gen-BNP          | 42.81 ± 12.88                  | 33.15 ± 19.68                  | 22.24 ± 21.15                   | 23.24 ± 31.65                   | 31.23 ± 41.65                     | 60.13 ± 65.35                     |
| TNP              | 1 . 37 ± 2 . 11                | 2 . 29 ± 4 . 32                | 4 . 70 ± 10 . 05                | 17 . 75 ± 41 . 28               | 25 . 52 ± 59 . 07                 | 8 . 63 ± 3 . 67                   |
| Gen-TNP          | 1.05 ± 1.57                    | 1.42 ± 2.86                    | 2.24 ± 5.46                     | 2.96 ± 1.22                     | 4.00 ± 1.65                       | 8.68 ± 3.84                       |

## C.6 Ablation Study

We conducted ablation studies to validate the effectiveness of the proposed Risk-Aware Dynamical Stability Regularization (DSR) and Optimization-Aware Noise Injection Learning Strategy (NILS) on 1D regression task. Table 9 presents the results of the original method, Gen-NPs with only DSR, Gen-NPs with only NILS, and the full Gen-NPs with both modules included.

Table 9: Ablation study results comparing the original method.

| Method                                                                            | LL (RBF)                                                            | GI (RBF)                                                          | LL (Periodic)       | GI (Periodic)     |
|-----------------------------------------------------------------------------------|---------------------------------------------------------------------|-------------------------------------------------------------------|---------------------|-------------------|
| Original CNP Gen-CNP (with DSR Gen-CNP (with NILS Full Gen-CNP (DSR + Original NP | 0 . 265 ± 0 . 015 0 . 276 ± 0 . 013 0 . 279 ± 0 . 012 0 . 286 ± 0 . | 0 . 880 ± 0 . 027 0 . 858 ± 0 . 030 0 . 846 ± 0 . 035 . 830 ± 0 . | - 1 . 435 ± 0 . 020 | 1 . 312 ± 0 . 053 |
| only)                                                                             |                                                                     |                                                                   | - 1 . 428 ± 0 . 022 | 1 . 202 ± 0 . 045 |
| only)                                                                             |                                                                     |                                                                   | - 1 . 423 ± 0 . 024 | 1 . 151 ± 0 . 041 |
| NILS)                                                                             | 010                                                                 | 0 042                                                             | - 1 . 418 ± 0 . 023 | 1 . 112 ± 0 . 039 |
|                                                                                   | 0 . 240 ± 0 . 022                                                   | 0 . 490 ± 0 . 025                                                 | - 1 . 145 ± 0 . 032 | 1 . 192 ± 0 . 043 |
| Gen-NP (with DSR only)                                                            | 0 . 261 ± 0 . 018                                                   | 0 . 479 ± 0 . 022                                                 | - 1 . 135 ± 0 . 030 | 1 . 179 ± 0 . 040 |
| Gen-NP (with NILS only)                                                           | 0 . 259 ± 0 . 014                                                   | 0 . 480 ± 0 . 017                                                 | - 1 . 133 ± 0 . 027 | 1 . 182 ± 0 . 038 |
| Full Gen-NP (DSR + NILS)                                                          | 0 . 270 ± 0 . 009                                                   | 0 . 470 ± 0 . 012                                                 | - 1 . 125 ± 0 . 024 | 1 . 165 ± 0 . 036 |

As shown in Table 9 and Appendix, these results highlight the significance of incorporating DSR for improving dynamical stability and NILS for robust optimization. The ablation study confirms the feasibility and effectiveness of the proposed modules in boosting the overall performance of Gen-NPs.

## C.7 Comparison with Stability Neural Processes

To provide a comprehensive analysis, we conducted additional experiments to compare stability-based generalization error (SGE) with our proposed Gen-NPs, emphasizing the differences in their noise introduction mechanisms and evaluation methods.

While SGE quantifies generalization as the difference between test error and training error, our approach focuses on gradient incoherence (GI) and directly modeling the parameter-data mutual information through controlled noise injection. To test robustness, we added Gaussian noise (mean = 0, variance = 1) to a random selection of 5% and 10% of the training data. The experiments were performed on a one-dimensional regression task with data generated using an RBF kernel.

Table 10 presents detailed results for CNP, ANP, and BNP models under original, 5% noise, and 10% noise conditions. Gen-NPs consistently achieves higher log-likelihood (LL) values and lower gradient incoherence (GI) compared to baseline methods across all noise levels. Notably, even as noise increases, Gen-NPs maintains better performance relative to standard NPs, demonstrating its enhanced robustness.

The key difference lies in how noise is utilized: Stability Neural Processes approaches add noise to the training data to measure stability, while Gen-NPs introduces noise strategically during the parameter update process. This fundamental difference enables Gen-NPs to better capture the mutual information between model parameters and training data, leading to improved generalization capacity under various conditions.

## D Computational Complexity Analysis

In this section, we analyze the computational complexity and overhead introduced by our GenNPs approach compared to standard NP methods. While noise injection and dynamical stability regularization enhance model performance, they also introduce additional computation. Here we quantify these costs and demonstrate that the performance benefits outweigh the computational overhead.

Theoretical Complexity Analysis Let d denote the model parameter dimension, n the number of context points, m the number of tasks, and b the batch size.

For the dynamical stability regularization (DSR) term computation, we avoid explicitly forming the full Hessian matrix, which would require O ( d 3 ) operations. Instead, we utilize efficient Hessianvector product approximations, reducing the complexity to O ( bd 2 ) . The noise injection component

Table 10: Comparison of log-likelihood (LL), gradient incoherence (GI), and stability-based generalization error (SGE) under different noise levels. Results are reported as mean ± standard deviation.

| Metric       | Method                              | LL                                                                                                                                          | GI                                                                                                                                          | SGE                                                                                                                                         |
|--------------|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| Original     | CNP Gen-CNP ANP Gen-ANP BNP Gen-BNP | 0.272 ± 0.013 0.283 ± 0.011 0.809 ± 0.004 0.810 ± 0.003 0.394 ± 0.015 0.402 ± 0.010 0.234 ± 0.016 0.271 ± 0.013 0.773 ± 0.007 0.795 ± 0.006 | 0.865 ± 0.025 0.834 ± 0.044 0.967 ± 0.043 0.954 ± 0.014 0.151 ± 0.006 0.150 ± 0.009 0.860 ± 0.020 0.828 ± 0.042 0.963 ± 0.040 0.950 ± 0.012 | 0.872 ± 0.030 0.850 ± 0.051 1.256 ± 0.046 1.147 ± 0.018 0.872 ± 0.009 0.783 ± 0.011 0.875 ± 0.028 0.845 ± 0.049 0.975 ± 0.045 0.960 ± 0.016 |
| Noise (+5%)  | CNP                                 |                                                                                                                                             |                                                                                                                                             |                                                                                                                                             |
| Noise (+5%)  | Gen-CNP                             |                                                                                                                                             |                                                                                                                                             |                                                                                                                                             |
| Noise (+5%)  | ANP                                 |                                                                                                                                             |                                                                                                                                             |                                                                                                                                             |
| Noise (+5%)  | Gen-ANP                             |                                                                                                                                             |                                                                                                                                             |                                                                                                                                             |
| Noise (+5%)  | BNP                                 | 0.362 ± 0.018                                                                                                                               | 0.149 ± 0.005                                                                                                                               | 0.157 ± 0.008                                                                                                                               |
| Noise (+5%)  | Gen-BNP                             | 0.388 ± 0.013                                                                                                                               | 0.136 ± 0.008                                                                                                                               | 0.155 ± 0.010                                                                                                                               |
| Noise (+10%) | CNP                                 | 0.220 ± 0.017                                                                                                                               | 0.855 ± 0.018                                                                                                                               | 0.870 ± 0.026                                                                                                                               |
| Noise (+10%) | Gen-CNP                             | 0.258 ± 0.016                                                                                                                               | 0.826 ± 0.041                                                                                                                               | 0.838 ± 0.047                                                                                                                               |
| Noise (+10%) | ANP                                 | 0.750 ± 0.010                                                                                                                               | 0.960 ± 0.038                                                                                                                               | 0.970 ± 0.043                                                                                                                               |
| Noise (+10%) | Gen-ANP                             | 0.765 ± 0.009                                                                                                                               | 0.945 ± 0.011                                                                                                                               | 0.955 ± 0.014                                                                                                                               |
| Noise (+10%) | BNP                                 | 0.340 ± 0.021                                                                                                                               | 0.146 ± 0.004                                                                                                                               | 0.155 ± 0.007                                                                                                                               |
| Noise (+10%) | Gen-BNP                             | 0.375 ± 0.015                                                                                                                               | 0.138 ± 0.007                                                                                                                               | 0.153 ± 0.009                                                                                                                               |

Table 11: Theoretical complexity comparison between standard NPs and Gen-NPs

| Operation       | Standard NP   | Gen-NP (Ours)   |
|-----------------|---------------|-----------------|
| Forward pass    | O ( bnd )     | O ( bnd )       |
| Backward pass   | O ( bd 2 )    | O ( bd 2 )      |
| DSR computation | -             | O ( bd 2 )      |
| Noise injection | -             | O ( d )         |

has minimal overhead of O ( d ) for sampling from a Gaussian distribution and adding the noise to the parameter updates.

Empirical Evaluation We measured the actual computational overhead across different tasks using the same hardware setup for all experiments. Table 12 summarizes these findings.

Table 12: Empirical computational overhead and performance gains

| Method                    | Training Time   | Memory Usage   | Avg. LL Improvement   |
|---------------------------|-----------------|----------------|-----------------------|
| Original CNP              | 1.00×           | 1.00×          | -                     |
| Gen-CNP (with DSR only)   | 1.12×           | 1.08×          | +4.1%                 |
| Gen-CNP (with NILS only)  | 1.05×           | 1.02×          | +5.3%                 |
| Full Gen-CNP (DSR + NILS) | 1.18×           | 1.10×          | +7.9%                 |

Task-Specific Training Time Analysis We further analyzed the training time across different tasks and architectures to provide a comprehensive view of the computational overhead.

Cost-Benefit Analysis While Gen-NPs introduce approximately 10% additional training time, the consistent performance improvements across all tasks and metrics easily justify this minimal overhead. The most computationally intensive component is the DSR term, specifically the computation of Hessian-related properties. However, this overhead is only present during training; inference time remains virtually identical to standard NP methods.

Table 13: Training time comparison across different tasks (in hours)

| Method   |   1D Regression |   Image Completion |   Bayesian Optimization |
|----------|-----------------|--------------------|-------------------------|
| CNP      |            0.33 |               2.45 |                    0.86 |
| Gen-CNP  |            0.36 |               2.68 |                    0.95 |
| NP       |            0.41 |               2.98 |                    1.04 |
| Gen-NP   |            0.45 |               3.28 |                    1.14 |
| ANP      |            0.58 |               4.12 |                    1.48 |
| Gen-ANP  |            0.64 |               4.53 |                    1.62 |

For tasks requiring high accuracy and reliable uncertainty quantification, such as Bayesian optimization and medical image completion, the 7-9% improvement in log-likelihood represents a significant practical advantage that substantially outweighs the modest 10% increase in training resources.

Furthermore, we found that in practice, Gen-NPs often require fewer training iterations to reach a target performance level compared to standard NPs, which can fully offset the per-iteration computational overhead in end-to-end training scenarios. This favorable performance-to-cost ratio makes Gen-NPs particularly attractive for practical applications where generalization and reliable uncertainty estimation are critical.

Implementation Considerations To minimize the computational overhead while maintaining performance benefits, we recommend:

1. Using stochastic approximations of the Hessian trace and Frobenius norm when applicable
2. Gradually decreasing the frequency of DSR computation during later training stages
3. Implementing the DSR term computation with efficient auto-differentiation libraries that optimize Hessian-vector products
4. For very large models, considering a reduced-precision implementation of the DSR component

These optimizations can further reduce the computational gap between standard NPs and Gen-NPs while preserving the generalization benefits of our approach.

## E Algorithm Pseudocode

We present the complete algorithm for Generalization Neural Processes (Gen-NPs) that integrates both the Risk-Aware Dynamical Stability Regularization (DSR) and Optimization-Aware Noise Injection Learning Strategy (NILS) components. Algorithm 1 provides a comprehensive pseudocode implementation that practitioners can follow to apply our method to various Neural Process variants.

## Algorithm 1 Generalization Neural Processes (Gen-NPs)

Require: Task environment τ , initial learning rate η 0 , inverse temperature γ , number of tasks per batch B , number of iterations S , DSR coefficients λ 1 ∈ [0 . 01 , 0 . 1] , λ 2 ∈ [0 . 001 , 0 . 01]

- 1: Randomly initialize θ 0
- 2: for s ← 1 to S do
- 3: Sample a batch of tasks {D i } B i =1 ∼ τ
- 4: Initialize task gradients and DSR term: G ( θ s -1 ) = 0 , R dyn = 0
- 5: for each task i ∈ [ B ] do
- 6: Randomly split task D i into context set D C i and target set D T i
- 7: Calculate task-specific empirical risk ˜ R D T i ( θ s -1 )
- 8: Calculate task-specific gradient g i ( θ s -1 , D C i , D T i )
- 9: Estimate Hessian trace Tr ( H i ) and Frobenius norm ∥ H i ∥ F using efficient approximations
- 10: Update DSR term: R dyn + = 1 B ( λ 1 · Tr ( H i ) + λ 2 · ∥ H i ∥ F )
- 11: end for
- 12: Aggregate gradients over all tasks: G ( θ s -1 ) = 1 B ∑ B i =1 g i ( θ s -1 , D C i , D T i )
- 13: Calculate gradient of DSR term: ∇ R dyn
- 14: Update learning rate to η s
- 15: Calculate Gaussian noise variance σ 2 s = η s γ
- 16: Sample Gaussian noise ξ s ∼ N (0 , σ 2 s I k )
- 17: Update parameter θ s = θ s -1 -η s ( G ( θ s -1 ) + ∇ R dyn ) + ξ s
- 18: end for

<!-- image -->

(e) Sample functions produced by TNP and Gen-TNP.

Figure 6: Sample functions produced by NPs and their corresponding Gen variants given 30 context points. Data is generated from a GP with an RBF kernel. Each solid blue curve corresponds to one sample function, and the blue area around each curve represents the variance in the predictive distribution. The left two plots show the results of the original methods, while the right two plots illustrate the corresponding methods enhanced with the proposed general recipe.

<!-- image -->

Figure 7: Experimental results across different temperature values. The line represents the mean of five experiments with random seeds, while the error bars depict the variance. The three subplots correspond to the log-likelihood results for CelebA and EMNIST datasets.

<!-- image -->

(a) Image completion produced by Gen-Method and the baseline Method on the CelebA dataset.

(b) Image completion produced by Gen-Method and the baseline Method on the EMNIST dataset (seen classes).

<!-- image -->

(c) Image completions produced by Gen-Method and the baseline Method on the EMNIST dataset (unseen classes).

<!-- image -->

Figure 8: Image completion produced by Gen-Method and the baseline Method methods given 100 context points.

<!-- image -->

Iterations

Iterations

Iterations

Figure 9: Regret performance on 1D Bayesian Optimization (BO) tasks.

Figure 10: Regret performance on 2D Bayesian Optimization (BO) tasks.

<!-- image -->

521

Figure 11: Regret performance on 3D Bayesian Optimization (BO) tasks.

<!-- image -->