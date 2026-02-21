## Diffusion Transformers for Imputation: Statistical Efficiency and Uncertainty Quantification

## Zeqi Ye

Northwestern University Evanston, IL, USA zeqiye2029@u.northwestern.edu

## Minshuo Chen

Northwestern University Evanston, IL, USA minshuo.chen@northwestern.edu

## Abstract

Imputation methods play a critical role in enhancing the quality of practical timeseries data, which often suffer from pervasive missing values. Recently, diffusionbased generative imputation methods have demonstrated remarkable success compared to autoregressive and conventional statistical approaches. Despite their empirical success, the theoretical understanding of how well diffusion-based models capture complex spatial and temporal dependencies between the missing values and observed ones remains limited. Our work addresses this gap by investigating the statistical efficiency of conditional diffusion transformers for imputation and quantifying the uncertainty in missing values. Specifically, we derive statistical sample complexity bounds based on a novel approximation theory for conditional score functions using transformers, and, through this, construct tight confidence regions for missing values. Our findings also reveal that the efficiency and accuracy of imputation are significantly influenced by the missing patterns. Furthermore, we validate these theoretical insights through simulation and propose a mixed-masking training strategy to enhance the imputation performance.

## 1 Introduction

Sequential data are ubiquitous in real-world applications such as finance [John et al., 2019, Chen et al., 2016], healthcare [Tonekaboni et al., 2021, Kazijevs and Samad, 2023], transportation [Li et al., 2020, Tedjopurnomo et al., 2020], and meteorology [Yozgatligil et al., 2013]. However, these datasets often suffer from missing values due to factors such as sensor malfunctions, data transmission errors, and human oversight [Greco et al., 2012, Yi et al., 2016]. Missing data can significantly degrade the performance of downstream tasks [Ribeiro and Castro, 2022, Alwateer et al., 2024], making accurate and robust imputation a critical challenge.

One of the earliest imputation methods dates back to Allan and Wishart [1930], which provided formulas for estimating single missing observations. Over the past century, this foundational idea of imputation has been extended to broader application domains. Statistical imputation methods have gained sustained attention due to their computational efficiency and ease of implementation. These approaches range from simple techniques, such as imputation using the mean or median of observations, to interpolation-based methods [Tukey, 1952], and more sophisticated modelbased techniques, including Kalman filters and autoregressive models [Gómez and Maravall, 1994, Shumway et al., 2000]. However, these methods often rely on strong assumptions such as linearity and stationarity, which may not hold in complex real-world scenarios, thereby limiting their applicability and accuracy [Fuller, 2009].

To address the limitations of statistical methods, recent research has increasingly turned to machine learning approaches for imputation. These methods are capable of capturing complex spatio-temporal patterns and nonlinear dependencies without requiring strict assumptions [Fang and Wang, 2020]. Typical examples include training neural networks such as recurrent neural networks and transformer

architectures for inferring missing values [Wang et al., 2024]. In parallel, generative models such as Variational AutoEncoders (VAEs) and Generative Adversarial Networks (GANs) have shown promise by introducing uncertainty-aware imputations [Fortuin et al., 2020, Miao et al., 2021]. However, these generative models often spell limitations in expressiveness or training stability. More recently, diffusion-based generative models have emerged as a powerful alternative, offering robust imputations and strong empirical performance across diverse and high-dimensional time series datasets [Tashiro et al., 2021, Zhou et al., 2024].

Despite their widespread empirical success, diffusion-based imputation methods exhibit two key challenges. First, their performance is highly sensitive to dataset characteristics, often displaying substantial variability across benchmarks [Zhang et al., 2024, Zheng and Charoenphakdee, 2022, Tashiro et al., 2021]. Second, they are significantly affected by missing patterns, leading to inconsistencies in imputation quality [Zhang et al., 2024, Ouyang et al., 2023, Zhou et al., 2024]. These observations motivate the following fundamental questions:

How well can diffusion models capture the underlying conditional distribution of missing values? How does the missing pattern affect the imputation performance?

In this paper, we answer the two questions from a statistical learning perspective. Our analysis centers on Diffusion Transformers (DiT, Peebles and Xie [2022]) applied to imputation tasks with Gaussian process (GP) data. Despite their conceptual simplicity, GPs exhibit rich spatio-temporal dependencies and long-horizon dependencies that pose challenges for modeling and imputation. On the other hand, GPs are powerful statistical tools widely used in regression, classification, and forecasting tasks [Seeger, 2004, Banerjee et al., 2013, Borovitskiy et al., 2021].

We establish sample complexity bounds for DiTs in learning the underlying conditional distribution of missing values given observed ones. The obtained bounds demonstrate the role of missing patterns in imputation performance, highlighting how the condition number of the covariance matrix for the missing values and distribution shifts contribute to variability in accuracy. Furthermore, we derive confidence intervals for imputed values and show the coverage probability of them converging to the desired level. We summarize our contributions as follows.

- Statistical Efficiency . We show that DiTs capture the conditional distribution of missing values effectively. The sample complexity in Theorem 2 scales at a rate ˜ O ( √ Hd 2 κ 5 / √ n ) , where n denotes the training sample size. We obtain a n -1 / 2 convergence rate with a mild polynomial dependence on the sequence length H . In addition, κ is the condition number induced by the missing patterns. To establish Theorem 2, we develop a novel score representation theory (Theorem 1) for DiTs, where we utilize an algorithm unrolling technique.
- Uncertainty Quantification . Leveraging the generative power of trained DiTs, we construct confidence regions (intervals) from massive generated missing values. This approach possesses its natural appeal and enjoys strong coverage guarantees (Corollary 1). We show that the coverage probability converges to the desired level at a ˜ O ( n -1 / 2 ) rate. Meanwhile, the missing patterns influences the convergence.
- Mixed-Masking Training Strategy . Motivated by our theoretical results, we propose a training strategy blending different masking schemes to cover diverse missing patterns. The performance of our method on synthetic datasets validates our findings and outperforms benchmark methods.

Notations We use bold lowercase letters to denote vectors and bold uppercase letters to denote matrices. For a vector v , ∥ v ∥ 2 denotes its Euclidean norm. For a matrix A , ∥ A ∥ 2 and ∥ A ∥ F denote its spectral norm and Frobenius norm, respectively, and ∥ A ∥ ∞ = max i,j | A ij | . When matrix A is positive definite, we denote λ max ( A ) and λ min ( A ) as its largest and smallest eigenvalues; its condition number is κ ( A ) = λ max ( A ) /λ min ( A ) . We denote f ≲ g if there exists a constant C &gt; 0 such that f ≤ Cg . Notation O ( · ) suppresses constants, while ˜ O ( · ) further hides logarithmic factors.

Due to space limit, the related work section is deferred to Appendix A.

## 2 Imputation in Gaussian Processes via Conditional Diffusion Models

In this section, we formalize the imputation task as a conditional distribution estimation problem. When the data are sampled from a Gaussian process, we identify rich structures in the conditional distribution. We then utilize a DiT to learn the distribution of missing values. Lastly, we summarize diffusion-based imputation method in Algorithm 1.

## 2.1 Imputation for Gaussian Process Data

Imputation refers to the task of inferring missing values given the observed ones. Denote by I = { 1 , . . . , H } the set of all time indices. For a multivariate sequence X = [ x 1 , . . . , x H ] ∈ R d × H of length H , we consider a block-missing setting, where certain time frames are entirely unobserved. The subset of observed indices is denoted by I obs = { i 1 , . . . , i |I obs | } , where |I obs | denotes the cardinality. Correspondingly, I miss = I \ I obs denotes the time indices of missing frames. To avoid degenerate cases, we assume 0 &lt; |I miss | &lt; H . In the sequel, we focus on the Missing Completely at Random case [Little, 1988], where each index in I miss is independently sampled from some underlying distribution.

We represent the vectorized observed partial sequence as x obs = [ x ⊤ i 1 , . . . , x ⊤ i |I obs | ] ⊤ ∈ R d |I obs | , and the vectorized missing part as x miss = [ x ⊤ j 1 , . . . , x ⊤ j |I miss | ] ⊤ ∈ R d |I miss | . We estimate the missing values by learning the conditional distribution P ( x miss | x obs ) . Notably, learning the conditional distribution goes beyond point estimates of the missing values, but provides easy access to confidence regions. We slightly abuse the notation by using x to simultaneously refer to random vectors.

Throughout our theoretical analysis, we focus on d -dimensional Gaussian process data. To uniquely distinguish a Gaussian process, it suffices to specify its mean and covariance functions. In particular, we denote the mean as µ i = E [ x i ] and we parameterize the covariance matrix by Cov[ x i , x j ] = γ ( i, j ) Λ , where Λ = Var[ x h ] ∈ R d × d for any h and γ is a kernel function. It is worth mentioning that Λ captures the spatial dependencies and function γ represents temporal correlation. The kernel function γ dictates the strength and decay of the temporal dependencies among different data frames. The joint distribution of a sequence vec( X ) = [ x ⊤ 1 , · · · , x ⊤ H ] ⊤ is Gaussian N ( µ , Γ ⊗ Λ ) , where

<!-- formula-not-decoded -->

Here Γ ij = γ ( i, j ) and ⊗ is the matrix Kronecker product. We impose the following assumption for characterizing the temporal dependencies.

̸

Assumption 1. There exists d e -dimensional embedding { e i ∈ R d e } H i =1 such that ∥ e i ∥ 2 = r for a constant r . Moreover, for any i, j , it holds that ∥ e i -e j ∥ 2 = f ( | i -j | ) , and for | i 1 -j 1 | = | i 2 -j 2 | , f ( | i 1 -j 1 | ) = f ( | i 2 -j 2 | ) . Kernel function γ ( i, j ) only depends on ∥ e i -e j ∥ 2 . Furthermore, we assume Γ and Λ are positive definite.

̸

̸

Assumption 1 ensures that the pairwise distances in the embedding uniquely identifies positional gaps. We do not specify a particular form of the kernel function, which encodes many commonly ones such as Gaussian Radial Basis Function (RBF), Ornstein-Uhlenbeck kernels, and Matérn kernels [Rasmussen and Williams, 2006]. As a concrete example, sinusoidal embedding is widely used in transformer networks [Vaswani et al., 2017]. Consider a two-dimensional embedding defined as e i = [ r sin(2 πi/C ) , r cos(2 πi/C )] ⊤ , where r &gt; 0 is a fixed radius and C &gt; 0 is a scaling constant. The Euclidean distance between any two embedding is ∥ e i -e j ∥ 2 = 2 r | sin ( π ( i -j ) /C ) | , which is strictly positive for i = j , and approximately linear in | i -j | when C is sufficiently large.

Under the Gaussian process setting, the conditional distribution of x miss | x obs is still Gaussian [Bishop and Nasrabadi, 2006]. The conditional mean and covariance are given by

µ cond ( x obs ) = µ miss + Σ ⊤ cor Σ -1 obs ( x obs -µ obs ) , Σ cond = Σ miss -Σ ⊤ cor Σ -1 obs Σ cor , where we denote µ obs = E [ x obs ] (the same holds for µ miss ), Σ cor = Cov[ x obs , x miss ] , and Σ obs (resp. Σ miss ) as the covariance of x obs (resp. x miss ). See Figure 1 for a graphical demonstration. We check that Σ obs = Γ obs ⊗ Λ with Γ obs ∈ R |I obs |×|I obs | capturing correlation among index set I obs .

## 2.2 Training Diffusion Transformers for Imputation

We estimate the conditional distribution P ( x miss | x obs ) using diffusion transformers. A diffusion model consists of two coupled processes-a forward and a backward process. We adopt a continuoustime description. In the forward process, we gradually corrupt data by

<!-- formula-not-decoded -->

and w t is a Wiener process. The forward process terminates at a sufficiently large time T and we denote the distribution of x t as P t ( ·| x obs ) with density p t ( ·| x obs ) . Note that we only corrupt the missing values by Gaussian noise, but keep the observed partial sequence x obs unchanged.

Corresponding to the forward process, the backward process simulates the reverse evolution of the forward process. As a result, it generates new samples by progressively removing noise:

<!-- formula-not-decoded -->

where ¯ w t is another Wiener process and ∇ v t log p t ( v t | x obs ) is the conditional score function. In the remaining of the paper, we drop the subscript v t in the score function for simplicity. Unfortunately, ∇ log p T -t ( v t | x obs ) is typically unknown and must be estimated using a neural network. We denote the estimated score function by ̂ s ( v t , x obs , t ) . Consequently, the sample generation process follows an alternative backward SDE:

<!-- formula-not-decoded -->

Here, we also replace the unknown P T by a standard Gaussian distribution.

When training the score estimator ̂ s , we assume access to fully observed sequences. To simulate a partially observed sequence, we sample a masking sequence { τ 1 , . . . , τ H } ∈ { 0 , 1 } H , where 0 denotes missing the observation and 1 keeping the observation. Then x obs is extracted according to the masking sequence. In later context, we will investigate how to choose masking strategies. We summarize the diffusion-based method for sequence imputation in Algorithm 1.

## Algorithm 1 Diffusion-Based Sequence Imputation

- 1: Module I: Training
- 2: Input: Fully observed sequences D := { X i } n i =1 , a masking strategy.
- 3: Simulate { x ( i ) obs , x ( i ) miss } n i =1 pairs via the masking strategy, and train a conditional diffusion model.
- 4: Output: A well-trained conditional diffusion model.
- 5: Module II: Imputation
- 6: Input: Conditional diffusion model from Module I , a new partial sequence x ∗ obs , repetition time Z , and confidence level 1 -α .
- 7: Conditioned on x ∗ obs , independently generate B missing sequences ̂ x ( z ) miss for z = 1 , . . . , Z .
- 9: ⋆ Confidence region : ̂ CR ∗ 1 -α = { x miss : ∥ x miss -̂ x ∗ miss ∥ 2 ≤ ̂ D ∗ 1 -α } , where ̂ D ∗ 1 -α is the 1 -α upper quantile of ∥ ̂ x ( z ) miss -̂ x ∗ miss ∥ 2 for z = 1 , . . . , Z .
- 8: ⋆ Point estimate : Mean ̂ x ∗ miss = 1 Z ∑ Z z =1 ̂ x ( z ) miss (or median of the generated sequences).
- 10: Return: ̂ x ∗ miss and ̂ CR ∗ 1 -α .

For the rest of the paper, we parameterize the conditional score function using a transformer network. A transformer [Vaswani et al., 2017], comprises a series of blocks and each block encompasses a multi-head attention layer and a feedforward layer. Let Y = [ y 1 , . . . , y H ] ∈ R D × H be the (column) stacking matrix of H patches. In a transformer block, the multi-head attention layer computes

<!-- formula-not-decoded -->

where V m , Q m , K m are weight matrices of corresponding sizes in the m -th attention head, and σ is an activation function. The attention layer is followed by a feedforward layer, which computes

<!-- formula-not-decoded -->

Here, W 1 , W 2 are weight matrices, b 1 and b 2 are offset vectors, 1 denotes a vector of ones, and the ReLU activation function is applied entry-wise. This feedforward layer performs a linear transformation to the output of the attention module with more flexibility. For our study, the raw input to a transformer is H patches of d -dimensional vectors and time t in the backward process. We refer to T ( D,L,M,B,R ) as a transformer architecture defined by

<!-- formula-not-decoded -->

Attn i uses entrywise ReLU activation for i = 1 , . . . , L, number of heads in each Attn is bounded by M, the Frobenius norm of each weight matrix is bounded by B, the output range ∥ f ∥ 2 is bounded by R } . (5)

## 3 Conditional Score Approximation via Algorithm Unrolling

Suggested by the sample generation process (3), the key is to learn the conditional score function. This section devotes to establishing a novel score approximation theory of transformers based on algorithm unrolling.

Since x miss | x obs is Gaussian, the forward process (1) yields the following closed-form score function:

<!-- formula-not-decoded -->

where α t = e -t 2 and σ t = √ 1 -e -t . The matrix inverse poses a challenge in representing the score by a transformer, as it may deteriorate structures in Σ cond . Therefore, we reformulate the conditional score function as the optimal solution of a quadratic optimization problem:

<!-- formula-not-decoded -->

It suffices to obtain an approximate optimal solution of (7) using a gradient descent algorithm. At the k -th iteration, with a step size η t , we have

<!-- formula-not-decoded -->

for k = 0 , . . . , K -1 . Unfortunately, we encounter another matrix inverse in Σ -1 obs Σ cor s ( k ) . Analogous to (7), we consider an auxiliary quadratic optimization problem:

<!-- formula-not-decoded -->

Via a gradient descent algorithm with step size θ , the update reads

<!-- formula-not-decoded -->

where iteration index k aux = 0 , . . . , K aux -1 .

We substitute the last iterate u ( K aux ) into the right-hand side of (8) to obtain ˜ ∇L t ( s ( k ) ) as an approximation to ∇L t ( s ( k ) ) . We summarize the nested gradient descent algorithm for calculating the conditional score function in Algorithm 2.

## Algorithm 2 Nested Gradient Descent for Representing Score Function

- 1: Input: Observation x obs , current state v t , time t , step sizes η t , θ , iteration counts K aux , K . (Major) Gradient Descent:
- 2: Initialize s (0) = 0 .
- 3: for k = 0 , 1 , . . . , K -1 do

Auxiliary Gradient Descent:

- 4: Initialize u (0) = 0 .
- 5: for k aux = 0 , 1 , . . . , K aux -1 do
- 6: u ( k aux +1) = u ( k aux ) -θ ∇L ( k ) aux ( u ( k aux ) ) .
- 7: Calculate ˜ ∇L t ( s ( k ) ) using u ( K aux ) .
- 8: s ( k +1) = s ( k ) -η t ˜ ∇L t ( s ( k ) ) .
- 9: Return: s ( K ) .

With sufficiently large K aux and K , the representation error of Algorithm 2 can be well-controlled. Lemma 1 (Representation error of Algorithm 2) . Suppose Assumption 1 holds. For an arbitrarily fixed time t ∈ (0 , T ] , given an error tolerance ϵ ∈ (0 , 1) , choose K,K aux as

<!-- formula-not-decoded -->

Then, given δ &gt; 0 , for any x obs and v t in a compact region C δ , there exist step sizes η t and θ such that running Algorithm 2 gives rise to

<!-- formula-not-decoded -->

Detailed proof of Lemma 1 is provided in Appendix B. The compact region C δ truncates the norm of x obs and v t , which is plausible due to the Gaussian tail; see a precise definition of C δ in Appendix Equation (13). Lemma 1 suggests that the computational complexity of Algorithm 2 for approximating the score function is governed by the condition numbers of Σ cond and Σ obs . A large condition number on Σ cond implies that the variability of missing values among different directions changes significantly. Equivalently, with a large condition number, given x obs , the missing values exhibit strong anistropic uncertainty that complicates the imputation.

Representing the conditional score function by a nested gradient descent algorithm enables an effective transformer network approximation. We show that transformers can realize each gradient descent iteration using a constant number of attention blocks. We provide the following score approximation theory using transformers.

Theorem 1. Suppose Assumption 1 holds. Given an early stopping time t 0 ∈ (0 , T ] and an error level ϵ ∈ (0 , 1) , for any x obs , v t ∈ C δ , there exists a transformer architecture T ( D,L,M,B,R ) such that, with proper weight parameters, it yields an approximation ˜ s satisfying

<!-- formula-not-decoded -->

The configuration of the transformer architecture satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof is provided in Appendix C. Figure 1 depicts the transformer architecture in our constructive proof, which unrolls Algorithm 2 efficiently. To obtain the approximation error bound, we develop a careful analysis of the error propagation in the auxiliary gradient descent for calculating ˜ ∇L t . Theorem 1 also reinforces the insights from Lemma 1, where we observe that the size of the transformer network scales with the worst-case condition number. We will further discuss the relation between missing patterns and the condition number in Theorem 2.

Figure 1: Constructed transformer architecture: Within each transformer block, attention heads focus on capturing information of different covariance components ( Σ obs , Σ cor , Σ miss ) separately, and approximate corresponding matrix-vector multiplications. A total of K block groups perform major GD steps, with K aux inner blocks in each group dedicated to solving the auxiliary problem.

<!-- image -->

## 4 Capturing Conditional Distribution and Uncertainty Quantification

Given a properly chosen transformer architecture, we establish guarantees for learning the conditional distribution of missing values and uncertainty quantification. We consider an estimated score network ̂ s obtained by minimizing the following empirical score matching loss (a detailed derivation is deferred to Appendix E):

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Substituting the learned score ̂ s into the backward SDE (3) yields generated distribution v t 0 ∼ ̂ P t 0 ( · | x obs ) . We introduce an early-stopping time t 0 to stabilize the training and sample generation [Song et al., 2020]. We now present a convergence guarantee of ̂ P t 0 to the true conditional distribution.

Theorem 2. Referring to the training procedure in Algorithm 1, by choosing the transformer architecture as in Theorem 1 with ϵ = n -1 2 , terminal time T = O (log n ) , and early-stopping time t 0 = O ( λ min ( Σ cond ) n -1 2 ) , it holds that

<!-- formula-not-decoded -->

The proof of Theorem 2 is provided in Appendix D. This result establishes that DiT can efficiently learn the true conditional distribution of missing values. The sample complexity mildly depends on the sequence length. More importantly, the bound highlights that the estimation error depends on the condition numbers of Σ cond and Σ obs , reflecting the discussion after Lemma 1.

We provide an example to demonstrate that different missing patterns can lead to distinct condition numbers. Consider data of length H = 96 with time correlation modeled by a Laplace kernel γ ( i, j ) = exp( -∥ e i -e j ∥ 2 / 128) , and missing length |I miss | = 16 . Clustered missingness-16 consecutive missing entries at the tail-yields a large condition number κ ( Σ cond ) = 415 . 40 , making the task challenging. In contrast, dispersed missing patterns, 16 randomly placed missing entries, result in much smaller κ ( Σ cond ) = 3 . 00 , making estimation easier. We provide numerical results on this example in Section 5.

Confidence Region Construction Given the learned conditional distribution ̂ P t 0 and a new observed sequence x ∗ obs , we deploy the model to generate samples and form point estimates and confidence regions as in Algorithm 1. Since x ∗ obs may not be seen in the training samples, we encounter a distribution shift, meaning that we need to transfer the knowledge in the learned model to the new testing instance. The subtlety here is how to quantify the knowledge transfer rate. Our proposal is the following class-dependent distribution shift coefficient.

Definition 1. The distribution shift between two probability distributions P 1 and P 2 with respect to a function class G is defined as DS ( P 1 , P 2 ; G ) = sup g ∈G E y ∼ P 1 [ g ( y )] E y ∼ P 2 [ g ( y )] .

In our analysis, we specialize G to a function class induced by the transformer network:

<!-- formula-not-decoded -->

Since G might be insensitive to certain distinctions, it introduces some smoothing effect to capture the difference between P 1 and P 2 . We consider P 1 and P 2 as the marginal training distribution of x obs and the point mass of the testing distribution ✶ {· = x ∗ obs } , denoted as P x obs and P x ∗ obs , respectively. The following corollary provides a guarantee for the coverage probability of the constructed CR. Corollary 1. Under the setting of Theorem 2, given x ∗ obs , Algorithm 1 yields ̂ CR ∗ 1 -α satisfying

<!-- formula-not-decoded -->

where ψ ( x ∗ obs ) is independent of n and proportional to ∥ x ∗ obs ∥ 2 and κ ( Σ cond ) .

Detailed proof is provided in Appendix D. Corollary 1 says that the coverage probability of the constructed CR converges to the desired level at the same rate of the conditional distribution estimation. More importantly, the distribution shift coefficient directly influences the coverage probability. We present a detailed discussion in the following remark.

Remark 1. There are two factors controlling the distribution shift coefficient: 1) the observed values in x ∗ obs and 2) the missing pattern. From our theoretical analysis, we identify a profound impact of the missingness patterns on the learning efficiency and the choice of transformer architectures.

Indeed, when the masking strategy in Algorithm 1 is relatively easy, ϵ ( n ) dist is small. However, x ∗ obs can deviate significantly from the training samples, causing a large distribution shift. On the contrary,

including harder masks can effectively reduce the distribution shift, but elevates learning difficulty. As a result, there is a trade-off between the masking strategy and the reliability of the trained diffusion transformer for imputation. In Section 5, we introduce a mixed-masking training strategy to enhance the performance of diffusion transformers, where diverse masking patterns are randomly sampled. This reduces distribution shift and improves robustness to varying imputation difficulty.

## 5 Experiments

We evaluate the performance of DiT through simulation to validate our theoretical results on imputation efficiency, uncertainty quantification, and the effectiveness of the mixed-masking training strategy. Experiments are conducted on Gaussian processes and, additionally, on more complex latent Gaussian processes to assess generalization beyond our theoretical scope. The DiT implementation builds on the DiT codebase [Peebles and Xie, 2022]. Further experimental details and real-world dataset experiments are provided in Appendix F. Our code is available at https://github.com/liamyzq/DiT\_time\_series\_imputation .

## 5.1 Gaussian Processes

We generate Gaussian process data with sequence length H = 96 , dimension d = 8 , and define the missing segment length as |I miss | = 16 . In addition to applying Algorithm 1 to construct 95% confidence regions (CRs), we sample from the true conditional distribution to evaluate CR coverage-the proportion of true values that fall within the estimated CR for comparison.

Figure 2: Visualization of the four missing patterns for a sequence of length 96. Each horizontal line shows the positions of missing values (highlighted in blue, orange, green and red for Patterns 1-4), and annotations on the right indicate the pattern number and its condition number κ ( Σ cond ) .

<!-- image -->

We first vary two factors: training sample size n ∈ { 10 3 , 10 3 . 5 , 10 4 , 10 4 . 5 , 10 5 } , and missing patterns 1-4 (denoted as P1-P4) as shown in Figure 2. As discussed in Theorem 2, κ ( Σ cond ) acts as a key varying parameter. To mitigate distribution shift, we apply the same missing patterns to both training and test data. Results in Figure 3 show that small training sets ( n = 10 3 , 10 3 . 5 ) result in low variability and poor distribution estimation. As sample size increases, DiT yields CRs that significantly better match the true distribution. We further vary sequence length ( H ) and report the results in Table 1. The results suggest that CR coverage rate decreases as sequence length increases, which supports our theoretical findings. Regarding missing patterns, those with lower condition numbers reduce the sample complexity needed for effective estimation. These findings are consistent with our theory, suggesting that the conditional covariance condition number serves as a practical measure of estimation difficulty. Patterns with lower condition numbers retain richer temporal correlations, enabling accurate estimation with fewer samples.

Figure 3: Percentage of real data samples that fall within the DiT-generated 95% CR.

<!-- image -->

Table 1: Sequence length vs CR coverage rates (%)( ↑ ).

| H   | 16            | 32            | 64            | 96            | 128           |
|-----|---------------|---------------|---------------|---------------|---------------|
| CR  | 92.67 (±1.95) | 88.63 (±2.01) | 82.14 (±1.70) | 80.25 (±1.64) | 77.81 (±1.87) |

Table 2: CR coverage rates (%) ( ↑ ) of models trained using different strategies on different missing patterns.

|           | P1            | P2            | P3            | P4            |
|-----------|---------------|---------------|---------------|---------------|
| S1        | 34.58 (±1.22) | 58.46 (±1.89) | 72.42 (±1.66) | 80.25 (±1.64) |
| S2        | 66.22 (±3.86) | 83.71 (±2.86) | 74.04 (±1.90) | 81.50 (±2.12) |
| S3        | 56.04 (±6.48) | 81.05 (±2.09) | 74.59 (±1.27) | 83.09 (±1.48) |
| S4        | 57.27 (±5.34) | 79.00 (±2.42) | 74.38 (±3.00) | 82.74 (±2.40) |
| Only 8×2  | 36.74 (±1.31) | 60.51 (±1.65) | 71.24 (±1.52) | 80.46 (±2.01) |
| Only 4×4  | 34.15 (±1.16) | 59.23 (±1.88) | 73.08 (±1.10) | 79.83 (±1.84) |
| Only 1×16 | 32.68 (±1.50) | 54.23 (±1.76) | 69.46 (±1.53) | 76.72 (±2.20) |

Mixed-Masking training strategy. Based on our insights from our distribution shift analysis, we introduce mixed-masking training strategy. Remark 1 highlights that discrepancies between training and test distributions can impair CR estimation, especially in real-world settings with limited training data. A common practice is to train on fully random masks, which tend to have lower condition numbers and thus pose easier estimation tasks. However, this intensifies the mismatch with test cases featuring more challenging, clustered missing patterns, limiting model adaptability. To address this, we propose mixed-masking training strategy. Using the same n = 10 5 training samples, we evaluate the four test patterns in Figure 2. During training, we define four different mixed-masking strategies (each with 16 missing entries):

- S1 : 100% random missing pattern (16×1, sixteen randomly placed missing entries).
- S2 : 50% random (16×1) + 50% weakly grouped (8×2, eight randomly placed blocks of two consecutive missing entries).
- S3 : 33.3% random (16×1) + 33.3% weakly grouped (8×2) + 33.3% moderately grouped (4×4, four randomly placed blocks of four consecutive missing entries).
- S4 : 25% random (16×1) + 25% weakly grouped (8×2) + 25% moderately grouped (4×4) + 25% strongly grouped (1×16, one randomly placed block of sixteen consecutive missing entries).

Results in Table 2 show that models trained with mixed masking consistently outperform the baseline trained with completely random placed masks (S1). We also evaluate the strategies only containing individual patterns (8×2, 4×4, and 1×16 separately), and the results suggest that they yield inferior imputation performance compared to appropriately mixing different patterns. This supports our proposed mixed-masking strategies and aligns well with our theoretical insights. Yet determining optimal mixing ratios is instance based and remains an open question for future work.

Regarding how these strategies relate to our theoretical results, intuitively, different missing patterns during training lead to different training distributions P x obs , resulting in varying condition numbers and consequently different DS values. Training with diverse missing patterns-ranging from easy to hard-helps the model adapt to imputation tasks with varying levels of difficulty by effectively covering more scenarios. As for a more concrete example, let us denote the training distributions corresponding to S1 and S4 as P (1) x obs and P (4) x obs , respectively. Consider a test sample x ∗ obs following the strongly grouped missing pattern P1 (consecutive missing entries). Intuitively, the resulting distribution P x ∗ obs is closer to P (4) x obs than to P (1) x obs , which implies the distribution shift coefficient of P (4) x obs is smaller than the one of P (1) x obs . Empirically, we calculate the average ratio across all test samples with missing pattern P1 and find that:

<!-- formula-not-decoded -->

This clearly indicates that the mixed-masking training strategy (S4) yields significantly smaller distribution-shift coefficients compared to purely random missingness (S1). According to Corollary 1, this provides strong theoretical support for the superior empirical performance achieved by our mixed-masking strategy.

## 5.2 Latent Gaussian Processes

We conduct additional experiments to assess whether our findings generalize beyond the theoretical setting-specifically, whether different missing patterns affect imputation and uncertainty quantification performance, and whether the mixed-masking training strategy improves them. For X drawn from the Gaussian process in Section 5.1, we consider a corresponding latent Gaussian process: Y = ϕ ( X ) + ϵ with vec ( ϵ ) ∼ N ( 0 , 0 . 1 · I dH ) , where the non-linear transform ϕ ( x ) = x +exp( -x 2 ) + 2 sin( x ) is applied entry-wise. We adopt a training sample size of n = 10 5 . This introduces nonlinearity and noise, increasing the difficulty of distribution estimation.

We evaluate DiT on this transformed dataset using the same four missing patterns and four training strategies from Section 5.1. For comparison, we implement two representative generative imputation models-CSDI [Tashiro et al., 2021] and GPVAE [Fortuin et al., 2020], ensuring all models have comparable numbers of trainable parameters. We report Mean Squared Error (MSE) against the true conditional mean and CR coverage rates, following the setup in Section 5.1. Results are shown in

Table 3: MSE ( ↓ ) on latent Gaussian process data.

|                | DiT                                                 | CSDI                                                | GPVAE                                               |
|----------------|-----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|
| P1 S1 S2 S3 S4 | 0.70 (±0.03) 0.68 (±0.02) 0.67 (±0.03) 0.67 (±0.02) | 0.75 (±0.03) 0.69 (±0.02) 0.70 (±0.03) 0.68 (±0.02) | 5.24 (±0.75) 5.45 (±1.05) 5.13 (±0.49) 5.28 (±0.68) |
| P2 S1 S2 S3 S4 | 0.64 (±0.03) 0.62 (±0.02) 0.60 (±0.03) 0.62 (±0.03) | 0.66 (±0.03) 0.63 (±0.03) 0.62 (±0.02) 0.63 (±0.03) | 5.09 (±0.70) 5.01 (±0.62) 4.94 (±0.56) 4.84 (±0.60) |
| P3 S1 S2       | 0.62 (±0.02) 0.60 (±0.03) 0.58 (±0.02)              | 0.65 (±0.02) 0.64 (±0.03) 0.63 (±0.03)              | 4.63 (±0.58) 5.12 (±1.00) 4.50 (±0.52)              |
| S3 S4          | 0.58 (±0.03)                                        | 0.61 (±0.02)                                        | 4.59 (±0.54)                                        |
| P4 S1 S2 S3    | 0.56 (±0.01) 0.53 (±0.03) 0.53 (±0.01)              | 0.59 (±0.03) 0.60 (±0.02) 0.58 (±0.03)              | 4.89 (±0.69) 4.79 (±0.61) 4.39                      |
| S4             | 0.53 (±0.02)                                        | 0.58 (±0.02)                                        | (±0.49) 4.45 (±0.54)                                |

Table 4: CR coverage rates (%) ( ↑ ).

DiT

S1

S2

S3

S4

S1

S2

S3

S4

S1

S2

S3

S4

S1

S2

S3

S4

36.46 (±1.62)

53.68 (±3.26)

54.26 (±2.79)

56.43 (±3.76)

55.81 (±1.55)

65.77 (±2.87)

66.24 (±3.22)

63.95 (±4.38)

63.53 (±1.72)

71.29 (±2.99)

70.89 (±2.45)

73.36 (±4.37)

76.46 (±1.33)

78.63 (±2.62)

78.79 (±2.67)

80.64 (±3.72)

CSDI

54.75 (±1.89)

56.68 (±2.75)

58.64 (±3.11)

55.67 (±4.03)

63.67 (±1.77)

64.89 (±3.43)

63.13 (±2.95)

65.97 (±3.59)

61.35 (±1.49)

65.69 (±2.79)

63.48 (±2.90)

67.17 (±3.93)

68.60 (±1.74)

70.48 (±2.34)

73.46 (±2.53)

72.89 (±3.78)

Tables 3 and 4. Since GPVAE performs poorly in point estimation, we omit its CR coverage. DiT consistently outperforms in both MSE and CR coverage, indicating transformers may better suit this task than CSDI's convolutional design. Moreover, mixed-masking training improves performance not only for DiT but also for other models, demonstrating its broader benefit. These findings reinforce our conclusions from Gaussian process experiments and support the generalization of our theory and training methodology to more complex, nonlinear settings.

## 6 Conclusion and Discussion

Our work addresses a critical gap in the theoretical understanding of diffusion-based time series imputation and uncertainty quantification by investigating the statistical efficiency of diffusion transformers on Gaussian process data. This result enables efficient and accurate imputation and confidence region construction. Motivated by the theory, we propose a mixed-masking training strategy that introduces diverse missing patterns during training, rather than relying solely on completely random masks. Our experiments validate the theoretical findings and further demonstrate that the proposed strategy performs well and generalizes to more complex data beyond our analytical scope.

Looking ahead, investigating the behavior of diffusion transformers on heavy-tailed time series (e.g., financial data) would further clarify their limitations and guide practical design choices. Moreover, a more detailed analysis of optimal mixed-masking training strategies-especially those leveraging prior knowledge-could significantly improve the performance of imputation models.

P1

P2

P3

P4

## References

- Juan Miguel Lopez Alcaraz and Nils Strodthoff. Diffusion-based time series imputation and forecasting with structured state space models. arXiv preprint arXiv:2208.09399 , 2022.
- FE Allan and John Wishart. A method of estimating the yield of a missing plot in field experimental work. The Journal of Agricultural Science , 20(3):399-406, 1930.
- Majed Alwateer, El-Sayed Atlam, Mahmoud Mohammed Abd El-Raouf, Osama A Ghoneim, and Ibrahim Gad. Missing data imputation: A comprehensive review. Journal of Computer and Communications , 12(11):53-75, 2024.
- Theodore W Anderson. The statistical analysis of time series . John Wiley &amp; Sons, 2011.
- Anjishnu Banerjee, David B Dunson, and Surya T Tokdar. Efficient gaussian process regression for large datasets. Biometrika , 100(1):75-89, 2013.
- Parikshit Bansal, Prathamesh Deshpande, and Sunita Sarawagi. Missing value imputation on multidimensional time series. arXiv preprint arXiv:2103.01600 , 2021.
- Joe Benton, Valentin De Bortoli, Arnaud Doucet, and George Deligiannidis. Nearly d -linear convergence bounds for diffusion models via stochastic localization. arXiv preprint arXiv:2308.03686 , 2023.
- Christopher M Bishop and Nasser M Nasrabadi. Pattern recognition and machine learning , volume 4. Springer, 2006.
- Viacheslav Borovitskiy, Iskander Azangulov, Alexander Terenin, Peter Mostowsky, Marc Deisenroth, and Nicolas Durrande. Matérn gaussian processes on graphs. In International Conference on Artificial Intelligence and Statistics , pages 2593-2601. PMLR, 2021.
- Sébastien Bubeck et al. Convex optimization: Algorithms and complexity. Foundations and Trends® in Machine Learning , 8(3-4):231-357, 2015.
- Clément L Canonne. A short note on an inequality between kl and tv. arXiv preprint arXiv:2202.07198 , 2022.
- Defu Cao, Wen Ye, Yizhou Zhang, and Yan Liu. Timedit: General-purpose diffusion transformers for time series foundation model. arXiv preprint arXiv:2409.02322 , 2024.
- Wei Cao, Dong Wang, Jian Li, Hao Zhou, Lei Li, and Yitan Li. Brits: Bidirectional recurrent imputation for time series. Advances in neural information processing systems , 31, 2018.
- Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu. Recurrent neural networks for multivariate time series with missing values. Scientific reports , 8(1):6085, 2018.
- Jou-Fan Chen, Wei-Lun Chen, Chun-Ping Huang, Szu-Hao Huang, and An-Pin Chen. Financial time-series data analysis using deep convolutional neural networks. In 2016 7th International conference on cloud computing and big data (CCBD) , pages 87-92. IEEE, 2016.
- Minshuo Chen, Kaixuan Huang, Tuo Zhao, and Mengdi Wang. Score approximation, estimation and distribution recovery of diffusion models on low-dimensional data. In International Conference on Machine Learning , pages 4672-4712. PMLR, 2023.
- Minshuo Chen, Song Mei, Jianqing Fan, and Mengdi Wang. Opportunities and challenges of diffusion models for generative ai. National Science Review , 11(12):nwae348, 2024.
- Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, and Anru R Zhang. Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions. arXiv preprint arXiv:2209.11215 , 2022.
- Andrea Cini, Ivan Marisca, and Cesare Alippi. Filling the g\_ap\_s: Multivariate time series imputation by graph neural networks. arXiv preprint arXiv:2108.00298 , 2021.

- David R Cox, Gudmundur Gudmundsson, Georg Lindgren, Lennart Bondesson, Erik Harsaae, Petter Laake, Katarina Juselius, and Steffen L Lauritzen. Statistical analysis of time series: Some recent developments [with discussion and reply]. Scandinavian Journal of Statistics , pages 93-115, 1981.
- Wenjie Du. PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series. arXiv preprint arXiv:2305.18811 , 2023.
- Wenjie Du, David Côté, and Yan Liu. Saits: Self-attention-based imputation for time series. Expert Systems with Applications , 219:119619, 2023.
- Wenjie Du, Jun Wang, Linglong Qian, Yiyuan Yang, Zina Ibrahim, Fanxing Liu, Zepu Wang, Haoxin Liu, Zhiyuan Zhao, Yingjie Zhou, et al. Tsi-bench: Benchmarking time series imputation. arXiv preprint arXiv:2406.12747 , 2024.
- Chenguang Fang and Chen Wang. Time series data imputation: A survey on deep learning approaches. arXiv preprint arXiv:2011.11347 , 2020.
- Vincent Fortuin, Dmitry Baranchuk, Gunnar Rätsch, and Stephan Mandt. Gp-vae: Deep probabilistic time series imputation. In International conference on artificial intelligence and statistics , pages 1651-1661. PMLR, 2020.
- Chun Fu, Matias Quintana, Zoltan Nagy, and Clayton Miller. Filling time-series gaps using image techniques: Multidimensional context autoencoder approach for building energy data imputation. Applied Thermal Engineering , 236:121545, 2024a.
- Hengyu Fu, Zehao Dou, Jiawei Guo, Mengdi Wang, and Minshuo Chen. Diffusion transformer captures spatial-temporal dependencies: A theory for gaussian process data. arXiv preprint arXiv:2407.16134 , 2024b.
- Hengyu Fu, Zhuoran Yang, Mengdi Wang, and Minshuo Chen. Unveil conditional diffusion models with classifier-free guidance: A sharp statistical theory. arXiv preprint arXiv:2403.11968 , 2024c.
- Wayne A Fuller. Introduction to statistical time series . John Wiley &amp; Sons, 2009.
- Víctor Gómez and Agustín Maravall. Estimation, prediction, and interpolation for nonstationary series with the kalman filter. Journal of the American Statistical Association , 89(426):611-624, 1994.
- Sergio Greco, Cristian Molinaro, and Francesca Spezzano. Incomplete data and data dependencies in relational databases , volume 29. Morgan &amp; Claypool Publishers, 2012.
- José M Jerez, Ignacio Molina, Pedro J García-Laencina, Emilio Alba, Nuria Ribelles, Miguel Martín, and Leonardo Franco. Missing data imputation using statistical and machine learning methods in a real breast cancer problem. Artificial intelligence in medicine , 50(2):105-115, 2010.
- Chisimkwuo John, Emmanuel J Ekpenyong, and Charles C Nworu. Imputation of missing values in economic and financial time series data using five principal component analysis approaches. CBN Journal of Applied Statistics (JAS) , 10(1):3, 2019.
- Maksims Kazijevs and Manar D Samad. Deep imputation of missing values in time series health data: A review with benchmarking. Journal of biomedical informatics , page 104440, 2023.
- SeungHyun Kim, Hyunsu Kim, Eunggu Yun, Hwangrae Lee, Jaehun Lee, and Juho Lee. Probabilistic imputation for time-series classification with missing data. In International Conference on Machine Learning , pages 16654-16667. PMLR, 2023.
- Gen Li, Yu Huang, Timofey Efimov, Yuting Wei, Yuejie Chi, and Yuxin Chen. Accelerating convergence of score-based diffusion models, provably. arXiv preprint arXiv:2403.03852 , 2024.
- Huiping Li, Meng Li, Xi Lin, Fang He, and Yinhai Wang. A spatiotemporal approach for traffic data imputation with complicated missing patterns. Transportation research part C: emerging technologies , 119:102730, 2020.
- Roderick JA Little. A test of missing completely at random for multivariate data with missing values. Journal of the American statistical Association , 83(404):1198-1202, 1988.

- Mingzhe Liu, Han Huang, Hao Feng, Leilei Sun, Bowen Du, and Yanjie Fu. Pristi: A conditional diffusion framework for spatiotemporal imputation. In 2023 IEEE 39th International Conference on Data Engineering (ICDE) , pages 1927-1939. IEEE, 2023.
- Yonghong Luo, Xiangrui Cai, Ying Zhang, Jun Xu, et al. Multivariate time series imputation with generative adversarial networks. Advances in neural information processing systems , 31, 2018.
- Pierre-Alexandre Mattei and Jes Frellsen. Miwae: Deep generative modelling and imputation of incomplete data sets. In International conference on machine learning , pages 4413-4423. PMLR, 2019.
- Song Mei and Yuchen Wu. Deep networks as denoising algorithms: Sample-efficient learning of diffusion models in high-dimensional graphical models. IEEE Transactions on Information Theory , 2025.
- Xiaoye Miao, Yangyang Wu, Jun Wang, Yunjun Gao, Xudong Mao, and Jianwei Yin. Generative semi-supervised learning for multivariate time series imputation. In Proceedings of the AAAI conference on artificial intelligence , volume 35, pages 8983-8991, 2021.
- Ahmad Wisnu Mulyadi, Eunji Jun, and Heung-Il Suk. Uncertainty-aware variational-recurrent imputation network for clinical time series. IEEE Transactions on Cybernetics , 52(9):9684-9694, 2021.
- Kazusato Oko, Shunta Akiyama, and Taiji Suzuki. Diffusion models are minimax optimal distribution estimators. In International Conference on Machine Learning , pages 26517-26582. PMLR, 2023.
- Yidong Ouyang, Liyan Xie, Chongxuan Li, and Guang Cheng. Missdiff: Training diffusion models on tabular data with missing values. arXiv preprint arXiv:2307.00467 , 2023.
- Leandro Pardo. Statistical inference based on divergence measures . Chapman and Hall/CRC, 2018.
- William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748 , 2022.
- Ignacio Peis, Chao Ma, and José Miguel Hernández-Lobato. Missing data imputation and acquisition with deep hierarchical models and hamiltonian monte carlo. Advances in Neural Information Processing Systems , 35:35839-35851, 2022.
- Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian Processes for Machine Learning . MIT Press, 2006.
- S Mara Ribeiro and CL Castro. Missing data in time series: A review of imputation methods and case study. Learning and Nonlinear Models , 20(1):31-46, 2022.
- Matthias Seeger. Gaussian processes for machine learning. International journal of neural systems , 14(02):69-106, 2004.
- Robert H Shumway, David S Stoffer, and David S Stoffer. Time series analysis and its applications , volume 3. Springer, 2000.
- Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 , 2020.
- Wenpin Tang and Hanyang Zhao. Score-based diffusion models via stochastic differential equations-a technical tutorial. arXiv preprint arXiv:2402.07487 , 2024.
- Yusuke Tashiro, Jiaming Song, Yang Song, and Stefano Ermon. Csdi: Conditional score-based diffusion models for probabilistic time series imputation. Advances in Neural Information Processing Systems , 34:24804-24816, 2021.
- David Alexander Tedjopurnomo, Zhifeng Bao, Baihua Zheng, Farhana Murtaza Choudhury, and Alex Kai Qin. A survey on modern deep neural network for traffic prediction: Trends, methods and challenges. IEEE Transactions on Knowledge and Data Engineering , 34(4):1544-1561, 2020.

- Sana Tonekaboni, Danny Eytan, and Anna Goldenberg. Unsupervised representation learning for time series with temporal neighborhood coding. arXiv preprint arXiv:2106.00750 , 2021.
- John W Tukey. The extrapolation, interpolation and smoothing of stationary time series with engineering applications, 1952.
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems , 30, 2017.
- Peerapon Vateekul and Kanoksri Sarinnapakorn. Tree-based approach to missing data imputation. In 2009 IEEE International Conference on Data Mining Workshops , pages 70-75. IEEE, 2009.
- Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation , 23(7):1661-1674, 2011.
- Jun Wang, Wenjie Du, Wei Cao, Keli Zhang, Wenjia Wang, Yuxuan Liang, and Qingsong Wen. Deep learning for multivariate time series imputation: A survey. arXiv preprint arXiv:2402.04059 , 2024.
- Xu Wang, Hongbo Zhang, Pengkun Wang, Yudong Zhang, Binwu Wang, Zhengyang Zhou, and Yang Wang. An observed value consistent diffusion model for imputing missing values in multivariate time series. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining , pages 2409-2418, 2023.
- Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. Timesnet: Temporal 2d-variation modeling for general time series analysis. arXiv preprint arXiv:2210.02186 , 2022.
- Shin-Fu Wu, Chia-Yung Chang, and Shie-Jue Lee. Time series forecasting with missing values. In 2015 1st International Conference on Industrial Networks and Intelligent Systems (INISCom) , pages 151-156. IEEE, 2015.
- Jun-He Yang, Ching-Hsue Cheng, and Chia-Pan Chan. A time-series water level forecasting model based on imputation and variable selection method. Computational intelligence and neuroscience , 2017(1):8734214, 2017.
- Xiuwen Yi, Yu Zheng, Junbo Zhang, and Tianrui Li. St-mvl: Filling missing values in geo-sensory time series data. In Proceedings of the 25th international joint conference on artificial intelligence , 2016.
- Jinsung Yoon, James Jordon, and Mihaela Schaar. Gain: Missing data imputation using generative adversarial nets. In International conference on machine learning , pages 5689-5698. PMLR, 2018a.
- Jinsung Yoon, William R Zame, and Mihaela van der Schaar. Estimating missing data in temporal data streams using multi-directional recurrent neural networks. IEEE Transactions on Biomedical Engineering , 66(5):1477-1490, 2018b.
- Ceylan Yozgatligil, Sipan Aslan, Cem Iyigun, and Inci Batmaz. Comparison of missing value imputation methods in time series: the case of turkish meteorological data. Theoretical and applied climatology , 112:143-167, 2013.
- Hengrui Zhang, Liancheng Fang, and Philip S Yu. Unleashing the potential of diffusion models for incomplete data imputation. arXiv preprint arXiv:2405.20690 , 2024.
- Shuyi Zhang, Bin Guo, Anlan Dong, Jing He, Ziping Xu, and Song Xi Chen. Cautionary tales on air-quality improvement in beijing. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences , 473(2205):20170457, 2017.
- Shuhan Zheng and Nontawat Charoenphakdee. Diffusion models for missing value imputation in tabular data. arXiv preprint arXiv:2210.17128 , 2022.
- Jianping Zhou, Junhao Li, Guanjie Zheng, Xinbing Wang, and Chenghu Zhou. Mtsci: A conditional diffusion model for multivariate time series consistent imputation. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management , pages 3474-3483, 2024.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Section 3 and 4 provide theoretical results, while Section 5 provides numerical results.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Section 6 discuss possible limitations.

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

Justification: Proper assumptions are provided in Section 2, and proofs are provided in the Appendix.

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

Justification: Experiment settings are provided in Section 5 and Appendix F.

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

Justification: Code is provided in the supplementary materials.

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

Justification: Simulation details are provided in Section 5 and Appendix F.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Standard deviations of the experimental results are reported.

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

Justification: Details are provided in Appendix F.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our theoretical research is not tied to particular negative societal impacts.

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

Justification: Our work poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The creators and original owners of assets used in the paper are properly credited.

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

Justification: Our work does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our research does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our research does not involve crowdsourcing nor research with human subjects. Guidelines:

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

## A Related Work

In the early stages of time series imputation, statisticians developed a wide range of traditional statistical methods aimed at both imputation (point estimation) and quantifying uncertainty, often by leveraging well-established statistical tools to construct confidence intervals [Cox et al., 1981, Shumway et al., 2000]. Initial techniques were relatively simple, such as imputing missing values using the mean or median of observed entries. These were later followed by more advanced interpolation approaches based on regression models, including linear regression and splines [Shumway et al., 2000]. To better exploit the spatio-temporal structure inherent in time series data, model-based methods emerged, such as ARIMA, GARCH, Kalman filters, and Bayesian inference frameworks [Fuller, 2009]. These approaches are advantageous for their interpretability, ability to incorporate domain knowledge, and support for formal statistical testing. Moreover, many of them naturally allow for uncertainty quantification through predictive intervals or posterior distributions. However, these methods come with notable limitations: they typically rely on strong assumptions about stationarity, linearity, or noise distributions, making them less effective for complex real-world data with nonlinear or high-dimensional spatio-temporal dependencies [Anderson, 2011]. Additionally, their computational cost often scales poorly with data dimensionality, posing challenges for modern large-scale applications.

To address the limitations of statistical approaches, machine-based imputation methods have become increasingly popular in recent years. Early approaches include classical machine learning models [Jerez et al., 2010] such as support vector machines [Wu et al., 2015] and tree-based methods (including bagging and boosting techniques) [Vateekul and Sarinnapakorn, 2009, Yang et al., 2017]. With the advancement of model architectures and increasing computational power, deep learning-based models have gained prominence for their ability to capture complex temporal dependencies [Fang and Wang, 2020, Wang et al., 2024, Du et al., 2024]. Predictive models such as RNNs [Che et al., 2018, Yoon et al., 2018b, Cao et al., 2018], CNNs [Wu et al., 2022, Fu et al., 2024a], GNNs [Cini et al., 2021], and transformer-based networks [Bansal et al., 2021, Du et al., 2023] directly estimate missing values using well-designed architectures. Generative imputation methods model the distribution of missing data and perform better in quantifying uncertainty; representative techniques include GAN-based methods [Luo et al., 2018, Yoon et al., 2018a, Miao et al., 2021], VAE-based approaches [Mattei and Frellsen, 2019, Fortuin et al., 2020, Mulyadi et al., 2021, Peis et al., 2022, Kim et al., 2023], and diffusion models. Among diffusion approaches, CSDI [Tashiro et al., 2021] introduced conditional diffusion for time series imputation, and subsequent work [Alcaraz and Strodthoff, 2022, Wang et al., 2023, Liu et al., 2023, Zhou et al., 2024] improved conditioning strategies and computational efficiency. DiT [Peebles and Xie, 2022, Cao et al., 2024] extends this line by integrating a transformer backbone into the diffusion framework, achieving better imputation accuracy and uncertainty quantification. These methods resolve certain issues and perform well empirically, however, are still limited by lacks of uncertainty quantification in many methods and theoretical understanding.

Our work also contributes towards the theoretical foundations of diffusion models [Chen et al., 2024, Tang and Zhao, 2024]. Some prior works have established sample efficiency and learning guarantees for diffusion models when modeling the original data distribution. Chen et al. [2022], Benton et al. [2023], Li et al. [2024] show that the generated distribution remains close to the target distribution, assuming access to an relatively accurate score function. By incorporating score approximation procedures and corresponding theoretical analysis, Chen et al. [2023], Oko et al. [2023], Mei and Wu [2025] provide end-to-end guarantees, covering various types of data including manifold data and graphical models. In the case of conditional diffusion models, sharp statistical bounds of distribution estimation have been derived in Fu et al. [2024c]. Additionally, Fu et al. [2024b] explores the theoretical regime of modeling spatio-temporal dependencies in sequential data. However, these results do not directly apply to more concrete and complex scenarios, such as how conditional DiT models can learn intricate dependencies to accomplish time series imputation tasks.

## B Proof of Lemma 1

We provide the detailed proof of Lemma 1 in this section.

To simplify our analysis, we begin by making some assumptions. Firstly, without loss of generality, we assume the mean of the Gaussian process data µ = 0 . Large norms in x and v t often lead to training instability, making it practical to perform clipping. Inspired by this, leveraging the Gaussian and light-tailed nature of x and v t , we truncate the domain of the data and diffused samples by

defining an event that occurs with high probability 1 -δ :

<!-- formula-not-decoded -->

where C δ data = O ( √ Hd ) is a threshold depending on δ . Our score approximation analysis of Lemma 1 and Theorem 1 is conducted under the condition of event C δ (ensuring the conclusions hold with high probability 1 -δ ), which significantly simplifies the process. The relationship between the truncation range C δ data , and high probability δ is deferred to Lemma 11. Outside event C δ (i.e., on C c δ ), the unbounded range complicates obtaining a meaningful score approximation in the second-norm sense. However, as C c δ occurs with a small probability, we can still achieve reliable results in distribution estimation, where evaluation is based on expectation.

Some Useful Results In this part, we present some key results regarding the eigenvalues and condition numbers of covariance matrices, which will be instrumental in our analysis.

We first define:

<!-- formula-not-decoded -->

Using the positive definiteness of Γ and Λ , we obtain:

<!-- formula-not-decoded -->

Furthermore, by the properties of the Kronecker product, we derive:

<!-- formula-not-decoded -->

Finally, we assume:

<!-- formula-not-decoded -->

## B.1 Key Steps for Proving Lemma 1

In Lemma 1, we aim to show that the gradient-based Algorithm 2 provides a good approximation of the conditional score function.

The algorithm employs gradient descent to solve two types of optimization problems: the major GD problem (7), and the auxiliary GD problem (9), which is solved within each update step of the major GD. It is critical to note that the major GD updates are inherently noisy due to various reasons, such as the auxiliary GD approximating certain quantities at each step, and later using transformers to approximate each step. Therefore, to establish the result in Lemma 1, our proof consists of two key steps:

Step 1. We demonstrate that, with a sufficient number of auxiliary iterations K aux , the approximation error of the auxiliary GD loop's result can be controlled below a specified threshold.

Step 2. We then show that, by controlling the perturbation level in each major GD update step, the score approximation error (i.e., the gap between the output of the major GD and the ground truth score function) can also be bounded, provided there are enough major iterations N .

In the following, we elaborate on each step by providing precise statements and subsequently use them to prove Lemma 1. All supporting results are deferred to later sections.

## B.2 Detailed Statements in Steps 1-2 and Proof of Lemma 1

Now we present formal statements in Step 1-2 and use them to prove Lemma 1.

## B.2.1 Formal Statements in Steps 1-2

This section contains the statements of Lemma 2 and Lemma 3.

Lemma 2. For an arbitrarily fixed time t ∈ (0 , T ] and given an error tolerance ϵ 0 ∈ (0 , 1) , let b := Σ cor s ( k ) ∈ R d |I miss | , running the auxiliary gradient descent in (10) with a suitable step size θ = 2 / ( λ min ( Σ obs ) + λ max ( Σ obs )) for

<!-- formula-not-decoded -->

iterations produces a solution u ( K aux ) that satisfies

<!-- formula-not-decoded -->

Here, we introduce ϵ 0 to distinguish the noise arising from the auxiliary GD loop approximation from the error level ϵ stated in Lemma 1. This distinction provides additional flexibility to adjust ϵ 0 in subsequent proofs.

Next, we establish a lemma for the convergence of the major GD. In each major GD step (referring to (8)), we incorporate an error term and represent our gradient update as:

<!-- formula-not-decoded -->

where ξ ( k ) represents the error term in each perturbed major GD step. Explicitly accounting for the noise present in each perturbed gradient step, we can establish:

Lemma 3. For an arbitrarily fixed time t ∈ (0 , T ] and given an error tolerance ϵ ∈ (0 , 1) , suppose ∥ ξ ( k ) ∥ 2 ≤ ϵ , then running the major gradient descent in (14) and a suitable step size η t = 2 / ( λ min ( α 2 t Σ cond + σ 2 t I ) + λ max ( α 2 t Σ cond + σ 2 t I )) for

<!-- formula-not-decoded -->

iterations produces a solution s ( K ) that satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

With the convergence of both the auxiliary and major GD established, we are ready to prove Lemma 1.

## B.2.2 Proof of Lemma 1

Proof. By the statement in Lemma 3, we need to control the noise level in each major GD step, i.e. ensure ξ ( k ) ≤ ϵ. We analyze this error as

<!-- formula-not-decoded -->

Here, the latter term arises from approximating µ cond ( x obs ) , and ̂ Σ -1 obs ( x obs -µ obs ) represents the K aux -iteration auxiliary GD approximation of the matrix-vector product.

We provide a useful lemma to help control the error above.

Lemma 4. For an arbitrarily fixed time t ∈ (0 , T ] , we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Invoking Lemma 2 and Lemma 4, letting ϵ 0 = ( ( α 2 t +1) ∥ Σ cor ∥ 2 ) -1 ϵ , to ensure that ∥ ξ ( k ) ∥ 2 ≤ ϵ ., we can bound the required auxiliary iteration steps by:

<!-- formula-not-decoded -->

Lastly, we invoke Lemma 3, substitute ϵ with ϵ = σ -1 t ( 2 κ t +2 ) ϵ , we have

<!-- formula-not-decoded -->

Finally, notice that and leveraging

we have

<!-- formula-not-decoded -->

This completes the proof of Lemma 1.

## B.3 Proofs of Lemma 2 and Lemma 3

To prove the lemmas, we first state a standard result in convex optimization.

Lemma 5 (Theorem 3.12 in [Bubeck et al., 2015]) . Let f be β -smooth and α -strongly convex on R d and x ∗ be the global minimizer. Then gradient descent with η = 2 α + β satisfies

<!-- formula-not-decoded -->

where x ( k +1) = x ( k ) -η ∇ f ( x ( k ) ) is the outcome at the ( k +1) -th iteration of gradient descent, and κ = β α .

Equipped with this lemma, the proof process is straightforward.

## B.3.1 Proof of Lemma 2

Proof. Referring to (10) (expression of auxiliary GD step), the update steps are

<!-- formula-not-decoded -->

Weshould notice that L aux is λ max ( Σ obs ) -smooth and λ min ( Σ obs ) -strongly convex. Then by Lemma 5, we have

<!-- formula-not-decoded -->

We also have

<!-- formula-not-decoded -->

With preset error ϵ 0 &gt; 0 , taking number of iterations K aux ≥ ⌈ κ ( Σ obs )+1 2 log( ∥ b ∥ 2 λ min ( Γ obs ) λ min ( Λ ) ϵ 0 ) ⌉ , we obtain

## B.3.2 Proof of Lemma 3

Proof. In each step, we incorporate an error term and represent our gradient update as in (14):

<!-- formula-not-decoded -->

Weshould also notice that L t is λ max ( α 2 t Σ cond + σ 2 t I ) -smooth and λ min ( α 2 t Σ cond + σ 2 t I ) -strongly convex. Then with ∥ ξ ( k ) ∥ 2 ≤ ϵ , by Lemma 5,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then we have

<!-- formula-not-decoded -->

Similar to the proof of Lemma 2, we obtain

<!-- formula-not-decoded -->

By invoking Lemma 4, we also have

<!-- formula-not-decoded -->

Lastly, taking K = O ( κ t log ( Hdκ ( Λ ) κ ( Γ obs ) σ t ϵ )) , we obtain

<!-- formula-not-decoded -->

This finishes the proof of Lemma 3.

<!-- formula-not-decoded -->

## C Proof of Theorem 1

We provide the detailed proof of Theorem 1 in this section by explicitly construct a transformer architecture to unroll Algorithm 2. Firstly, we assume that the mean function µ obs , µ miss can be constructed by an additional preprocessing network. Thus, the analysis in this section can also be conducted under the condition of event C δ and still assuming µ = 0 as stated in Appendix B.

## C.1 Key Steps for Proving Theorem 1

The proof of Theorem 1 is presented in a constructive framework. Revisiting the architecture in (5), we observe that it comprises the encoder f in , which transforms the original input into a form compatible with the unrolling of Algorithm 2; the raw transformer blocks, which perform the algorithm unrolling; and the decoder f out , which extracts and truncates the output to provide the final score approximation.

We define the major GD step with k = 1 as the first major GD step and those with k &gt; 1 as the later major GD steps. Similarly, we categorize the auxiliary GD steps. Notably, the first major GD step is relatively simpler, while the later major GD steps are analogous to it. Accordingly, we separate our analysis into the first and later major GD steps. To establish Theorem 1, the proof proceeds through the following steps:

Step 1. Construct the encoder, decoder, and essential components that are critical for constructing the subsequent raw transformer architectures.

Step 2. Construct the raw transformer architecture for the first major GD step.

Step 3. Construct the raw transformer architecture for the later major GD steps analogously.

Step 4. Analyze the error and configuration of the raw transformer architectures constructed in the previous steps.

Step 5. Summarize the constructions and analyses to establish the result in Theorem 1.

## C.2 Constructing Encoder, Decoder and Some Crucial Transformer Components

For sake of simplicity, given a time step t ∈ ( t 0 , T ] , we denote ( v t ) j = x j ∈ R d in the following analysis. Additionally, we define each (FFN l ◦ Attn l ) as a transformer block. The architecture composed solely of transformer blocks, excluding the encoder f in and decoder f out , is referred to as the raw transformer, denoted by T raw ( D,L,M,B ) .

Encoder The encoder we need is to mapping our input x to higher dimensions in an attepmpt to include some useful values (e.g. time embeddings) and also some buffer spaces to finish the gradient descent process. For simplicity, at a specific time t , we suppose the encoder converts the initial input into Y = f in ([ x 1 , x 2 , . . . , x H , t ]) = [ y ⊤ 1 , . . . , y ⊤ N ] ⊤ ∈ R D × H , which satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ϕ ( t ) = [ η t , α t , σ 2 t , α 2 t ] ⊤ ∈ R d t with d t = 4 . Specifically, we use different subscriptions for observed indices and missing indices, i.e. i ∈ I obs , j ∈ I miss . For simplicity, we omitted the subscript t here, and 0 ⊤ 6 d , 0 ⊤ 5 d , 0 ⊤ 4 d serve as the buffer space for storing the components necessary for unrolling the algorithm.

Decoder Suppose the output tokens from the transformer blocks has produced a conditional score approximator in matrix shape, and the stability for the computation inside the network, we design the decoder f out = f norm ◦ f linear . Where f linear : R D × H → R d |I miss | extracts and flattens the input into a vector that aligns with the dimension of the conditional score function, and f norm : R d |I miss | → R d |I miss | controls the output range of the network by the upper bound of the score function. By Lemma 4, denote σ -2 t ( 1 + κ ( Λ ) λ min ( Γ obs ) ) C δ data as R t , we can set

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

f mult Module At the end of this part, we also provide the construction of the multiplication module, which approximates the product between scalars and vectors. This is a crucial component in constructing f GD later. We introduce a lemma, which is a modified version of Corollary 3 in [Fu et al., 2024b]:

Lemma 6. Suppose input to be Y = [ y 1 , y 2 , · · · , y H ] ∈ R D × H with y i = [ x ⊤ i , 0 ⊤ 3 d , w i , z ⊤ i ] , where x i ∈ [ -B,B ] d , w i ∈ [ -B,B ] and z i ∈ R d z . Given any ϵ mult &gt; 0 , there exists a (FFN-only) transformer architecture such that

<!-- formula-not-decoded -->

with L = O (log( B/ϵ mult )) layers that approximately multiply each component x i with the weight w i and put it into a buffer, keeping other dimensions the same. This can be formally written as

<!-- formula-not-decoded -->

The number of nonzero coefficients in each weight matrices or bias vectors is at most O ( d ) , and the norm of the matrices and bias are all bounded by O ( Bd ) .

## C.3 First Major GD Step

In this section, we construct the transformer architecture unrolling the first step of major GD procedure, and the result can be summarized as:

Lemma 7 (Construct first major GD step) . There exists a raw transformer architecture f GD , 1 ∈ T raw ( D,L,M,B ) , which can construct an approximate first step major GD result ˜ s (1) from the output of the encoder.

Given an error level ϵ &gt; 0 and learning rate η t &gt; 0 , the approximated first step GD result satisfies s (1) -s (1) = ξ (1) , where ∥ ξ (1) ∥ ≤ ϵ, where s (1) is the groundtruth gradient step.

<!-- formula-not-decoded -->

The configuration of the raw transformer architecture satisfies

<!-- formula-not-decoded -->

We defer the analysis of transformer configuration to C.5.

Following the encoder network construction above, we obtain the following input:

<!-- formula-not-decoded -->

Revisiting the major GD gradient step in (8), our goal is to form the following at major GD first step:

<!-- formula-not-decoded -->

Initially, we apply a multiplication module to construct:

<!-- formula-not-decoded -->

This encapsulates the fundamental operations required for constructing the first major GD iteration.

## C.3.1 Auxiliary GD for First Major GD Step

In this section, we approximate the term Σ -1 obs ( η t α t x obs ) using an iterative auxiliary GD procedure.

First Auxiliary GD Step Starting from the initialization u (0) i = 0 d for i ∈ I obs , we want the first auxiliary GD iteration finish the update:

<!-- formula-not-decoded -->

Using f mult, we can easily obtain

<!-- formula-not-decoded -->

This finished the first step of auxiliary GD.

Auxiliary GD Later Steps For subsequent iterations, the updated rule for the auxiliary GD becomes:

<!-- formula-not-decoded -->

for k aux = 1 , 2 , · · · , K aux -1 .

<!-- formula-not-decoded -->

Similar to the first auxiliary GD step performed above, we firstly use f mult to obtain

<!-- formula-not-decoded -->

.

Then, by constructing a 4 H -head attention block as described in C.6.1, we obtain

<!-- formula-not-decoded -->

Lastly, after combining an linear transformation FFN block with the attention block above to build up the basic transformer block T B obs , we will have

<!-- formula-not-decoded -->

where

˜ u ( k aux +1) i = u ( k aux ) i -H -1 ∑ m =0 ∑ k ∈I obs γ m ✶ {| i -k | = m } Λ f mult ( θ, u ( k aux ) k )+ f mult ( θη t α t , x i -µ i , obs ) . This completes a later step auxiliary GD update.

Final result for Auxiliary GD We denote the iterative blocks ( T B obs ◦ f mult ◦ f mult ) K aux as f inner . The result of the auxiliary GD for approximating ( Σ -1 obs ( η t α t ( x obs -µ obs ))) i is expressed as:

<!-- formula-not-decoded -->

After completing K aux auxiliary GD iterations, we obtain at the following transformation:

<!-- formula-not-decoded -->

which incorporates the iterative updates. The resulting output includes ˜ u i for each observation entry, alongside the original data. We therefore finish the auxiliary GD procedure for the first major GD step.

## C.3.2 Matrix Multiplication

After K aux steps of auxiliary GD iterations, we proceed with an additional matrix multiplication step to compute Σ ⊤ cor ˜ u .

The multiplication can be expressed as:

<!-- formula-not-decoded -->

Referring to the construction in C.6.2, this computation can be implemented using a 4 H -head attention block:

<!-- formula-not-decoded -->

Combining the attention block described above with a linear transformation through a FFN block, which we denote as T B cort , we obtain:

f GD , 1 = T B cort ◦ f inner ◦ f ( Y

We now define ˜ s (1) j as:

<!-- formula-not-decoded -->

Comparing this result with (15), we observe that the first step of the major gradient descent is now complete. We represent this step as f GD , 1 .

For simplicity, we introduce the notation:

<!-- formula-not-decoded -->

## C.4 Major GD Later Steps

In this section, we construct the transformer architecture unrolling the later steps of major GD procedure, and the result can be summarized as

Lemma 8 (Construct major GD later steps) . There exists a raw transformer architecture f GD ∈ T raw ( D,L,M,B ) (i.e. without encoder and decoder), which can construct a new approximate later step GD result ˜ s + from the output of the latest step of major GD. Given an error level ϵ ∈ (0 , 1) and learning rate η t &gt; 0 , the approximated first step GD result satisfies

<!-- formula-not-decoded -->

where s ( k +1) is the groundtruth gradient step.

Furthermore, the configuration of the raw transformer architecture satisfies

<!-- formula-not-decoded -->

We defer the analysis of transformer configuration to C.5.

In the later steps of the major GD, we need to compute the following update:

<!-- formula-not-decoded -->

In the following proof, for the sake of simplicity, we use s + , s as abbreviation for s ( k +1) , s ( k ) , respectively.

In each new major GD step, the input to the iteration is the output of the most recent GD step. For simplicity, we continue to represent this input using Y .

<!-- formula-not-decoded -->

Similar to the construction in the first step, we first apply a multiplication module to obtain:

<!-- formula-not-decoded -->

.

Next, we proceed to the matrix multiplication step, followed by the auxiliary GD process.

## C.4.1 Matrix Multiplication 1

To begin, we compute Σ cor s , which can be expressed as:

<!-- formula-not-decoded -->

For each i corresponding to the observations, this can be further rewritten as:

<!-- formula-not-decoded -->

To perform this computation, similar to the construction in the first step major GD, we employ a 4 H -head attention block combined with an identical FFN block to form T B cor :

<!-- image -->

## C.4.2 Auxiliary GD

In this step, we compute Σ -1 obs Σ cor s .

Following the auxiliary GD procedure described in C.3.1, we employ a similar iterative approach using f inner to approximate the multiplication between a matrix inverse and vectors:

<!-- formula-not-decoded -->

## C.4.3 Matrix Multiplication 2

In this step, we compute Σ ⊤ cor Σ -1 obs Σ cor s .

Following a similar procedure as in C.3.2, we construct a transformer block T B cort . This block similarly employs a 4 H -head attention mechanism combined with an identity FFN to perform the matrix multiplication:

<!-- formula-not-decoded -->

## C.4.4 Matrix Multiplication 3

In this step, we compute Σ miss s .

Following a similar procedure as in C.6.1, we construct a transformer block T B miss . This attention layer utilizes a 4 H -head attention block to obtain:

.

<!-- formula-not-decoded -->

Then, with a linear transformation FFN layer, we get the final output

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Comparing this expression with (17), we conclude that one major GD update has been completed.

We can represent the later major GD steps compactly as:

<!-- formula-not-decoded -->

## C.5 Error Analysis and Transformer Configurations

In this section, we analyze the error induced by using transformer architectures to unroll the gradient descent procedure, and derive the corresponding transformer configurations to formally establish the result in Lemma 7 and 8. Lastly, we combine these results to finish the proof of Theorem 1.

Above all, we should notice that, by our construction above, leveraging transformer to approximate each step of Auxiliary GD also induces noise. Consequently, similar to (14), in each auxiliary GD step, we incorporate an error term and represent the update as:

<!-- formula-not-decoded -->

where ξ ( k aux ) 0 represents the approximation error term in each auxiliary GD step. We can also state a corresponding Lemma that sharing the same proof strategy with its counterpart in major GD (Lemma 3):

Lemma 9. For an arbitrarily fixed time t ∈ (0 , T ] and given an error tolerance ϵ 0 ∈ (0 , 1) , if we can control ∥ ξ ( k aux ) 0 ∥ 2 ≤ ϵ 0 , then running the auxiliary GD in (14) with a suitable step size θ for iterations gives:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C.5.1 First Major GD Step

In this part, we analyze the noise introduced by the construction in C.3, and corresponding transformer architecture configuration (i.e. D,L,M,B ).

Bounding Approximation Error Let s (1) denote the exact major GD update, and ˜ s (1) represent our approximation. According to (15), in first major GD step, transformer blocks are utilized to compute:

<!-- formula-not-decoded -->

for j ∈ |I miss | , as an approximation to:

<!-- formula-not-decoded -->

In first major GD step, auxiliary GD is responsible for computing Σ -1 ( η t α t ( x obs -µ obs )) .

̂ obs

From (16), each auxiliary GD step updates as (here we only analyze the later auxiliary GD step, and we use u + , u for ˜ u ( k aux +1) , ˜ u ( k aux ) for the sake of simplicity):

and we approximate it using:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To ensure control over the error ∥ ˜ u + -u + ∥ 2 = ∥ ξ 0 ∥ 2 ≤ ϵ 0 , Lemma 6 indicates that setting ϵ mult , aux , 1 = ϵ 0 ( H 3 / 2 ∥ Λ ∥ F + H 1 / 2 ) √ d suffices. This requires L mult , aux , 1 = O ( log ( dN ∥ Λ ∥ F ϵ 0 )) iterations.

With each step noise controlled, the entire auxiliary GD procedure, combined with the subsequent matrix product blocks, yields Σ cor ̂ Σ -1 obs f mult ( η t α t , x obs -µ obs ) . According to Lemma 9, setting

<!-- formula-not-decoded -->

ensures that:

<!-- formula-not-decoded -->

This provides an approximation of Σ ⊤ cor Σ -1 obs with controlled error bounds.

Finally, the f mult module approximates scalar and vector multiplications to complete the first step of gradient descent. The overall error for each j is computed as:

<!-- formula-not-decoded -->

By setting ϵ 0 = ( ∥ Σ cor ∥ F ( κ ( Σ obs ) + 3) √ H ) -1 ϵ , which leads to an auxiliary gradient descent step count of

<!-- formula-not-decoded -->

and by Lemma 6, setting ϵ mult = ( 8 √ dN ) -1 ϵ , which leads to

<!-- formula-not-decoded -->

we successfully control the error ∥ ξ (1) ∥ 2 = ∥ ˜ s (1) -s (1) ∥ 2 ≤ ϵ 2 &lt; ϵ .

Configuration of Transformer Architecture for Approximating the First Major GD Step We finally summarize our construction by characterizing the configuration of the architecture:

- The input to the transformer is of dimension D × H with D = 12 d + d e + d t +3 .
- In each auxiliary GD step, we use 1 transformer block to form the matrix product and some f mult modules, requiring a total of L mult , aux transformer blocks. We need to perform P 1 auxiliary GD steps. After completing the auxiliary GD, additional f mult modules are used to compute scalar and vector products, which require L mult , 1 blocks. Thus, the number of the transformer blocks is bounded by

<!-- formula-not-decoded -->

- The number of transformer blocks is bounded by M = 4 H .
- With the constructions above, referring to Lemma 6, the norm of the multiplication module is bounded by O ( d ( ∥ x ∥ ∞ + ∥ s ∥ ∞ ) ;, and Lemma 4 helps us bound x ∞ and ∥ v ∥ ∞ . Referring to the attention module constructed in C.6.2 and C.6.1, the norm of the attention matrices are bounded by O ( d ( r 2 + λ max ( Λ ))) ; and considering the weight matrices in the FFN, since they only do linear transformations and only have at most O ( d ) nonzero weights, their norm are bounded by O ( d ) . To sum up, we have the norm of the transformer parameters bounded by

<!-- formula-not-decoded -->

And this finishes the proof of Lemma 7.

## C.5.2 Major GD Later Steps

In this part, we analyze the noise introduced by the construction in C.4, and corresponding transformer architecture configuration.

Bounding Approximation Error In later steps, according to 17, we use

<!-- formula-not-decoded -->

to approximate

<!-- formula-not-decoded -->

We first consider the auxiliary gradient descent which computes ̂ Σ -1 obs Σ cor f mult ( η t α 2 t , s ) . Compared to the first step analysis, we simply replace x obs with s . So we can completely follow the procedure in (19). To control ∥ ˜ u + -u + ∥ 2 = ∥ ξ ∥ 2 ≤ ϵ 0 , we set the corresponding inside multiplication module error as

<!-- formula-not-decoded -->

which requires

<!-- formula-not-decoded -->

Combining the auxiliary GD output with the following matrix multiplication blocks, we obtain Σ ⊤ cor ̂ Σ -1 obs Σ cor f mult ( η t α 2 t , s ) . By Lemma 9, with K aux = ⌈ κ ( Σ obs )+1 2 log( η t α t ∥ Σ cor ∥ F ∥ s ∥ 2 λ min ( Γ obs ) λ min ( Λ ) ϵ 0 ) ⌉ , we have

<!-- formula-not-decoded -->

Next, we decompose the overall error term. Recall that

<!-- formula-not-decoded -->

similar to the analysis in the first iteration, we can derive the error bound for approximating s + j as:

<!-- formula-not-decoded -->

where the second term, ̂ η t α t µ cond ( x obs ) , was computed in the first iteration bound and is thus bounded by ϵ 2 .

By setting ϵ 0 = ( 2 ∥ Σ cor ∥ F ( κ ( Σ obs ) + 3) √ H ) -1 ϵ , which leads to the auxiliary gradient descent step count:

<!-- formula-not-decoded -->

and setting ϵ mult = ( 8 √ dN ( ∥ Σ miss ∥ F +2) ) -1 ϵ , which leads to

<!-- formula-not-decoded -->

we successfully control the error ∥ ξ + ∥ 2 = ∥ ˜ s + -s + ∥ 2 ≤ ϵ .

Size of Transformer Architecture for Approximating the Later Steps Major GD We finally summarize our construction by characterizing the size of the architecture:

- The input to the transformer is of dimension D × H with D = 12 d + d e + d t +3 .
- In the later major GD steps, , we use 1 + L mult , aux , + transformer blocks for each auxiliary GD step, and a total of K aux , + auxiliary GD steps are required. After completing the auxiliary GD, we perform additional matrix multiplications (e.g., multiplying vectors by Σ cor , Σ ⊤ cor , and Σ miss ), which require 3 transformer blocks for attention. Subsequently, f mult modules are used to complete the major GD, requiring L mult , + blocks. Thus, the total number of transformer blocks required for each subsequent major GD step is bounded by:

<!-- formula-not-decoded -->

- The number of transformer blocks is bounded by M = 4 H .
- Same as the analysis in the first step of major GD, we have the norm of the transformer parameters bounded by

<!-- formula-not-decoded -->

And this finishes the proof of Lemma 8.

## C.5.3 Proof of Theorem 1

Proof. We formally construct the conditional score approximation transformer as follows:

<!-- formula-not-decoded -->

Recalling that κ t = κ ( α 2 t Σ cond + σ 2 t I ) , by the major GD convergence result in Lemma 3, to ensure ∥ ˜ s -s ∥ 2 ≤ σ -1 t ϵ , the total major GD iteration number required is upper bounded by K = O ( κ t log ( Hdκ t κ ( Λ ) κ ( Γ obs ) ϵ )) , which is obtained by substituting ϵ with σ -1 t ( 2 κ t +2 ) ϵ .

Utilizing Lemma 7 and 8, and substituting ϵ with σ -1 t ( 2 κ t +2 ) ϵ , the following transformer configuration can control the error in each major GD step:

<!-- formula-not-decoded -->

where ℓ is computed by K times the transformer block required in each major GD step.

Finally, by substituting σ -1 t , κ t with σ -1 t 0 , κ t 0 , and considering the truncation range which is induced by the decoder ( R = O ( σ -2 t 0 √ Hdκ ( Λ ) κ ( Γ obs )) ), we obtain a uniform bound for any t ∈ [ t 0 , T ] . Taking supremum over all admissible I obs ⊂ I , and leverage the relationship that κ ( Σ cond ) = κ ( Λ ) κ ( Γ obs ) , we finish the proof of Theorem 1.

## C.6 Construction of Attention Layers

In this section, we construct the attention layers used in the transformer architectures built up in C.3 and C.4.

We utilize the added

<!-- formula-not-decoded -->

to construct different types of interaction between different types of samples. (i.e. When we want attention exclusively among observed samples or missing samples, we use the construction method as described in T B obs below; When we want attention between observed samples and missing samples, while setting all other interactions to zero, we use the construction method as described in T B cort below. )

The intuition of (16) suggests a construction of a multi-head attention layer. Formally, for an arbitrary value of m , we construct four attention heads with ReLU activation. The indicator function ✶ {| i -j | = m } can be realized by calculating the auxiliary product e ⊤ i e j of time embedding. To see this, we observe

<!-- formula-not-decoded -->

Therefore, it holds that

<!-- formula-not-decoded -->

since f Assumption 1 ensures that the time embeddings uniquely identify discrete time gaps through their pairwise distances. Directly approximating an indicator function using a ReLU network can be difficult. Yet we note that | i -j | can only take integer values. Therefore, we can slightly widen the decision band for the indicator function. Specifically, we denote a minimum gap ∆ = min i =1 ,...,H -1 { f 2 ( i +1) -f 2 ( i ) } . Thus, we deduce

<!-- formula-not-decoded -->

We can use four ReLU functions to approximate the right-hand side of the last display, and simultaneously take different type of interaction types into account. We use another indicator function (which can be realized by the 0s and 1s added above) to represent what types of interaction we want in this specific transformer block.

We construct a trapezoid function as follows:

<!-- formula-not-decoded -->

## C.6.1 Construction of attention matrices related to the observed part

We construct the attention matrices for T B obs here.

For particular m , we utilize

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

It is easy to verify that

We can claim that 4 H attention heads and identity FFN are enough for constructing this block.

<!-- formula-not-decoded -->

## C.6.2 Construction of attention matrices related to the correlation part

We only need to do some small changes to T B obs .

For particular m , let

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

It is easy to verify that

<!-- formula-not-decoded -->

We can thus state that 4 H attention heads and identity FFN are enough for constructing this block.

## D Proofs of Theorem 2 and Corollary 1

In this section, we provide the detailed proof of Theorem 2 and Corollary 1.

Firstly, we introduce some notations specifically for this part for sake of simplicity. We denote our training set with n i.i.d. samples as

<!-- formula-not-decoded -->

We introduce the corollary below which will act as an significant role in our later proof:

Corollary 2. By choosing the transformer architecture T ( D,L,M,B,R ) as in Theorem 1, the early-stopping time t 0 &lt; 1 and the terminal time T = O (log n ) , it holds that

<!-- formula-not-decoded -->

where κ t 0 := κ ( α 2 t Σ cond + σ 2 t I ) .

The proof of Corollary 2 is deferred to Appendix E.

## D.1 Proof of Theorem 2

Although our assumption on Gaussian processes does not ensure the Novikov's condition to hold, according to [Chen et al., 2022], as long as we have bounded the second moment for the score estimation error and finite KL divergence w.r.t the standard Gaussian, we could still adopt Girsanov's Theorem and bound the KL divergence between the two distribution. We restate the Lemma as follows:

Lemma 10 (Corollary D.1 in [Oko et al., 2023], see also Theorem 2 in [Chen et al., 2022]) . Let p 0 be a probability distribution, and let Y = { Y t } t ∈ [0 ,T ] and Y ′ = { Y ′ t } t ∈ [0 ,T ] be two stochastic processes that satisfy the following SDEs:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We further define the distributions of Y t and Y ′ t by p t and p ′ t . Suppose that

<!-- formula-not-decoded -->

for any t ∈ [0 , T ] . Then we have

<!-- formula-not-decoded -->

Equipped with Corollary 2 and Lemma 10, we are ready to prove Theorem 2.

Proof of Theorem 2. Firstly, following the proof of Lemma 12, we can easily verify that for any s ∈ T ( D,L,M,B,R ) ,

<!-- formula-not-decoded -->

Thus, the condition (10) holds for all t ∈ [ t 0 , T ] , which means that we could apply Girsanov's theorem in this time range.

To further distinguish the SDE defined in (1), (2), and (3), we denote the distribution of x t , v t , ̂ v t as P t , P ← t , ̂ P ← t , respectively. Additionally, we need to introduce another intermediate backward process between P ← t , ̂ P ← t as follows

<!-- formula-not-decoded -->

and we denote the marginal distribution of v ′← t (conditioned on y ) as P ′ T -t ( ·| y ) .

Equipped with these notations, we can decompose the total variation between P and ̂ P ← t 0 as

<!-- formula-not-decoded -->

We denote { ξ i } d |I miss | i =1 as the eigenvalues of Σ cond , and we can do eigenvalue decompositions to Σ cond as Σ cond = Q Ξ Q ⊤ , where ( Ξ ) ii = ξ i .

Considering the second last term' by Data Processing Inequality and Pinsker's Inequality (see e.g. Lemma 2 in [Canonne, 2022]), we have

<!-- formula-not-decoded -->

where we leverage the close-form solution of the KL-divergence between two gaussian distributions in the first equality.

Regarding the first term, by Pinsker's Inequality and the close-form solution of the KL-divergence between Gaussian distributions [Pardo, 2018], we have

<!-- formula-not-decoded -->

Leveraging the close-form solution of the KL-divergence between two gaussian distributions, we further have

<!-- formula-not-decoded -->

Considering term A , we have

<!-- formula-not-decoded -->

Then we obtain

Regarding term B , we have

<!-- formula-not-decoded -->

Considering term C , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining (21), (23), (22), and invoking Lemma 10, we have:

<!-- formula-not-decoded -->

Plugging in the result in Corollary 2 (taking T = O (log n ) , and t 0 = O ( ξ min n -1 2 ) ), we finally obtain

<!-- formula-not-decoded -->

where Γ cond = Γ miss -Γ ⊤ cor Γ -1 obs Γ cor . We complete our proof.

## D.2 Proof of Corollary 1

With Theorem 2 established, the conclusion in Corollary 1 goes straightforward.

Proof. Firstly, by the definition of total variation distance, we have the relationship

<!-- formula-not-decoded -->

Following the decomposition in (21), we can obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Regarding the right hand side, following the derivation in (24), we can bound each term similarly by taking t 0 = O ( λ min ( Σ cond ) n -1 2 ) and T = O (log n ) .

For the second term, we leverage the close-form solution of the KL-divergence between two gaussian distributions:

<!-- formula-not-decoded -->

For the last term, recalling the definition of DS ( P x obs , P x ∗ obs ; G ) , we have

<!-- formula-not-decoded -->

Let

<!-- formula-not-decoded -->

For the first term, leveraging the result in (23), and the decomposition of term C in the proof Theorem 2, we have

<!-- formula-not-decoded -->

Finally, we can combine all the bounds above to obtain

<!-- formula-not-decoded -->

and the corollary follows.

## E Proof of Corollary 2

In this section, we provide the detailed proof of Corollary 2.

Training Loss During training, given a state v t = α t v 0 + σ t z , z ∼ N ( 0 , I d |I miss | ) , v 0 = x miss , we aim to minimize the ideal risk function:

<!-- formula-not-decoded -->

However, in practice, the objective (25) is not directly accessible. According to Lemma C.3 in Vincent [2011], an equivalent objective function L ( s ) , which differs from R ( s ) only by a constant, can be used for optimization:

<!-- formula-not-decoded -->

Here, ϕ t is the Gaussian transition kernel of the forward process, satisfying ∇ log ϕ t ( v t | v 0 ) = -( v t -α t v 0 ) σ 2 t .

Thus, we can leverage the corresponding empirical loss (11):

<!-- formula-not-decoded -->

where the loss function is defined in (12)

<!-- formula-not-decoded -->

## E.1 Steps for Proving Corollary 2

## E.1.1 Risk Decomposition

The proof procedure is analogous to the proof of Theorem 4.1 in [Fu et al., 2024c], provided in Appendix D of the same work. Our goal is to derive a bound on E { ( x ( i ) , y ( i ) ) } n i =1 [ R ( ̂ s )] . We denote the ground truth score function as s ∗ and set R ( s ∗ ) = 0 .

Following the setup, the risk can be decomposed as:

<!-- formula-not-decoded -->

where ̂ s is the score function trained on dataset D ( \ ) using the empirical risk. By creating n i.i.d. ghost samples

<!-- formula-not-decoded -->

the population risk of ̂ s can be rewritten as:

<!-- formula-not-decoded -->

To bound the rewritten population risk, we can further decompose it by analyzing its behavior in a truncated area (aligning with our score approximation analysis in Theorem 1, we analyze the error conditioning on the event C δ = {∥ x ∥ 2 , ∥ v ∥ 2 ≤ C δ data } ), and the error induced by truncation.

The truncated loss function is defined as

<!-- formula-not-decoded -->

Accordingly, we denote the truncated domain of the score function by X = [ -C δ data , C δ data ] d |I miss | × [ -C δ data , C δ data ] d |I obs | , and the truncated loss function class defined as

<!-- formula-not-decoded -->

Define the following intermediate terms ( ̂ s depends on D ( \ ) ):

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The decomposition for the expected empirical risk over D ( \ ) then becomes:

<!-- formula-not-decoded -->

The terms A , B , and C respectively represent the error incurred due to truncation, approximation in truncation, and the in-sample empirical risk expectation.

## E.1.2 Bound of Each Component

We first bound the data range with high probability. The proof of the lemmas stated in this section are deferred to E.3.

Lemma 11 (Range of the data) . Given a sufficiently large data truncation range C δ data &gt; 0 , we have

<!-- formula-not-decoded -->

where δ n,d = 2 n exp { -C 2 C 2 data 8( Hd +1)( ∥ Λ ∥ F +1) } , and C 2 is the absolute constant defined in Lemma 16.

We also have P [ E ] ≥ 1 -δ d , where δ d = 1 n δ n,d .

In the following analysis, for the sake of simplicity, we denote C Σ = 1 + ∥ Γ cor ∥ 2 κ ( Λ ) λ min ( Γ obs ) , which origins from Lemma 4. Then we state Lemma 12 to bound term A in (29), which is the counterpart of ( D. 12) in [Fu et al., 2024c].

Lemma 12. For any s ∈ T ,

<!-- formula-not-decoded -->

It is straightforward to conclude that

<!-- formula-not-decoded -->

Then we proceed to the term C in (29). For any s ∈ T , we have the following relationship

<!-- formula-not-decoded -->

the inequality holds due to ̂ s minimizes ̂ L .

Taking minimum w.r.t. s ∈ T , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 13. Given an error level ϵ ∈ (0 , 1) ,

<!-- formula-not-decoded -->

Finally, we can proceed to the bound of term B . In an attempt to providing the bound, we first need to calculate the covering number of the loss function class S ( C δ data ) , and correspondingly, the covering number of our transformer architecture function class. The covering number is defined as follows:

Definition 2. We denote N ( δ, F , ∥ · ∥ ) to be the δ -covering number of any function class F w.r.t the norm ∥ · ∥ , i.e.,

<!-- formula-not-decoded -->

A modified version of Lemma 23 in [Fu et al., 2024b] provides the following result on transformer covering numbers:

Lemma 14. Consider the entire transformer architecture F = T ( D,L,M,B,R ) (i.e. with encoder and decoder). If the input to the transformer satisfy ∥ v t ∥ 2 , ∥ y ∥ 2 ≤ C δ data , the time embedding e and the diffusion time-step embedding ϕ ( t ) satisfy ∥ e ∥ 2 = r, ∥ ϕ ( t ) ∥ 2 ≤ C diff and r, C diff ≤ O ( √ Hd ) , then the log-covering number of the transformer architecture is bounded by

<!-- formula-not-decoded -->

Then, we can leverage the following lemma to calculate the covering number of the corresponding truncated loss function class.

Lemma 15. Suppose ̂ s (1) , ̂ s (2) ∈ T ( D,L,M,B,R ) such that ∥ ̂ s (1) ( v t , y ; t ) -̂ s (2) ( v t , y ; t ) ∥ 2 ≤ δ s for any ∥ v t ∥ 2 , ∥ y ∥ 2 , ∥ x ∥ 2 ≤ C δ data and t ≥ t 0 , then we have

<!-- formula-not-decoded -->

Equipped with this lemma, it is straight forward to derive that log N ( δ l ; S ( C δ data ) , ∥·∥ ∞ ) ≲ D 2 M ( L 2 log ( BMNRC δ data C Σ ) +log ( BMLHdC δ data C Σ δ s )) , and δ s satisfies

<!-- formula-not-decoded -->

Invoking the bound provided in (D.16) of [Fu et al., 2024c], we have

<!-- formula-not-decoded -->

Combining the bound of A, B and C ((30),(32), (31)), we can leverage the empirical risk decomposition (29) to finalize the proof of Corollary 2.

## E.2 Proof of Corollary 2

Proof of Corollary 2. By (29), (30),(32) and (31), we have

E

D

(

)

n

R

[

(

s

)]

̂

≤

A

+

B

+

C

<!-- formula-not-decoded -->

Plugging in the configuration of our transformer architecture in Theorem 1, and take

<!-- formula-not-decoded -->

the inequality above gives rise to

<!-- formula-not-decoded -->

## E.3 Proof of Supporting Lemmas in E.1.2

Proof of Lemma 11. We first state the polynomial concentration lemma for Gaussian random variables.

Lemma 16 (Lemma 24 in [Fu et al., 2024b]) . Let g be a polynomial of degree p and x ∼ N (0 , I d ) . Then there exists an absolute positive constant C p , depending only on p , such that for any δ &lt; 1 ,

<!-- formula-not-decoded -->

For a random variable r ∼ N ( 0 , Σ 0 ) , consider g ( · ) = ∥ · ∥ 2 2 , we have

<!-- formula-not-decoded -->

Applying Lemma 16, we can conclude that with high probability at least 1 -2 exp( -C 2 δ ) ,

<!-- formula-not-decoded -->

Considering v t , we have Σ 1 = α 2 t ( Γ miss ⊗ Λ ) + σ 2 t I ; and for x , we have Σ 2 = Γ ⊗ Λ . Therefore,

<!-- formula-not-decoded -->

the last inequality holds for δ &lt; 1 . Similar inequalities hold for ∥ x ( i ) ∥ 2 .

Consider C δ data ≥ 2 √ ( Hd +1)( ∥ Λ ∥ F +1) and let δ = ( C δ data ) 2 8 Hd +1)( ∥ Λ ∥ F +1) . We can then obtain a union bound. With probability at least 1 -2 n exp { -C 2 ( C data ) 2 8( Hd +1)( ∥ Λ ∥ F +1) } ,

<!-- formula-not-decoded -->

We finish the proof by setting δ n,d = 2 n exp { -C 2 C 2 data 8 Hd +1)( ∥ Λ ∥ F +1) } .

Proof of Lemma 12. For any s ∈ T ( s can depend on x , y ),

<!-- formula-not-decoded -->

where R t is the truncation range of the decoder, and we apply triangular inequality in the third line.

Proof of Lemma 13. Since ˜ s ∈ T , we can invoke Theorem 1 and triangular inequality:

<!-- formula-not-decoded -->

where we apply Cauchy-Schwarz inequality in the second step, Jensen's inequality in the last step. For the second last term, similar to the proof of Lemma 12, we have

<!-- formula-not-decoded -->

For the last term, we have

<!-- formula-not-decoded -->

where we utilize the positive definiteness of Σ ⊤ cor Σ -1 obs Σ cor in the last inequality. Combining all the terms together, we have

<!-- formula-not-decoded -->

Proof of Lemma 15. We have

<!-- formula-not-decoded -->

Therefore, we obtain

<!-- formula-not-decoded -->

Figure 4: Comparison of imputation methods on the Electricity dataset, with 95% CR.

<!-- image -->

## F Experiment Details

For our numerical experiments, we trained the models using a batch size of 64. Our adapted DiT model architecture used a hidden size of 256, 12 transformer layers, and 16 attention heads per layer. We utilized the PyPOTS [Du, 2023] framework to implement and handle hyperparameter tuning for the baseline methods CSDI and GP-VAE. This tuning process aimed to find the best settings and ensure the models had a comparable number of trainable parameters. Experiments were conducted on hardware consisting of an NVIDIA RTX A6000 GPU (48GB) and an Intel(R) Xeon(R) Gold 6242R CPU @ 3.10GHz. We report all the results as the average of 5 runs. Our implementation of DiT for imputation is attached in supplementary materials.

## F.1 Real World Datasets

Dataset Descriptions. We utilize two real-world datasets, BeijingAir [Zhang et al., 2017] and ETT\_m1, to benchmark the imputation performance of DiT. The BeijingAir dataset comprises hourly measurements of six air pollutants and meteorological variables collected from 12 monitoring sites in Beijing. The ETT\_m1 dataset, part of the Electricity Transformer Temperature benchmark, records clients' electricity consumption data, including power load and oil temperature. Detailed statistics for both datasets are provided in Table 5.

Table 5: 80% of the data is used for training, and 20% for testing.

| Dataset     |   Total Samples |   Sequence Length | Time Interval   |   Number of Variables |
|-------------|-----------------|-------------------|-----------------|-----------------------|
| Air Quality |            1168 |                30 | 1H              |                   132 |
| Electricity |            2321 |                48 | 15min           |                     7 |

Results. We report the Mean Absolute Error (MAE) in Table 6, the Mean Squared Error (MSE) in Table 7 and the Mean Relative Error (MRE) in Table 8. Results are shown across different missing data rates (10%, 20%, and 50%) for both datasets. The experimental results indicate that DiT consistently outperforms the baseline methods on both datasets, demonstrating its effectiveness, and our mixed-masking strategy can also enhance DiT's performance on real-world datasets.

Figure 4 presents a comparison of imputation results on the ETT\_m1 dataset, where we randomly select samples from a 50% missing data scenario. From the plots, it is evident that although both DiT and CSDI generate CRs that largely encompass the true data points, DiT achieves a tighter bandwidth, leading to improved uncertainty quantification performance.

<!-- image -->

| Model                                               | ETTm_1 (Missing %)   | ETTm_1 (Missing %)                                 | ETTm_1 (Missing %)                                 | BeijingAir (Missing %)                   | BeijingAir (Missing %)                             | BeijingAir (Missing %)                             |
|-----------------------------------------------------|----------------------|----------------------------------------------------|----------------------------------------------------|------------------------------------------|----------------------------------------------------|----------------------------------------------------|
|                                                     | 10%                  | 20%                                                | 50%                                                | 10%                                      | 20%                                                | 50%                                                |
| CSDI [Tashiro et al., 2021] GP-VAE [Fortuin et al., | 0.1448 (±0.0105)     | 0.1521 (±0.0114) 0.3267 (±0.0044) 0.1377 (±0.0095) | 0.1650 (±0.0097) 0.4666 (±0.0073) 0.1543 (±0.0102) | 0.1780 (±0.0138) 0.4152 (±0.0088) 0.1753 | 0.1800 (±0.0129) 0.4401 (±0.0080) 0.1815 (±0.0208) | 0.2141 (±0.0119) 0.5265 (±0.0054) 0.2057 (±0.0145) |
| 2020]                                               | 0.2786 (±0.0077)     |                                                    |                                                    |                                          |                                                    |                                                    |
| DiT                                                 | 0.1269 (±0.0076)     |                                                    |                                                    | (±0.0094)                                |                                                    |                                                    |

Table 6: Time Series Imputation MAE Results

Table 7: Time Series Imputation MSE Results

<!-- image -->

| Model                                                   | ETT_m1 (Missing %)   | ETT_m1 (Missing %)                                                  | ETT_m1 (Missing %)                                        | BeijingAir (Missing %)                | BeijingAir (Missing %)                                              | BeijingAir (Missing %)                                              |
|---------------------------------------------------------|----------------------|---------------------------------------------------------------------|-----------------------------------------------------------|---------------------------------------|---------------------------------------------------------------------|---------------------------------------------------------------------|
|                                                         | 10%                  | 20%                                                                 | 50%                                                       | 10%                                   | 20%                                                                 | 50%                                                                 |
| CSDI [Tashiro et al., 2021] GP-VAE [Fortuin et al., DiT | 0.0615 (±0.0097)     | 0.0698 (±0.0106) 0.2138 (±0.0067) 0.0606 (±0.0076) 0.0588 (±0.0081) | 0.0797 (±0.0106) 0.4249 (±0.0127) 0.0684 0.0711 (±0.0092) | 0.4196 (±0.1726) 0.4096 0.3683 0.3428 | 0.3926 (±0.0790) 0.4777 (±0.0179) 0.4025 (±0.0424) 0.3864 (±0.0403) | 0.4534 (±0.0379) 0.7017 (±0.0189) 0.4255 (±0.0670) 0.4229 (±0.0539) |
| 2020]                                                   | 0.1567 (±0.0094)     |                                                                     |                                                           | (±0.0202)                             |                                                                     |                                                                     |
|                                                         | 0.0534 (±0.0063)     |                                                                     | (±0.0070)                                                 | (±0.0351)                             |                                                                     |                                                                     |
| DiT w/ mixed-masking strategy                           | 0.0502 (±0.0055)     |                                                                     |                                                           | (±0.0275)                             |                                                                     |                                                                     |

Table 8: Time Series Imputation MRE Results

<!-- image -->

| Model                         | ETT_m1 (Missing %)   | ETT_m1 (Missing %)      | ETT_m1 (Missing %)   | BeijingAir (Missing %)   | BeijingAir (Missing %)   | BeijingAir (Missing %)   |
|-------------------------------|----------------------|-------------------------|----------------------|--------------------------|--------------------------|--------------------------|
|                               | 10%                  | 20%                     | 50%                  | 10%                      | 20%                      | 50%                      |
| CSDI [Tashiro et al., 2021]   | 0.1706 (±0.0123)     | 0.1808 (±0.0135) 0.3882 | 0.1938 (±0.0114)     | 0.2380 (±0.0186)         | 0.2420 (±0.0174)         | 0.2929 (±0.0159)         |
| GP-VAE [Fortuin et al., 2020] | 0.3285 (±0.0091)     | (±0.0052)               | 0.5478 (±0.0085)     | 0.5598 (±0.0118)         | 0.5917 (±0.0107)         | 0.7042 (±0.0072)         |
| DiT                           | 0.1592 (±0.0084)     | 0.1701 (±0.0102)        | 0.1825 (±0.0094)     | 0.2154 (±0.0125)         | 0.2578 (±0.0375)         | 0.3073 (±0.0241)         |