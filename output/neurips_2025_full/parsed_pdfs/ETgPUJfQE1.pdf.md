## Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks

Xuan Tang 1 Han Zhang 1 Yuan Cao 1 Difan Zou 1 , 2

1

School of Computing and Data Science, The University of Hong Kong 2 Institute of Data Science, The University of Hong Kong {xuantang8,hzhang23}@connect.hku.hk , {yuancao,dzou}@hku.hk

## Abstract

Adam is a popular and widely used adaptive gradient method in deep learning, which has also received tremendous focus in theoretical research. However, most existing theoretical work primarily analyzes its full-batch version, which differs fundamentally from the stochastic variant used in practice. Unlike SGD, stochastic Adam does not converge to its full-batch counterpart even with infinitesimal learning rates. We present the first theoretical characterization of how batch size affects Adam's generalization, analyzing two-layer over-parameterized CNNs on image data. Our results reveal that while both Adam and AdamW with proper weight decay λ converge to poor test error solutions, their mini-batch variants can achieve near-zero test error. We further prove Adam has a strictly smaller effective weight decay bound than AdamW, theoretically explaining why Adam requires more sensitive λ tuning. Extensive experiments validate our findings, demonstrating the critical role of batch size and weight decay in Adam's generalization performance.

## 1 Introduction

Adaptive gradient methods, such as Adam (Kingma and Ba, 2015) and its variant AdamW (Loshchilov and Hutter, 2019), have emerged as widely adopted optimizers for training deep learning models across diverse tasks (He et al., 2016; Ma and Hovy, 2016). More recently, Adam and its variants have also been used to train large language models (LLMs) like GPT (Brown et al., 2020), LLaMA (Touvron et al., 2023), and Deepseek (Bi et al., 2024). In practice, Adam is known for its fast convergence during training, yet its generalization performance varies significantly depending on the task. Despite its empirical success, the theoretical understanding of Adam remains incomplete, especially regarding its generalization performance.

Recent theoretical work has sought to analyze the task-dependent behavior of Adam and compare it with other optimizers like gradient descent (GD). For instance, Wilson et al. (2017) demonstrated that adaptive methods like Adam exhibit poor generalization on linear models, while GD and stochastic gradient descent (SGD) can achieve zero test error. Further, Zhou et al. (2020) theoretically characterized the generalization gap between SGD and Adam through local convergence analysis, though their work did not account for neural network architectures or test error behavior. Other studies have focused on the implicit bias of adaptive methods: Wang et al. (2022) analyzed momentum's role in generalization, proving that GD with momentum and its adaptive variants converge to the ℓ 2 max-margin solution; Xie and Li (2024) showed that full-batch AdamW converges only to a KKT point under an ℓ ∞ norm constraint; and Zhang et al. (2024) established Adam's convergence to a maximum ℓ ∞ -margin classifier in linear logistic regression with separable data. In nonconvex settings, Zou et al. (2023b) revealed that full-batch Adam and GD converge to distinct solutions with differing generalization performance, which shows that even with weight decay, Adam fails to achieve low test error in overparameterized CNNs. Following the nonconvex analysis of Adam vs.

GD by Zou et al. (2023b), Li et al. (2025) show that Sign Gradient Descent-a sign-only surrogate for Adam (Balles and Hennig, 2018; Bernstein et al., 2018)-achieves fast convergence but poor generalization when training two-layer Transformers.

While existing theoretical analyses have provided valuable insights into the behavior of full-batch Adam, these results may not fully capture the characteristics of stochastic gradient Adam commonly used in practice. Notably, although stochastic gradient descent (SGD) and full-batch GD exhibit similar training dynamics in expectation (Bottou, 2012), stochastic gradient Adam demonstrates fundamentally different behavior from its full-batch counterpart-a distinction that persists even with vanishingly small learning rates. This gap raises important questions about how stochastic gradient Adam, particularly with small batch sizes, affects model generalization, an aspect that remains largely unexplored in current literature.

Motivated by this, in this paper, we investigate how the generalization of mini-batch Adam and AdamW differs from that of large-batch Adam. We analyze the convergence and generalization of Adam (and AdamW) with different batch sizes on two-layer over-parameterized convolutional neural networks (CNNs) for an image data model. This analysis follows the settings outlined in the recent study of full-batch Adam in Zou et al. (2023b). We also compare the sensitivity of the weight decay parameters λ for effective weight decay in Adam and AdamW.

The main contributions of this paper are summarized as follows.

- Theorem 4.1 and 4.4 rigorously prove that in the large-batch regime, both Adam and AdamW converge to solutions with poor test error in nonconvex settings, even with proper weight decay. This extends prior results for full-batch Adam to AdamW, showing that adaptive methods inherently overfit noise in low-stochasticity training. Real-world data experiments in Figure 1 demonstrate that large-batch Adam and AdamW suffer drastic test error increases, while synthetic experiments in Appendix D confirm this failure stems from noise-dominated solutions.
- For mini-batch training, theorem 4.2 and 4.5 prove that stochastic Adam and AdamW achieve nearzero test error in nonconvex settings with appropriate weight decay. The key mechanism is twofold: (i) stochastic gradients implicitly regularize the optimization trajectory by slowing noise fitting while preserving feature learning dynamics, preventing Adam from overfitting noise patches; (ii) weight decay explicitly suppresses residual noise components. This synergy ensures convergence to solutions dominated by true features. Real-world data experiments in Figure 1 demonstrate that mini-batch Adam and AdamW significantly improve test performance, with synthetic-data experiments in Appendix D further validating our theoretical insights. Moreover, under constant β 1 , β 2 hyperparameters, we prove stochastic Adam and AdamW can be rigorously approximated by SignSGD (Bernstein et al., 2018) and SignSGDW (with decoupled decay) respectively. This extends the known full-batch Adam → SignGD correspondence to stochastic regimes-a crucial advancement given mini-batch noise fundamentally modifies approximation dynamics. Our analysis in Appendix C reveals this approximation holds precisely when gradient magnitudes dominate optimization noise (e.g., | g ( t ) t,j,r [ k ] | ≥ ˜ Θ( η ) where η is the learning rate).
- Corollary 4.3 and 4.6 derive distinct theoretical upper bounds for weight decay parameters in nonconvex settings: Adam permits a strictly smaller maximum effective λ than AdamW. This arises because Adam's adaptive gradient normalization amplifies the effective impact of weight decay, causing excessive regularization to destabilize updates. In contrast, AdamW's decoupled weight decay mechanism avoids this issue. Experiments in Figure 2 validate that exceeding Adam's upper bound (e.g., λ &gt; 0 . 05 ) leads to catastrophic test error increases, while AdamW tolerates much larger λ values (e.g., λ = 0 . 5 ) without significant performance degradation. This demonstrates that the interplay between batch size and weight decay is critical: mini-batch training enables effective regularization, but Adam's narrow tolerance demands precise λ calibration.

The rest of paper is organized as follows. Section 2 discusses the works that are most closely related to this paper. Section 3 describes the problem settings. Section 4 presents the main results of this paper. Section 5 provides the proof outline of stochastic gradient Adam. Section 6 concludes this paper and discusses future research directions. Additional experiments and all experimental details can be found in Appendix D. All proofs are provided in the remaining appendices (Appendix A- C).

Notation. Scalars are denoted by lowercase letters x, y, . . . , vectors by bold lowercase letters x , . . . , and matrices by bold uppercase letters A , . . . . For any integer d ≥ 1 , denote the set [ d ] = { 1 , . . . , d } .

̸

̸

For x ∈ R , define [ x ] + = max { x, 0 } and sgn( x ) = x/ | x | for x = 0 , sgn(0) = 0 . For x = ( x 1 , . . . , x d ) ⊤ ∈ R d , define ∥ x ∥ p = ( ∑ d i =1 | x i | p ) 1 /p ( p ≥ 1 ) and supp ( x ) = { i ∈ [ d ] : x i = 0 } . For real sequences { a n } , { b n } , denote a n = O ( b n ) if there exist C, N &gt; 0 , s.t. | a n | ≤ C | b n | , ∀ n ≥ N ; denote a n = Ω( b n ) if b n = O ( a n ) ; a n = Θ( b n ) if both O ( b n ) and Ω( b n ) hold; denote a n = o ( b n ) if for any C &gt; 0 , there exist N &gt; 0 , s.t. | a n | &lt; C | b n | , ∀ n ≥ N ; and denote a n = ω ( b n ) if b n = o ( a n ) . We write ˜ O ( · ) , ˜ Ω( · ) , ˜ Θ( · ) to suppress logarithmic factors, a n = poly( b n ) if a n = Θ( b D n ) for some D &gt; 0 , and a n = polylog( b n ) if a n = poly(log b n ) .

## 2 Related Work

Adaptive Optimization Methods. There are a series of papers on adaptive gradient methods, including AdaGrad (Duchi et al., 2011), Adam (Kingma and Ba, 2015), AdamW (Loshchilov and Hutter, 2019), and second-order information methods (Yao et al., 2021; Liu et al., 2024). The convergence of Adam and related methods has been analyzed in a line of papers under various conditions (Chen et al., 2019; Guo et al., 2021; Défossez et al., 2022). However, some work presented the possible case where Adam fails to converge to an optimal solution even in simple one-dimensional convex settings (Reddi et al., 2018). The generalization performance of Adam has been investigated and compared with that of gradient descent in Wilson et al. (2017); Zhou et al. (2020); Zou et al. (2023b). To better understand the performance of Adam, Bernstein et al. (2018, 2019); Kunstner et al. (2023) analyzed its similarity with signGD. Similar works have also been done for AdamW (Xie and Li, 2024). Loshchilov and Hutter (2019) demonstrated that improper use of weight decay in Adam could lead to poor generalization performance, and proposed the AdamW that improves generalization in comparison to Adam. Recent work has also highlighted the role of weight decay in modern deep learning setups, showing its impact on optimization dynamics and generalization (D'Angelo et al., 2024). While L 2 regularization and weight decay are equivalent for standard SGD and GD (with rescaled by learning rate), that is not the case for adaptive methods like stochastic gradient Adam and full-batch Adam (Loshchilov and Hutter, 2019; Zhang et al., 2019; Zhuang et al., 2022). However, the true reason why Adam with weight decay fails to improve the generalization remains unclear. Therefore, the current understanding of how batch size and weight decay influence the generalization performance of Adam is still relatively limited.

Implicit bias. Implicit bias refers to the tendency of machine learning algorithms to favor certain solutions. This phenomenon has also been studied in neural networks theoretically to understand how they generalize and converge to solutions. Lyu and Li (2019) and Ji and Telgarsky (2020) studied the implicit bias of gradient descent on the homogeneous neural networks. Kunin et al. (2023) extended the results to a wider class of networks with varying degree of homogeneity. Cai et al. (2024) focused on the large stepsize gradient descent on two-layer non-homogeneous networks. Frei et al. (2022) analyzed the implicit bias of gradient flow in two-layer fully-connected neural networks with leaky ReLU activations for nearly-orthogonal data. Kou et al. (2024) extended this results and analyzed the implicit bias of gradient descent on similar settings. For Adam and AdamW, the implicit bias of Adam have been analyzed in Wang et al. (2021, 2022); Zhang et al. (2024), and the implicit bias of AdamW has been analyzed in (Xie and Li, 2024). Recently, Cattaneo et al. (2024) showed that Adam penalizes the ℓ 1 -norm of perturbed gradients, favoring flat minima. Our work complements this view by analyzing, in a discrete-time feature learning setting, how batch size and weight decay jointly regulate noise suppression and generalization.

Feature learning. There are a series of papers that studied the feature learning theory in neural networks. Allen-Zhu and Li (2020) investigated the feature learning of ensemble methods and knowledge distillation in deep learning when applied to data with multi-view features. Cao et al. (2022) examined the benign overfitting in the supervised learning of two-layer convolutional neural networks, and proved that under certain conditions on signal-to-noise ratio (SNR), arbitrary small training and test loss can be achieved. Zou et al. (2023b) compared the feature learning of full-batch Adam and GD on two-layer convolutional neural networks. It demonstrated that GD learns the features, but full-batch Adam, even with proper regularization, may still fail. Some works have studied the feature learning of contrastive learning method (Zhang and Cao, 2024), federated learning (Huang et al., 2024b) on two-layer convolutional neural networks, and multi-modal contrastive learning on single-layer ReLU networks (Huang et al., 2024a). Additionally, some papers have analyzed feature learning on other architectures, such as transformers (Jelassi et al., 2022; Li et al.,

2025), and diffusion models (Han et al., 2025a,b); and other training configurations (Zou et al., 2023a; Lu et al., 2024). Unlike the aforementioned works, this paper focuses on the feature learning of Adam and AdamW algorithms with different batch sizes on the two-layer convolutional neural networks.

## 3 Problem Setup

In this paper, we train the two-layer convolutional neural network (CNN) with Adam and AdamW on the training dataset S := { ( x i , y i ) } n i =1 of size n , which is generated from a data model D . In this section, we introduce the data model D , the two-layer CNN model, and the training details of two algorithms (Adam and AdamW) analyzed in this paper.

Data Model. We adopt the feature-noise patch concatenation framework from Definition 3.1, aligning with previous studies (Allen-Zhu and Li, 2020; Cao et al., 2022; Jelassi et al., 2022; Zou et al., 2023b; Huang et al., 2024b,a; Zhang and Cao, 2024; Li et al., 2025; Han et al., 2025a).

Definition 3.1. Let each data point ( x , y ) consist of a feature vector x ∈ R 2 d and a label y ∈ {-1 , 1 } . The data is generated as follows:

<!-- formula-not-decoded -->

where x 1 and x 2 represent two distinct feature patches. One of these patches corresponds to the signal patch and consists of a feature vector y · v , where v ∈ R d is assumed to be a sparse vector, specifically 1 -sparse. The other patch represents the noise patch and is a noise vector denoted by ξ . Without loss of generality, we assume v = [1 , 0 , . . . , 0] ⊤ . The data is generated from the following distribution D :

1. The label y is generated as a Rademacher random variable with y ∈ {-1 , +1 } .
2. Randomly select s coordinates from the set [ d ] \{ 1 } with equal probability. This selection is represented by a binary vector s ∈ { 0 , 1 } d . Then generate ξ from the Gaussian distribution N ( 0 , σ 2 p I d ) and apply the masking operation such that ξ = ξ ⊙ s , where ⊙ denotes element-wise multiplication. Finally, add feature noise to the vector ξ by updating it as ξ = ξ -αy v , where α ∈ (0 , 1) controls the strength of the feature noise.
3. One of the two patches x 1 , x 2 is randomly selected and is assigned as y · v , representing the signal patch, while the other patch is assigned as ξ , representing the noise patch.

<!-- formula-not-decoded -->

The data model formalizes image classification dynamics where localized label-relevant features coexist with global noise-aligning with CNN behaviors: sparse mid-layer activations (Papyan et al., 2017) vs. non-informative regions as independent noise (Yang, 2019). By isolating 1 -sparse feature and s -sparse noise patches, we distill the feature learning vs. noise memorization interplay. Though our analysis uses a simplified single feature/noise patch model for clarity, the results can be extended to broader settings (e.g., multi-patch or denser features/noises) by assuming sub-Gaussian noise and using concentration inequalities (e.g., Bernstein bounds) to control overlapping or structured perturbations, with similar qualitative behavior expected as long as the total noise remains controlled.

Two-layer CNN model. We define the two-layer CNN considered in this paper as follows.

Definition 3.2. Given the data ( x , y ) ∼ D and the activation function σ ( x ) = [ x ] q + with q ≥ 3 , the j -th output of the neural network F with width m is

<!-- formula-not-decoded -->

where w j,r is the weight at the r -th neuron and initialized from Gaussian distribution N ( 0 , σ 2 0 I d ) . In this paper, we assume j ∈ {± 1 } for clarity, ensuring the logit index matches the data label. Additionally, we also assume m = polylog( n ) and σ 0 = Θ( d -1 / 4 ) .

Training algorithm. We investigate the behavior of stochastic Adam and AdamW, starting from same initializations and training on the same dataset S = { ( x i , y i ) } n i =1 . The loss function for each data point ( x i , y i ) is denoted as L i ( W ) = -log e Fy i ( W , x i ) ∑ j ∈{-1 , 1 } e F j ( W , x i ) .

For stochastic Adam and AdamW , the CNN model is trained by minimizing the empirical loss function

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ∥ · ∥ F denotes the Frobenius norm, λ is the weight decay regularization of Adam . Therefore, the stochastic gradient g ( t ) t,j,r can be calculated as

<!-- formula-not-decoded -->

where the subscript t of g ( t ) t,j,r represents the batch I t at the t -th iteration and the superscript t of g ( t ) t,j,r represents the model W ( t ) at the t -th iteration. Herein, we emphasize a fundamental distinction: Adam's stochastic gradients inherently incorporate weight decay regularization, whereas AdamW's gradients remain regularization-free-a deliberate design choice to prevent momentumbased normalization from destabilizing regularization effects (Loshchilov and Hutter, 2019). This architectural distinction, also analytically demonstrated in our proof, crucially impacts the training process. The momentum estimates m ( t ) j,r , v ( t ) j,r of Adam/AdamW are updated as follows

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where β 1 , β 2 are the hyperparameters of Adam/AdamW and we initialize m (0) j,r = v (0) j,r = 0 . Finally, the update rule of stochastic Adam/AdamW for model W can be formulated as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where η is the learning rate, ϵ = Θ( λη ) is stability constant and λ is the decoupled weight decay parameter of AdamW . In particular, in (3.4), (3.5) and (3.6), the square ( · ) 2 , square root √ · , and division · / · all denote entry-wise calculations. The details of gradient calculation and its expansion can be found in Appendix A.

## 4 Main Results

In this section, we present the main results of our study. We begin by introducing the primary metric used to evaluate generalization performance: the classification error rate.

Given training dataset S = { ( x i , y i ) } n i =1 generated from data model D in Definition 3.1. We define the training error err S ( W ) and test error err D ( W ) of model W as follows,

<!-- formula-not-decoded -->

Figure 1: Test error vs. batch size for VGG16 and ResNet18 on CIFAR-10.

<!-- image -->

<!-- formula-not-decoded -->

While theoretical analyses often prioritize mathematically tractable surrogate losses (e.g., crossentropy, hinge loss), classification error rate remains the most direct and practical performance metric. Unlike continuously approximated surrogate losses, error rate directly quantifies discrete misclassification events, better reflecting models' true decision-making ability in classification tasks.

## 4.1 Theoretical Results for Adam

The following Theorem 4.1 characterizes the behavior of Adam in the large-batch regime.

Theorem 4.1 (Large-batch Adam) . Suppose η = 1 poly( n ) and λ satisfies 0 &lt; λ = o ( σ q -2 0 σ p n ) , we train our CNN model in Definition 3.2 on loss function (3.1) for T = poly( n ) η epochs using Adam (3.5) with batch size B satisfies n B = Θ(1) . Then with probability at least 1 -n -1 , we have

- The training error is zero: err S ( W ( T ) ) = 0 .
- The test error is high: err D ( W ( T ) ) ≥ 1 2 -o (1) .

Theorem 4.1 extends Zou et al. (2023b)'s full-batch analysis to large-batch regimes ( B = Θ( n ) ). Basically, it states that under the nearly same data model in Zou et al. (2023b), large-batch Adam cannot effectively learn the feature vector from the training dataset, and finally attains a nearly 0.5 test accuracy, despite its perfect fitting on the training data points.

In stark contrast, we further provide Theorem 4.2, which proves that stochastic gradient Adam with a smaller batch size can achieve good generalization performance.

Theorem 4.2 (Mini-batch Adam) . Suppose η = 1 poly( n ) and 0 &lt; λ = o ( σ q -2 0 σ p n ) , we train our CNN model in Definition 3.2 on loss (3.1) for T = poly( n ) η epochs using stochastic Adam (3.5) with batch size B satisfies n B ≥ Θ(log ϵ -1 ) , where ϵ is the hyperparameter of Adam. Then with high probability at least 1 -n -1 , we have

- The training error is zero: err S ( W ( T ) ) = 0 .
- The test error is near-zero: err D ( W ( T ) ) = o (1) .

Our theoretical results demonstrate that mini-batch Adam achieves near-perfect test accuracy when the ratio n/B is large, significantly outperforming its large-batch counterpart which exhibits only random-guessing performance. This advantage can be attributed to three fundamental properties of stochastic Adam optimization: First, since the feature vector is shared across all data points, its learning remains robust regardless of batch size. In contrast, noise vectors are data-specific and vary across different samples. When using mini-batches, only a subset of noise vectors is exposed during each update, creating an inherent asymmetry in learning dynamics. More importantly, Adam's coordinate-wise normalization amplifies this effect: it maintains consistent learning rates for shared features while substantially slowing down noise memorization. This selective suppression of

noise learning explains the superior generalization performance of mini-batch Adam compared to large-batch implementations.

Besides the results on the generalization performance, we further deliver the following corollary, which states the feasible range of the weight decay in Adam. Theorems 4.1 and 4.2 directly yield:

Corollary 4.3 (Effective weight decay in Adam) . Suppose the same conditions as in Theorem 4.1 and 4.2. If λ = ω ( σ q -2 0 ) , then with probability at least 1 -n -1 , training stuck at the initialization.

This corollary provides a theoretical upper bound on the effective weight decay that allows Adam to successfully train models, aligning well with previous empirical observations that Adam typically performs better with small weight decay values compared to AdamW (Loshchilov and Hutter, 2019). This sensitivity arises because weight decay regularization is implicitly entangled with the normalization step in Adam, i.e., when the gradient of weight decay is greater than that of the cross-entropy loss, it will fully dominate the Adam update. In the next subsection, we will show that weight decay will exhibit a different behavior in AdamW, leading to a different feasible range for λ .

## 4.2 Theoretical Results for AdamW

We first establish the theoretical results of Adamw in Theorem 4.4 under large-batch training.

Theorem 4.4 (Large-batch AdamW) . Suppose η = 1 poly( n ) , λ = ˜ Ω( B 2 n ∧ 1) and λ = ˜ O (1) , we train our CNN model in Definition 3.2 on loss function (3.2) for T = poly( n ) η epochs using AdamW (3.6) with batch size B satisfies n B = Θ(1) or n B = o ( sσ p ) . Then with probability at least 1 -n -1 , we have

- The training error is zero: err S ( W ( T ) ) = 0 .
- The test error is high: err D ( W ( T ) ) ≥ 1 2 -o (1) .

The results for large-batch AdamW closely resemble those of large-batch Adam in Theorem 4.1: the learned model consistently exhibit test errors of at least 1 / 2 -o (1) , performing no better than random guessing. This phenomenon arises from the training dynamics in the early stages, where the influence of weight decay is minimal. As a result, both Adam and AdamW exhibit similar behavior, tending to fit noise. By the time the decoupled weight decay in AdamW begins to take effect, the model has already overfit to the feature noise -αy v . The weight decay then guides the model toward nearby local minima, effectively preserving the previously memorized noise. Then at test time, this overfitting to feature noise causes the model to predict labels that are systematically misaligned with the true labels, leading to test performance that is no better than random guessing.

In contrast to the results for large-batch in Theorem 4.4, the following Theorem 4.5 characterizes the generalization ability of mini-batch AdamW.

Theorem 4.5 (Mini-batch AdamW) . Suppose η = 1 poly( n ) , λ = ˜ Ω( B 2 n ∧ 1) and λ = ˜ O (1) , we train our CNN model in Definition 3.2 on loss function (3.2) for T = poly( n ) η epochs using stochastic AdamW (3.6) with batch size B satisfies n B ≥ Θ(log ϵ -1 ) and n B = ω ( sσ p ∨ n 1 / 2 ) . Then with probability at least 1 -n -1 , we have

- The training error is zero: err S ( W ( T ) ) = 0 .
- The test error is near-zero: err D ( W ( T ) ) = o (1) .

Under mini-batch training, AdamW achieves near-zero test error with partial similarity to Adam. The extended iterations per epoch slow early-stage noise overfitting (notably for s -sparse noise). However, AdamW's decoupled weight decay penalizes weights independently of gradients, exerting significant regularization only in later phases to converge toward generalizable minima. Thus, AdamW can leverage a much larger λ than Adam, which is shown as follows:

Corollary 4.6. Regarding the effective weight decay coefficients of Adam and AdamW for achieving good generalization performance, we have λ Adam ∼ σ q -2 0 ≪ B 2 n ∧ 1 ∼ λ AdamW .

Corollary 4.6 reveals a fundamental gap between the effective weight decay regimes of Adam and AdamW, consistent with empirical observations. For Adam , the admissible weight decay λ Adam

Figure 2: Test error vs. weight decay (batch size = 16), comparing Adam and AdamW on each model.

<!-- image -->

is extremely small, bounded above by an initialization-dependent term σ q -2 0 . Consequently, even moderate weight decay values can destabilize training due to the entanglement between gradient updates and weight decay, leading to suboptimal generalization. In contrast, AdamW decouples weight decay from the gradient update, applying regularization directly on the weights. As a result, it requires a larger weight decay to effectively suppress noise overfitting and exhibits greater robustness. The lower bound ˜ Ω( B 2 n ∧ 1) serves as a sufficient condition ensuring that weight decay is strong enough to prevent noise amplification. This robustness is reflected in its much broader effective range-from ˜ Ω( B 2 n ∧ 1) up to ˜ O (1) -representing a large, constant-order window independent of initialization. This theoretical separation explains the empirical fact that Adam requires delicate tuning and is highly sensitive to weight decay, whereas AdamW is considerably more robust and easier to tune Loshchilov and Hutter (2019). Our experiments (Figure 2) further corroborate this prediction.

Experiments. We train VGG16 and ResNet18 on CIFAR-10 with Adam ( λ = 5 × 10 -4 ) and AdamW( λ = 1 × 10 -2 ), selecting the optimal learning rate from { 5 × 10 -4 , 1 × 10 -4 , 1 × 10 -5 } and varying the batch size to measure test error (Figure 1). Both optimizers exhibit a sharp performance degradation once the batch size exceeds a critical threshold, in line with theoretical predictions of large-batch generalization collapse (Theorem 4.1 and 4.4). Separately, at a fixed batch size of 16 (Figure 2), we find that Adam's error spikes for λ &gt; 0 . 05 , whereas AdamW remains robust even up to λ = 0 . 5 , highlighting the benefit of decoupled weight decay in adaptive optimizers. This observation is consistent with our theoretical analysis (Corollary 4.3 and 4.6). Additional experiments, including the dynamics of feature learning and noise memorization (Figures 3, 4), sensitivity to weight decay (Figures 5, 6), error bars across random seeds and momentum parameters (Figures 7, 8, 9, 10), and large-scale vision experiments with ResNet-50 on ImageNet-1K (Figures 11, 12), all corroborate our theoretical findings. Complete experimental details and results are provided in Appendix D.

## 5 Proof Outline of the Main Results

In this section, we mainly outline the proof sketch for the theorem 4.2 in Section 4. Proof sketches for remaining theorems are deferred to Appendix B. Following the two-stage analysis framework of Cao et al. (2022); Zou et al. (2023b), we decompose the proof into two distinct stages:

Stage I: Pattern Learning. During the initial phase of training, the effect of regularization is negligible. The model operates in an underfitting regime, where it rapidly learns dominant patterns in the training data, leading to improved empirical performance on test error.

Stage II: Regularization. As training progresses, the model's classification accuracy on the training set approaches convergence, resulting in diminished gradient magnitudes. At this stage, regularization dominates the optimization dynamics, steering the model converge to a local minima. Due to the nonconvex nature of the loss landscape, the model retains the patterns acquired during the pattern learning stage.

Furthermore, motivated by the behavioral similarity between Adam and SignGD when the learning rate is sufficiently small or β 1 , β 2 approach zero (Balles and Hennig, 2018; Bernstein et al., 2018),

we present results for SignSGD. We subsequently extend these results to stochastic Adam, which provided in Appendix C. The update rules for SignSGD are given as follows:

<!-- formula-not-decoded -->

where g ( t ) t,j,r in (5.1) denotes the stochastic gradient of (3.1). The detailed update rules of Adam with the SignSGD approximation are provided in Eqs. (B.3) and (B.4), while those of AdamW are given in Eqs. (B.5) and (B.6).

Next, following the framework of feature learning (Allen-Zhu and Li, 2020; Cao et al., 2022; Zou et al., 2023b; Han et al., 2025a), we primarily focus on two key quantities: 1) Feature Learning ⟨ w j,r , j v ⟩ : This term captures the alignment between the learned weight vector w j,r and the true feature direction j v , reflecting the model's ability to extract meaningful latent structures from the data. 2) Noise Memorization ⟨ w y i ,r , ξ i ⟩ : This term measures the correlation between w y i ,r and the noise patch ξ i of individual samples, characterizing the extent to which the model overfits to stochastic perturbations or idiosyncrasies in the training set. This decomposition allows us to separately analyze the model's generalization behavior (driven by feature learning) and its memorization capacity (influenced by noise fitting).

We first clarify that, although the sketch appears straightforward, the underlying process is non-trivial. Numerous intricate and interesting details arise, which we elaborate on in the proof presented in Appendix C.

Webegin by characterizing the dynamics of feature learning and noise memorization under large-batch training, to facilitate a comparative analysis with mini-batch regimes, as formalized in Lemma 5.1.

Large-batch Adam. We consider n B = Θ(1) , which is the large-batch setting.

Lemma 5.1. Given the training dataset S , if n B = Θ(1) , η = 1 / poly( d ) and 0 &lt; λ = o ( σ q -2 0 σ p /n ) , then for any t ≤ T 0 with T 0 = ˜ O ( 1 ηsσ p ) and any i ∈ [ n ] ,

<!-- formula-not-decoded -->

We observe that Lemma 5.1 is identical to Lemma 5.2 in (Zou et al., 2023b), allowing us to directly extend their full-batch Adam results to the large-batch setting. The remainder of the proof follows identically, as the underlying theoretical machinery remains unchanged under this batch size scaling n B = Θ(1) . We observe that under large-batch setting, the optimization dynamics of Adam closely resemble those of the full-batch setting. This similarity arises because the algorithm traverses the entire dataset within few iterations, resulting in nearly identical momentum estimates and, consequently, comparable training dynamics between large-batch and full-batch regimes.

We next consider the mini-batch setting, which yields conclusions that differ fundamentally from those in the large-batch setting.

Mini-batch Adam. We consider n B ≥ Θ(log ϵ -1 ) , which is the mini-batch setting.

Lemma 5.2 (Stage I) . Given the training dataset S , if n B ≥ Θ(log ϵ -1 ) , η = 1 / poly( d ) and 0 &lt; λ = o ( σ q -2 0 σ p /n ) , then for any t ≤ T 0 with T 0 = ˜ O ( 1 ηsσ p ) and any i ∈ [ n ] ,

<!-- formula-not-decoded -->

Compared to Lemma 5.1, Lemma 5.2 reveals fundamentally different optimization dynamics. Specifically, feature learning progressively increases throughout Stage I , whereas noise memorization remains suppressed near the initialization. The key distinction from the large-batch setting lies in the fact that, under the mini-batch regime, traversing the entire dataset requires many iterations. Since noise is sparse and uncorrelated across samples while features are dense and shared, feature learning can proceed effectively during the early training phase without being hindered by weight decay regularization. In contrast, noise memorization is strongly suppressed by weight decay due to its sparsity. As training progresses, the momentum estimates in Adam gradually forget the gradient contributions from noise, allowing weight decay to dominate. As a result, recently acquired noise

memorization is continually erased, keeping noise-related parameters close to their initialization throughout Stage I .

In the following lemma 5.3, we show that the patterns learned by the model during Stage I are preserved in Stage II , due to the nature of our non-convex optimization landscape.

Lemma 5.3 (Stage II) . Suppose the same conditions hold as in Lemma 5.2. For t &gt; T 0 , j ∈ {± 1 } , r ∈ [ m ] , i ∈ [ n ] , let r ∗ = argmax r ∈ [ m ] ⟨ w ( t ) j,r , j v ⟩ , then ⟨ w ( t ) j,r ∗ , j v ⟩ = ˜ Θ(1) and ⟨ w ( t ) y i ,r , ξ i ⟩ ≤ ˜ Θ( ηsσ p ) .

Given Lemma 5.2 and Lemma 5.3, we can characterize the convergence rate of Adam as follows. Lemma 5.4 (Convergence) . Suppose the same conditions hold as in Lemma 5.2 and 5.3, if the step size η = O ( d -1 2 ) , then for any t ,

<!-- formula-not-decoded -->

Combine Lemma 5.3 and Lemma 5.4, we observe that the model successfully learns the true features and eventually converges to a local minimum, retaining strong generalization performance.

## 6 Conclusion and Limitation

In this work, we theoretically and empirically analyze the impact of varying batch sizes and weight decay parameters on the generalization of Adam and AdamW when learning two-layer CNNs. Our results demonstrate that large-batch Adam and AdamW inherently overfit noise-dominated solutions even with weight decay, while their mini-batch counterparts achieve strong generalization through the synergy of implicit stochastic gradient regularization and explicit weight decay. Furthermore, we establish that Adam's adaptive gradient normalization imposes stricter constraints on weight decay parameters compared to AdamW, necessitating precise calibration for stable optimization.

While our theoretical framework provides insights into the interplay between batch size, weight decay, and generalization, several limitations highlight critical directions for future research. First, the current analysis is restricted to two-layer networks. Extending the results to deeper architectures and investigating how batch size influences the dynamics of hierarchical feature learning presents a promising direction. Second, our work focuses on image data, and an important direction is to extend the analysis to domains with fundamentally different data structures, such as NLP, where batch size and weight decay may impact model performance through different mechanisms. Finally, other critical hyperparameters, such as momentum, learning rate schedules, and gradient clipping, are not considered in our analysis, and some modern vision architectures succeed with large batches (Liu et al., 2022, 2023; Chen et al., 2024) despite our theoretical predictions, suggesting that additional factors like architectural design and normalization may play a significant role.

## Acknowledgments

We would like to thank the anonymous reviewers and area chairs for their helpful comments. Xuan Tang and Difan Zou acknowledge the support from NSFC 62306252, Hong Kong ECS award 27309624, Guangdong NSF 2024A1515012444, and the central fund from HKU IDS. Yuan Cao is partially supported by NSFC 12301657 and Hong Kong ECS award 27308624.

## References

- ALLEN-ZHU, Z. and LI, Y. (2020). Towards understanding ensemble, knowledge distillation and self-distillation in deep learning. arXiv preprint arXiv:2012.09816 .
- BALLES, L. and HENNIG, P. (2018). Dissecting adam: The sign, magnitude and variance of stochastic gradients. In International Conference on Machine Learning . PMLR.
- BERNSTEIN, J., WANG, Y.-X., AZIZZADENESHELI, K. and ANANDKUMAR, A. (2018). signsgd: Compressed optimisation for non-convex problems. In International Conference on Machine Learning . PMLR.

- BERNSTEIN, J., ZHAO, J., AZIZZADENESHELI, K. and ANANDKUMAR, A. (2019). signSGD with majority vote is communication efficient and fault tolerant. In International Conference on Learning Representations .
- BI, X., CHEN, D., CHEN, G., CHEN, S., DAI, D., DENG, C., DING, H., DONG, K., DU, Q., FU, Z. ET AL. (2024). Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954 .
- BOTTOU, L. (2012). Neural Networks: Tricks of the Trade: Second Edition . Springer Berlin Heidelberg.
- BROWN, T., MANN, B., RYDER, N., SUBBIAH, M., KAPLAN, J. D., DHARIWAL, P., NEELAKANTAN, A., SHYAM, P., SASTRY, G., ASKELL, A. ET AL. (2020). Language models are few-shot learners. Advances in neural information processing systems 33 1877-1901.
- CAI, Y., WU, J., MEI, S., LINDSEY, M. and BARTLETT, P. (2024). Large stepsize gradient descent for non-homogeneous two-layer networks: Margin improvement and fast optimization. Advances in Neural Information Processing Systems 37 71306-71351.
- CAO, Y., CHEN, Z., BELKIN, M. and GU, Q. (2022). Benign overfitting in two-layer convolutional neural networks. Advances in neural information processing systems 35 25237-25250.
- CATTANEO, M. D., KLUSOWSKI, J. M. and SHIGIDA, B. (2024). On the implicit bias of adam. In Proceedings of the 41st International Conference on Machine Learning , vol. 235 of Proceedings of Machine Learning Research . PMLR.
- CHEN, H., CHU, X., REN, Y., ZHAO, X. and HUANG, K. (2024). Pelk: Parameter-efficient large kernel convnets with peripheral convolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition .
- CHEN, X., LIU, S., SUN, R. and HONG, M. (2019). On the convergence of a class of adamtype algorithms for non-convex optimization. In 7th International Conference on Learning Representations, ICLR 2019 .
- D'ANGELO, F., ANDRIUSHCHENKO, M., VARRE, A. V. and FLAMMARION, N. (2024). Why do we need weight decay in modern deep learning? Advances in Neural Information Processing Systems 37 23191-23223.
- DÉFOSSEZ, A., BOTTOU, L., BACH, F. R. and USUNIER, N. (2022). A simple convergence proof of adam and adagrad. Trans. Mach. Learn. Res. .
- DUCHI, J., HAZAN, E. and SINGER, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research 12 .
- FREI, S., VARDI, G., BARTLETT, P., SREBRO, N. and HU, W. (2022). Implicit bias in leaky relu networks trained on high-dimensional data. In The Eleventh International Conference on Learning Representations .
- GUO, Z., XU, Y., YIN, W., JIN, R. and YANG, T. (2021). A novel convergence analysis for algorithms of the adam family. arXiv preprint arXiv:2112.03459 .
- HAN, A., HUANG, W., CAO, Y. and ZOU, D. (2025a). On the feature learning in diffusion models. In The Thirteenth International Conference on Learning Representations .
- HAN, Y., HAN, A., HUANG, W., LU, C. and ZOU, D. (2025b). Can diffusion models learn hidden inter-feature rules behind images? In Forty-second International Conference on Machine Learning .
- HE, K., ZHANG, X., REN, S. and SUN, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition .
- HUANG, W., HAN, A., CHEN, Y., CAO, Y., ZHIQIANG XU and SUZUKI, T. (2024a). On the comparison between multi-modal and single-modal contrastive learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems .

- HUANG, W., SHI, Y., CAI, Z. and SUZUKI, T. (2024b). Understanding convergence and generalization in federated learning through feature learning theory. In The Twelfth International Conference on Learning Representations .
- JELASSI, S., SANDER, M. and LI, Y. (2022). Vision transformers provably learn spatial structure. In Advances in Neural Information Processing Systems (S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho and A. Oh, eds.), vol. 35. Curran Associates, Inc.
- JI, Z. and TELGARSKY, M. (2020). Directional convergence and alignment in deep learning. Advances in Neural Information Processing Systems 33 17176-17186.
- KINGMA, D. P. and BA, J. (2015). Adam: A method for stochastic optimization. International Conference on Learning Representations .
- KOU, Y., CHEN, Z. and GU, Q. (2024). Implicit bias of gradient descent for two-layer relu and leaky relu networks on nearly-orthogonal data. Advances in Neural Information Processing Systems 36 .
- KUNIN, D., YAMAMURA, A., MA, C. and GANGULI, S. (2023). The asymmetric maximum margin bias of quasi-homogeneous neural networks. In The Eleventh International Conference on Learning Representations .
- KUNSTNER, F., CHEN, J., LAVINGTON, J. W. and SCHMIDT, M. (2023). Noise is not the main factor behind the gap between sgd and adam on transformers, but sign descent might be. In The Eleventh International Conference on Learning Representations .
- LI, B., HUANG, W., HAN, A., ZHOU, Z., SUZUKI, T., ZHU, J. and CHEN, J. (2025). On the optimization and generalization of two-layer transformers with sign gradient descent. In The Thirteenth International Conference on Learning Representations .
- LIU, H., LI, Z., HALL, D. L. W., LIANG, P. and MA, T. (2024). Sophia: A scalable stochastic second-order optimizer for language model pre-training. In The Twelfth International Conference on Learning Representations .
- LIU, S., CHEN, T., CHEN, X., CHEN, X., XIAO, Q., WU, B., KÄRKKÄINEN, T., PECHENIZKIY, M., MOCANU, D. C. and WANG, Z. (2023). More convnets in the 2020s: Scaling up kernels beyond 51x51 using sparsity. In ICLR .
- LIU, Z., MAO, H., WU, C.-Y., FEICHTENHOFER, C., DARRELL, T. and XIE, S. (2022). A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition .
- LOSHCHILOV, I. and HUTTER, F. (2019). Decoupled weight decay regularization. In International Conference on Learning Representations .
- LU, M., WU, B., YANG, X. and ZOU, D. (2024). Benign oscillation of stochastic gradient descent with large learning rates. In The Twelfth International Conference on Learning Representations (ICLR)(07/05/2024-11/05/2024, Vienna) .
- LYU, K. and LI, J. (2019). Gradient descent maximizes the margin of homogeneous neural networks. arXiv preprint arXiv:1906.05890 .
- MA, X. and HOVY, E. (2016). End-to-end sequence labeling via bi-directional lstm-cnns-crf. arXiv preprint arXiv:1603.01354 .
- PAPYAN, V., ROMANO, Y. and ELAD, M. (2017). Convolutional neural networks analyzed via convolutional sparse coding. Journal of Machine Learning Research 18 1-52.
- REDDI, S. J., KALE, S. and KUMAR, S. (2018). On the convergence of adam and beyond. In International Conference on Learning Representations .
- TOUVRON, H., LAVRIL, T., IZACARD, G., MARTINET, X., LACHAUX, M.-A., LACROIX, T., ROZIÈRE, B., GOYAL, N., HAMBRO, E., AZHAR, F. ET AL. (2023). LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 .

- WANG, B., MENG, Q., CHEN, W. and LIU, T.-Y. (2021). The implicit bias for adaptive optimization algorithms on homogeneous neural networks. In International Conference on Machine Learning . PMLR.
- WANG, B., MENG, Q., ZHANG, H., SUN, R., CHEN, W., MA, Z.-M. and LIU, T.-Y. (2022). Does momentum change the implicit regularization on separable data? Advances in Neural Information Processing Systems 35 26764-26776.
- WILSON, A. C., ROELOFS, R., STERN, M., SREBRO, N. and RECHT, B. (2017). The marginal value of adaptive gradient methods in machine learning. Advances in neural information processing systems 30 .
- XIE, S. and LI, Z. (2024). Implicit bias of adamw: ℓ ∞ -norm constrained optimization. In Forty-first International Conference on Machine Learning .
- YANG, G. (2019). Scaling limits of wide neural networks with weight sharing: Gaussian process behavior, gradient independence, and neural tangent kernel derivation. arXiv preprint arXiv:1902.04760 .
- YAO, Z., GHOLAMI, A., SHEN, S., MUSTAFA, M., KEUTZER, K. and MAHONEY, M. (2021). Adahessian: An adaptive second order optimizer for machine learning. In proceedings of the AAAI conference on artificial intelligence , vol. 35.
- ZHANG, C., ZOU, D. and CAO, Y. (2024). The implicit bias of adam on separable data. In The Thirty-eighth Annual Conference on Neural Information Processing Systems .
- ZHANG, G., WANG, C., XU, B. and GROSSE, R. (2019). Three mechanisms of weight decay regularization. In International Conference on Learning Representations .
- ZHANG, H. and CAO, Y. (2024). Understanding the benefits of simclr pre-training in two-layer convolutional neural networks. arXiv preprint arXiv:2409.18685 .
- ZHOU, P., FENG, J., MA, C., XIONG, C., HOI, S. C. H. and E, W. (2020). Towards theoretically understanding why sgd generalizes better than adam in deep learning. In Advances in Neural Information Processing Systems , vol. 33.
- ZHUANG, Z., LIU, M., CUTKOSKY, A. and ORABONA, F. (2022). Understanding adamw through proximal methods and scale-freeness. Transactions on Machine Learning Research .
- ZOU, D., CAO, Y., LI, Y. and GU, Q. (2023a). The benefits of mixup for feature learning. In International Conference on Machine Learning . PMLR.
- ZOU, D., CAO, Y., LI, Y. and GU, Q. (2023b). Understanding the generalization of adam in learning neural networks with proper regularization. In International Conference on Learning Representations .

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claims made in the abstract and introduction accurately reflect the paper's contributions and scope.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The paper discusses the limitations of the work in the conclusion section, which are also our future research directions.

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

## Answer: [Yes]

Justification: We provide the main results in the main paper, and state all the assumptions and the complete proof in Appendix. We also provide a proof outline in Section 5 in the main paper.

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

Justification: We provide all the configuration of the experiments in Appendix D.

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

Justification: We use VGG16, ResNet18, ResNet50 models and the CIFAR-10, ImageNet1K datasets, which are easily available on the Internet. All the experimental details are provided in Appendix D.

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

Justification: We provide all the experimental details in Appendix D.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We report the error bars in Figures 7, 8, 9, 10 in Appendix D.

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

Justification: We provide all the experimental details in Appendix D.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research follows the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: This paper is a theoretical analysis paper on the generalization of stochastic gradient Adam (AdamW) and does not have potential societal impact.

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

Justification: This paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: This paper does not use existing assets.

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

Justification: This paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

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

Justification: The core method development in this research does not involve LLMs.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Preliminaries

In this section, we give the asymptotic equations for all the parameters we use, some useful lemmas, and the gradient and weight update equations.

## A.1 Asymptotic Equations

First, we give the asymptotic equations for all the parameters we use in the proof.

Condition A.1.

Suppose that the following conditions on data model 3.1 hold,

1. The dimension d satisfies d = poly( n ) .
2. The number of noise coordinates s satisfies s = Θ ( d 1 / 2 n 2 ) .
3. The variance parameter σ 2 p of the noise vector satisfies σ 2 p = Θ ( 1 s · polylog( n ) ) .
4. The feature noise strength α satisfies α = Θ( σ p · polylog( n )) .

Condition A.2. Suppose that the following conditions on hyper-parameters hold,

1. The initialization variance of the model weights σ 2 0 satisfies σ 2 0 = Θ ( 1 d 1 / 2 ) .
2. The width of the network m satisfies m = polylog( n ) .
3. The learning rate η satisfies η = 1 poly( n ) .
4. The parameter ϵ in Stochastic Adam and Stochastic AdamW satisfies ϵ = Θ( λη ) .

Based on the parameter configuration, we claim that the following equations hold, which will be frequently used in the subsequent proof.

<!-- formula-not-decoded -->

The following Lemma A.3 describes the initialization.

Lemma A.3. At the initialization, for ∀ j ∈ {± 1 } , r ∈ [ m ] , i ∈ [ n ] , we have

<!-- formula-not-decoded -->

Proof. By Definition 3.2, we have

By standard Gaussian tails, we get

For ⟨ w (0) j,r , ξ i ⟩ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For data ( x i , y i ) ∼ D , by Definition 3.1, let B i = supp ( ξ i ) \{ 1 } , we have

<!-- formula-not-decoded -->

Therefore, condition on the training dataset S , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

since ξ i [ k ] ∼ N (0 , σ 2 p ) . Therefore, we have

<!-- formula-not-decoded -->

## A.2 Preliminary Lemmas

The following lemma studies non-overlap support property of noise patch in data model D in Definition 3.1.

Lemma A.4 (Non-overlap support, Lemma C.1 in Zou et al. (2023b)) . Let { ( x i , y i ) } i =1 ,...,n be the training dataset sampled according to Definition 3.1. Moreover, let B i = supp ( ξ i ) \{ 1 } be the support of x i except the first coordinate. Then with probability at least 1 -n -2 , B i ∩ B j = ∅ for all i, j ∈ [ n ] .

## A.3 Gradients and Updates

We first calculate the gradient of the individual loss function L i with respect to w ( t ) j,r .

Lemma A.5. Consider the CNN model defined in Eq. 3.2. Let ( x i , y i ) be a data generated from data model D in Definition 3.1. The gradient of L i ( W ) = -log e Fy i ( W , x i ) ∑ j ∈{-1 , 1 } e F j ( W , x i ) with respect to w j,r is:

<!-- formula-not-decoded -->

where ℓ j,i := 1 y i = j -logit j ( F, x i ) and logit j ( F, x i ) = e F j ( W , x i ) ∑ k ∈{-1 , 1 } e F k ( W , x i ) .

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Based on the definition of ℓ j,i , we have a useful lemma as follow:

Lemma A.6. Given data ( x i , y i ) generated from data model 3.1, define ℓ j,i = 1 y i = j -logit j ( F, x i ) and logit j ( F, x i ) = e F j ( W , x i ) ∑ k ∈{-1 , 1 } e F k ( W , x i ) , we have

<!-- formula-not-decoded -->

Proof of Lemma A.6. For j = y i ,

<!-- formula-not-decoded -->

̸

For j = y i ,

<!-- formula-not-decoded -->

Now we calculate the gradient of loss (3.1) and loss (3.2), responding to stochastic Adam and stochastic AdamW respectively. Here we slightly abuse the notation. We use g ( t ) t,j,r to represent the stochastic gradient with respect to w ( t ) j,r at the t -th iteration. The subscript t of g ( t ) t,j,r represents the batch at the t -th iteration and the superscript t of g ( t ) t,j,r represents the weight matrix W ( t ) at the t -th iteration.

Lemma A.7 (Gradient of Stochastic Adam) . Consider the CNN model in Definition 3.2. Let { ( x i , y i ) } n i =1 be the training dataset generated from data model in Definition 3.1. Using stochastic Adam to train the neural network, at the t -th iteration with batch data index set I t of size B , the stochastic gradient of the loss (3.1) with respect to w ( t ) j,r is as follows:

<!-- formula-not-decoded -->

More specific, for the k -th coordinate, we have

- k = 1 :

<!-- formula-not-decoded -->

- k ∈ B i , i ∈ I t :

̸

<!-- formula-not-decoded -->

- k = 1 and k / ∈ ∪ i ∈I t B i :

<!-- formula-not-decoded -->

Proof of Lemma A.7. The loss of stochastic Adam at the t -th iteration with batch data index set I t is

<!-- formula-not-decoded -->

By Lemma A.5, we have

<!-- formula-not-decoded -->

For the k -th coordinate, if k = 1 , we know v = [1 , 0 , . . . , 0] ⊤ and ξ i [1] = -αy i . So we have

<!-- formula-not-decoded -->

̸

If k ∈ B i , i ∈ I t , by Lemma A.4, we know B i ∩ B j = ∅ for i = j . So we have

<!-- formula-not-decoded -->

If k = 1 and k / ∈ ∪ i ∈I t B i , it is obvious that

<!-- formula-not-decoded -->

Lemma A.8 (Gradient of Stochastic AdamW) . Consider the CNN model defined in Eq. 3.2. Let { ( x i , y i ) : i ∈ [ n ] } be the training dataset generated from data model 3.1. Use stochastic AdamW training the neural network, at the t -th iteration with batch data index set I t of size B , the stochastic gradient of the Loss defined in Eq. (3.2) with respect to w ( t ) j,r is as follows:

<!-- formula-not-decoded -->

More specific, for the k -th coordinate, we have

- k = 1 :

<!-- formula-not-decoded -->

- k ∈ B i , i ∈ I t :

̸

- k = 1 and k / ∈ ∪ i ∈I t B i :

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma A.8. The loss of stochastic AdamW at the t -th iteration with batch data index set I t is

<!-- formula-not-decoded -->

By Lemma A.5, we have

<!-- formula-not-decoded -->

For the k -th coordinate, if k = 1 , we know v = [1 , 0 , . . . , 0] ⊤ and ξ i [1] = -αy i . So we have

<!-- formula-not-decoded -->

̸

If k ∈ B i , i ∈ I t , by Lemma A.4, we know B i ∩ B j = ∅ for i = j . So we have

<!-- formula-not-decoded -->

If k = 1 and k / ∈ ∪ i ∈I t B i , it is obvious that

<!-- formula-not-decoded -->

̸

## B Proof Sketch

In this section, we mainly outline the proof sketch for the main results in Section 4. Following the two-stage analysis framework of Cao et al. (2022); Zou et al. (2023b), we decompose the proof into two distinct stages:

Stage I: Pattern Learning. During the initial phase of training, the effect of regularization is negligible. The model operates in an underfitting regime, where it rapidly learns dominant patterns in the training data, leading to improved empirical performance on test error.

Stage II: Regularization. As training progresses, the model's classification accuracy on the training set approaches convergence, resulting in diminished gradient magnitudes. At this stage, regularization dominates the optimization dynamics, steering the model converge to a local minima. Due to the nonconvex nature of the loss landscape, the model retains the patterns acquired during the pattern learning stage.

Furthermore, motivated by the behavioral similarity between Adam and SignGD when the learning rate is sufficiently small or β 1 , β 2 approach zero (Balles and Hennig, 2018; Bernstein et al., 2018), we present results for SignSGD and SignSGDW (SignSGD with decoupled weight decay). We subsequently extend these results to stochastic Adam and AdamW, which provided in Appendix C. The update rules for SignSGD are given as follows:

<!-- formula-not-decoded -->

where g ( t ) t,j,r in (B.1) is stochastic gradient of (3.1). The updata rules for SignSGDW are given as follows:

<!-- formula-not-decoded -->

where g ( t ) t,j,r in (B.2) is stochastic gradient of (3.2), and λ is the weight decay parameter.

Next, following the framework of feature learning (Allen-Zhu and Li, 2020; Cao et al., 2022; Zou et al., 2023b; Han et al., 2025a), we primarily focus on two key quantities: 1) Feature Learning ⟨ w j,r , j v ⟩ : This term captures the alignment between the learned weight vector w j,r and the true feature direction j v , reflecting the model's ability to extract meaningful latent structures from the data. 2) Noise Memorization ⟨ w y i ,r , ξ i ⟩ : This term measures the correlation between w y i ,r and the noise patch ξ i of individual samples, characterizing the extent to which the model overfits to stochastic perturbations or idiosyncrasies in the training dataset. This decomposition allows us to separately analyze the model's generalization behavior (driven by feature learning) and its memorization capacity (influenced by noise fitting).

## B.1 Proof Sketch for Stochastic Adam

We present the dynamics of feature learning ⟨ w ( t ) j,r , j v ⟩ and noise memorization ⟨ w ( t ) y i ,r , ξ i ⟩ for SignSGD as follows. The details of calculation are provided in Appendix A.

<!-- formula-not-decoded -->

## B.1.1 Proof Sketch for Theorem 4.1

In this section, we present the proof sketch for Theorem 4.1. We consider n B = Θ(1) , which is the large-batch setting.

LemmaB.1. Given the training dataset S , if n B = Θ(1) , η = 1 / poly( d ) and 0 &lt; λ = o ( σ q -2 0 σ p /n ) then for any t ≤ T 0 with T 0 = ˜ O ( 1 ηsσ p ) and any i ∈ [ n ] ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We note that the above lemma is equivalent to Lemma 5.2 in (Zou et al., 2023b), which enables us to directly extend their results for full-batch Adam to the large-batch regime. The remainder of the proof proceeds in the same way, as the core theoretical framework remains invariant under the batch size scaling n B = Θ(1) . Recall that the condition sσ p = ω (1) implies that noise memorization outpaces feature learning and we have ℓ j,r = Θ(1) throughout Stage I after a certain number of iterations, the direction of feature learning is reversed, as indicated by the update rule in Equation B.3. Specifically, the noise-driven term ασ ′ ( ⟨ w ( t ) , ξ i ⟩ )

satisfying ασ ′ ( ⟨ w j,r , ξ i ⟩ ) ≫ σ ′ ( ⟨ w j,r , y i v ⟩ ) + nλ | w j,r [1] | . By the end of Stage I , the model's feature learning direction has been inverted, while noise memorization reaches a quasi-stationary However, it lacks the capacity to eliminate the memorized noise. Consequently, the model fits the feature noise -αy v and converges to a local minimum that preserves the patterns acquired in Stage I , ultimately leading to poor generalization performance.

since the outputs are small. As a result, j,r becomes dominant, ( t ) ( t ) ( t ) state. In the subsequent regularization phase, weight decay drives the model toward convergence.

We observe that under large-batch setting, the optimization dynamics of Adam closely resemble those of the full-batch setting. This similarity arises because the algorithm traverses the entire dataset within few iterations, resulting in nearly identical momentum estimates and, consequently, comparable training dynamics between large-batch and full-batch regimes.

We next consider the mini-batch setting, which yields conclusions that differ fundamentally from those in the large-batch setting.

## B.1.2 Proof Sketch for Theorem 4.2

In this section, we present the proof sketch for Theorem 4.2. We consider n B ≥ Θ(log ϵ -1 ) , which is the mini-batch setting.

Lemma B.2 (Stage I) . Given the training dataset S , if n B ≥ Θ(log ϵ -1 ) , η = 1 / poly( d ) and 0 &lt; λ = o ( σ q -2 0 σ p /n ) , then for any t ≤ T 0 with T 0 = ˜ O ( B ηn ) and any i ∈ [ n ] ,

<!-- formula-not-decoded -->

Compared to Lemma B.1, Lemma B.2 reveals fundamentally different optimization dynamics in the mini-batch regime. In Stage I , feature learning advances steadily-since ℓ j,r = Θ(1) -while noise memorization remains at its initialization scale. This divergence arises because mini-batch sampling requires many more iterations to traverse the dataset: dense, shared features receive consistent gradient updates and resist weight decay, whereas sparse, uncorrelated noise is continuously attenuated. As features strengthen, network outputs grow, the loss gradients diminish, and weight decay takes over, marking the transition into Stage II . We now show that the structures acquired in Stage I persist throughout this regularization phase.

Lemma B.3 (Stage II) . Suppose the same conditions hold as in Lemma B.2. For t &gt; T 0 , j ∈ {± 1 } , r ∈ [ m ] , i ∈ [ n ] , let r ∗ = argmax r ∈ [ m ] ⟨ w ( t ) j,r , j v ⟩ , then ⟨ w ( t ) j,r ∗ , j v ⟩ = ˜ Θ(1) and ⟨ w ( t ) y i ,r , ξ i ⟩ ≤ ˜ Θ( ηsσ p ) .

This lemma follows because, once feature learning has increased accuracy and reduced gradients, weight decay takes effect but cannot reverse the established feature alignment; finally the model converges to a local minimum that preserves the patterns learned in Stage I .

Lemma B.4 (Convergence) . Suppose the same conditions hold as in Lemma B.2 and B.3, if the step size η = O ( d -1 2 ) , then for any t ,

<!-- formula-not-decoded -->

Combining Lemma B.3 and B.4, we observe that the model successfully learns the true features and eventually converges to a local minimum with infinitesimal learning rate η and T = poly( n ) /η , retaining strong generalization performance.

## B.1.3 Proof Sketch for Corollary 4.3

We next show that if the weight decay parameter satisfies λ = ω ( σ q -2 0 ) , then the learning dynamics of Adam are effectively suppressed. This implies that the effective weight decay for Adam is of the order σ q -2 0 , which is significantly smaller than that required for AdamW, as will be discussed later.

Corollary 4.3 formalizes this observation by showing that if the Adam weight decay parameter satisfies λ = ω ( σ q -2 0 ) , the training process becomes stagnant and remains near the initialization. This corollary follows directly from the proof sketches of both large-batch and mini-batch Adam. By Lemma A.3, we know that at initialization:

<!-- formula-not-decoded -->

Then, from the update rules given in Equations (B.3) and (B.4), we observe that the updates are dominated by the weight decay term, i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

due to the condition λ = ω ( σ q -2 0 ) . As a result, the learning dynamics are overwhelmed by the regularization term, preventing meaningful updates. Consequently, the model parameters remain close to their initialization throughout training. We formalize this in Lemma B.5.

Lemma B.5. Suppose the same conditions hold as in Lemma B.1 and B.4, if λ = ω ( σ q -2 0 ) , then

<!-- formula-not-decoded -->

and

## B.2 Proof Sketch for Stochastic AdamW

Motivated by the similarity between Adam and SignSGD, and to better illustrate the core idea, we present the dynamics of feature learning ⟨ w ( t ) j,r , j v ⟩ and noise memorization ⟨ w ( t ) y i ,r , ξ i ⟩ under SignSGDW. The detailed derivation of update formula is deferred to Appendix A. However, we emphasize that there are key differences between AdamW and SignSGDW.

In SignSGDW, due to the presence of the sign operator, the weight decay affects ⟨ w ( t ) j,r , j v ⟩ only after it grows beyond a certain threshold, and similarly for ⟨ w ( t ) y i ,r , ξ i ⟩ . In contrast, for AdamW, weight decay becomes effective once ⟨ w ( t ) j,r , j v ⟩ or ⟨ w ( t ) y i ,r , ξ i ⟩ reaches a level where the gradient magnitudes become sufficiently small. At this point, the update is normalized by the stability constant ϵ and dominated by the weight decay term, which causes both ⟨ w ( t ) j,r , j v ⟩ and ⟨ w ( t ) y i ,r , ξ i ⟩ to cease increasing. Besides, as the lemmas in this section are simplified instances of those presented in Section C.2, we omit them for brevity. For more details, refer to Section C.2.

<!-- formula-not-decoded -->

Moreover, we should note that the duration of Stage I under SignSGDW differs markedly from that under SignSGD, owing to the decoupled weight decay mechanism. During Stage I , model parameters grow unchecked by gradient-based regularization, allowing features to accumulate strength until the decoupled weight decay term begins to exert significant influence. Once this threshold is reached, training transitions into Stage II , in which weight decay counteracts further parameter growth and stabilizes the weight norms.

## B.2.1 Proof Sketch for Theorem 4.4

In this section, we present the proof sketch for Theorem 4.4. We consider n B = Θ(1) or n B = o ( sσ p ) , which is the large-batch setting.

The following Lemma B.6 characterizes the duration of Stage I in the large-batch SignSGDW setting and provides upper bounds on feature learning and noise memorization.

Lemma B.6 (Stage I, pattern learning) . Given the training dataset S , if n B = Θ(1) or n B = o ( sσ p ) , η = 1 / poly( d ) , λ = ˜ Ω( B 2 n ∧ 1) and λ = ˜ O (1) , then for any t ≤ T 0 with T 0 = ˜ O ( B ληn ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since in the large-batch regime n B = o ( sσ p ) , Lemma B.6 implies that noise memorization accumulates faster than feature learning. At the beginning of Stage I, feature gradients dominate because

σ ′ ( ⟨ w ( t ) y i ,r , y i v ⟩ ) ≫ ασ ′ ( ⟨ w ( t ) y i ,r , ξ i ⟩ ) , given α = o (1) and negligible weight decay influence. After a certain number of epochs, the noise term grows until ασ ′ ( ⟨ w ( t ) y i ,r , ξ i ⟩ ) ≫ σ ′ ( ⟨ w ( t ) y i ,r , y i v ⟩ ) , at which point feature learning reverses and eventually flips direction. Lemma B.7 below provides a precise description of this transition.

Lemma B.7 (Stage I, fitting feature noise) . Suppose the same conditions hold as in Lemma B.6, if α ≥ ˜ Θ ( ( B n sσ p ) 1 -q ) , then for any t ∈ [ T r , T 0 ] with T r = ˜ O ( σ 0 ηsσ p α 1 / ( q -1) ) ≤ T 0 ,

<!-- formula-not-decoded -->

and at epoch T 0 , we have (a) w ( T 0 · n B ) j,r [1] = -sgn( j ) · ˜ Ω(1 /λ ) ; (b) w ( T 0 · n B ) j,r [ k ] = sgn( ξ i [ k ]) · ˜ Ω( B nλ ) or ± ˜ O ( η ) for k ∈ B i with y i = j ; (c) w ( T 0 · n B ) j,r [ k ] = ± ˜ O ( η ) otherwise.

Lemma B.7 implies that, by the end of Stage I , the model has fitted the training noise. The following Lemma B.8 shows that these pattern persist throughout Stage II , ultimately leading to poor generalization.

LemmaB.8 (Stage II, preserve the noise) . Suppose the same conditions hold as in Lemma B.6 and B.7, for t &gt; T 0 , j ∈ {± 1 } , r ∈ [ m ] , i ∈ [ n ] , let r ∗ = argmax r ∈ [ m ] ⟨ w ( t ) y i ,r , ξ i ⟩ , then ⟨ w ( t ) j,r , j · v ⟩ = -˜ Θ(1 /λ ) and ⟨ w ( t ) y i ,r ∗ , ξ i ⟩ = ˜ Θ( Bsσ p nλ )

The following Lemma B.9 prove the convergence under certain conditions.

Lemma B.9 (Convergence) . Suppose the same conditions hold as in Lemma B.6, B.7 and B.8, if the step size satisfies η = O ( d -1 / 2 ) , then for any t ,

<!-- formula-not-decoded -->

Combining Lemmas B.8 and B.9, we observe that, with an infinitesimal learning rate η and T = poly( n ) /η , the model ultimately fits the feature noise and converges to a local minimum, resulting in poor generalization performance.

## B.2.2 Proof Sketch for Theorem 4.5

In this section, we present the proof sketch for Theorem 4.5. We consider n B ≥ Θ( n 1 / 2 ∨ log ϵ -1 ) and n B = ω ( sσ p ) , which is the mini-batch setting.

The following Lemma B.10 characterizes the duration of Stage I in the mini-batch SignSGDW setting and provides upper bounds on feature learning and noise memorization.

Lemma B.10 (Stage I) . Given the training dataset S , if n B ≥ Θ( n 1 / 2 ∨ log ϵ -1 ) and n B = ω ( sσ p ) , η = 1 / poly( d ) , λ = ˜ Ω( B 2 n ∧ 1) and λ = ˜ O (1) , then for any t ≤ T 0 with T 0 = ˜ O ( B ληn ) ,

<!-- formula-not-decoded -->

In the mini-batch regime, n B = ω ( sσ p ) , so feature learning outpaces noise memorization-unlike in the large-batch case. Consequently, noise cannot reverse the feature learning; instead, features are learned continuously until decoupled weight decay intervenes. Noise memorization also grows until this point, but because noise is both sparse and independent, it accrues only during a few iterations per epoch and is concurrently suppressed by weight decay. Hence, both feature learning and noise memorization reach their peak at the end of Stage I , after which weight decay governs Stage II . The next Lemma B.11 formalizes this behavior.

Lemma B.11 (Stage II) . Suppose the same conditions hold as in Lemma B.10, for t &gt; T 0 , j ∈ {± 1 } , r ∈ [ m ] , i ∈ [ n ] , let r ∗ = argmax r ∈ [ m ] ⟨ w ( t ) j,r , j v ⟩ , then ⟨ w ( t ) j,r ∗ , j v ⟩ = ˜ Θ(1 /λ ) and ⟨ w ( t ) y i ,r , ξ i ⟩ ≤ ˜ Θ( Bsσ p nλ ) .

The following lemma establishes convergence under the specified conditions.

Lemma B.12 (Convergence) . Suppose the same conditions hold as in Lemma B.10 and B.11, if the step size satisfies η = O ( d -1 / 2 ) , then for any t ,

<!-- formula-not-decoded -->

## B.2.3 Proof Sketch for Corollary 4.6

By Conditions A.1 and A.2, along with Definition 3.2, we know that d = poly( n ) , and hence

<!-- formula-not-decoded -->

This directly implies that the effective weight decay parameter for Adam satisfies

<!-- formula-not-decoded -->

This completes the proof.

## C Proofs

First we give a general upper bound of the moving average in stochastic Adam and stochastic AdamW. Lemma C.1. Let m ( t ) j,r be the first momentum estimate, v ( t ) j,r be the second momentum estimate at the t -th iterate in the update rule of stochastic Adam or stochastic AdamW. Then for all j ∈ {± 1 } , r ∈ [ m ] and k ∈ [ d ] , if β 2 &gt; β 2 1 , β 1 , β 2 ∈ [0 , 1) , we have

<!-- formula-not-decoded -->

Proof of Lemma C.1. Let us expand the moment estimates

<!-- formula-not-decoded -->

. Then we have

<!-- formula-not-decoded -->

where the first inequality we use Cauchy-Schwartz inequality and the third equality we use the fact that z 2 t = [ β t 1 (1 -β 1 ) ] 2 β t 2 (1 -β 2 ) is a convergent series. So we have

<!-- formula-not-decoded -->

## C.1 Proof of Stochastic Adam

First, we try to approximate the update of stochastic Adam to sign update since the similar performance between Adam and SignGD (Bernstein et al., 2018; Balles and Hennig, 2018; Zou et al., 2023b; Xie and Li, 2024; Li et al., 2025).

Lemma C.2. Consider the update of stochastic Adam in (3.5) . Let W ( t ) be the weight at the t -th iteration. Suppose that ⟨ w ( t ) j,r , y i v ⟩ , ⟨ w ( t ) j,r , ξ i ⟩ = ˜ Θ(1) for all j ∈ {± 1 } , r ∈ [ m ] , i ∈ [ n ] and β 2 1 &lt; β 2 . We have the approximate update rule for each coordinate weight as follows:

- For k = 1 , we have either | g ( t ) t,j,r [1] | ≤ ˜ Θ( η ) or

<!-- formula-not-decoded -->

- For every k ∈ B i , i ∈ I t -τ , τ ∈ T k := { τ 0 + i · n B : i ∈ { 0 } ∪ [ ¯ τ n/B -1] , τ 0 &lt; n B } , where τ 0 represents the number of iterations away from the current iteration t , coordinate k is affected by ξ i sampled at the iteration t -τ 0 since the moving average, and we define ¯ τ = Θ(log( λη ) -1 )

-If n = Θ(1) , for any τ 0 &lt; n , we have either

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

- For the remaining coordinates k = 1 and k / ∈ B i , i ∈ I t -τ 0 , τ 0 ∈ { 0 } ∪ [¯ τ ] , where τ 0 represents the number of iterations away from the current iteration t , coordinate k is affected by ξ i sampled at the iteration t -τ 0 since the moving average, and we define ¯ τ = log( λη ) -1 . Then we have either | g ( t ) t,j,r [ k ] | ≤ ˜ Θ( λη ) or

<!-- formula-not-decoded -->

Proof of Lemma C.2. Let first focus on the first momentum estimate,

<!-- formula-not-decoded -->

where the last equality we select ¯ τ = Θ(log( λη ) -1 ) such that ∑ t τ =¯ τ +1 β τ 1 (1 -β 1 ) = O ( λη ) and | g ( t -τ ) t -τ,j,r [ k ] | = ˜ O (1) for all k ∈ [ d ] by Lemma A.7, since the facts that ⟨ w ( t ) j,r , y i v ⟩ , ⟨ w ( t ) j,r , ξ i ⟩ = ˜ O (1) .

Similarly, for the second momentum estimate,

<!-- formula-not-decoded -->

Here we use the same ¯ τ because we can always reselect ¯ τ of smaller one to larger one, and the absolute value of the tail will not increase. Then we have

<!-- formula-not-decoded -->

since ϵ = Θ( λη ) . Now we want to use sign update to approximate (C.1). First, we should note that once the signs of g ( t -τ ) t -τ,j,r [ k ] for τ ∈ [0 , ¯ τ ] aligned, (C.1) can be approximated as sgn( g ( t ) t,j,r [ k ]) · ˜ Θ(1) , since

<!-- formula-not-decoded -->

Recall the gradient of stochastic Adam given in Lemma A.7, we want to approximate g ( t -τ ) t -τ,j,r [ k ] to g ( t ) t -τ,j,r [ k ] , such that we can use the current weight to approximate sign update. By Lemma C.1, the upper bound of each coordinate in one step is Θ( η ) . Then for τ ∈ [ t -¯ τ, t ] , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, we have

Then recall the predict function

<!-- formula-not-decoded -->

We have

<!-- formula-not-decoded -->

where the second inequality we use the convexity of σ ( · ) and the facts that |⟨ w ( t ) j,r , y i v ⟩| = ˜ Θ(1) and |⟨ w ( t ) j,r , ξ i ⟩| = ˜ Θ(1) . The last inequality we use m = ˜ Θ(1) and sσ p = ω (1) .

Then we can approximate ℓ ( τ ) j,r to ℓ ( t ) j,r in the gradient A.7.

<!-- formula-not-decoded -->

where we use the fact that ˜ Θ( η ¯ τsσ p ) = o (1) and (C.5). So we have

<!-- formula-not-decoded -->

for all τ ∈ [ t -¯ τ, t ] . Further, by (C.2), (C.3) and the facts that |⟨ w ( t ) j,r , y i v ⟩| = ˜ O (1) and |⟨ w ( t ) j,r , ξ i ⟩| = ˜ O (1) , recall σ ( x ) = max(0 , x ) q , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

So we conclude that

<!-- formula-not-decoded -->

Similarly, we have

<!-- formula-not-decoded -->

Now, we have all the tools we need to approximate g ( t -τ ) t -τ,j,r [ k ] to g ( t ) t -τ,j,r [ k ] . Recall Lemma A.7, substitute (C.4), (C.6) and (C.7) into g ( t -τ ) t -τ,j,r [ k ] , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- For all k ∈ B i , i ∈ I t -τ ,

<!-- formula-not-decoded -->

̸

- For k = 1 and k / ∈ B i , i ∈ I t -τ ,

<!-- formula-not-decoded -->

Plugging (C.8), (C.9) and (C.10) into (C.1), with facts that ¯ τ = ˜ Θ(1) , λ = o (1) , |⟨ w ( t ) j,r , ξ i ⟩| = ˜ Θ(1) , |⟨ w ( t ) j,r , y i v ⟩| = ˜ Θ(1) , | ℓ ( t ) j,i | = Θ(1) , ϵ = Θ( λη ) and Lemma A.6, we have

- •

<!-- formula-not-decoded -->

.

- For k ∈ B i , i ∈ I t -τ 0 , τ 0 ∈ { 0 } ∪ [¯ τ ] , where τ 0 represents the number of iterations away from the current iteration t , coordinate k is affected by ξ i sampled at the iteration t -τ 0 since the moving average. We note that if the number of iteration in one epoch n B is less than ¯ τ , the moving average will use some sample x multiply times. We denote T k := { τ 0 + i · n B : i ∈ { 0 }∪ [ ¯ τ n/B -1] , τ 0 ≤ n B } as the timestamp set involved using noise ξ i (i.e., i ∈ I t -τ for any τ ∈ T k ), and k ∈ B i , τ 0 ≤ n B . If n B &gt; ¯ τ , in this case we have T k := { τ 0 } for any k ∈ B i , and ξ i was used in iteration t -τ 0 Then we have

<!-- formula-not-decoded -->

where the third equality we denote g ( t -τ ) t -τ,j,r [ k ] = Θ( g ( t ) t -τ,j,r [ k ]) ± Θ( λη ¯ τ ) = Θ( λ w ( t ) j,r [ k ]) ± Θ( λη ¯ τ ) as ˜ g for τ ∈ { 0 } ∪ [¯ τ ] \T k . For the denominator, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we use the fact that ϵ = Θ( λη ) . Now we handle β τ 0 1 and β τ 0 2 2 with more care. First we have β τ 0 1 &lt; β τ 0 2 2 , | g ( t ) t -τ 0 ,j,r [ k ] | = ˜ O (1) and τ 0 &lt; n B .

Then if n B = Θ(1) , we have β τ 0 1 = β τ 0 2 2 = Θ(1) , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For τ 0 &lt; n B , τ 0 ≥ Θ(log( λη ) -1 ) such that β τ 0 2 2 ≤ Θ( λη ) , we have

<!-- formula-not-decoded -->

since | g ( t ) t -τ 0 ,j,r [ k ] | = ˜ O (1) , λη = o (1) and ηsσ p = o (1) . We claim that the intersection (gap) of Θ(log( λη ) -1 ) and Θ(1) is very small for Θ(log( λη ) -1 ) , that is, considering the intersection part c log( λη ) -1 for a sufficiently small constant c &gt; 0 , the impact of the intersection (gap) is very small, since c log( λη ) -1 = o (log( λη ) -1 ) . Therefore, for most of τ 0

<!-- formula-not-decoded -->

- For k = 1 and k / ∈ B i , i ∈ I t -τ , τ ∈ [0 , ¯ τ ] , recall g ( t ) t -τ,j,r [ k ] = λ w ( t ) j,r [ k ] , we have

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This completes the proof.

## C.1.1 Proof of Theorem 4.1

Lemma C.3 (Stage I) . Given the training dataset S , if n B = Θ(1) , η = 1 / poly( d ) and 0 &lt; λ = o ( σ q -2 0 σ p /n ) , then for any t ≤ T 0 with T 0 = ˜ O ( 1 ηsσ p ) and any i ∈ [ n ] ,

<!-- formula-not-decoded -->

Proof of Lemma C.3. By Lemma C.1, we have

<!-- formula-not-decoded -->

Then, we prove ⟨ w ( t +1) y i ,r , ξ i ⟩ = ⟨ w ( t ) y i ,r , ξ i ⟩ + ˜ Θ( ηsσ p ) by induction. By Lemma A.3, we have

<!-- formula-not-decoded -->

which imply that | ℓ (0) j,i | = Θ(1) . Additionally, we have ηs = o ( σ q -1 0 ) and 0 &lt; λ = o ( σ q -2 0 σ p /n ) . For a sufficiently large fraction of k ∈ B i (e.g., 0 . 99 ), we have | B -1 σ q -1 0 ξ i [ k ] | ≥ ˜ Θ( ηB -1 sσ p | ℓ (0) j,i | + λ | w (0) j,r [ k ] | ) for i ∈ I τ . Therefore, by Lemma C.2 and A.6, we have

<!-- formula-not-decoded -->

Recall n B = Θ(1) . By Lemma C.2 we have the following update according to (B.4), (C.11) and Lemma C.1.

<!-- formula-not-decoded -->

For general t ≤ T 0 , assuming ⟨ w ( t ) y i ,r , ξ i ⟩ ≥ ⟨ w ( t -1) y i ,r , ξ i ⟩ + ˜ Θ( ηsσ p ) . Then we have

<!-- formula-not-decoded -->

By Lemma C.1, we have

<!-- formula-not-decoded -->

So we have | ℓ ( t ) j,i | = Θ(1) . Besides, we still establish the condition | B -1 ( w (0) j,r [ k ]+ tηsσ p ) q -1 ξ i [ k ] | ≥ ˜ Θ( ηB -1 sσ p | ℓ (0) j,i | + λ | w (0) j,r [ k ]+ tη | ) since 0 &lt; λ = o ( σ q -2 0 σ p /n ) . Then we have (C.11) for t . Follow the same proof above, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The term O ( ηsσ p ) in the above inequality arises because, for coordinates that | ξ i [ k ] | ≤ O ( σ p ) , we cannot exploit sign information. Instead, we directly apply Lemma C.1. This completes the proof.

Lemma C.3 coincides with Lemma A.3 of (Zou et al., 2023b), allowing us to transfer their full-batch Adam analysis to the large-batch case under n B = Θ(1) . Therefore, the remaining proofs are omitted for brevity, as they coincide with those in Zou et al. (2023b).

## C.1.2 Proof of Theorem 4.2

Lemma C.4 (StageI, nearly zero noise memorization) . Given the training dataset S , if n B ≥ Θ(log ϵ -1 ) , η = 1 / poly( d ) and 0 &lt; λ = o ( σ q -2 0 σ p /n ) , then for any t ≤ T 0 with T 0 = ˜ O ( 1 η ) and any i ∈ [ n ] , suppose ⟨ w ( t ) y i ,r , y i · v ⟩ &gt; -˜ Θ( σ 0 ) , then

<!-- formula-not-decoded -->

Proof of Lemma C.4. By Lemma A.3, at initialization

<!-- formula-not-decoded -->

since α = o (1) . LemmaC.2 ensures that stochastic updates slow down noise memorization-allowing ⟨ w ( t ) j,r , ξ i ⟩ to grow for only

<!-- formula-not-decoded -->

iterations after ξ i is sampled-while in the remaining

<!-- formula-not-decoded -->

iterations, weight decay dominates. In particular, whenever | w ( t ) j,r [ k ] | ≥ ˜ Θ( η ) we have

<!-- formula-not-decoded -->

Concretely, if ξ i is sampled at iteration τ 1 of the first epoch, then

<!-- formula-not-decoded -->

where we directly bound the update by Θ(1) according to Lemma C.1. Over the next o (log( λη ) -1 ) iterations the noise memorization ⟨ w ( t ) j,r , ξ i ⟩ increases by at most

<!-- formula-not-decoded -->

and thereafter weight decay decreases it in each of the remaining n B -o (log( λη ) -1 ) steps. Hence, we can calculate the maximum value of the noise memorization

<!-- formula-not-decoded -->

since n/B ≥ Θ ( log( λη ) -1 ) and ξ i [1] = -αy i . This is true in every epoch. So we complete the proof.

Lemma C.5 (Stage I, feature learning) . Given the training dataset S , if n B ≥ Θ(log ϵ -1 ) , η = 1 / poly( d ) and 0 &lt; λ = o ( σ q -2 0 σ p /n ) , then for any t ≤ T 0 with T 0 = ˜ O ( 1 η ) and j ∈ {± 1 } , we have

<!-- formula-not-decoded -->

Proof of Lemma C.5. by Lemma A.7, we have

<!-- formula-not-decoded -->

Then with Lemma A.3 and facts that ℓ (0) j,i = Θ(1) , by Lemma A.6, we have

<!-- formula-not-decoded -->

Substituting them into g (0) 0 ,j,r , with α = o (1) , s 1 / 2 σ p = ˜ O (1) , λ = o ( σ q -2 0 σ p /n ) , we get

<!-- formula-not-decoded -->

By Lemma C.2 and η = o ( σ q -1 0 ) , we have

<!-- formula-not-decoded -->

Now suppose that the equality holds for iterations 0 , . . . , t . Then ⟨ w ( t ) j,r , j · v ⟩ = ˜ O (1) , ⟨ w ( t ) y i ,r , ξ i ⟩ = ˜ Θ( ηsσ p ) = O (1) . Therefore, ℓ ( t ) j,i = Θ(1) . By Lemma C.4, we have

<!-- formula-not-decoded -->

Substituting them into g ( t ) t,j,r , with α = o (1) , s 1 / 2 σ p = ˜ O (1) , λ = o ( σ q -2 0 σ p /n ) , we get

<!-- formula-not-decoded -->

By Lemma C.2 and η = o ( σ q -1 0 ) , we have

<!-- formula-not-decoded -->

This completes the proof.

Lemma C.6 (Stage I, general dynamics) . Given the training dataset S , if n B ≥ Θ(log ϵ -1 ) , η = 1 / poly( d ) and 0 &lt; λ = o ( σ q -2 0 σ p /n ) , then for any t ≤ T 0 with T 0 = ˜ O ( 1 η ) and any i ∈ [ n ] ,

<!-- formula-not-decoded -->

Proof of Lemma C.6. We prove the claim by induction on t , using Lemma C.4 and C.5. At t = 0 , by Lemma A.3, at initialization we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and hence

Therefore Lemma C.4 holds at t = 0 . Suppose Lemma C.4 and C.5 for some t ≥ 0 , then

<!-- formula-not-decoded -->

by exactly the same proof used in the proof of Lemma C.5. This lower bound remains valid at step t +1 . Consequently, Lemma C.4 continues to hold, and the induction carries through all iterations. This completes the proof.

Lemma C.7 (Stage II) . Given the training dataset S , if n B ≥ Θ(log ϵ -1 ) , η = 1 / poly( d ) and 0 &lt; λ = o ( σ q -2 0 σ p /n ) , then for any t &gt; T 0 , j ∈ {± 1 } , r ∈ [ m ] , i ∈ [ n ] , let r ∗ = argmax r ∈ [ m ] ⟨ w ( t ) j,r , j v ⟩ , then ⟨ w ( t ) j,r ∗ , j v ⟩ = ˜ Θ(1) and ⟨ w ( t ) j,r , ξ i ⟩ ≤ ˜ Θ( ηsσ p + α ) .

Proof of Lemma C.7. We begin by establishing the bound ⟨ w ( t ) j,r , ξ i ⟩ ≤ ˜ Θ( ηsσ p + α ) . According to Lemma C.4, during the first T 0 epochs, the number of iterations in which weight decay dominates the update dynamics is at least

<!-- formula-not-decoded -->

In each such iteration, the contribution from weight decay is lower bounded by ˜ Θ( ηsσ p ) , leading to a cumulative effect of

<!-- formula-not-decoded -->

This term is asymptotically larger than ˜ Θ( √ sσ p σ 0 ) , i.e., nsσ p /B = ω ( √ sσ p σ 0 ) . Therefore, we conclude that over the first T 0 epochs, the weight decay effectively suppresses noise memorization, ensuring that ⟨ w ( t ) j,r , ξ i ⟩ ≤ ⟨ w ( t ) j,r , -αy i v ⟩ + ˜ Θ( ηsσ p ) ≤ ˜ Θ( ηsσ p + α ) holds.

Next, we focus on ⟨ w ( t ) j,r ∗ , j v ⟩ = ˜ Θ(1) for r ∗ = argmax r ∈ [ m ] ⟨ w ( t ) j,r , j v ⟩ . By Lemma C.6, we know ⟨ w ( T 0 ) j,r ∗ , j · v ⟩ = ˜ Θ(1) and ℓ ( T 0 ) j,r = Θ(1) . For t &gt; T 0 , We show if ⟨ w ( t ) j,r ∗ , j · v ⟩ ≤ ( 1 m log ( ( λ ) -1 -1 )) 1 q , then for ( x i , y i ) with y i = j ,

<!-- formula-not-decoded -->

where the inequality we use ⟨ w ( t ) j,r , ξ i ⟩ ≤ ˜ Θ( ηsσ p + α ) and ηsσ p = o (1) , α = o (1) . Then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we use Lemma A.6, ⟨ w ( t ) j,r , ξ i ⟩ ≤ ˜ Θ( ηsσ p + α ) and α = o (1) . So we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then we have

<!-- formula-not-decoded -->

So we have

<!-- formula-not-decoded -->

Therefore, ⟨ w ( t ) j,r ∗ , j · v ⟩ = ˜ Θ(1) for t &gt; T 0 = 1 η . This completes the proof.

Lemma C.8 (Convergence) . Suppose the same conditions hold as in Lemma C.6 and C.7, if the step size η = O ( d -1 2 ) , then for any t ,

<!-- formula-not-decoded -->

Proof of Lemma C.8. We aim to prove the convergence of the objective function under the Adam optimization algorithm in a non-convex setting. Recall the loss function for each data point i is

<!-- formula-not-decoded -->

where W represents the parameter matrix, x i is the input data, y i is the true label, and F j ( W , x i ) are the logits for class j . The total objective is:

<!-- formula-not-decoded -->

with λ = o (1) as a small regularization parameter.

Since L i ( W ) is non-convex, we exploit its smoothness with respect to the logits [ F j ( W , x i )] j . Specifically, L i ( W ) is 1-smooth in [ F j ( W , x i )] j due to the properties of the cross-entropy loss. Define:

<!-- formula-not-decoded -->

Using the smoothness property, we apply a second-order Taylor-like expansion around W ( t ) :

<!-- formula-not-decoded -->

This upper bound arises because the second derivative of L i with respect to the logits is bounded by 1, a standard result for cross-entropy loss. The logits are defined as: F j ( W ( t ) , x i ) = ∑ m r =1 [ σ ( ⟨ w ( t ) j,r , y i v ⟩ ) + σ ( ⟨ w ( t ) j,r , ξ i ⟩ )] , where w ( t ) j,r the r -th neuron in j -th output of W ( t ) , σ ( z ) = [ z ] q + is a smooth activation function (e.g., with q ≥ 3 ). By Lemma C.7 and C.1, we have ⟨ w ( t ) j,r , v ⟩ ≤ ˜ Θ(1) and ⟨ w ( t ) j,r , ξ i ⟩ ≤ ˜ Θ(1) , ensuring the local smoothness of σ remains ˜ O (1) between ⟨ w ( t +1) j,r , y i v ⟩ and ⟨ w ( t ) j,r , y i v ⟩ (similar for ⟨ w ( t ) j,r , ξ i ⟩ ). Then with Taylor expansion, we have

<!-- formula-not-decoded -->

where the last inequality we use Lemma C.1. Similarly, we have

<!-- formula-not-decoded -->

Summing over r (with m = ˜ Θ(1) ), we get

<!-- formula-not-decoded -->

Additionally, ∥∇ W F j ( W ( t ) , x i ) ∥ F ≤ ˜ Θ(1) since m = ˜ Θ(1) , ⟨ w ( t ) j,r , y i v ⟩ ≤ ˜ Θ(1) , ⟨ w ( t ) j,r , ξ i ⟩ ≤ ˜ Θ(1) . So we have

<!-- formula-not-decoded -->

Substitute (C.15) and (C.16) into (C.12):

<!-- formula-not-decoded -->

For the full objective:

<!-- formula-not-decoded -->

Since λ ∥ W ∥ 2 F is 2 λ -smooth and λ = o (1) , the regularization term contributes:

<!-- formula-not-decoded -->

where the quadratic term is absorbed into ˜ Θ( η 2 d ) . Substitute (C.17) and (C.19) into (C.18), we have

<!-- formula-not-decoded -->

Take expectation for the stochastic gradient of both side in (C.20),

<!-- formula-not-decoded -->

where we use Lemma C.2 that the update aligns with the gradient's sign for large gradient and the fact that ns 2 σ p = O ( d ) and Jensen's inequality. This completes the proof.

Lemma C.9 (Generalization Performance of Stochastic Adam) . Suppose the same conditions hold as in Lemma C.8. We have the following results for T = poly( n ) η , with training dataset S

- The training error is zero: err S ( W ( T ) ) = 0 .
- The test error is near-zero: err D ( W ( T ) ) = o (1) .

Proof of Lemma C.9. By Lemma C.7, we have

<!-- formula-not-decoded -->

Recall F j ( W , x ) in Definition 3.2, with ηsσ p = o (1) , α = o (1) , we directly have

<!-- formula-not-decoded -->

since F y i ( W ( T ) , x i ) = ˜ Ω(1) , while F -y i ( W ( T ) , x ) ≤ ˜ Θ( ηsσ p + α ) . Besides, for test data ( x , y ) ∼ D with x = [ y v ⊤ , ξ ⊤ ] ⊤ , it is clear that with high probability ⟨ w ( T ) y,r ∗ , y v ⟩ = ˜ Θ(1) and [ ⟨ w ( T ) y,r , ξ ⟩ ] + ≤ ˜ Θ( ηsσ p + α ) , then similar as training error, we have

<!-- formula-not-decoded -->

while

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

This implies that mini-batch Adam can achieve nearly zero test error. This completes the proof.

## C.1.3 Proof of Corollary 4.3

Corollary 4.3 follows directly from Lemma C.10.

Lemma C.10. Suppose the same conditions hold as in Lemma C.3 and C.6, if λ = ω ( σ q -2 0 ) , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma C.10. This corollary is an immediate consequence of Lemmas C.3 and C.6. In particular, Lemma A.3 guarantees that at t = 0

<!-- formula-not-decoded -->

Since λ = ω ( σ q -2 0 ) , Lemma A.7 implies that, at initialization, the weight decay regularization term overwhelmingly dominates the gradient:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, by Lemma C.2, the updates remain in the regularization-dominated regime, and no coordinate ever grows beyond its ˜ Θ( σ 0 ) scale throughout training. This completes the proof.

## C.2 Proof of Stochastic AdamW

Lemma C.11. Consider the update of stochastic AdamW in (3.6) . Let W ( t ) be the weight at the t -th iteration. Suppose that ⟨ w ( t ) j,r , y i v ⟩ , ⟨ w ( t ) j,r , ξ i ⟩ = ˜ Θ(1) for all j ∈ {± 1 } , r ∈ [ m ] , i ∈ [ n ] and β 2 1 &lt; β 2 . We have the approximate update rule for each coordinate weight as follows:

- For k = 1 , we have either | g ( t ) t,j,r [1] | ≤ ˜ Θ( η ) or

<!-- formula-not-decoded -->

- For every k ∈ B i , i ∈ I t -τ , τ ∈ T k := { τ 0 + i · n B : i ∈ { 0 } ∪ [ ¯ τ n/B ] , τ 0 &lt; n B } , where τ 0 represents the number of iterations away from the current iteration t , coordinate k is affected by ξ i sampled at the iteration t -τ 0 since the moving average, and we define ¯ τ = Θ(log( λη ) -1 )
- -If n B ≤ Θ(1) , for any τ 0 &lt; n B , we have either ∣ ∣ ∣ g ( t ) t -τ 0 ,j,r [ k ] ∣ ∣ ∣ ≤ ˜ Θ ( B -1 ηsσ p | ℓ ( t ) j,i | ) or

<!-- formula-not-decoded -->

- -If n B ≥ Θ(log( λη ) -1 ) = ˜ Θ(1) , for τ 0 ≤ Θ(log( λ -1 sσ p )) such that β τ 0 1 ≥ Θ( λ sσ p ) , we have either

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For τ 0 ≥ Θ(log( λη ) -1 ) such that β τ 0 2 ≤ Θ( λ 2 η 2 ) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

or

̸

- For the remaining coordinates k = 1 and k / ∈ B i , i ∈ I t -τ 0 , τ 0 ∈ { 0 } ∪ [¯ τ ] , where τ 0 represents the number of iterations away from the current iteration t , coordinate k is affected by ξ i sampled at the iteration t -τ 0 since the moving average, and we define ¯ τ = log( λη ) -1 . Then we have

<!-- formula-not-decoded -->

Proof of Lemma C.11. The proof is similar to Lemma C.2. We select ¯ τ = Θ(log( λη ) -1 ) such that ∑ t τ =¯ τ +1 β τ 1 (1 -β 1 ) = O ( λ 2 η 2 ) and ∑ t τ =¯ τ +1 β τ 1 (1 -β 1 ) · g ( t -τ ) t -τ,j,r [ k ] = ˜ O ( λ 2 η 2 ) .

<!-- formula-not-decoded -->

Recall the gradient of stochastic AdamW given in Lemma A.8, we want to approximate g ( t -τ ) t -τ,j,r [ k ] to g ( t ) t -τ,j,r [ k ] , such that we can use the current weight to approximate sign update. By Lemma C.1, the upper bound of the normalized moving average of each coordinate in one step is Θ( η ) since λ = ˜ O (1) . Then for τ ∈ [ t -¯ τ, t ] , we have

<!-- formula-not-decoded -->

The last inequality we use the fact that ⟨ w ( k ) j,r , y i v ⟩ = ˜ Θ(1) , λ = ˜ O (1) and Lemma C.1. Similarly, we have

<!-- formula-not-decoded -->

Then recall the predict function

<!-- formula-not-decoded -->

We have

<!-- formula-not-decoded -->

where the second inequality we use the convexity of σ ( · ) and the facts that |⟨ w ( t ) j,r , y i v ⟩| = ˜ Θ(1) and |⟨ w ( t ) j,r , ξ i ⟩| = ˜ Θ(1) , the last inequality we use m = ˜ Θ(1) and sσ p = ω (1) .

Then we can approximate ℓ ( τ ) j,r to ℓ ( τ ) j,r in the gradient A.8.

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

where we use the fact that ˜ Θ( η ¯ τsσ p ) = o (1) and (C.23). So we have

<!-- formula-not-decoded -->

for all τ ∈ [ t -¯ τ, t ] . Further, by (C.21), (C.22) and the facts that |⟨ w ( t ) j,r , y i v ⟩| = ˜ O (1) and |⟨ w ( t ) j,r , ξ i ⟩| = ˜ O (1) , recall σ ( x ) = max(0 , x ) q , we have

<!-- formula-not-decoded -->

So we conclude that

<!-- formula-not-decoded -->

Similarly, we have

<!-- formula-not-decoded -->

Now, we have all the tools we need to approximate g ( t -τ ) t -τ,j,r [ k ] to g ( t ) t -τ,j,r [ k ] . Recall Lemma A.8, substitute (C.24) and (C.25) into g ( t -τ ) t -τ,j,r [ k ] , we have

- For k = 1 ,

<!-- formula-not-decoded -->

- For all k ∈ B i , i ∈ I t -τ ,

<!-- formula-not-decoded -->

̸

- For k = 1 and k / ∈ B i , i ∈ I t -τ ,

<!-- formula-not-decoded -->

Plugging (C.26), (C.27) and (C.28) into (C.1), with facts that ¯ τ = ˜ Θ(1) , λ = o (1) , |⟨ w ( t ) j,r , ξ i ⟩| = ˜ Θ(1) , |⟨ w ( t ) j,r , y i v ⟩| = ˜ Θ(1) , | ℓ ( t ) j,i | = Θ(1) , ϵ = Θ( λη ) and Lemma A.6, we have

<!-- formula-not-decoded -->

- For k ∈ B i , i ∈ I t -τ 0 , τ 0 ∈ { 0 } ∪ [¯ τ ] , where τ 0 represents the number of iterations away from the current iteration t , coordinate k is affected by ξ i sampled at the iteration t -τ 0 since the moving average. We note that if the number of iteration in one epoch n B is less than ¯ τ , the moving average will use some sample x multiply times. We denote T k := { τ 0 + i · n B : i ∈ [ ¯ τ n/B -1] } as the timestamp set involved using noise ξ i (i.e., i ∈ I t -τ for any τ ∈ T k ), and k ∈ B i , τ 0 ≤ n B .

If n B &gt; ¯ τ , in this case we have T k := { τ 0 } for any k ∈ B i , and ξ i was used in iteration t -τ 0 . Then we have

<!-- formula-not-decoded -->

Now we handle β τ 0 1 and β τ 0 2 2 with more care. First we have β τ 0 1 &lt; β τ 0 2 2 and | g ( t ) t -τ 0 ,j,r [ k ] | = ˜ O (1) .

Then if n B ≤ Θ(1) , then β τ 0 1 = Θ(1) and β τ 0 2 2 = Θ(1) since τ 0 &lt; n B . In this case, we have

<!-- formula-not-decoded -->

If n B ≥ Θ(log ϵ -1 ) = Θ(log( λη ) -1 ) = ˜ Θ(1) such that β n 2 B 2 ≤ Θ( λ 2 η 2 ) , then for τ 0 = O (log( λ -1 sσ p )) such that β τ 0 1 ≥ Θ( ϵ ηsσ p ) = Θ( λ sσ p ) , we have

<!-- formula-not-decoded -->

For τ 0 &lt; n B , τ 0 ≥ Θ(log( λη ) -1 ) such that β τ 0 2 2 = O ( λ 2 η 2 ) , we have

<!-- formula-not-decoded -->

since ϵ = Θ( λη ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

This completes the proof.

## C.2.1 Proof of Theorem 4.4

Lemma C.12 (Stage I, pattern learning) . Given the training dataset S , if n B = Θ(1) or n B = o ( sσ p ) , η = 1 / poly( d ) , λ = ˜ Ω( B 2 n ∧ 1) and λ = ˜ O (1) , then for any t ≤ T 0 with T 0 = ˜ O ( 1 ηsσ p ) ,

<!-- formula-not-decoded -->

Proof of Lemma C.12. We prove this Lemma by induction. By Lemma C.1,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second equality we use Taylor expansion and λη · n B = o (1) , and the last equality we have ⟨ w (0) j,r , j · v ⟩ = ˜ Θ( σ 0 ) by Lemma A.3. Now suppose the inequality holds for t = 0 , . . . , t 0 with t 0 ≤ T 0 . We have

<!-- formula-not-decoded -->

since n = o ( sσ p ) . For t = t 0 +1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, we have ⟨ w ( t ) j,r , j · v ⟩ = ˜ O (1) . Then, we prove ⟨ w ( ( t +1) · n B ) y i ,r , ξ i ⟩ = ⟨ w ( t · n B ) y i ,r , ξ i ⟩ + ˜ Θ( ηsσ p ) by induction. By Lemma A.3, we have

<!-- formula-not-decoded -->

which imply that | ℓ (0) j,i | = Θ(1) . Assume that sample ( x i , y i ) is in batch I τ in the first epoch. Then we have

<!-- formula-not-decoded -->

since λη = o (1) , η = o ( σ 0 ) , α = o (1) and s 1 / 2 σ p = ˜ O (1) . Additionally, we have ηs = o ( σ q -1 0 ) and | ξ i [ k ] | ≥ ˜ Θ( σ p ) with high probability. Then | B -1 σ q -1 0 ξ i [ k ] | ≥ ˜ Θ( ηB -1 sσ p | ℓ (0) j,i | ) for i ∈ I τ . Therefore, by Lemma C.11 and A.6, we have

<!-- formula-not-decoded -->

Then, by Lemma C.11 we have the following update according to (B.6), (C.29) and Lemma C.1.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

At the end of the first epoch, we have

<!-- formula-not-decoded -->

This completes the base case for t = 1 . For general t ≤ t 0 with t 0 ≤ T 0 , assuming ⟨ w ( t · n B ) y i ,r , ξ i ⟩ = ⟨ w ( t -1) · n B ) y i ,r , ξ i ⟩ + ˜ Θ( ηsσ p ) . Then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

So we have | ℓ ( t · n B ) j,i | = Θ(1) . Follow the same proof above with t = t 0 +1 , assuming that sample ( x i , y i ) is in batch I t 0 · n/B + τ in the t -th epoch. Then we have

<!-- formula-not-decoded -->

since η = o ( σ 0 ) and α = o (1) . Additionally, we have ηs = o ( σ q -1 0 ) and | ξ i [ k ] | ≥ ˜ Θ( σ p ) with high probability. Then | B -1 ( w (0) j,r [ k ] + tηsσ p ) q -1 ξ i [ k ] | ≥ ˜ Θ( ηB -1 sσ p | ℓ (0) j,i | ) for i ∈ I t 0 · n/B + τ . Therefore, by Lemma C.11 and A.6, we have

<!-- formula-not-decoded -->

Then, by Lemma C.11 we have the following update according to (B.6), (C.30) and Lemma C.1.

<!-- formula-not-decoded -->

By Lemma C.1, we have

<!-- formula-not-decoded -->

At the end of this epoch, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This completes the proof.

Because in the large-batch regime we have n/B = o ( sσ p ) , Lemma C.12 tells us that noise memorization outpaces feature learning. Early in Stage I, feature gradients predominate since σ ′ ( ⟨ w ( t ) y i ,r , y i v ⟩ ) ≫ ασ ′ ( ⟨ w ( t ) y i ,r , ξ i ⟩ ) , given that α = o (1) and weight decay effect is negligible. After a certain number of epochs, however, the noise component grows until ασ ′ ( ⟨ w ( t ) y i ,r , ξ i ⟩ ) ≫ σ ′ ( ⟨ w ( t ) y i ,r , y i v ⟩ ) , at which point feature learning slows, then reverses direction entirely. Lemma C.13 below characterizes this transition in detail.

Lemma C.13 (Stage I, fitting feature noise) . Given the training dataset S , if n B = Θ(1) or n B = o ( sσ p ) , η = 1 / poly( d ) , λ = ˜ Ω( B 2 n ∧ 1) and λ = ˜ O (1) , then if α ≥ ˜ Θ ( ( B n sσ p ) 1 -q ) , for any t ∈ [ T r , T 0 ] with T r = ˜ O ( σ 0 ηsσ p α 1 / ( q -1) ) ≤ T 0 ,

<!-- formula-not-decoded -->

At epoch T 0 , we have (a) w ( T 0 · n B ) j,r [1] = -sgn( j ) · ˜ Ω( n Bsσ p ) ; (b) w ( T 0 · n B ) j,r [ k ] = sgn( ξ i [ k ]) · ˜ Ω( 1 sσ p ) or ± ˜ O ( η ) for k ∈ B i with y i = j ; (c) w ( T 0 · n B ) j,r [ k ] = ± ˜ O ( η ) otherwise.

Proof of Lemma C.13. By Lemma C.12, we have

<!-- formula-not-decoded -->

Hence, there exists some constant C &gt; 0 , for t ∈ [ T r · n B , T 0 · n B ] ,

<!-- formula-not-decoded -->

Then by Lemma A.8, C.11 and A.6, we have

<!-- formula-not-decoded -->

So we conclude that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, at the end of epoch T 0 ,

<!-- formula-not-decoded -->

Multiply j on both side, we get

<!-- formula-not-decoded -->

For w ( T 0 · n B ) j,r [ k ] , where k ∈ B i and i ∈ [ n ] , Lemma C.12 shows that it increases by Θ( η ) in the direction of sgn( ξ i [ k ]) if ⟨ w (0) y i ,r , ξ i ⟩ &gt; 0 , that is,

<!-- formula-not-decoded -->

Otherwise, weight decay drives w ( t ) j,r [ k ] toward zero if it is initially negative, in this case w ( T 0 · n B ) j,r [ k ] ∈ [ -˜ Θ( η ) , ˜ Θ( η )] . For the remaining coordinates, Lemma A.8 implies the gradients are zero, so the updates are dominated by weight decay. Given the fact that T 0 η = ω ( σ 0 ) , we have w ( T 0 · n B ) j,r [ k ] ∈ [ -˜ Θ( η ) , ˜ Θ( η )] . This completes the proof.

Lemma C.13 implies that, by the end of Stage I , the model has fitted the feature noise -αy v . The following Lemma C.14 shows that these pattern persist throughout Stage II , ultimately leading to poor generalization.

Lemma C.14 (Stage II, preserve the noise) . Suppose the same conditions hold as in Lemma C.12 and C.13, for t &gt; T 0 · n B , j ∈ {± 1 } , r ∈ [ m ] , i ∈ [ n ] , let r ∗ = argmax r ∈ [ m ] ⟨ w ( t ) y i ,r , ξ i ⟩ , then ⟨ w ( t ) j,r , j · v ⟩ = -˜ Θ( n Bsσ p ) and ⟨ w ( t ) y i ,r ∗ , ξ i ⟩ = ˜ Θ(1) .

Proof of Lemma C.14. By Lemma C.12, C.11 and (B.6), we have ⟨ w ( t ) -y i ,r , ξ i ⟩ ∈ [ -˜ Θ( ηsσ p ) , ˜ Θ( σ 0 )] . Because if ⟨ w ( t ) -y i ,r , ξ i ⟩ ≥ ˜ Θ( σ 0 ) , then we have

<!-- formula-not-decoded -->

while if ⟨ w ( t ) -y i ,r , ξ i ⟩ &lt; 0 , we have

<!-- formula-not-decoded -->

Now, suppose ⟨ w ( t ) y i ,r ∗ , ξ i ⟩ ≤ ( 1 m log ( ( λη ) -1 -1 )) 1 q , then for ( x i , y i ) with y i = j ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the inequality we use ⟨ w ( t ) j,r , j v ⟩ &lt; 0 . Then, in epoch T a , which contains iteration t , it follows from Lemmas C.11, C.12 and (B.6) that for all t a ∈ [ t, T a + n B ] , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the inequality we use ⟨ w ( t ) -j,r , j v ⟩ ≤ ˜ Θ( n Bsσ p ) , n B = o ( sσ p ) , ⟨ w ( t ) -j,r , ξ i ⟩ ) ≤ ˜ Θ( σ 0 ) . Then by Lemma C.11, C.12 and (B.6), we have

<!-- formula-not-decoded -->

since ηsσ p = o (1) . For ⟨ w ( t ) j,r , j v ⟩ = -˜ Θ( n Bsσ p ) , the same proof applies, since sσ p = o (1) and n B = o ( sσ p ) . If ⟨ w ( t ) y i ,r ∗ , ξ i ⟩ ≥ ( log ( ( λη ) -2 -1 )) 1 q , then for ( x i , y i ) with y i = j , ℓ ( t ) j,i ≤ Θ( λ 2 η 2 ) .

Then by Lemma C.11, C.12 and (B.5), we have

<!-- formula-not-decoded -->

since η = o ( 1 sσ p ) . This completes the proof.

Lemma C.15 (Convergence) . Suppose the same conditions hold as in Lemma C.12, C.13 and C.14, if the step size satisfies η = O ( d -1 / 2 ) , then for any t ,

<!-- formula-not-decoded -->

Proof of Lemma C.15. The proof is similar to Lemma C.8. We aim to prove the convergence of the objective function under the AdamW optimization algorithm in a non-convex setting. Recall the loss function for each data point i is

<!-- formula-not-decoded -->

where W represents the parameter matrix, x i is the input data, y i is the true label, and F j ( W , x i ) are the logits for class j . The total objective is:

<!-- formula-not-decoded -->

Since L i ( W ) is non-convex, we exploit its smoothness with respect to the logits [ F j ( W , x i )] j . Specifically, L i ( W ) is 1-smooth in [ F j ( W , x i )] j due to the properties of the cross-entropy loss. Define:

<!-- formula-not-decoded -->

Using the smoothness property, we apply a second-order Taylor-like expansion around W ( t ) :

<!-- formula-not-decoded -->

This upper bound arises because the second derivative of L i with respect to the logits is bounded by 1, a standard result for cross-entropy loss. The logits are defined as: F j ( W ( t ) , x i ) = ∑ m r =1 [ σ ( ⟨ w ( t ) j,r , y i v ⟩ ) + σ ( ⟨ w ( t ) j,r , ξ i ⟩ )] , where w ( t ) j,r the r -th neuron in j -th output of W ( t ) , σ ( z ) = [ z ] q + is a smooth activation function (e.g., with q ≥ 3 ). By Lemma C.14 and C.1, we have ⟨ w ( t ) j,r , v ⟩ ≤ ˜ Θ(1) and ⟨ w ( t ) j,r , ξ i ⟩ ≤ ˜ Θ(1) , ensuring the local smoothness of σ remains ˜ O (1) between ⟨ w ( t +1) j,r , y i v ⟩ and ⟨ w ( t ) j,r , y i v ⟩ (similar for ⟨ w ( t ) j,r , ξ i ⟩ ). Then with Taylor expansion, we have

<!-- formula-not-decoded -->

where the last inequality we use Lemma C.1 and ∥ w ( t ) j,r ∥ 2 2 ≪ Θ( d ) by Lemma C.13 and C.14. Similarly, we have

<!-- formula-not-decoded -->

Summing over r (with m = ˜ Θ(1) ), we get

<!-- formula-not-decoded -->

Additionally, ∥∇ W F j ( W ( t ) , x i ) ∥ F ≤ ˜ Θ(1) since m = ˜ Θ(1) , ⟨ w ( t ) j,r , y i v ⟩ ≤ ˜ Θ(1) , ⟨ w ( t ) j,r , ξ i ⟩ ≤ ˜ Θ(1) . So we have

<!-- formula-not-decoded -->

Substitute (C.34) and (C.35) into (C.31):

<!-- formula-not-decoded -->

For the full objective:

<!-- formula-not-decoded -->

Substitute (C.36) into (C.37), we have

<!-- formula-not-decoded -->

Take expectation for the stochastic gradient of both side in (C.38),

<!-- formula-not-decoded -->

where we use Lemma C.11 that the update aligns with the gradient's sign for large gradient and the fact that ns 2 σ p = O ( d ) , and Jensen's inequality, Hölder's inequality and n B = o ( sσ p ) , ∥ W ( t ) ∥ ∞ ≤ ˜ Θ( n Bsσ p ) by Lemma C.13 and C.14. This completes the proof.

Lemma C.16 (Generalization of Stochastic AdamW, large-batch) . Suppose the same conditions hold as in Lemma C.15. We have the following results for T = poly( n ) η , with training dataset S

- The training error is zero: err S ( W ( T ) ) = 0 .
- The test error is high: err D ( W ( T ) ) ≥ 1 2 -o (1) .

Proof of Lemma C.16. By Lemma C.14, we have

<!-- formula-not-decoded -->

Recall F j ( W , x ) in Definition 3.2, with ηsσ p = o (1) , α = o (1) , we directly have

<!-- formula-not-decoded -->

since F y i ( W ( T ) , x i ) = ˜ Ω(1) , while F -y i ( W ( T ) , x ) ≤ ˜ Θ( 1 sσ p + σ 0 ) and sσ p = ω (1) . Besides, for test data ( x , y ) ∼ D with x = [ y v ⊤ , ξ ⊤ ] ⊤ , it is clear that with high probability ⟨ w ( T ) y,r , y v ⟩ = -˜ Θ( 1 sσ p ) , then similar as training error, we have

<!-- formula-not-decoded -->

while

<!-- formula-not-decoded -->

Here, ζ y,r and ζ -y,r are independent and symmetric random variables. Therefore, if the term ˜ Θ ( 1 sσ p ) dominates ζ y,r and ζ -y,r , then it is immediate that F y ( W ( T ) , x ) &lt; F -y ( W ( T ) , x ) , since α = o (1) . This implies that large-batch AdamW yields high test error. On the other hand, if ˜ Θ ( 1 sσ p ) is dominated by both ζ y,r and ζ -y,r , then with probability at least 1 / 2 -o (1) , we have F y ( W ( T ) , x ) &lt; F -y ( W ( T ) , x ) , as ζ y,r and ζ -y,r are independent of v . In this case, large-batch AdamW incurs at least 1 / 2 -o (1) test error. Therefore, we conclude:

<!-- formula-not-decoded -->

This completes the proof.

## C.2.2 Proof of Theorem 4.5

Lemma C.17 (Stage I) . Given the training dataset S , if n B ≥ Θ( n 1 / 2 ∨ log ϵ -1 ) and n B = ω ( sσ p ) , η = 1 / poly( d ) , λ = ˜ Ω( B 2 n ∧ 1) and λ = ˜ O (1) , then for any t ≤ T 0 with T 0 = ˜ O ( B ηn ) ,

<!-- formula-not-decoded -->

Proof of Lemma C.17. We prove this Lemma by induction. First, we prove ⟨ w ( ( t +1) · n B ) y i ,r , ξ i ⟩ = ⟨ w ( t · n B ) y i ,r , ξ i ⟩ + ˜ Θ( ηsσ p ) . It is same as Lemma C.12. By Lemma A.3, we have

<!-- formula-not-decoded -->

which imply that | ℓ (0) j,i | = Θ(1) . Assume that sample ( x i , y i ) is in batch I τ in the first epoch. Then we have

<!-- formula-not-decoded -->

since λη = o (1) , η = o ( σ 0 ) , α = o (1) and s 1 / 2 σ p = ˜ O (1) . Additionally, we have ηs = o ( σ q -1 0 ) and | ξ i [ k ] | ≥ ˜ Θ( σ p ) with high probability. Then | B -1 σ q -1 0 ξ i [ k ] | ≥ ˜ Θ( ηB -1 sσ p | ℓ (0) j,i | ) for i ∈ I τ . Therefore, by Lemma C.11 and A.6, we have

<!-- formula-not-decoded -->

Then, by Lemma C.11 we have the following update according to (B.6), (C.29) and Lemma C.1.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

At the end of the first epoch, we have

<!-- formula-not-decoded -->

By Lemma C.1, we have

<!-- formula-not-decoded -->

This completes the base case for t = 1 . For general t ≤ t 0 with t 0 ≤ T 0 , assuming ⟨ w ( t · n B ) y i ,r , ξ i ⟩ = ⟨ w ( t -1) · n B ) y i ,r , ξ i ⟩ + ˜ Θ( ηsσ p ) . Then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It's also obvious that ⟨ w ( t · n B ) j,r , j v ⟩ = ˜ O (1) . So we have | ℓ ( t · n B ) j,i | = Θ(1) . Follow the same proof above with t = t 0 +1 , assuming that sample ( x i , y i ) is in batch I t 0 · n/B + τ in the t -th epoch. Then we have

<!-- formula-not-decoded -->

since η = o ( σ 0 ) and α = o (1) . Additionally, we have ηs = o ( σ q -1 0 ) and | ξ i [ k ] | ≥ ˜ Θ( σ p ) with high probability. Then | B -1 ( w (0) j,r [ k ] + tηsσ p ) q -1 ξ i [ k ] | ≥ ˜ Θ( ηB -1 sσ p | ℓ (0) j,i | ) for i ∈ I t 0 · n/B + τ . Therefore, by Lemma C.11 and A.6, we have

<!-- formula-not-decoded -->

Then, by Lemma C.11 we have the following update according to (B.6), (C.30) and Lemma C.1.

<!-- formula-not-decoded -->

At the end of this epoch, we have

<!-- formula-not-decoded -->

Apply Lemma C.11, we get

<!-- formula-not-decoded -->

since ⟨ w (0) j,r , j · v ⟩ = ˜ Θ( σ 0 ) and λ = ˜ O (1) . We have

<!-- formula-not-decoded -->

So we have ⟨ w ( t ) y i ,r , ξ i ⟩ ≤ ˜ Θ( s 1 / 2 σ p σ 0 + ηsσ p ) for t ∈ [0 , n B ] . Thus, for t ∈ [0 , n B ] , we have σ ′ ( ⟨ w ( t ) j,r , j v ⟩ ) ≫ ασ ′ ( ⟨ w ( t ) j,r , ξ i ) , since α = o (1) . By Lemma A.8 and A.6, we have

<!-- formula-not-decoded -->

Apply Lemma C.11, for t ∈ [0 , n B ] , we get

<!-- formula-not-decoded -->

since ⟨ w (0) j,r , j · v ⟩ = ˜ Θ( σ 0 ) and λ = ˜ O (1) . So we have for t = 0 ,

<!-- formula-not-decoded -->

Now suppose the equality holds for t = 0 , . . . , t 0 with t 0 ≤ T 0 . We have

<!-- formula-not-decoded -->

Since n B = ω ( sσ p ) . We have

<!-- formula-not-decoded -->

Therefore, for t ∈ [( t 0 +1) · n B , ( t 0 +2) · n B ] ,

<!-- formula-not-decoded -->

Apply Lemma C.11, for t = t 0 +1 , we have

<!-- formula-not-decoded -->

since ⟨ w ( t ) j,r , j · v ⟩ = ˜ O (1) and λ = ˜ O (1) . This completes the proof.

<!-- formula-not-decoded -->

Next, we prove ⟨ w ( ( t +1) · n B ) ) j,r , j v ⟩ = ⟨ w ( t · n B ) j,r , j v ⟩ +Θ( η · n B ) . By Lemma A.3, A.8 and C.11, we have

<!-- formula-not-decoded -->

since α = o (1) , and we have η = o ( σ q -1 0 ) . By Lemma A.6, we have

<!-- formula-not-decoded -->

Lemma C.18 (Stage II) . Given the training dataset S , if n B ≥ Θ( n 1 / 2 ∨ log ϵ -1 ) and n B = ω ( sσ p ) , η = 1 / poly( d ) , λ = ˜ Ω( B 2 n ∧ 1) and λ = ˜ O (1) , then for any t &gt; T 0 , j ∈ {± 1 } , r ∈ [ m ] , i ∈ [ n ] , let r ∗ = argmax r ∈ [ m ] ⟨ w ( t ) j,r , j v ⟩ , then ⟨ w ( t ) j,r ∗ , j v ⟩ = ˜ Θ(1) and ⟨ w ( t ) y i ,r , ξ i ⟩ ≤ ˜ Θ( Bsσ p n ) .

Proof of Lemma C.18. By Lemma C.17, we know that ⟨ w ( t ) j,r , j v ⟩ increases at a faster rate than ⟨ w ( t ) y i ,r , ξ i ⟩ since n B = ω ( sσ p ) . We also have ⟨ w ( t ) -y i ,r , ξ i ⟩ ∈ [ -˜ Θ( ηsσ p ) , ˜ Θ( σ 0 )] following Lemma C.14.

Now suppose that ⟨ w ( t ) j,r ∗ , j v ⟩ ≥ ( log ( ( λη ) -2 -1 )) 1 q , then for ( x i , y i ) with y i = j ,

<!-- formula-not-decoded -->

where the inequality we use ⟨ w ( t ) -j,r , j v ⟩ &lt; 0 , ⟨ w ( t ) -j,r , ξ i ⟩ ) ≤ ˜ Θ( σ 0 ) . Then by Lemma C.11, C.17 and (B.5), we have

<!-- formula-not-decoded -->

since η = o (1) . Similarly, if ⟨ w (0) y i ,r , ξ i ⟩ ≥ ˜ Θ( σ 0 )

<!-- formula-not-decoded -->

Otherwise, ⟨ w ( t ) y i ,r , ξ i ⟩ ∈ [ -˜ Θ( σ 0 ) , ˜ Θ( σ 0 )] and satisfies ⟨ w ( t ) y i ,r , ξ i ⟩ ≤ ˜ Θ( Bsσ p n ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the inequality we use ⟨ w ( t ) -j,r , j v ⟩ &lt; 0 , ⟨ w ( t ) -j,r , ξ i ⟩ ) ≤ ˜ Θ( σ 0 ) . Then by Lemma C.11, C.17 and (B.5), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

since λ = ˜ O (1) . This completes the proof.

Lemma C.19 (Convergence) . Suppose the same conditions hold as in Lemma C.17 and C.18, if the step size satisfies η = O ( d -1 / 2 ) , then for any t ,

<!-- formula-not-decoded -->

Proof of Lemma C.19. The proof is same as Lemma C.15. We aim to prove the convergence of the objective function under the AdamW optimization algorithm in a non-convex setting. Recall the loss function for each data point i is

<!-- formula-not-decoded -->

where W represents the parameter matrix, x i is the input data, y i is the true label, and F j ( W , x i ) are the logits for class j . The total objective is:

<!-- formula-not-decoded -->

Since L i ( W ) is non-convex, we exploit its smoothness with respect to the logits [ F j ( W , x i )] j . Specifically, L i ( W ) is 1-smooth in [ F j ( W , x i )] j due to the properties of the cross-entropy loss. Define:

<!-- formula-not-decoded -->

Using the smoothness property, we apply a second-order Taylor-like expansion around W ( t ) :

<!-- formula-not-decoded -->

This upper bound arises because the second derivative of L i with respect to the logits is bounded by 1, a standard result for cross-entropy loss. The logits are defined as: F j ( W ( t ) , x i ) = ∑ m r =1 [ σ ( ⟨ w ( t ) j,r , y i v ⟩ ) + σ ( ⟨ w ( t ) j,r , ξ i ⟩ )] , where w ( t ) j,r the r -th neuron in j -th output of W ( t ) , σ ( z ) = [ z ] q + is a smooth activation function (e.g., with q ≥ 3 ). By Lemma C.18 and C.1, we have ⟨ w ( t ) j,r , v ⟩ ≤ ˜ Θ(1) and ⟨ w ( t ) j,r , ξ i ⟩ ≤ ˜ Θ(1) , ensuring the local smoothness of σ remains ˜ O (1) between ⟨ w ( t +1) j,r , y i v ⟩ and ⟨ w ( t ) j,r , y i v ⟩ (similar for ⟨ w ( t ) j,r , ξ i ⟩ ). Then with Taylor expansion, we have

<!-- formula-not-decoded -->

where the last inequality we use Lemma C.1 and ∥ w ( t ) j,r ∥ 2 2 ≪ Θ( d ) by C.18. Similarly, we have

<!-- formula-not-decoded -->

Summing over r (with m = ˜ Θ(1) ), we get

<!-- formula-not-decoded -->

Additionally, ∥∇ W F j ( W ( t ) , x i ) ∥ F ≤ ˜ Θ(1) since m = ˜ Θ(1) , ⟨ w ( t ) j,r , y i v ⟩ ≤ ˜ Θ(1) , ⟨ w ( t ) j,r , ξ i ⟩ ≤ ˜ Θ(1) . So we have

<!-- formula-not-decoded -->

Substitute (C.44) and (C.45) into (C.41):

<!-- formula-not-decoded -->

For the full objective:

<!-- formula-not-decoded -->

Substitute (C.46) into (C.47), we have

<!-- formula-not-decoded -->

Take expectation for the stochastic gradient of both side in (C.48),

<!-- formula-not-decoded -->

where we use Lemma C.11 that the update aligns with the gradient's sign for large gradient and the fact that ns 2 σ p = O ( d ) , and Jensen's inequality, Hölder's inequality and λ = ˜ O (1) , ∥ W ( t ) ∥ ∞ ≤ ˜ Θ(1) by Lemma C.18. This completes the proof.

Lemma C.20 (Generalization of Stochastic AdamW, mini-batch) . Suppose the same conditions hold as in Lemma C.19. We have the following results for T = poly( n ) η , with training dataset S

- The training error is zero: err S ( W ( T ) ) = 0 .
- The test error is near-zero: err D ( W ( T ) ) = o (1) .

Proof of Lemma C.20. By Lemma C.18, we have

<!-- formula-not-decoded -->

Recall F j ( W , x ) in Definition 3.2, with α = o (1) , we directly have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

since F y i ( W ( T ) , x i ) = ˜ Ω(1) , while F -y i ( W ( T ) , x i ) ≤ ˜ Θ( σ 0 + α ) .

For test data ( x , y ) ∼ D with x = [ y v ⊤ , ξ ⊤ ] ⊤ , it is clear that with high probability ⟨ w ( T ) y,r ∗ , y v ⟩ = ˜ Θ(1) . Let B = supp ( ξ ) , ∥ w B ∥ 2 2 = ∑ k ∈B w y,r [ k ] 2 , ζ y,r = ∑ k ∈B w y,r [ k ] · ξ [ k ] ∼ N (0 , ∥ w B ∥ 2 2 · σ 2 p ) , then we have

<!-- formula-not-decoded -->

We have

So the upper bound of ∥ w B ∥ 2 2 is

<!-- formula-not-decoded -->

Finally, with high probability while

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

This implies that mini-batch AdamW can achieve nearly zero test error. This completes the proof.

## C.2.3 Proof of Corollary 4.6

By Conditions A.1 and A.2, along with Definition 3.2, we know that d = poly( n ) , and hence

<!-- formula-not-decoded -->

This directly implies that the effective weight decay parameter for Adam satisfies

<!-- formula-not-decoded -->

This completes the proof.

<!-- formula-not-decoded -->

since n B = ω ( n 1 / 2 ) . The same result holds for ζ -y,r . Then, we have

<!-- formula-not-decoded -->

̸

Now we calculate the upper bound of ∥ w B ∥ 2 2 . By Lemma A.4, we know ⟨ ξ i , ξ j ⟩ = 0 for i = j, i, j ∈ [ n ] . Then let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where c = ˜ Θ(1) . By Lemma C.18, we have

<!-- formula-not-decoded -->

Since ∥ ξ i ∥ 2 2 ∼ σ 2 p χ 2 s and s = ω (log n ) , we have ∥ ξ i ∥ 2 2 = ˜ Θ( sσ 2 p ) with high probability. Hence, λ + min (Σ) = min i ∥ ξ i ∥ 2 2 = ˜ Θ( sσ 2 p ) . With a little abuse of notation, we have

<!-- formula-not-decoded -->

By Lemma C.11, λ = ˜ Ω( B 2 n ∧ 1) and ληT = ω (1) , we have

<!-- formula-not-decoded -->

## D Experimental details and results

This section presents the complete details of our experiments.

## D.1 Experimental Details for Real-world Data

For the real-world experiments in Figures 1 and 2, we use the CIFAR-10 dataset, VGG16 and ResNet18 architectures, and the Adam and AdamW optimizers, all implemented in PyTorch. We do not use data augmentation in order to avoid any additional regularization effects.

In Figure 1, we report the test error as a function of batch size. The batch sizes considered are { 16 , 32 , 64 , 256 , 1024 , 4096 , 8192 } , with training conducted for 100 epochs. The weight decay is set to 5 × 10 -4 for Adam and 1 × 10 -2 for AdamW; the momentum parameters are fixed at ( β 1 , β 2 ) = (0 . 9 , 0 . 99) for both optimizers. Each configuration is evaluated with three learning rates: { 5 × 10 -4 , 1 × 10 -4 , 1 × 10 -5 }, and we report the best test performance for each batch size. All experiments can be run within one hour on a single RTX 4090 GPU. The only exception is training ResNet18 with a batch size of 8192, which requires three GPUs due to memory constraints.

Figure 1(a) presents the test error versus batch size for Adam with VGG16 and ResNet18, while Figure 1(b) shows the corresponding results for AdamW. Both demonstrate that test performance degrades as batch size increases, which is consistent with our theoretical findings in Section 4, showing that small-batch Adam and AdamW outperform their large-batch counterparts.

In Figure 2, we report the test error as a function of weight decay λ for Adam and AdamW, using VGG16 (Figure 2(a)) and ResNet18 (Figure 2(b)). We fix the batch size to 16, the learning rate to 1 × 10 -4 , and set ( β 1 , β 2 ) = (0 . 9 , 0 . 99) . The weight decay values for Adam are { 1 × 10 -1 , 5 × 10 -2 , 1 × 10 -2 , 5 × 10 -3 , 1 × 10 -3 , 5 × 10 -4 , 1 × 10 -4 , 5 × 10 -5 , 1 × 10 -5 , 5 × 10 -6 , 1 × 10 -6 , 5 × 10 -7 } , and for AdamW are { 5 × 10 -1 , 1 × 10 -1 , 5 × 10 -2 , 1 × 10 -2 , 5 × 10 -3 , 1 × 10 -3 , 5 × 10 -4 , 1 × 10 -4 } . All models are trained for 100 epochs.

Figure 2(a) shows results for training VGG16, and Figure 2(b) for ResNet18, both using Adam and AdamW. For a fair comparison, we scale the weight decay λ of AdamW by a factor of 1 / 25 . The results show that Adam suffers from poor generalization under large weight decay values (e.g., λ &gt; 0 . 05 ), while AdamW maintains stable performance even with larger weight decays (e.g., λ = 0 . 5 ), which aligns with our theoretical results in Section 4.

## D.2 Experimental Details for Synthetic Data

For the data model defined in Definition 3.1, we set the input dimension to d = 1000 and the number of training samples to n = 200 , consisting of 100 positive and 100 negative samples. The sparsity level is set to s = 0 . 1 d = 100 , and the noise strength is σ p = 1 / √ s = 0 . 1 . The feature noise strength is set to α = 0 . 2 , and the model weights are initialized with standard deviation σ 0 = 0 . 01 . The network, defined in Definition 3.2, has width m = 20 .

All synthetic experiments are trained for T = 10 4 epochs with a learning rate of η = 5 × 10 -5 , and evaluated on a test dataset of size 10 4 . For Adam and AdamW optimizers, we adopt the default momentum hyperparameters β 1 = 0 . 9 and β 2 = 0 . 999 .

We primarily focus on the following metrics:

- Training error: err S ( W ) .
- Test error: err D ( W ) .
- Feature learning: max r ∈ [ m ] ⟨ w j,r , j v ⟩ .
- Noise memorization: min i ∈ [ n ]: y i = j max r ∈ [ m ] ⟨ w j,r , ξ i ⟩ or max i ∈ [ n ]: y i = j max r ∈ [ m ] ⟨ w j,r , ξ i ⟩ .

Large-batch Adam vs. Mini-batch Adam. We set λ = 1 × 10 -5 for both large-batch Adam (batch size B = 100 ) and mini-batch Adam (batch size B = 2 ). Table 1 presents the training and test errors of the solutions obtained by the two training methods. Although both large-batch and mini-batch Adam achieve zero training error, their generalization performance differs significantly. Specifically, large-batch Adam suffers from high test error (greater than 0 . 5 ), while mini-batch Adam achieves zero test error. This observation verifies Theorems 4.1 and 4.2.

Table 1: Training and test errors of Adam with large ( B = 100 ) and mini-batch ( B = 2 ) settings.

| Batch size     |   B = 100 |   B = 2 |
|----------------|-----------|---------|
| Training error |    0      |       0 |
| Test error     |    0.9545 |       0 |

Figure 3: Feature learning and noise memorization of Adam in the training.

<!-- image -->

Moreover, Figure 3(a) illustrates the dynamics of feature learning, measured by max r ∈ [ m ] ⟨ w j,r , j v ⟩ , and noise memorization, measured by min i ∈ [ n ]: y i = j max r ∈ [ m ] ⟨ w j,r , ξ i ⟩ , under large-batch Adam. The results are consistent with Figure 2 in Zou et al. (2023b). Figure 3(b) shows the corresponding dynamics for mini-batch Adam, where feature learning max r ∈ [ m ] ⟨ w j,r , j v ⟩ increases steadily, while noise memorization max i ∈ [ n ]: y i = j max r ∈ [ m ] ⟨ w j,r , ξ i ⟩ remains suppressed at the end of Pattern Learning Stage. In the subsequent Regularization Stage, feature learning saturates at a stable threshold and stops increasing. This behavior is consistent with Lemma C.7.

Large-batch AdamW vs. Mini-batch AdamW. We set λ = 0 . 01 for both large-batch AdamW (batch size B = 100 ) and mini-batch AdamW (batch size B = 2 ). Table 2 reports the training and test errors for both training methods. Although both large-batch and mini-batch AdamW achieve zero training error, their test performance differs significantly: large-batch AdamW suffers from high test error (exceeding 0 . 5 ), while mini-batch AdamW attains zero test error. This observation supports Theorems 4.4 and 4.5.

Figure 4(a) illustrates the dynamics of feature learning, measured by max r ∈ [ m ] ⟨ w j,r , j v ⟩ , and noise memorization, measured by min i ∈ [ n ]: y i = j max r ∈ [ m ] ⟨ w j,r , ξ i ⟩ , under large-batch AdamW. Initially, feature learning increases, but it is eventually flipped by noise memorization, which grows at a faster rate. As a result, the model begins fitting to the feature noise, which is negatively aligned with the true feature direction. Specifically, noise memorization increases rapidly during the Pattern Learning Stage and saturates at a logarithmic rate in the Regularization Stage. These behaviors are consistent with Lemmas C.12, C.13, and C.14.

Figure 4(b) shows the corresponding dynamics for mini-batch AdamW. Feature learning increases steadily and remains unaffected by noise memorization during the Pattern Learning Stage. In the Regularization Stage, feature learning saturates at a stable threshold, which causes the gradient to become small and consequently suppresses further growth of noise memorization (recall that ξ i [1] = -αy i ). This behavior is consistent with Lemmas C.17 and C.18.

Large weight decay regularization λ hinders Adam training. We repeat the experiments from Large-batch Adam vs. Mini-batch Adam using a larger weight decay parameter λ = 0 . 05 , and

Table 2: Training and test errors of AdamW with large ( B = 100 ) and mini-batch ( B = 2 ) settings.

| Batch size                | B = 100   | B = 2   |
|---------------------------|-----------|---------|
| Training error Test error | 0 0.5485  | 0 0     |

Figure 6: Training error and test error over epochs of AdamW training with λ = 0 . 5 .

<!-- image -->

those from Large-batch AdamW vs. Mini-batch AdamW with λ = 0 . 5 . Figure 5 reports the training accuracy over epochs for Adam, while Figure 6 shows the same for AdamW. It can be observed that Adam fails to train under large weight decay. In contrast, AdamW remains robust and achieves results consistent with those in Large-batch AdamW vs. Mini-batch AdamW , even with a larger λ = 0 . 5 . These results support Corollaries 4.3 and 4.6.

## D.3 Additional Experimental Results

Error bars across random seeds, Figures 7 and 8. We provide additional results to support our theoretical findings. To assess statistical significance, we repeat the CIFAR-10 experiments from Figures 1 and 2 with five random seeds (0-4), using the same settings as in Section D.1. Figures 7 and 8 report the results, with error bars denoting the standard deviation across runs. The results confirm that both Adam and AdamW degrade in performance as the batch size increases, and that Adam is more sensitive to weight decay than AdamW.

ADAM

<!-- image -->

T est Error vs. Batch Size

- (a) Test error vs. batch size under Adam

ADAMW

T est Error vs. Batch Size

(b) Test error vs. batch size under AdamW

Figure 7: Error bars across seeds: Test error vs. batch size for VGG16 and ResNet18 on CIFAR-10.

<!-- image -->

Figure 8: Error bars across seeds: Test error vs. weight decay (batch size = 16), comparing Adam and AdamW.

<!-- image -->

- (a) Test error vs. batch size under Adam

(b) Test error vs. batch size under AdamW

Figure 9: Error bars across ( β 1 , β 2 ) : Test error vs. batch size for VGG16 and ResNet18 on CIFAR-10.

Sensitivity to momentum parameters ( β 1 , β 2 ) , Figures 9 and 10. We further study the sensitivity of Adam and AdamW to the momentum parameters ( β 1 , β 2 ) , which are treated as constants in our theory. We sweep over β 1 ∈ { 0 , 0 . 5 , 0 . 9 } and β 2 ∈ { 0 . 5 , 0 . 9 , 0 . 95 } , yielding 8 valid combinations under β 2 1 &lt; β 2 , plus the standard setting ( β 1 , β 2 ) = (0 . 9 , 0 . 99) , for a total of 9 configurations. Figures 9 and 10 report the results, with error bars showing the standard deviation across the 9 runs. The findings again confirm that both Adam and AdamW suffer performance degradation as batch size increases, and that Adam is more sensitive to weight decay than AdamW.

Large-scale vision experiments with ResNet-50 on ImageNet-1K subset, Figures 11 and 12. To further validate our theory, we conduct large-scale experiments on ImageNet-1K. We construct a subset by randomly sampling 100 training images per class (seed=0), ensuring a controlled largebatch regime ( n B = Θ(1) ) while keeping computation feasible. ResNet-50 is trained for 90 epochs

Figure 10: Error bars across ( β 1 , β 2 ) : Test error vs. weight decay (batch size = 16), comparing Adam and AdamW.

<!-- image -->

Figure 11: ImageNet-1K subset: Top-5 validation error vs. batch size for Adam and AdamW with ResNet-50.

<!-- image -->

with standard ImageNet preprocessing. We report top-5 validation error against batch size (Figure 11) and weight decay (Figure 12), comparing Adam and AdamW.

For Figure 11, we set learning rate η = 1 × 10 -4 , ( β 1 , β 2 ) = (0 . 9 , 0 . 99) , and weight decay λ = 1 × 10 -4 for Adam and λ = 1 × 10 -2 for AdamW. Batch sizes are { 64 , 128 , 256 , 512 , 1024 , 2048 , 3072 , 4096 , 8192 , 16384 , 32768 } .

For Figure 12, we fix B = 64 , η = 1 × 10 -3 , and ( β 1 , β 2 ) = (0 . 9 , 0 . 99) . Weight decay values for Adam are { 5 × 10 -7 , 1 × 10 -6 , 5 × 10 -6 , 1 × 10 -5 , 5 × 10 -5 , 1 × 10 -4 , 5 × 10 -4 , 1 × 10 -3 , 5 × 10 -3 , 1 × 10 -2 , 5 × 10 -2 , 1 × 10 -1 } , and for AdamW are { 1 × 10 -4 , 5 × 10 -4 , 1 × 10 -3 , 5 × 10 -3 , 1 × 10 -2 , 5 × 10 -2 , 1 × 10 -1 , 5 × 10 -1 } .

The results again confirm that both optimizers degrade as batch size increases, and that Adam is more sensitive to weight decay than AdamW.

Figure 12: ImageNet-1K subset: Top-5 validation error vs. weight decay for Adam and AdamW with ResNet-50 (batch size = 64).

<!-- image -->